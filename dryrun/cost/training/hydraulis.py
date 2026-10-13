"""Hydraulis Appendix C inspired V1 training latency and memory model."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from dryrun.cost.artifact import ArtifactEntry, CostArtifact, FitMetrics
from dryrun.profiler.io import profile_digest
from dryrun.profiler.schema import ModelDescriptor, Parallelism, ProfileRecord, TrainingPayload

from .activation import ActivationGeometry, activation_bytes
from .base import (
    TrainingCostEstimate,
    TrainingCostModel,
    TrainingLatencyModel,
    TrainingMemoryModel,
    TrainingParallelism,
    TrainingWorkload,
)
from .fitting import (
    check_feature_range,
    scaled_nonnegative_fit,
    split_records_by_run,
    validate_extrapolate_names,
    validation_is_in_sample,
)
from .inventory import dtype_bytes, heaviest_stage_parameters, parameter_inventory
from .latency import (
    LATENCY_TERM_COUNT,
    LatencyFeatures,
    check_latency_rank,
    latency_coefficients,
    latency_feature_ranges,
    latency_features,
    latency_from_coefficients,
    robust_step_time,
    step_time_dispersion,
)


@dataclass(frozen=True)
class _CellKey:
    tp: int
    pp: int
    cp: int
    # Expert parallelism, part of the key for MoE blocks: a different EP means different
    # all-to-all traffic and a different expert count per rank, so it is a separate
    # measurement rather than a slice of the same one. 1 for every dense model.
    ep: int
    schedule: str
    recompute: str
    optimizer: str
    parameter_dtype: str
    gradient_dtype: str
    optimizer_state_dtype: str
    distributed_optimizer: bool
    parameter_count: int
    geometry: ActivationGeometry

    @property
    def parallelism(self) -> Parallelism:
        return Parallelism(tp=self.tp, pp=self.pp, dp=1, cp=self.cp, ep=self.ep)

    def selector(self, component: str) -> dict[str, Any]:
        return {
            "component": component,
            "schedule": self.schedule,
            "recompute": self.recompute,
            "optimizer": self.optimizer,
            "parameter_dtype": self.parameter_dtype,
            "gradient_dtype": self.gradient_dtype,
            "optimizer_state_dtype": self.optimizer_state_dtype,
            "distributed_optimizer": self.distributed_optimizer,
        }


def _training_records(records: list[ProfileRecord]) -> tuple[dict[_CellKey, list[ProfileRecord]], list[ProfileRecord]]:
    materialized = [record for record in records if record.record_type == "training"]
    if not materialized:
        raise ValueError("No training profiling records are available.")
    groups: dict[_CellKey, list[ProfileRecord]] = defaultdict(list)
    for record in materialized:
        payload = record.payload
        if not isinstance(payload, TrainingPayload):
            continue
        if record.parallelism.cp != 1 or payload.schedule.lower() != "1f1b":
            continue
        parameter_count = payload.parameter_count or record.model.parameter_count
        if parameter_count is None:
            raise ValueError("Hydraulis memory fitting requires parameter_count.")
        key = _CellKey(
            tp=record.parallelism.tp,
            pp=record.parallelism.pp,
            cp=record.parallelism.cp,
            ep=record.parallelism.ep,
            schedule=payload.schedule.lower(),
            recompute=payload.recompute,
            optimizer=payload.optimizer,
            parameter_dtype=payload.parameter_dtype,
            gradient_dtype=payload.gradient_dtype,
            optimizer_state_dtype=payload.optimizer_state_dtype,
            distributed_optimizer=payload.distributed_optimizer,
            parameter_count=int(parameter_count),
            geometry=ActivationGeometry.from_model(
                record.model,
                activation_dtype_bytes=dtype_bytes(payload.parameter_dtype),
            ),
        )
        groups[key].append(record)
    if not groups:
        raise ValueError("No CP=1, 1F1B training records are available.")
    return dict(groups), materialized


def _latency_features(payload: TrainingPayload, pp: int) -> LatencyFeatures:
    """Build the structured latency design row for one payload.

    `pp` is the record's own pipeline degree, not the cell's, because a latency
    cell now pools several. The features are per-layer, so each record must be
    normalised by the pipeline it actually ran on.
    """
    return latency_features(
        payload.microbatches,
        TrainingParallelism(tp=1, pp=pp, dp=1, cp=1),
    )


def _pool_latency_cells(
    groups: dict[_CellKey, list[ProfileRecord]],
) -> dict[_CellKey, list[ProfileRecord]]:
    """Merge cells that differ only in `pp` into one latency cell per TP group.

    The latency form charges `nmb + pp - 1` slots at `1/pp` of the model's cost
    per slot, both known from the topology, so measuring one PP per TP is enough
    to predict the rest. Pooling makes that concrete: the fit sees every PP at
    once and has to satisfy all of them with one set of coefficients.

    The surviving key carries `pp=1` so the entry is addressed by TP alone. That
    pairs with `CostArtifact.entry_for(..., ignore_pp=True)`, which is how the
    predictor finds it for any requested PP.
    """
    pooled: dict[_CellKey, list[ProfileRecord]] = defaultdict(list)
    for key, records in groups.items():
        # `pp=1` is a label meaning "any pipeline degree", not a claim that these
        # records ran without pipelining.
        pooled[replace(key, pp=1)].extend(records)
    return dict(pooled)


def _activation_feature(
    payload: TrainingPayload,
    pp: int,
    geometry: ActivationGeometry,
    parallelism: Parallelism,
) -> float:
    """
    Return predicted activation bytes for one payload.

    This is an architecture-derived quantity in bytes, not an opaque token count, so
    the fitted coefficient is a dimensionless correction near one rather than a
    per-model scale factor.
    """
    return activation_bytes(
        geometry,
        payload.microbatches,
        TrainingParallelism(tp=parallelism.tp, pp=pp, dp=parallelism.dp, cp=parallelism.cp, ep=parallelism.ep),
    )


def _latency_arrays(records: list[ProfileRecord]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one design row per candidate, not per step.

    Every retained step used to be its own row, so a candidate whose steps were noisy
    outvoted a clean one. Each profiled configuration should carry equal weight, and its
    steps are repeat measurements of one quantity: reduced by a median, which is
    unbiased against the right-skew that stragglers produce.

    Each row is normalised by its own record's `pp`, since a latency cell pools
    several pipeline degrees.
    """
    rows: list[tuple[float, float, float, float]] = []
    targets: list[float] = []
    for record in records:
        payload = record.payload
        if record.run.status != "success" or not isinstance(payload, TrainingPayload):
            continue
        if not payload.step_times_s:
            continue
        rows.append(_latency_features(payload, record.parallelism.pp).design_row())
        targets.append(robust_step_time(payload.step_times_s))
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def _fit_latency_entry(
    key: _CellKey,
    records: list[ProfileRecord],
    *,
    validation_fraction: float,
    require_full_rank: bool,
) -> ArtifactEntry | None:
    successful = [
        record
        for record in records
        if record.run.status == "success"
        and isinstance(record.payload, TrainingPayload)
        and record.payload.step_times_s
    ]
    if not successful:
        return None
    train_records, validation_records = split_records_by_run(
        successful,
        validation_fraction=validation_fraction,
    )
    train_x, train_y = _latency_arrays(train_records)
    validation_x, validation_y = _latency_arrays(validation_records)
    if len(train_y) < LATENCY_TERM_COUNT:
        raise ValueError(
            f"Hydraulis latency needs at least {LATENCY_TERM_COUNT} candidates for {key!r}."
        )
    parameters, inliers, rank = scaled_nonnegative_fit(train_x, train_y)
    if require_full_rank:
        check_latency_rank(
            rank, backend="Hydraulis", key=key, slots_feature="the microbatch count"
        )
    train_predicted = train_x @ parameters
    validation_predicted = validation_x @ parameters
    all_features = [
        _latency_features(record.payload, record.parallelism.pp) for record in successful
    ]
    dispersions = [
        step_time_dispersion(record.payload.step_times_s) for record in successful
    ]
    feature_ranges = latency_feature_ranges(
        all_features,
        slots_feature="pipeline_slots",
        extra={
            "micro_batch_size": (
                float(min(record.payload.micro_batch_size for record in successful)),
                float(max(record.payload.micro_batch_size for record in successful)),
            ),
        },
    )
    return ArtifactEntry(
        parallelism=key.parallelism,
        selector=key.selector("latency"),
        coefficients=latency_coefficients(parameters),
        train_metrics=FitMetrics.from_predictions(
            train_y,
            train_predicted,
            n_outliers=int(np.sum(~inliers)),
            design_rank=rank,
        ),
        validation_metrics=FitMetrics.from_predictions(
            validation_y,
            validation_predicted,
            design_rank=rank,
        ),
        feature_ranges=feature_ranges,
        diagnostics={
            # Spelled in the same terms `LatencyFeatures.design_row` actually builds, which
            # differ from the earlier `slots * max_microbatch_tokens` wording in two ways.
            #
            # The token terms are SUMS over the rank's microbatches plus a `pp - 1` bubble
            # charged at the largest: a stage runs every microbatch it is given, so a step
            # costs the sum of their costs, and only the slots it cannot fill are charged
            # at the maximum. And every term carries a `1 / PP` stage share, because the
            # fitted coefficients price the whole model while a stage holds one slice.
            #
            # The two forms coincide on the uniform microbatches profiling measures, so the
            # old string still fitted correctly; it just could not be reimplemented from.
            "formula": (
                "per_step_overhead "
                "+ (token_cost * (sum_over_microbatches(sum_tokens) "
                "+ (PP - 1) * max_microbatch_tokens) "
                "+ token_quadratic_cost * (sum_over_microbatches(sum_squared_tokens) "
                "+ (PP - 1) * max_microbatch_sum_squared_tokens) "
                "+ microbatch_overhead * (num_microbatches + PP - 1)) / PP"
            ),
            "pipeline_latency": "derived-per-layer-1f1b",
            "profiled_pipeline_degrees": sorted(
                {record.parallelism.pp for record in successful}
            ),
            "dp_latency": "ideal-local-workload-scaling",
            "step_aggregation": "median-per-candidate",
            "validation_fraction": validation_fraction,
            # `split_records_by_run` hands back the same list twice at fraction zero or
            # with a single run, so `validation_metrics` is then the training metric.
            "validation_is_in_sample": validation_is_in_sample(
                successful,
                validation_fraction=validation_fraction,
            ),
            # Relative median absolute deviation of the retained step times. Above
            # roughly 0.05 the measurement, not the model, limits accuracy.
            "max_step_dispersion": max(dispersions) if dispersions else 0.0,
            "median_step_dispersion": float(np.median(dispersions)) if dispersions else 0.0,
        },
    )


def _static_bytes(
    *,
    parameter_count: int,
    geometry: ActivationGeometry,
    parallelism: TrainingParallelism,
    parameter_dtype: str,
    gradient_dtype: str,
    optimizer_state_dtype: str,
    distributed_optimizer: bool,
) -> float:
    """
    Return static state bytes on the most loaded rank.

    Peak memory is set by the heaviest pipeline stage, which holds the embedding or
    the output projection on top of its share of transformer layers.
    """
    return parameter_inventory(
        parameter_count=parameter_count,
        parallelism=parallelism,
        parameter_dtype=parameter_dtype,
        gradient_dtype=gradient_dtype,
        optimizer_state_dtype=optimizer_state_dtype,
        distributed_optimizer=distributed_optimizer,
        local_parameters=heaviest_stage_parameters(
            parameter_count=parameter_count,
            num_layers=geometry.num_layers,
            hidden_size=geometry.hidden_size,
            vocab_size=geometry.vocab_size,
            parallelism=parallelism,
        ),
    ).total_bytes


def _record_inventory(
    key: _CellKey,
    record: ProfileRecord,
) -> float:
    return _static_bytes(
        parameter_count=key.parameter_count,
        geometry=key.geometry,
        parallelism=TrainingParallelism.from_profile_parallelism(record.parallelism),
        parameter_dtype=key.parameter_dtype,
        gradient_dtype=key.gradient_dtype,
        optimizer_state_dtype=key.optimizer_state_dtype,
        distributed_optimizer=key.distributed_optimizer,
    )


def _memory_arrays(
    key: _CellKey,
    records: list[ProfileRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[float] = []
    targets: list[float] = []
    static_bytes: list[float] = []
    for record in records:
        payload = record.payload
        if record.run.status != "success" or not isinstance(payload, TrainingPayload):
            continue
        peak = payload.peak_allocated_bytes
        if peak is None:
            continue
        static = _record_inventory(key, record)
        features.append(
            _activation_feature(payload, key.pp, key.geometry, record.parallelism)
        )
        targets.append(float(peak))
        static_bytes.append(static)
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        np.asarray(static_bytes, dtype=np.float64),
    )


def _fit_memory_entry(
    key: _CellKey,
    records: list[ProfileRecord],
    *,
    validation_fraction: float,
    require_full_rank: bool,
) -> ArtifactEntry | None:
    """
    Fit peak allocated bytes as analytic static state plus a corrected activation term.

    The target is `peak_allocated_bytes`, the live-tensor high-water mark, not
    `peak_reserved_bytes`. Reserved memory is the caching allocator's segment pool: it
    is quantized, sticky, and under full recompute it barely responds to sequence
    length because a longer activation reuses segments the static state already
    caused to be reserved. Measured on 8x H20 at TP1/PP1/DP8, doubling the sequence
    length from 512 to 1024 moved allocated by +304 MiB and reserved by -4 MiB, so
    fitting reserved produced a near-zero coefficient of determination. Reserved is
    still recorded, and callers that need a provisioning number should add an
    allocator-overhead margin on top of this prediction.

    Both terms are in bytes, so `activation_scale` is a dimensionless correction that
    should land near one; a value far from one means the geometry is wrong rather than
    the machine being unusual.
    """
    successful = [
        record
        for record in records
        if record.run.status == "success"
        and isinstance(record.payload, TrainingPayload)
        and record.payload.peak_allocated_bytes is not None
    ]
    oom_records = [record for record in records if record.run.status == "oom"]
    if not successful and not oom_records:
        return None
    if successful:
        train_records, validation_records = split_records_by_run(
            successful,
            validation_fraction=validation_fraction,
        )
        train_feature, train_target, train_static = _memory_arrays(key, train_records)
        validation_feature, validation_target, validation_static = _memory_arrays(
            key,
            validation_records,
        )
        train_residual = np.maximum(train_target - train_static, 0)
        design = np.column_stack((train_feature, np.ones(len(train_feature))))
        parameters, inliers, rank = scaled_nonnegative_fit(design, train_residual)
        if require_full_rank and rank < 2:
            raise ValueError(
                f"Hydraulis memory design matrix is rank {rank}, expected 2 for {key!r}. "
                "Add at least two activation-load points."
            )
    else:
        train_feature = np.asarray([], dtype=np.float64)
        train_target = np.asarray([], dtype=np.float64)
        train_static = np.asarray([], dtype=np.float64)
        validation_feature = train_feature
        validation_target = train_target
        validation_static = train_static
        parameters = np.asarray([0.0, 0.0])
        inliers = np.asarray([], dtype=bool)
        rank = 0

    constrained_oom = 0
    intercept_increase = 0.0
    for record in oom_records:
        payload = record.payload
        if not isinstance(payload, TrainingPayload) or record.hardware.gpu_memory_bytes is None:
            continue
        feature = _activation_feature(payload, key.pp, key.geometry, record.parallelism)
        static = _record_inventory(key, record)
        required_residual = record.hardware.gpu_memory_bytes + 1 - static
        deficit = required_residual - (parameters[0] * feature + parameters[1])
        if deficit > intercept_increase:
            intercept_increase = deficit
        constrained_oom += 1
    parameters[1] += max(intercept_increase, 0.0)

    train_predicted = train_static + parameters[0] * train_feature + parameters[1]
    validation_predicted = (
        validation_static + parameters[0] * validation_feature + parameters[1]
    )
    activation_features = [
        _activation_feature(record.payload, key.pp, key.geometry, record.parallelism)
        for record in successful + oom_records
        if isinstance(record.payload, TrainingPayload)
    ]
    dp_values = [float(record.parallelism.dp) for record in successful + oom_records]
    return ArtifactEntry(
        parallelism=key.parallelism,
        selector=key.selector("memory"),
        coefficients={
            "activation_scale": float(parameters[0]),
            "residual_constant": float(parameters[1]),
            "parameter_count": float(key.parameter_count),
            **{f"geometry_{name}": value for name, value in key.geometry.to_dict().items()},
        },
        train_metrics=FitMetrics.from_predictions(
            train_target,
            train_predicted,
            n_outliers=int(np.sum(~inliers)) if inliers.size else 0,
            design_rank=rank,
        ),
        validation_metrics=FitMetrics.from_predictions(
            validation_target,
            validation_predicted,
            design_rank=rank,
        ),
        feature_ranges={
            "activation_bytes": (
                min(activation_features),
                max(activation_features),
            ),
            "profiled_dp": (min(dp_values), max(dp_values)),
        },
        diagnostics={
            "memory_metric": "peak_allocated_bytes",
            "static_inventory": "full-param-and-grad-buffer-with-dp-sharded-optimizer",
            "optimizer_sharding": "distributed-optimizer-dp",
            "activation_model": (
                "(num_layers+1)*hidden*bytes + vocab*bytes per token, scaled by "
                "min(num_microbatches, pp) in-flight microbatches over pp stages"
            ),
            "oom_constraints": constrained_oom,
            "oom_intercept_increase": max(intercept_increase, 0.0),
            "validation_fraction": validation_fraction,
            # `split_records_by_run` hands back the same list twice at fraction zero or
            # with a single run, so `validation_metrics` is then the training metric.
            "validation_is_in_sample": validation_is_in_sample(
                successful,
                validation_fraction=validation_fraction,
            ),
        },
    )


def _selector(
    *,
    component: str,
    workload: TrainingWorkload,
    parameter_dtype: str,
    gradient_dtype: str,
    optimizer_state_dtype: str,
    distributed_optimizer: bool,
) -> dict[str, Any]:
    return {
        "component": component,
        "schedule": workload.schedule.lower(),
        "recompute": workload.recompute,
        "optimizer": workload.optimizer,
        "parameter_dtype": parameter_dtype,
        "gradient_dtype": gradient_dtype,
        "optimizer_state_dtype": optimizer_state_dtype,
        "distributed_optimizer": distributed_optimizer,
    }


class HydraulisLatencyModel(TrainingLatencyModel):
    """Evaluate fitted Hydraulis latency cells with explicit OOD checks."""

    def __init__(
        self,
        artifact: CostArtifact,
        *,
        parameter_dtype: str,
        gradient_dtype: str,
        optimizer_state_dtype: str,
        distributed_optimizer: bool,
        extrapolate: frozenset[str] = frozenset(),
    ):
        self.artifact = artifact
        self.parameter_dtype = parameter_dtype
        self.gradient_dtype = gradient_dtype
        self.optimizer_state_dtype = optimizer_state_dtype
        self.distributed_optimizer = distributed_optimizer
        self.extrapolate = extrapolate

    def predict(
        self,
        workload: TrainingWorkload,
        parallelism: TrainingParallelism,
    ) -> float:
        latency, _ = self.predict_with_notes(workload, parallelism)
        return latency

    def predict_with_notes(
        self,
        workload: TrainingWorkload,
        parallelism: TrainingParallelism,
    ) -> tuple[float, list[str]]:
        """Return (latency, extrapolation_notes) so the caller can record them."""
        canonical = parallelism.to_profile_parallelism(canonical_dp=True)
        entry = self.artifact.entry_for(
            canonical,
            selector=_selector(
                component="latency",
                workload=workload,
                parameter_dtype=self.parameter_dtype,
                gradient_dtype=self.gradient_dtype,
                optimizer_state_dtype=self.optimizer_state_dtype,
                distributed_optimizer=self.distributed_optimizer,
            ),
            # One TP entry serves every PP; see `_pool_latency_cells`.
            ignore_pp=True,
        )
        microbatches = workload.critical_rank_microbatches(parallelism)
        features = latency_features(microbatches, parallelism)
        notes: list[str] = []
        for name, value in (
            ("tokens", features.tokens),
            ("pipeline_slots", features.slots),
            # Gated because it is a batch total, not a per-microbatch maximum.
            # See `latency_feature_ranges`.
            ("sum_squared_tokens", features.weighted_sum_squared_tokens),
            ("micro_batch_size", float(workload.micro_batch_size)),
        ):
            note = check_feature_range(entry, name, value, extrapolate=self.extrapolate)
            if note is not None:
                notes.append(note)
        coefficients = entry.coefficients
        # `latency_from_coefficients` raises on a non-positive result, so there is no
        # separate guard here.
        return latency_from_coefficients(coefficients, features), notes


class HydraulisMemoryModel(TrainingMemoryModel):
    """Combine static parameter inventory with a fitted activation residual."""

    def __init__(
        self,
        artifact: CostArtifact,
        *,
        parameter_dtype: str,
        gradient_dtype: str,
        optimizer_state_dtype: str,
        distributed_optimizer: bool,
        extrapolate: frozenset[str] = frozenset(),
    ):
        self.artifact = artifact
        self.parameter_dtype = parameter_dtype
        self.gradient_dtype = gradient_dtype
        self.optimizer_state_dtype = optimizer_state_dtype
        self.distributed_optimizer = distributed_optimizer
        self.extrapolate = extrapolate

    def predict(
        self,
        workload: TrainingWorkload,
        parallelism: TrainingParallelism,
    ) -> int:
        memory, _ = self.predict_with_notes(workload, parallelism)
        return memory

    def predict_with_notes(
        self,
        workload: TrainingWorkload,
        parallelism: TrainingParallelism,
    ) -> tuple[int, list[str]]:
        """Return (peak_memory_bytes, extrapolation_notes) so the caller can record them."""
        canonical = parallelism.to_profile_parallelism(canonical_dp=True)
        entry = self.artifact.entry_for(
            canonical,
            selector=_selector(
                component="memory",
                workload=workload,
                parameter_dtype=self.parameter_dtype,
                gradient_dtype=self.gradient_dtype,
                optimizer_state_dtype=self.optimizer_state_dtype,
                distributed_optimizer=self.distributed_optimizer,
            ),
        )
        microbatches = workload.critical_rank_microbatches(parallelism)
        geometry = ActivationGeometry.from_dict(
            {
                name.removeprefix("geometry_"): value
                for name, value in entry.coefficients.items()
                if name.startswith("geometry_")
            }
        )
        feature = activation_bytes(geometry, microbatches, parallelism)
        notes: list[str] = []
        note = check_feature_range(entry, "activation_bytes", feature, extrapolate=self.extrapolate)
        if note is not None:
            notes.append(note)
        parameter_count = int(entry.coefficients["parameter_count"])
        static = _static_bytes(
            parameter_count=parameter_count,
            geometry=geometry,
            parallelism=parallelism,
            parameter_dtype=self.parameter_dtype,
            gradient_dtype=self.gradient_dtype,
            optimizer_state_dtype=self.optimizer_state_dtype,
            distributed_optimizer=self.distributed_optimizer,
        )
        predicted = (
            static
            + entry.coefficients["activation_scale"] * feature
            + entry.coefficients["residual_constant"]
        )
        return int(math.ceil(predicted)), notes


class HydraulisTrainingCost(TrainingCostModel):
    """Compose Hydraulis V1 latency and memory from one artifact."""

    model_type = "hydraulis"

    @classmethod
    def fit(
        cls,
        records: list[ProfileRecord],
        *,
        validation_fraction: float = 0.2,
        require_full_rank: bool = True,
    ) -> CostArtifact:
        """Fit latency and memory entries for every supported discrete cell."""
        groups, materialized = _training_records(records)
        model_names = {record.model.name for record in materialized}
        gpu_names = {record.hardware.gpu_name for record in materialized}
        if len(model_names) != 1 or len(gpu_names) != 1:
            raise ValueError("A training artifact must target one model and one GPU type.")
        first = materialized[0]
        model = asdict(first.model)
        hardware = asdict(first.hardware)
        hardware["gpu_count"] = sorted({record.hardware.gpu_count for record in materialized})
        entries: list[ArtifactEntry] = []

        # Latency pools every PP in a TP group into one fit, because the features
        # are per-layer and the slot count is analytic, so PP is arithmetic rather
        # than a cell to measure. Memory does not pool: the 1F1B peak depends on
        # which stage is in flight and how many microbatches it is holding, which
        # is not a per-stage division of one number.
        for latency_key, latency_records in _pool_latency_cells(groups).items():
            latency_entry = _fit_latency_entry(
                latency_key,
                latency_records,
                validation_fraction=validation_fraction,
                require_full_rank=require_full_rank,
            )
            if latency_entry is not None:
                entries.append(latency_entry)

        for key, cell_records in groups.items():
            memory_entry = _fit_memory_entry(
                key,
                cell_records,
                validation_fraction=validation_fraction,
                require_full_rank=require_full_rank,
            )
            if memory_entry is not None:
                entries.append(memory_entry)
        if not entries:
            raise ValueError("No Hydraulis entries could be fitted.")
        return CostArtifact(
            kind="training",
            model_type=cls.model_type,
            model=model,
            hardware=hardware,
            profile_digest=profile_digest(materialized),
            entries=tuple(entries),
            diagnostics={
                "supported_cp": [1],
                "supported_schedule": ["1f1b"],
                "layer_partition": "uniform",
                "dp_communication": "ignored",
                "activation_checkpoint_recompute": "discrete-profiled-selector",
            },
        )

    @classmethod
    def from_artifact(
        cls,
        artifact_or_path: CostArtifact | str | Path,
        *,
        parameter_dtype: str | None = None,
        gradient_dtype: str | None = None,
        optimizer_state_dtype: str | None = None,
        distributed_optimizer: bool | None = None,
        extrapolate: list[str] | None = None,
    ) -> HydraulisTrainingCost:
        """Load composable latency and memory models from one artifact.

        Args:
            extrapolate: Feature names for which out-of-domain queries are permitted.
                An empty list or None (the default) keeps the strict behaviour. When a
                name is listed and the query is out of range, the prediction proceeds and
                the exceedance is recorded in TrainingCostEstimate.diagnostics under the
                key 'extrapolation_notes'. Unknown names raise ValueError at construction.
                Valid names per component:
                  latency: micro_batch_size, pipeline_slots, tokens
                  memory:  activation_bytes, profiled_dp
        """
        artifact = (
            artifact_or_path
            if isinstance(artifact_or_path, CostArtifact)
            else CostArtifact.load(artifact_or_path)
        )
        if artifact.kind != "training" or artifact.model_type != cls.model_type:
            raise ValueError("Artifact does not contain a Hydraulis training model.")
        selectors = [entry.selector for entry in artifact.entries]

        def infer(name: str, explicit: Any) -> Any:
            if explicit is not None:
                return explicit
            values = {selector[name] for selector in selectors}
            if len(values) != 1:
                raise ValueError(f"Artifact has multiple {name} values; select one explicitly.")
            return values.pop()

        selected_parameter_dtype = str(infer("parameter_dtype", parameter_dtype))
        selected_gradient_dtype = str(infer("gradient_dtype", gradient_dtype))
        selected_optimizer_state_dtype = str(
            infer("optimizer_state_dtype", optimizer_state_dtype)
        )
        selected_distributed_optimizer = bool(
            infer("distributed_optimizer", distributed_optimizer)
        )
        extrapolate_set = frozenset(extrapolate) if extrapolate else frozenset()
        all_valid_names = frozenset().union(
            *(frozenset(entry.feature_ranges.keys()) for entry in artifact.entries)
        )
        validate_extrapolate_names(
            extrapolate_set, all_valid_names, context="Hydraulis artifact"
        )
        common = {
            "artifact": artifact,
            "parameter_dtype": selected_parameter_dtype,
            "gradient_dtype": selected_gradient_dtype,
            "optimizer_state_dtype": selected_optimizer_state_dtype,
            "distributed_optimizer": selected_distributed_optimizer,
            "extrapolate": extrapolate_set,
        }
        return cls(
            latency_model=HydraulisLatencyModel(**common),
            memory_model=HydraulisMemoryModel(**common),
            model=ModelDescriptor.from_dict(artifact.model),
        )

    def estimate(
        self,
        workload: TrainingWorkload,
        parallelism: TrainingParallelism,
    ) -> TrainingCostEstimate:
        """Estimate latency and memory, recording any permitted extrapolations."""
        latency, lat_notes = self.latency_model.predict_with_notes(workload, parallelism)
        memory, mem_notes = self.memory_model.predict_with_notes(workload, parallelism)
        diagnostics: dict[str, Any] = {
            "tp": parallelism.tp,
            "pp": parallelism.pp,
            "dp": parallelism.dp,
            "cp": parallelism.cp,
            "ep": parallelism.ep,
            "micro_batch_size": workload.micro_batch_size,
            "schedule": workload.schedule,
            "recompute": workload.recompute,
        }
        all_notes = lat_notes + mem_notes
        if all_notes:
            diagnostics["extrapolation_notes"] = all_notes
        return TrainingCostEstimate(
            latency_s=latency,
            peak_memory_bytes=memory,
            diagnostics=diagnostics,
        )
