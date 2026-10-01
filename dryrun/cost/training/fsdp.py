"""FSDP2 training latency and memory model.

The latency form is the structured one from `latency.py`, not a new polynomial. FSDP is
precisely the degenerate case that module documents: with no pipeline, `pipeline_slots`
collapses to the microbatch count, so

    step = per_step_overhead + num_microbatches * (token_cost * tokens + microbatch_overhead)

Reusing that form rather than restating it means the quadratic token term that was
measured and rejected there cannot reappear here under a different name.

Memory differs from Megatron in two ways that both matter more than a fitted residual
can absorb. The static pool comes from `fsdp_inventory`, which shards parameters,
gradients, and optimizer moments: roughly half of what Megatron's inventory would
report for the same model.

Memory feature design
---------------------
With `gradient_checkpointing=true`, activations are recomputed on the backward pass
and are not kept resident between the forward and the backward. The per-step activation
working set is therefore small and roughly token-independent. What drives peak memory
in that case is the number of in-flight microbatch autograd graphs: each graph pins its
intermediate tensors until `backward()` completes, and the graphs accumulate when
`num_microbatches > 1`.

Measured on 8x H20, `shard_size=8`, `gradient_checkpointing=true`:
    nmb=1  tokens= 512 -> allocated 9.4784 GiB
    nmb=1  tokens=1024 -> allocated 9.4784 GiB
    nmb=1  tokens=2048 -> allocated 9.4785 GiB
    nmb=2  tokens= 512 -> allocated 10.0585 GiB
    nmb=2  tokens=1024 -> allocated 10.0585 GiB
    nmb=2  tokens=2048 -> allocated 10.0585 GiB

Memory is flat to four decimal places across a 4x token range; one extra microbatch
adds ~580 MiB. Fitting only `max_tokens` against a flat target produces a degenerate
regression: `activation_slope` collapses to ~5e-14 and both candidates score
catastrophically negative r² (allocated -2.6e9, reserved -3.4e5) because the single
feature explains nothing and the optimizer assigns all variance to the constant.

The residual therefore carries two terms: `microbatch_slope * num_microbatches` and
`activation_slope * max_tokens`. Both are fitted non-negatively. When
`gradient_checkpointing=true` the non-negative constraint drives `activation_slope` to
near zero rather than distorting the `microbatch_slope`. When
`gradient_checkpointing=false` activations are resident and `max_tokens` genuinely
drives memory, so the token term must be present for that case to work correctly.

Choosing the memory target
--------------------------
`peak_allocated_bytes` is the live-tensor high-water mark and `peak_reserved_bytes` is
the caching allocator's segment pool. Allocated responds to load and reserved is
quantized and sticky, so allocated is the better regression target, but that is a
measurement, not an assumption, so both are fitted and the winner is selected on
validation coefficient of determination with both scores recorded in the entry.

The simulator, however, compares the prediction against device capacity, and an OOM is
caused by reserved bytes rather than allocated ones. On 8x H20 the measured ratio was
20.928 / 9.478 = 2.208, so predicting allocated directly would under-report OOM by more
than twofold. The artifact therefore also stores the measured
`reserved_over_allocated` ratio and applies it at predict time: the regression keeps the
well-behaved target while the capacity comparison keeps the quantity that actually
causes OOM.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dryrun.cost.artifact import ArtifactEntry, CostArtifact, FitMetrics
from dryrun.profiler.io import profile_digest
from dryrun.profiler.schema import Parallelism, ProfileRecord, SequenceStats, TrainingPayload

from .base import (
    TrainingCostEstimate,
    TrainingCostModel,
    TrainingLatencyModel,
    TrainingMemoryModel,
    TrainingOutOfDomainError,
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
from .fsdp_inventory import fsdp_inventory
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
from .microbatch import BatchingMode, MicrobatchPlanner

FSDP_SCHEDULE = "fsdp"
# Candidate memory targets, allocated first so a tie resolves to the physically
# better-behaved quantity rather than to dictionary order.
MEMORY_TARGETS = ("peak_allocated_bytes", "peak_reserved_bytes")


def _tag_int(tags: dict[str, str], name: str) -> int:
    try:
        return int(tags[name])
    except KeyError as error:
        raise ValueError(f"An FSDP record must carry the {name} tag.") from error
    except ValueError as error:
        raise ValueError(f"FSDP tag {name} must be an integer, got {tags[name]!r}.") from error


def _tag_bool(tags: dict[str, str], name: str) -> bool:
    """
    Parse a boolean tag written by `str(value)`.

    Tags are strings, and `bool("False")` is `True`, so the comparison has to be
    explicit or every offloaded cell would silently be treated as non-offloaded.
    """
    try:
        value = tags[name]
    except KeyError as error:
        raise ValueError(f"An FSDP record must carry the {name} tag.") from error
    if value not in ("True", "False"):
        raise ValueError(f"FSDP tag {name} must be True or False, got {value!r}.")
    return value == "True"


def _selector(
    *,
    component: str,
    shard_size: int,
    replicate_size: int,
    recompute: str,
    optimizer: str,
    parameter_dtype: str,
    gradient_dtype: str,
    optimizer_state_dtype: str,
    optimizer_offload: bool,
    reshard_after_forward: bool,
) -> dict[str, Any]:
    """
    Build the discrete cell selector both the fit and the predict side use.

    One definition rather than two, because `CostArtifact.entry_for` matches on the keys
    it is given: a key present when fitting but absent when predicting would leave the
    lookup under-constrained instead of pointing at the omission.
    """
    return {
        "component": component,
        "schedule": FSDP_SCHEDULE,
        "shard_size": shard_size,
        "replicate_size": replicate_size,
        "recompute": recompute,
        "optimizer": optimizer,
        "parameter_dtype": parameter_dtype,
        "gradient_dtype": gradient_dtype,
        "optimizer_state_dtype": optimizer_state_dtype,
        "optimizer_offload": optimizer_offload,
        "reshard_after_forward": reshard_after_forward,
    }


@dataclass(frozen=True)
class _CellKey:
    """Identify one discrete FSDP configuration that shares a single regression."""

    shard_size: int
    replicate_size: int
    recompute: str
    optimizer: str
    parameter_dtype: str
    gradient_dtype: str
    optimizer_state_dtype: str
    optimizer_offload: bool
    reshard_after_forward: bool
    parameter_count: int

    @property
    def world_size(self) -> int:
        return self.shard_size * self.replicate_size

    @property
    def parallelism(self) -> Parallelism:
        """
        Return the canonical topology the artifact keys entries on.

        FSDP has no tensor, pipeline, or context parallelism, and the mesh travels in the
        selector because sharding and replication are configuration rather than data
        partitioning. Data parallelism is canonicalized to one for the same reason
        Hydraulis canonicalizes it: `dp` scales the local workload, not the cell.
        """
        return Parallelism(tp=1, pp=1, dp=1, cp=1)

    def selector(self, component: str) -> dict[str, Any]:
        return _selector(
            component=component,
            shard_size=self.shard_size,
            replicate_size=self.replicate_size,
            recompute=self.recompute,
            optimizer=self.optimizer,
            parameter_dtype=self.parameter_dtype,
            gradient_dtype=self.gradient_dtype,
            optimizer_state_dtype=self.optimizer_state_dtype,
            optimizer_offload=self.optimizer_offload,
            reshard_after_forward=self.reshard_after_forward,
        )

    def static_bytes(self) -> float:
        return fsdp_inventory(
            parameter_count=self.parameter_count,
            world_size=self.world_size,
            parameter_dtype=self.parameter_dtype,
            gradient_dtype=self.gradient_dtype,
            optimizer_state_dtype=self.optimizer_state_dtype,
            optimizer_offload=self.optimizer_offload,
        ).total_bytes


def _fsdp_records(
    records: list[ProfileRecord],
) -> tuple[dict[_CellKey, list[ProfileRecord]], list[ProfileRecord]]:
    """
    Group FSDP records into regression cells and reject anything that is not FSDP.

    A 1F1B record is rejected rather than skipped. Merging Megatron and FSDP
    measurements into one cell would produce a regression over two different memory
    layouts and two different schedules, and silently dropping the foreign records
    instead would hide a mis-pointed input file.
    """
    materialized = [record for record in records if record.record_type == "training"]
    if not materialized:
        raise ValueError("No training profiling records are available.")
    groups: dict[_CellKey, list[ProfileRecord]] = defaultdict(list)
    for record in materialized:
        payload = record.payload
        if not isinstance(payload, TrainingPayload):
            continue
        schedule = payload.schedule.lower()
        if schedule != FSDP_SCHEDULE:
            raise ValueError(
                f"The FSDP cost model only fits records whose schedule is fsdp, got {schedule!r}. "
                "Fit Megatron records with the hydraulis model instead."
            )
        sequence_parallel = record.tags.get("sequence_parallel", "1")
        if sequence_parallel != "1":
            raise ValueError(
                f"FSDP V1 only fits sequence_parallel=1 records, got {sequence_parallel!r}. "
                "Ulysses changes the per-rank token count and needs its own feature."
            )
        if (record.parallelism.tp, record.parallelism.pp, record.parallelism.cp) != (1, 1, 1):
            raise ValueError(
                "FSDP records must have tp=pp=cp=1; the mesh is expressed by shard_size "
                f"and replicate_size, got {record.parallelism!r}."
            )
        parameter_count = payload.parameter_count or record.model.parameter_count
        if parameter_count is None:
            raise ValueError("FSDP memory fitting requires parameter_count.")
        shard_size = _tag_int(record.tags, "shard_size")
        replicate_size = _tag_int(record.tags, "replicate_size")
        if shard_size * replicate_size != record.parallelism.dp:
            raise ValueError(
                f"The FSDP mesh {shard_size}x{replicate_size} disagrees with dp="
                f"{record.parallelism.dp}; the static inventory would be mis-sized."
            )
        key = _CellKey(
            shard_size=shard_size,
            replicate_size=replicate_size,
            recompute=payload.recompute,
            optimizer=payload.optimizer,
            parameter_dtype=payload.parameter_dtype,
            gradient_dtype=payload.gradient_dtype,
            optimizer_state_dtype=payload.optimizer_state_dtype,
            optimizer_offload=_tag_bool(record.tags, "optimizer_offload"),
            reshard_after_forward=_tag_bool(record.tags, "reshard_after_forward"),
            parameter_count=int(parameter_count),
        )
        groups[key].append(record)
    if not groups:
        raise ValueError("No FSDP training records are available.")
    return dict(groups), materialized


def _latency_features(payload: TrainingPayload) -> LatencyFeatures:
    """Build the latency design row, with `pp=1` collapsing slots to the microbatch count."""
    return latency_features(payload.microbatches, TrainingParallelism(dp=1))


def _memory_features(microbatches: tuple[SequenceStats, ...]) -> tuple[float, float]:
    """
    Return the two memory features: number of in-flight microbatch graphs and max tokens.

    With gradient checkpointing, each microbatch accumulates its autograd graph until
    `backward()` completes, so `num_microbatches` drives peak memory even when activations
    are small. Without gradient checkpointing, activations are resident and `max_tokens`
    drives peak memory. Both terms are kept so the fit is correct in either case; the
    non-negative constraint drives whichever coefficient does not explain the target to
    approximately zero rather than distorting the other.
    """
    num_microbatches = float(len(microbatches))
    max_tokens = float(max(item.sum_tokens for item in microbatches))
    return num_microbatches, max_tokens


def _successful(records: list[ProfileRecord]) -> list[ProfileRecord]:
    return [
        record
        for record in records
        if record.run.status == "success"
        and isinstance(record.payload, TrainingPayload)
        and record.payload.step_times_s
    ]


def _latency_arrays(records: list[ProfileRecord]) -> tuple[np.ndarray, np.ndarray]:
    """Build one design row per candidate, its steps reduced by a median."""
    rows: list[tuple[float, float, float, float]] = []
    targets: list[float] = []
    for record in records:
        payload = record.payload
        if record.run.status != "success" or not isinstance(payload, TrainingPayload):
            continue
        if not payload.step_times_s:
            continue
        rows.append(_latency_features(payload).design_row())
        targets.append(robust_step_time(payload.step_times_s))
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def _fit_latency_entry(
    key: _CellKey,
    records: list[ProfileRecord],
    *,
    validation_fraction: float,
    require_full_rank: bool,
) -> ArtifactEntry | None:
    successful = _successful(records)
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
            f"FSDP latency needs at least {LATENCY_TERM_COUNT} candidates for {key!r}."
        )
    parameters, inliers, rank = scaled_nonnegative_fit(train_x, train_y)
    if require_full_rank:
        check_latency_rank(
            rank, backend="FSDP", key=key, slots_feature="the microbatch count"
        )
    train_predicted = train_x @ parameters
    validation_predicted = validation_x @ parameters
    all_features = [_latency_features(record.payload) for record in successful]
    dispersions = [step_time_dispersion(record.payload.step_times_s) for record in successful]
    feature_ranges = latency_feature_ranges(
        all_features,
        slots_feature="num_microbatches",
        extra={
            # The profile sweeps `n_seqs_per_rank`, and the planner decides how many
            # sequences a predicted microbatch holds, so this guards the GEMM shape rather
            # than a config.
            "sequences_per_microbatch": (
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
            "formula": (
                "per_step_overhead + num_microbatches * (token_cost * max_microbatch_tokens "
                "+ token_quadratic_cost * max_microbatch_sum_squared_tokens "
                "+ microbatch_overhead)"
            ),
            # There is no pipeline, so the slots multiplier of the shared latency form
            # degenerates to the microbatch count rather than taking a fitted term.
            "slots": "num_microbatches",
            "dp_latency": "ideal-local-workload-scaling",
            "step_aggregation": "median-per-candidate",
            "validation_fraction": validation_fraction,
            "validation_is_in_sample": validation_is_in_sample(
                successful,
                validation_fraction=validation_fraction,
            ),
            "max_step_dispersion": max(dispersions) if dispersions else 0.0,
            "median_step_dispersion": float(np.median(dispersions)) if dispersions else 0.0,
        },
    )


def _memory_arrays(
    records: list[ProfileRecord],
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    nmb_list: list[float] = []
    tokens_list: list[float] = []
    targets: list[float] = []
    for record in records:
        payload = record.payload
        if record.run.status != "success" or not isinstance(payload, TrainingPayload):
            continue
        peak = getattr(payload, metric)
        if peak is None:
            continue
        nmb, max_tokens = _memory_features(payload.microbatches)
        nmb_list.append(nmb)
        tokens_list.append(max_tokens)
        targets.append(float(peak))
    return (
        np.column_stack((nmb_list, tokens_list)) if nmb_list else np.empty((0, 2), dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
    )


@dataclass(frozen=True)
class _MemoryFit:
    """
    One candidate memory regression, kept whole so its metrics can be rebuilt.

    The fitted arrays are retained rather than only their summary statistics because an
    OOM record raises the intercept *after* the regression converges, and the entry must
    advertise the quality of the coefficients it actually ships. See `metrics`.

    `train_feature` and `validation_feature` are 2D arrays with columns
    `[num_microbatches, max_tokens]`, matching the two-term residual design.
    """

    metric: str
    microbatch_slope: float
    activation_slope: float
    constant: float
    static_bytes: float
    train_feature: np.ndarray
    train_target: np.ndarray
    validation_feature: np.ndarray
    validation_target: np.ndarray
    n_outliers: int
    design_rank: int

    def metrics(self, constant: float) -> tuple[FitMetrics, FitMetrics]:
        """
        Return train and validation metrics for the constant that will actually ship.

        Computing these inside the fit and shipping a later, OOM-raised constant alongside
        them would advertise a perfect fit for coefficients that are badly wrong: measured
        on the OOM fixture, a constant bumped from 6.03e-10 to 8.62e10 still reported
        `r_squared=1.0, mre_percent=0.00` where the honest MRE against the same successful
        records is 526%.

        Args:
            constant (float): The residual constant the entry carries, OOM bump included.
        """
        slopes = np.array([self.microbatch_slope, self.activation_slope])
        train_predicted = self.static_bytes + self.train_feature @ slopes + constant
        validation_predicted = self.static_bytes + self.validation_feature @ slopes + constant
        return (
            FitMetrics.from_predictions(
                self.train_target,
                train_predicted,
                n_outliers=self.n_outliers,
                design_rank=self.design_rank,
            ),
            FitMetrics.from_predictions(
                self.validation_target,
                validation_predicted,
                design_rank=self.design_rank,
            ),
        )


def _fit_memory_target(
    key: _CellKey,
    successful: list[ProfileRecord],
    metric: str,
    *,
    validation_fraction: float,
    require_full_rank: bool,
) -> _MemoryFit:
    """Fit `metric - static_inventory` against `[num_microbatches, max_tokens, 1]`."""
    static = key.static_bytes()
    train_records, validation_records = split_records_by_run(
        successful,
        validation_fraction=validation_fraction,
    )
    train_feature, train_target = _memory_arrays(train_records, metric)
    validation_feature, validation_target = _memory_arrays(validation_records, metric)
    residual = np.maximum(train_target - static, 0)
    # Three-column design: microbatch count, max tokens per microbatch, and intercept.
    # The microbatch term captures autograd-graph accumulation, which dominates under
    # gradient checkpointing. The token term captures resident activations, which
    # dominate when checkpointing is off. The non-negative fit drives whichever
    # coefficient does not explain the target to approximately zero.
    design = np.column_stack((train_feature, np.ones(len(train_feature))))
    parameters, inliers, rank = scaled_nonnegative_fit(design, residual)
    if require_full_rank and rank < 3:
        raise ValueError(
            f"FSDP memory design matrix is rank {rank}, expected 3 for {key!r}. "
            "Vary both the sequence length and the microbatch count to provide "
            "independent signal for both the microbatch slope and the activation slope."
        )
    return _MemoryFit(
        metric=metric,
        microbatch_slope=float(parameters[0]),
        activation_slope=float(parameters[1]),
        constant=float(parameters[2]),
        static_bytes=static,
        train_feature=train_feature,
        train_target=train_target,
        validation_feature=validation_feature,
        validation_target=validation_target,
        n_outliers=int(np.sum(~inliers)),
        design_rank=rank,
    )


def _empty_memory_fit() -> _MemoryFit:
    """Return the zero fit an OOM-only cell carries, which measures nothing."""
    empty_features = np.empty((0, 2), dtype=np.float64)
    empty_targets = np.empty(0, dtype=np.float64)
    return _MemoryFit(
        metric=MEMORY_TARGETS[0],
        microbatch_slope=0.0,
        activation_slope=0.0,
        constant=0.0,
        static_bytes=0.0,
        train_feature=empty_features,
        train_target=empty_targets,
        validation_feature=empty_features,
        validation_target=empty_targets,
        n_outliers=0,
        design_rank=0,
    )


def _reserved_over_allocated(records: list[ProfileRecord]) -> float:
    """
    Return the measured ratio of reserved to allocated peak bytes.

    Floored at one because the allocator can never reserve less than it has handed out,
    so a ratio below one is measurement noise rather than a saving to bank on.

    A single scalar applied multiplicatively is the weak part of this model. Fragmentation
    tracks allocation *churn*, which depends on the microbatch count and the recompute
    policy, rather than tracking the peak, so `reserved - allocated` may well be closer to
    constant than to proportional. A ratio then over-predicts at low load, where the static
    pool dominates and fragments little, and under-predicts at high load. The measured
    2.208 comes from one configuration, and one data point cannot distinguish a
    multiplicative mean from an additive offset, which is why the ratio is recorded in the
    entry rather than only applied. Revisit once several load points exist: if
    `reserved - allocated` is the more stable quantity, this should become an additive term.
    """
    ratios = [
        float(record.payload.peak_reserved_bytes) / float(record.payload.peak_allocated_bytes)
        for record in records
        if isinstance(record.payload, TrainingPayload)
        and record.payload.peak_reserved_bytes is not None
        and record.payload.peak_allocated_bytes
    ]
    if not ratios:
        return 1.0
    return max(1.0, float(np.mean(ratios)))


def _fit_memory_entry(
    key: _CellKey,
    records: list[ProfileRecord],
    *,
    validation_fraction: float,
    require_full_rank: bool,
) -> ArtifactEntry | None:
    """Fit both peak metrics, select on validation quality, and fold in OOM inequalities."""
    successful = [
        record
        for record in records
        if record.run.status == "success"
        and isinstance(record.payload, TrainingPayload)
        and record.payload.peak_allocated_bytes is not None
        and record.payload.peak_reserved_bytes is not None
    ]
    oom_records = [record for record in records if record.run.status == "oom"]
    if not successful and not oom_records:
        return None
    static = key.static_bytes()
    if successful:
        candidates = [
            _fit_memory_target(
                key,
                successful,
                metric,
                validation_fraction=validation_fraction,
                require_full_rank=require_full_rank,
            )
            for metric in MEMORY_TARGETS
        ]
        # Selection reads the regression's own validation score, before any OOM bump: the
        # question is which target responds to load, and the bump is a floor rather than a
        # statement about fit quality.
        candidate_r_squared = {
            item.metric: item.metrics(item.constant)[1].r_squared for item in candidates
        }
        # Strict `>` keeps the first candidate on a tie, and MEMORY_TARGETS lists
        # allocated first, so an uninformative comparison falls back to the target whose
        # response to load is physical rather than allocator-quantized.
        selected = max(candidates, key=lambda item: candidate_r_squared[item.metric])
    else:
        selected = _empty_memory_fit()
        candidate_r_squared = {metric: 0.0 for metric in MEMORY_TARGETS}

    # An OOM is an inequality rather than a measurement: the true peak was at least the
    # device capacity. Raising the intercept is the same treatment Hydraulis applies.
    #
    # The deficit is computed in the *selected metric's* units, so the capacity target is
    # divided by the ratio that `predict` will later multiply back in. Comparing a raw
    # allocated-scale fit against a reserved-scale capacity would otherwise raise the
    # intercept by the full ratio again and over-predict the floor by that factor.
    reserved_ratio = _reserved_over_allocated(successful)
    predict_scale = reserved_ratio if selected.metric == "peak_allocated_bytes" else 1.0
    constrained_oom = 0
    intercept_increase = 0.0
    for record in oom_records:
        payload = record.payload
        if not isinstance(payload, TrainingPayload) or record.hardware.gpu_memory_bytes is None:
            continue
        nmb, max_tokens = _memory_features(payload.microbatches)
        required_total = (record.hardware.gpu_memory_bytes + 1) / predict_scale
        required_residual = required_total - static
        deficit = (
            required_residual
            - (selected.microbatch_slope * nmb + selected.activation_slope * max_tokens + selected.constant)
        )
        intercept_increase = max(intercept_increase, deficit)
        constrained_oom += 1
    constant = selected.constant + max(intercept_increase, 0.0)
    # Rebuilt from the final constant, so the advertised quality describes the coefficients
    # the entry carries rather than the ones the regression happened to converge on.
    train_metrics, validation_metrics = selected.metrics(constant)

    observed = [
        record
        for record in successful + oom_records
        if isinstance(record.payload, TrainingPayload)
    ]
    nmb_values = [_memory_features(record.payload.microbatches)[0] for record in observed]
    tokens_values = [_memory_features(record.payload.microbatches)[1] for record in observed]
    return ArtifactEntry(
        parallelism=key.parallelism,
        selector=key.selector("memory"),
        coefficients={
            "microbatch_slope": selected.microbatch_slope,
            "activation_slope": selected.activation_slope,
            "residual_constant": constant,
            "parameter_count": float(key.parameter_count),
            "world_size": float(key.world_size),
        },
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        feature_ranges={
            "num_microbatches": (min(nmb_values), max(nmb_values)),
            "activation_tokens": (min(tokens_values), max(tokens_values)),
        },
        diagnostics={
            "memory_metric": selected.metric,
            "candidate_r_squared": candidate_r_squared,
            # False when the two targets scored equally, which `r_squared=0.0` at
            # `n_samples=1` guarantees regardless of how large the residual is: with five
            # runs and the default fraction the split yields a single validation run, both
            # targets can score 0.0, and the winner is then decided by MEMORY_TARGETS
            # order. The selection rule is unchanged; the audit trail now says so.
            "validation_selection_informative": len(set(candidate_r_squared.values())) > 1,
            "validation_fraction": validation_fraction,
            "validation_is_in_sample": validation_is_in_sample(
                successful,
                validation_fraction=validation_fraction,
            ),
            # Applied at predict time so the capacity comparison uses the quantity that
            # causes OOM while the regression keeps the well-behaved target.
            "reserved_over_allocated": reserved_ratio,
            "static_inventory": "fsdp2-sharded-parameters-gradients-and-optimizer",
            "activation_model": (
                "num-microbatches-plus-max-tokens-two-term-residual"
            ),
            "oom_constraints": constrained_oom,
            "oom_intercept_increase": max(intercept_increase, 0.0),
        },
    )


class _FsdpCellModel:
    """Hold the discrete cell selectors and the planner both components share."""

    def __init__(
        self,
        artifact: CostArtifact,
        *,
        shard_size: int,
        replicate_size: int,
        recompute: str,
        optimizer_offload: bool,
        reshard_after_forward: bool,
        parameter_dtype: str,
        gradient_dtype: str,
        optimizer_state_dtype: str,
        planner: MicrobatchPlanner,
        extrapolate: frozenset[str] = frozenset(),
    ):
        self.artifact = artifact
        self.shard_size = shard_size
        self.replicate_size = replicate_size
        self.recompute = recompute
        self.optimizer_offload = optimizer_offload
        self.reshard_after_forward = reshard_after_forward
        self.parameter_dtype = parameter_dtype
        self.gradient_dtype = gradient_dtype
        self.optimizer_state_dtype = optimizer_state_dtype
        self.planner = planner
        self.extrapolate = extrapolate

    @property
    def world_size(self) -> int:
        return self.shard_size * self.replicate_size

    def _entry(
        self,
        component: str,
        workload: TrainingWorkload,
        parallelism: TrainingParallelism,
    ) -> ArtifactEntry:
        if workload.schedule.lower() != FSDP_SCHEDULE:
            raise TrainingOutOfDomainError(
                f"The FSDP cost model only serves the fsdp schedule, got {workload.schedule!r}."
            )
        if workload.recompute != self.recompute:
            raise TrainingOutOfDomainError(
                f"This FSDP cell was fitted at recompute={self.recompute!r}, "
                f"got {workload.recompute!r}."
            )
        # A cell is a measurement of exactly one mesh, and the mesh sets both the sharded
        # static inventory and the load each rank carries. Answering for a different dp
        # would be a fabrication, not an extrapolation: the mesh never enters the fitted
        # form, so the prediction would come back unchanged while the true static pool
        # scales as 1/dp. Left unchecked that silently defeated the simulator's OOM guard,
        # under-reporting a dp=4 query against an 8-rank artifact by exactly twofold.
        if parallelism.dp != self.world_size:
            raise TrainingOutOfDomainError(
                f"This FSDP cell was measured on a {self.shard_size}x{self.replicate_size} "
                f"shard-by-replicate mesh, which is dp={self.world_size}, so it cannot "
                f"answer for dp={parallelism.dp}. Profile that mesh instead."
            )
        return self.artifact.entry_for(
            Parallelism(tp=1, pp=1, dp=1, cp=1),
            selector=_selector(
                component=component,
                shard_size=self.shard_size,
                replicate_size=self.replicate_size,
                recompute=self.recompute,
                optimizer=workload.optimizer,
                parameter_dtype=self.parameter_dtype,
                gradient_dtype=self.gradient_dtype,
                optimizer_state_dtype=self.optimizer_state_dtype,
                optimizer_offload=self.optimizer_offload,
                reshard_after_forward=self.reshard_after_forward,
            ),
        )

    def _microbatches(
        self,
        workload: TrainingWorkload,
        parallelism: TrainingParallelism,
    ) -> tuple[SequenceStats, ...]:
        """
        Plan the critical rank's microbatches for a real, ragged RL batch.

        `TrainingWorkload.critical_rank_microbatches` assumes uniform equal-count
        microbatches, which is exactly what RL trajectory lengths are not, and the
        batching mode is worth a measured 1.17x to 4.34x on DAPO-like lengths.
        """
        return self.planner.plan(
            workload.sequence_lengths, parallelism.dp, micro_batch_size=workload.micro_batch_size
        )


class FsdpLatencyModel(_FsdpCellModel, TrainingLatencyModel):
    """Evaluate a fitted FSDP latency cell with explicit out-of-domain checks."""

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
        entry = self._entry("latency", workload, parallelism)
        microbatches = self._microbatches(workload, parallelism)
        features = latency_features(microbatches, TrainingParallelism(dp=parallelism.dp))
        notes: list[str] = []
        for name, value in (
            ("tokens", features.tokens),
            ("num_microbatches", features.slots),
            (
                "sequences_per_microbatch",
                float(max(item.count for item in microbatches)),
            ),
        ):
            note = check_feature_range(entry, name, value, extrapolate=self.extrapolate)
            if note is not None:
                notes.append(note)
        coefficients = entry.coefficients
        # Shared with hydraulis so the formula exists once. It raises on a non-positive
        # result, and it reads `features.sum_squared_tokens`: the same LatencyFeatures the
        # fit was built from, so predict and fit cannot disagree on the same input.
        return latency_from_coefficients(coefficients, features), notes


class FsdpMemoryModel(_FsdpCellModel, TrainingMemoryModel):
    """Combine the FSDP2 static inventory with a fitted activation residual."""

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
        """Return (peak_memory_bytes, extrapolation_notes) so the caller can log them."""
        entry = self._entry("memory", workload, parallelism)
        microbatches = self._microbatches(workload, parallelism)
        num_microbatches, max_tokens = _memory_features(microbatches)
        notes: list[str] = []
        for name, value in (
            ("num_microbatches", num_microbatches),
            ("activation_tokens", max_tokens),
        ):
            note = check_feature_range(entry, name, value, extrapolate=self.extrapolate)
            if note is not None:
                notes.append(note)
        coefficients = entry.coefficients
        # The world size comes from the fitted cell rather than from `parallelism.dp`. The
        # two are equal by the mesh check in `_entry`, and reading the artifact keeps the
        # static term tied to the measurement it was fitted against.
        static = fsdp_inventory(
            parameter_count=int(coefficients["parameter_count"]),
            world_size=int(coefficients["world_size"]),
            parameter_dtype=self.parameter_dtype,
            gradient_dtype=self.gradient_dtype,
            optimizer_state_dtype=self.optimizer_state_dtype,
            optimizer_offload=self.optimizer_offload,
        ).total_bytes
        predicted = (
            static
            + coefficients["microbatch_slope"] * num_microbatches
            + coefficients["activation_slope"] * max_tokens
            + coefficients["residual_constant"]
        )
        if entry.diagnostics["memory_metric"] == "peak_allocated_bytes":
            # Allocated excludes allocator fragmentation, which is what actually exhausts
            # the device, so the capacity comparison needs the measured ratio.
            predicted *= float(entry.diagnostics["reserved_over_allocated"])
        return int(math.ceil(predicted)), notes


class FsdpTrainingCost(TrainingCostModel):
    """Compose FSDP2 latency and memory from one artifact."""

    model_type = "fsdp"

    @classmethod
    def fit(
        cls,
        records: list[ProfileRecord],
        *,
        validation_fraction: float = 0.2,
        require_full_rank: bool = True,
    ) -> CostArtifact:
        """Fit latency and memory entries for every profiled FSDP cell."""
        groups, materialized = _fsdp_records(records)
        model_names = {record.model.name for record in materialized}
        gpu_names = {record.hardware.gpu_name for record in materialized}
        if len(model_names) != 1 or len(gpu_names) != 1:
            raise ValueError("A training artifact must target one model and one GPU type.")
        first = materialized[0]
        model = asdict(first.model)
        hardware = asdict(first.hardware)
        hardware["gpu_count"] = sorted({record.hardware.gpu_count for record in materialized})
        entries: list[ArtifactEntry] = []
        for key, cell_records in groups.items():
            latency_entry = _fit_latency_entry(
                key,
                cell_records,
                validation_fraction=validation_fraction,
                require_full_rank=require_full_rank,
            )
            memory_entry = _fit_memory_entry(
                key,
                cell_records,
                validation_fraction=validation_fraction,
                require_full_rank=require_full_rank,
            )
            if latency_entry is not None:
                entries.append(latency_entry)
            if memory_entry is not None:
                entries.append(memory_entry)
        if not entries:
            raise ValueError("No FSDP entries could be fitted.")
        return CostArtifact(
            kind="training",
            model_type=cls.model_type,
            model=model,
            hardware=hardware,
            profile_digest=profile_digest(materialized),
            entries=tuple(entries),
            diagnostics={
                "supported_schedule": [FSDP_SCHEDULE],
                "supported_sequence_parallel": [1],
                "mesh_axes": ["shard_size", "replicate_size"],
                "dp_communication": "ignored",
                "activation_checkpoint_recompute": "discrete-profiled-selector",
            },
        )

    @classmethod
    def from_artifact(
        cls,
        artifact_or_path: CostArtifact | str | Path,
        *,
        batching_mode: str = "pad_to_max",
        max_tokens_per_rank: int | None = None,
        extrapolate: list[str] | None = None,
        **selectors: Any,
    ) -> FsdpTrainingCost:
        """Load composable latency and memory models for one discrete cell.

        Args:
            extrapolate: Feature names for which out-of-domain queries are permitted.
                An empty list or None (the default) keeps the strict behaviour: any query
                outside the profiled range raises TrainingOutOfDomainError. When a name
                is listed and the query is out of range, the prediction proceeds and the
                exceedance is recorded in TrainingCostEstimate.diagnostics under the key
                'extrapolation_notes'. Unknown names raise ValueError at construction so a
                typo cannot silently enable nothing. Valid names per component:
                  latency: num_microbatches, sequences_per_microbatch, tokens
                  memory:  num_microbatches, activation_tokens
        """
        artifact = (
            artifact_or_path
            if isinstance(artifact_or_path, CostArtifact)
            else CostArtifact.load(artifact_or_path)
        )
        if artifact.kind != "training" or artifact.model_type != cls.model_type:
            raise ValueError("Artifact does not contain an FSDP training model.")
        known = (
            "shard_size",
            "replicate_size",
            "recompute",
            "optimizer_offload",
            "reshard_after_forward",
            "parameter_dtype",
            "gradient_dtype",
            "optimizer_state_dtype",
        )
        unexpected = set(selectors) - set(known)
        if unexpected:
            raise ValueError(f"Unknown FSDP selectors {sorted(unexpected)}.")
        entry_selectors = [entry.selector for entry in artifact.entries]

        def infer(name: str) -> Any:
            explicit = selectors.get(name)
            if explicit is not None:
                return explicit
            values = {selector[name] for selector in entry_selectors}
            if len(values) != 1:
                raise ValueError(f"Artifact has multiple {name} values; select one explicitly.")
            return values.pop()

        extrapolate_set = frozenset(extrapolate) if extrapolate else frozenset()
        # Validate names against the union of all feature_ranges keys so the check is
        # correct regardless of which entry the query will eventually hit. Both latency
        # and memory entries are checked; a name valid for one component but not the other
        # is still valid here because the user is naming features, not components.
        all_valid_names = frozenset().union(
            *(frozenset(entry.feature_ranges.keys()) for entry in artifact.entries)
        )
        validate_extrapolate_names(
            extrapolate_set, all_valid_names, context="FSDP artifact"
        )

        common = {
            "artifact": artifact,
            "shard_size": int(infer("shard_size")),
            "replicate_size": int(infer("replicate_size")),
            "recompute": str(infer("recompute")),
            "optimizer_offload": bool(infer("optimizer_offload")),
            "reshard_after_forward": bool(infer("reshard_after_forward")),
            "parameter_dtype": str(infer("parameter_dtype")),
            "gradient_dtype": str(infer("gradient_dtype")),
            "optimizer_state_dtype": str(infer("optimizer_state_dtype")),
            "planner": MicrobatchPlanner(
                BatchingMode(batching_mode),
                max_tokens_per_rank=max_tokens_per_rank,
            ),
            "extrapolate": extrapolate_set,
        }
        return cls(
            latency_model=FsdpLatencyModel(**common),
            memory_model=FsdpMemoryModel(**common),
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
