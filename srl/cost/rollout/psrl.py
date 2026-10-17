"""PSRL empirical rollout model with separate prefill and decode fits."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from srl.cost.artifact import ArtifactEntry, CostArtifact, FitMetrics
from srl.profiler.schema import Parallelism, ProfileRecord

from ._fit import (
    RolloutSample,
    artifact_context,
    feature_ranges,
    robust_nonnegative_fit,
    rollout_groups,
    split_by_run,
)
from .base import RolloutCostModel, RolloutWorkload


class PSRLFitted(RolloutCostModel):
    """Decompose each phase into attention and low-load non-attention terms."""

    model_type = "psrl"

    def __init__(
        self,
        *,
        prefill_attn_b: float,
        prefill_attn_k: float,
        prefill_other_floor: float,
        prefill_other_b: float,
        prefill_other_k: float,
        decode_attn_b: float,
        decode_attn_k: float,
        decode_other_floor: float,
        decode_other_b: float,
        decode_other_k: float,
    ):
        coefficients = {
            "prefill_attn_b": prefill_attn_b,
            "prefill_attn_k": prefill_attn_k,
            "prefill_other_floor": prefill_other_floor,
            "prefill_other_b": prefill_other_b,
            "prefill_other_k": prefill_other_k,
            "decode_attn_b": decode_attn_b,
            "decode_attn_k": decode_attn_k,
            "decode_other_floor": decode_other_floor,
            "decode_other_b": decode_other_b,
            "decode_other_k": decode_other_k,
        }
        if any(value < 0 for value in coefficients.values()):
            raise ValueError("PSRL coefficients must be non-negative.")
        for name, value in coefficients.items():
            setattr(self, name, value)

    def _prefill_terms(self, workload: RolloutWorkload) -> tuple[float, float]:
        attention = self.prefill_attn_b + self.prefill_attn_k * workload.sum_l2
        linear = self.prefill_other_b + self.prefill_other_k * workload.n_tokens
        other = max(self.prefill_other_floor, linear)
        return attention, other

    def _decode_terms(self, workload: RolloutWorkload) -> tuple[float, float]:
        attention = self.decode_attn_b + self.decode_attn_k * workload.ctxsum
        linear = self.decode_other_b + self.decode_other_k * workload.request_count
        other = max(self.decode_other_floor, linear)
        return attention, other

    def predict(self, workload: RolloutWorkload):
        if workload.phase == "prefill":
            attention, other = self._prefill_terms(workload)
            latency = attention + other
        elif workload.phase == "decode":
            attention, other = self._decode_terms(workload)
            latency = attention + other
        else:
            prefill_attention, prefill_other = self._prefill_terms(workload)
            decode_attention, decode_other = self._decode_terms(workload)
            latency = prefill_attention + decode_attention + max(prefill_other, decode_other)
        if latency <= 0:
            raise ValueError("A rollout cost model must predict positive latency.")
        return latency

    def saturated(self, n_tokens, t2, ctxsum, n_reqs=0) -> bool:
        workload = RolloutWorkload.from_step(n_tokens, t2, ctxsum, n_reqs)
        if workload.phase == "prefill":
            return self.prefill_other_b + self.prefill_other_k * n_tokens >= self.prefill_other_floor
        if workload.phase == "decode":
            return self.decode_other_b + self.decode_other_k * n_reqs >= self.decode_other_floor
        return (
            self.prefill_other_b + self.prefill_other_k * n_tokens >= self.prefill_other_floor
            and self.decode_other_b + self.decode_other_k * n_reqs >= self.decode_other_floor
        )

    def decode_coeffs(self, n_d: int, ctxsum):
        other = max(self.decode_other_floor, self.decode_other_b + self.decode_other_k * n_d)
        alpha = self.decode_attn_b + self.decode_attn_k * ctxsum + other
        beta = self.decode_attn_k * n_d
        return 0, alpha, beta

    @classmethod
    def fit(
        cls,
        records: list[ProfileRecord],
        *,
        validation_fraction: float = 0.2,
    ) -> CostArtifact:
        """Fit phase-specific PSRL coefficients for every profiled topology."""
        groups, materialized = rollout_groups(records)
        model, hardware, digest = artifact_context(materialized)
        entries: list[ArtifactEntry] = []
        for parallelism, samples in groups.items():
            phase_samples = {
                phase: [sample for sample in samples if sample.payload.phase == phase]
                for phase in ("prefill", "decode", "mixed")
            }
            if len(phase_samples["prefill"]) < 5 or len(phase_samples["decode"]) < 5:
                raise ValueError(
                    f"PSRL needs at least five prefill and five decode observations for {parallelism!r}."
                )
            prefill_train, prefill_validation = split_by_run(
                phase_samples["prefill"],
                validation_fraction=validation_fraction,
            )
            decode_train, decode_validation = split_by_run(
                phase_samples["decode"],
                validation_fraction=validation_fraction,
            )
            mixed_train, mixed_validation = split_by_run(
                phase_samples["mixed"],
                validation_fraction=validation_fraction,
            ) if phase_samples["mixed"] else ([], [])

            prefill_parameters, prefill_outliers, prefill_rank = _fit_phase(
                prefill_train,
                phase="prefill",
            )
            decode_parameters, decode_outliers, decode_rank = _fit_phase(
                decode_train,
                phase="decode",
            )
            fitted = cls(
                prefill_attn_b=prefill_parameters[0],
                prefill_attn_k=prefill_parameters[1],
                prefill_other_floor=prefill_parameters[2],
                prefill_other_b=prefill_parameters[3],
                prefill_other_k=prefill_parameters[4],
                decode_attn_b=decode_parameters[0],
                decode_attn_k=decode_parameters[1],
                decode_other_floor=decode_parameters[2],
                decode_other_b=decode_parameters[3],
                decode_other_k=decode_parameters[4],
            )
            train = prefill_train + decode_train + mixed_train
            validation = prefill_validation + decode_validation + mixed_validation
            train_y, train_predicted = _evaluate(fitted, train)
            validation_y, validation_predicted = _evaluate(fitted, validation)
            coefficients = {
                "prefill_attn_b": float(prefill_parameters[0]),
                "prefill_attn_k": float(prefill_parameters[1]),
                "prefill_other_floor": float(prefill_parameters[2]),
                "prefill_other_b": float(prefill_parameters[3]),
                "prefill_other_k": float(prefill_parameters[4]),
                "decode_attn_b": float(decode_parameters[0]),
                "decode_attn_k": float(decode_parameters[1]),
                "decode_other_floor": float(decode_parameters[2]),
                "decode_other_b": float(decode_parameters[3]),
                "decode_other_k": float(decode_parameters[4]),
            }
            entries.append(
                ArtifactEntry(
                    parallelism=parallelism,
                    selector={},
                    coefficients=coefficients,
                    train_metrics=FitMetrics.from_predictions(
                        train_y,
                        train_predicted,
                        n_outliers=prefill_outliers + decode_outliers,
                        design_rank=min(prefill_rank, decode_rank),
                    ),
                    validation_metrics=FitMetrics.from_predictions(
                        validation_y,
                        validation_predicted,
                        design_rank=min(prefill_rank, decode_rank),
                    ),
                    feature_ranges=feature_ranges(samples),
                    diagnostics={
                        "fit_method": "phase-specific-nonnegative-soft-l1",
                        "prefill_design_rank": prefill_rank,
                        "decode_design_rank": decode_rank,
                        "mixed_step_rule": "attention-sum-other-max",
                    },
                )
            )
        return CostArtifact(
            kind="rollout",
            model_type=cls.model_type,
            model=model,
            hardware=hardware,
            profile_digest=digest,
            entries=tuple(entries),
        )

    @classmethod
    def from_artifact(
        cls,
        artifact_or_path: CostArtifact | str | Path,
        parallelism: Parallelism,
    ) -> PSRLFitted:
        """Load one exact PSRL topology from an artifact."""
        artifact = (
            artifact_or_path
            if isinstance(artifact_or_path, CostArtifact)
            else CostArtifact.load(artifact_or_path)
        )
        if artifact.kind != "rollout" or artifact.model_type != cls.model_type:
            raise ValueError(f"Artifact does not contain a {cls.model_type!r} rollout model.")
        return cls(**artifact.entry_for(parallelism).coefficients)


def _fit_phase(
    samples: list[RolloutSample],
    *,
    phase: str,
) -> tuple[np.ndarray, int, int]:
    y = np.asarray([sample.payload.latency_s for sample in samples], dtype=np.float64)
    if phase == "prefill":
        attention_feature = np.asarray(
            [sample.payload.sum_l2 for sample in samples],
            dtype=np.float64,
        )
        other_feature = np.asarray(
            [sample.payload.n_tokens for sample in samples],
            dtype=np.float64,
        )
    else:
        attention_feature = np.asarray(
            [sample.payload.ctxsum for sample in samples],
            dtype=np.float64,
        )
        other_feature = np.asarray(
            [sample.payload.request_count for sample in samples],
            dtype=np.float64,
        )
    design = np.column_stack((np.ones(len(samples)), attention_feature, other_feature))
    linear = np.maximum(np.linalg.lstsq(design, y, rcond=None)[0], 0)
    floor = max(float(np.percentile(y, 10) * 0.8), 1e-9)
    initial = np.asarray([0.0, linear[1], floor, max(linear[0] - floor, 0.0), linear[2]])

    def residual(parameters: np.ndarray) -> np.ndarray:
        attn_b, attn_k, other_floor, other_b, other_k = parameters
        predicted = (
            attn_b
            + attn_k * attention_feature
            + np.maximum(other_floor, other_b + other_k * other_feature)
        )
        return predicted - y

    parameters, inlier_mask = robust_nonnegative_fit(
        residual,
        initial=initial,
        n_samples=len(samples),
    )
    return parameters, int(np.sum(~inlier_mask)), int(np.linalg.matrix_rank(design))


def _evaluate(
    model: PSRLFitted,
    samples: list[RolloutSample],
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray([sample.payload.latency_s for sample in samples], dtype=np.float64)
    predicted = np.asarray(
        [
            model.predict(
                RolloutWorkload(
                    phase=sample.payload.phase,
                    n_tokens=sample.payload.n_tokens,
                    sum_l2=sample.payload.sum_l2,
                    ctxsum=sample.payload.ctxsum,
                    request_count=sample.payload.request_count,
                )
            )
            for sample in samples
        ],
        dtype=np.float64,
    )
    return actual, predicted
