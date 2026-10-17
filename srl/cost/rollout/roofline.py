"""Unified rollout roofline model with model-owned regression."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

from srl.cost.artifact import ArtifactEntry, CostArtifact, FitMetrics
from srl.profiler.schema import Parallelism, ProfileRecord

from ._fit import artifact_context, feature_ranges, robust_nonnegative_fit, rollout_groups, split_by_run
from .base import RolloutCostModel, RolloutWorkload


class UnifiedRoofline(RolloutCostModel):
    """Predict ``max(F, W + G*n_tokens + A_p*sum_l2/b + A_d*ctxsum)``."""

    model_type = "roofline"

    def __init__(
        self,
        F,
        W=0,
        G=0,
        A_p=0,
        A_d=0,
        b: int = 16,
    ):
        if b <= 0:
            raise ValueError("b must be positive.")
        if any(value < 0 for value in (F, W, G, A_p, A_d)):
            raise ValueError("Roofline coefficients must be non-negative.")
        self.F = F
        self.W = W
        self.G = G
        self.A_p = A_p
        self.A_d = A_d
        self.b = b

    def _raw(self, workload: RolloutWorkload):
        prefill_sum_l2 = workload.sum_l2 if workload.phase in {"prefill", "mixed"} else 0
        decode_ctxsum = workload.ctxsum if workload.phase in {"decode", "mixed"} else 0
        return (
            self.W
            + self.G * workload.n_tokens
            + self.A_p * prefill_sum_l2 / self.b
            + self.A_d * decode_ctxsum
        )

    def predict(self, workload: RolloutWorkload):
        raw = self._raw(workload)
        latency = self.F if raw < self.F else raw
        if latency <= 0:
            raise ValueError("A rollout cost model must predict positive latency.")
        return latency

    def saturated(self, n_tokens, t2, ctxsum, n_reqs=0) -> bool:
        workload = RolloutWorkload.from_step(n_tokens, t2, ctxsum, n_reqs)
        return self._raw(workload) >= self.F

    def decode_coeffs(self, n_d: int, ctxsum):
        alpha = self.W + self.G * n_d + self.A_d * ctxsum
        beta = self.A_d * n_d
        return self.F, alpha, beta

    @classmethod
    def fit(
        cls,
        records: list[ProfileRecord],
        *,
        block_size: int = 16,
        validation_fraction: float = 0.2,
    ) -> CostArtifact:
        """Fit independent non-negative roofline cells for every topology."""
        groups, materialized = rollout_groups(records)
        model, hardware, digest = artifact_context(materialized)
        entries: list[ArtifactEntry] = []
        for parallelism, samples in groups.items():
            train, validation = split_by_run(samples, validation_fraction=validation_fraction)
            train_y = np.asarray([sample.payload.latency_s for sample in train], dtype=np.float64)
            train_x = np.asarray(
                [
                    [
                        1.0,
                        sample.payload.n_tokens,
                        sample.payload.sum_l2 / block_size
                        if sample.payload.phase in {"prefill", "mixed"}
                        else 0.0,
                        sample.payload.ctxsum
                        if sample.payload.phase in {"decode", "mixed"}
                        else 0.0,
                    ]
                    for sample in train
                ],
                dtype=np.float64,
            )
            scales = np.maximum(np.max(np.abs(train_x), axis=0), 1.0)
            scaled_x = train_x / scales
            initial = _roofline_initial(scaled_x, train_y)

            def residual(parameters: np.ndarray) -> np.ndarray:
                floor = parameters[0]
                raw = scaled_x @ parameters[1:]
                return np.maximum(floor, raw) - train_y

            parameters, inlier_mask = robust_nonnegative_fit(
                residual,
                initial=initial,
                n_samples=len(train),
            )
            linear_parameters = parameters[1:] / scales
            fitted = cls(
                F=parameters[0],
                W=linear_parameters[0],
                G=linear_parameters[1],
                A_p=linear_parameters[2],
                A_d=linear_parameters[3],
                b=block_size,
            )
            train_predicted = np.asarray(
                [fitted.predict(_payload_workload(sample.payload)) for sample in train],
                dtype=np.float64,
            )
            validation_y = np.asarray(
                [sample.payload.latency_s for sample in validation],
                dtype=np.float64,
            )
            validation_predicted = np.asarray(
                [fitted.predict(_payload_workload(sample.payload)) for sample in validation],
                dtype=np.float64,
            )
            rank = int(np.linalg.matrix_rank(train_x))
            outliers = int(np.sum(~inlier_mask))
            entries.append(
                ArtifactEntry(
                    parallelism=parallelism,
                    selector={"block_size": block_size},
                    coefficients={
                        "F": float(parameters[0]),
                        "W": float(linear_parameters[0]),
                        "G": float(linear_parameters[1]),
                        "A_p": float(linear_parameters[2]),
                        "A_d": float(linear_parameters[3]),
                        "b": float(block_size),
                    },
                    train_metrics=FitMetrics.from_predictions(
                        train_y,
                        train_predicted,
                        n_outliers=outliers,
                        design_rank=rank,
                    ),
                    validation_metrics=FitMetrics.from_predictions(
                        validation_y,
                        validation_predicted,
                        design_rank=rank,
                    ),
                    feature_ranges=feature_ranges(samples),
                    diagnostics={
                        "fit_method": "nonnegative-soft-l1",
                        "grouped_holdout": True,
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
    ) -> UnifiedRoofline:
        """Load one exact topology from a versioned artifact."""
        artifact = (
            artifact_or_path
            if isinstance(artifact_or_path, CostArtifact)
            else CostArtifact.load(artifact_or_path)
        )
        if artifact.kind != "rollout" or artifact.model_type != cls.model_type:
            raise ValueError(f"Artifact does not contain a {cls.model_type!r} rollout model.")
        entry = artifact.entry_for(parallelism)
        coefficients = entry.coefficients
        return cls(
            F=coefficients["F"],
            W=coefficients["W"],
            G=coefficients["G"],
            A_p=coefficients["A_p"],
            A_d=coefficients["A_d"],
            b=int(coefficients["b"]),
        )


def _roofline_initial(
    design: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    candidates = np.unique(
        np.concatenate(
            (
                np.asarray([0.0]),
                target[target <= np.percentile(target, 50)],
            )
        )
    )
    best_score = float("inf")
    best = None
    for floor in candidates:
        active = target > floor + max(1e-12, abs(floor) * 1e-8)
        source_x = design[active] if int(np.sum(active)) >= design.shape[1] else design
        source_y = target[active] if int(np.sum(active)) >= design.shape[1] else target
        # Deferred because only fitting needs scipy, and simulation should not pay its import.
        from scipy.optimize import lsq_linear

        linear = lsq_linear(source_x, source_y, bounds=(0, np.inf)).x
        predicted = np.maximum(floor, design @ linear)
        score = float(np.sum((predicted - target) ** 2))
        if score < best_score:
            best_score = score
            best = np.concatenate((np.asarray([floor]), linear))
    if best is None:
        raise ValueError("Unable to initialize roofline regression.")
    return best


def _payload_workload(payload) -> RolloutWorkload:
    return RolloutWorkload(
        phase=payload.phase,
        n_tokens=payload.n_tokens,
        sum_l2=payload.sum_l2,
        ctxsum=payload.ctxsum,
        request_count=payload.request_count,
    )


class LinearLPS(UnifiedRoofline):
    """Limited processor sharing model used for closed-form validation."""

    def __init__(
        self,
        r1=Fraction(1),
        K: int = 1,
    ):
        r1 = Fraction(r1)
        self.r1 = r1
        self.K = K
        super().__init__(
            F=1 / r1,
            W=0,
            G=1 / (r1 * K),
            A_p=0,
            A_d=0,
            b=1,
        )
