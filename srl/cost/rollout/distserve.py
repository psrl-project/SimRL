"""DistServe-style architecture-calibrated rollout model."""

from __future__ import annotations

from pathlib import Path

from srl.cost.artifact import ArtifactEntry, CostArtifact
from srl.profiler.schema import Parallelism, ProfileRecord

from .base import RolloutCostModel, RolloutWorkload
from .roofline import UnifiedRoofline


class DistServe(RolloutCostModel):
    """Apply DistServe Appendix A constants to one model architecture."""

    model_type = "distserve"

    def __init__(
        self,
        *,
        h: int,
        m: int,
        C1: float,
        C2: float,
        C3: float,
        C4: float,
        C5: float,
        b: int = 16,
        F: float = 0,
    ):
        if h <= 0 or m <= 0 or b <= 0:
            raise ValueError("Architecture dimensions and b must be positive.")
        if any(value < 0 for value in (C1, C2, C3, C4, C5, F)):
            raise ValueError("DistServe coefficients must be non-negative.")
        self.h = h
        self.m = m
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        self.b = b
        self.F = F
        self.gemm = 4 * h * h + 2 * h * m
        self._W = C3 + C4 * self.gemm
        self._G = C1 * self.gemm
        self._A_p = C2 * 3 * h
        self._A_d = C5 * 3 * h

    def _raw(self, workload: RolloutWorkload):
        prefill_sum_l2 = workload.sum_l2 if workload.phase in {"prefill", "mixed"} else 0
        decode_ctxsum = workload.ctxsum if workload.phase in {"decode", "mixed"} else 0
        return (
            self._W
            + self._G * workload.n_tokens
            + self._A_p * prefill_sum_l2 / self.b
            + self._A_d * decode_ctxsum
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
        alpha = self._W + self._G * n_d + self._A_d * ctxsum
        beta = self._A_d * n_d
        return self.F, alpha, beta

    @classmethod
    def fit(
        cls,
        records: list[ProfileRecord],
        *,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        block_size: int = 16,
        validation_fraction: float = 0.2,
    ) -> CostArtifact:
        """Fit effective hardware constants through a roofline calibration."""
        if not records:
            raise ValueError("At least one profiling record is required.")
        descriptor = records[0].model
        h = hidden_size or descriptor.hidden_size
        m = intermediate_size or descriptor.extra.get("intermediate_size")
        if h is None or m is None:
            raise ValueError("DistServe fitting requires hidden_size and intermediate_size.")
        h = int(h)
        m = int(m)
        gemm = 4 * h * h + 2 * h * m
        roofline_artifact = UnifiedRoofline.fit(
            records,
            block_size=block_size,
            validation_fraction=validation_fraction,
        )
        entries: list[ArtifactEntry] = []
        for source in roofline_artifact.entries:
            roofline = source.coefficients
            overhead = roofline["W"]
            denominator = 1.0 + float(gemm) ** 2
            c3 = overhead / denominator
            c4 = overhead * gemm / denominator
            coefficients = {
                "h": float(h),
                "m": float(m),
                "C1": roofline["G"] / gemm,
                "C2": roofline["A_p"] / (3 * h),
                "C3": c3,
                "C4": c4,
                "C5": roofline["A_d"] / (3 * h),
                "b": roofline["b"],
                "F": roofline["F"],
            }
            diagnostics = dict(source.diagnostics)
            diagnostics.update(
                {
                    "c3_c4_identifiability": "single-model-canonical-minimum-norm",
                    "effective_overhead": overhead,
                    "gemm_flops_per_token_layer": gemm,
                }
            )
            entries.append(
                ArtifactEntry(
                    parallelism=source.parallelism,
                    selector={
                        "block_size": block_size,
                        "hidden_size": h,
                        "intermediate_size": m,
                    },
                    coefficients=coefficients,
                    train_metrics=source.train_metrics,
                    validation_metrics=source.validation_metrics,
                    feature_ranges=source.feature_ranges,
                    diagnostics=diagnostics,
                )
            )
        return CostArtifact(
            kind="rollout",
            model_type=cls.model_type,
            model=roofline_artifact.model,
            hardware=roofline_artifact.hardware,
            profile_digest=roofline_artifact.profile_digest,
            entries=tuple(entries),
            diagnostics={
                "calibration_scope": "single-model",
                "c3_c4_solution": "canonical-minimum-norm",
            },
        )

    @classmethod
    def from_artifact(
        cls,
        artifact_or_path: CostArtifact | str | Path,
        parallelism: Parallelism,
    ) -> DistServe:
        """Load one exact DistServe topology from an artifact."""
        artifact = (
            artifact_or_path
            if isinstance(artifact_or_path, CostArtifact)
            else CostArtifact.load(artifact_or_path)
        )
        if artifact.kind != "rollout" or artifact.model_type != cls.model_type:
            raise ValueError(f"Artifact does not contain a {cls.model_type!r} rollout model.")
        coefficients = artifact.entry_for(parallelism).coefficients
        return cls(
            h=int(coefficients["h"]),
            m=int(coefficients["m"]),
            C1=coefficients["C1"],
            C2=coefficients["C2"],
            C3=coefficients["C3"],
            C4=coefficients["C4"],
            C5=coefficients["C5"],
            b=int(coefficients["b"]),
            F=coefficients["F"],
        )
