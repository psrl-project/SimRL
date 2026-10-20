"""Regression diagnostics for fitted vLLM rollout cost models."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from srl.cost.artifact import ArtifactEntry, CostArtifact, FitMetrics  # noqa: E402
from srl.profiler.schema import Parallelism, ProfileRecord  # noqa: E402

from . import ROLLOUT_MODEL_REGISTRY  # noqa: E402
from ._fit import RolloutSample, rollout_groups, split_by_run  # noqa: E402
from .base import RolloutCostModel, RolloutWorkload  # noqa: E402

_MAX_SCATTER_POINTS = 4_000
_SURFACE_POINTS_PER_AXIS = 36


def plot_vllm_rollout_regression(
    records: Sequence[ProfileRecord],
    artifact: CostArtifact,
    output_dir: str | Path,
    *,
    validation_fraction: float = 0.2,
) -> list[Path]:
    """Plot phase-specific fit diagnostics for every profiled topology."""
    if artifact.kind != "rollout":
        raise ValueError("Regression plots require a rollout cost artifact.")
    if artifact.model_type not in ROLLOUT_MODEL_REGISTRY:
        raise ValueError(f"Unsupported rollout model {artifact.model_type!r}.")
    if any(record.software.backend != "vllm" for record in records):
        raise ValueError("Rollout regression plots currently support only vLLM records.")

    groups, _ = rollout_groups(records)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    model_class = ROLLOUT_MODEL_REGISTRY[artifact.model_type]

    for parallelism, samples in groups.items():
        entry = artifact.entry_for(parallelism)
        model = model_class.from_artifact(artifact, parallelism)
        phase_splits = _phase_splits(
            samples,
            model_type=artifact.model_type,
            validation_fraction=validation_fraction,
        )
        path = destination / _plot_filename(artifact.model_type, parallelism)
        _plot_topology(
            path,
            artifact=artifact,
            entry=entry,
            model=model,
            phase_splits=phase_splits,
        )
        paths.append(path)
    return paths


def _phase_splits(
    samples: list[RolloutSample],
    *,
    model_type: str,
    validation_fraction: float,
) -> dict[str, tuple[list[RolloutSample], list[RolloutSample]]]:
    pure = {
        phase: [sample for sample in samples if sample.payload.phase == phase]
        for phase in ("prefill", "decode")
    }
    if model_type == "psrl":
        return {
            phase: split_by_run(phase_samples, validation_fraction=validation_fraction)
            for phase, phase_samples in pure.items()
        }

    train, validation = split_by_run(samples, validation_fraction=validation_fraction)
    return {
        phase: (
            [sample for sample in train if sample.payload.phase == phase],
            [sample for sample in validation if sample.payload.phase == phase],
        )
        for phase in ("prefill", "decode")
    }


def _plot_topology(
    path: Path,
    *,
    artifact: CostArtifact,
    entry: ArtifactEntry,
    model: RolloutCostModel,
    phase_splits: dict[str, tuple[list[RolloutSample], list[RolloutSample]]],
) -> None:
    fig = plt.figure(figsize=(17, 13))
    for row, phase in enumerate(("prefill", "decode")):
        train, validation = phase_splits[phase]
        surface_ax = fig.add_subplot(2, 2, row * 2 + 1, projection="3d")
        parity_ax = fig.add_subplot(2, 2, row * 2 + 2)
        _plot_surface(surface_ax, model, phase, train, validation)
        _plot_phase_curves(parity_ax, model, phase, train, validation)

    topology = entry.parallelism
    fig.suptitle(
        f"vLLM {artifact.model_type} rollout regression: "
        f"tp={topology.tp}, pp={topology.pp}, dp={topology.dp}, cp={topology.cp}",
        fontsize=15,
    )
    fig.text(
        0.02,
        0.01,
        _coefficient_text(artifact.model_type, entry),
        ha="left",
        va="bottom",
        family="monospace",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_surface(
    ax,
    model: RolloutCostModel,
    phase: str,
    train: list[RolloutSample],
    validation: list[RolloutSample],
) -> None:
    x_name, y_name = _phase_features(phase)
    all_samples = train + validation
    if not all_samples:
        ax.text2D(0.5, 0.5, f"No {phase} observations", ha="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    x = np.asarray([getattr(sample.payload, x_name) for sample in all_samples], dtype=np.float64)
    y = np.asarray([getattr(sample.payload, y_name) for sample in all_samples], dtype=np.float64)
    log_x_grid = np.linspace(np.log10(np.min(x)), np.log10(np.max(x)), _SURFACE_POINTS_PER_AXIS)
    log_y_grid = np.linspace(np.log10(np.min(y)), np.log10(np.max(y)), _SURFACE_POINTS_PER_AXIS)
    log_x_mesh, log_y_mesh = np.meshgrid(log_x_grid, log_y_grid)
    x_mesh = np.power(10.0, log_x_mesh)
    y_mesh = np.power(10.0, log_y_mesh)
    predicted_mesh = np.asarray(
        [
            _predict_features(model, phase, x_value, y_value)
            for x_value, y_value in zip(x_mesh.ravel(), y_mesh.ravel())
        ],
        dtype=np.float64,
    ).reshape(x_mesh.shape)
    ax.plot_surface(
        log_x_mesh,
        log_y_mesh,
        predicted_mesh,
        cmap="viridis",
        alpha=0.45,
        linewidth=0,
        antialiased=True,
    )

    for samples, marker, color, label in (
        (train, "o", "tab:blue", "train actual"),
        (validation, "^", "tab:orange", "validation actual"),
    ):
        selected = _downsample(samples)
        if not selected:
            continue
        scatter_x = np.log10([getattr(sample.payload, x_name) for sample in selected])
        scatter_y = np.log10([getattr(sample.payload, y_name) for sample in selected])
        latency = [sample.payload.latency_s for sample in selected]
        ax.scatter(scatter_x, scatter_y, latency, marker=marker, color=color, s=9, alpha=0.5, label=label)

    ax.set_title(f"{phase}: measured points and fitted surface")
    ax.set_xlabel(f"log10({x_name})")
    ax.set_ylabel(f"log10({y_name})")
    ax.set_zlabel("latency_s")
    ax.legend(fontsize=8)


def _plot_phase_curves(
    ax,
    model: RolloutCostModel,
    phase: str,
    train: list[RolloutSample],
    validation: list[RolloutSample],
) -> None:
    all_samples = train + validation
    if not all_samples:
        ax.text(0.5, 0.5, f"No {phase} observations", ha="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    x_name = "sum_l2" if phase == "prefill" else "ctxsum"
    request_counts = sorted({sample.payload.request_count for sample in all_samples})
    count_min, count_max = min(request_counts), max(request_counts)
    norm = Normalize(vmin=count_min, vmax=count_max if count_max > count_min else count_min + 1)
    cmap = plt.get_cmap("viridis")

    by_request_count: dict[int, list[RolloutSample]] = {
        request_count: [
            sample
            for sample in all_samples
            if sample.payload.request_count == request_count
        ]
        for request_count in request_counts
    }
    for request_count, samples in by_request_count.items():
        observed_x = np.asarray(
            [getattr(sample.payload, x_name) for sample in samples],
            dtype=np.float64,
        )
        x_min, x_max = float(np.min(observed_x)), float(np.max(observed_x))
        if x_min == x_max:
            x_grid = np.asarray([x_min])
        else:
            x_grid = np.geomspace(x_min, x_max, 120)
        predicted = [
            _predict_phase_curve(model, phase, request_count, x_value)
            for x_value in x_grid
        ]
        ax.plot(
            x_grid,
            predicted,
            color=cmap(norm(request_count)),
            linewidth=1,
            alpha=0.75,
            zorder=1,
        )

    metrics: list[str] = []
    for samples, marker, label in (
        (train, "o", "train actual"),
        (validation, "^", "validation actual"),
    ):
        actual, predicted = _evaluate(model, samples)
        if actual.size:
            metrics.append(_metrics_line(label, FitMetrics.from_predictions(actual, predicted)))
        selected = _downsample(samples)
        if not selected:
            continue
        ax.scatter(
            [getattr(sample.payload, x_name) for sample in selected],
            [sample.payload.latency_s for sample in selected],
            c=[sample.payload.request_count for sample in selected],
            cmap=cmap,
            norm=norm,
            marker=marker,
            s=10,
            alpha=0.5,
            linewidths=0,
            zorder=2,
        )

    colorbar = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    colorbar.set_label("request_count")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_name)
    ax.set_ylabel("latency_s")
    ax.set_title(
        f"{phase}: x={x_name}, y=actual latency, "
        "fitted lines grouped by request_count"
    )
    ax.grid(True, alpha=0.25)
    legend_handles = [
        Line2D([0], [0], color="black", linewidth=1, label="fitted curve"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="gray",
            label="train actual",
        ),
    ]
    if validation:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="^",
                color="none",
                markerfacecolor="gray",
                label="validation actual",
            )
        )
    ax.legend(handles=legend_handles, fontsize=8)
    ax.text(
        0.02,
        0.98,
        "\n".join(metrics),
        transform=ax.transAxes,
        ha="left",
        va="top",
        family="monospace",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )


def _predict_phase_curve(
    model: RolloutCostModel,
    phase: str,
    request_count: int,
    x_value: float,
) -> float:
    if phase == "prefill":
        n_tokens = float(np.sqrt(request_count * x_value))
        workload = RolloutWorkload(
            phase="prefill",
            n_tokens=n_tokens,
            sum_l2=x_value,
            ctxsum=0,
            request_count=request_count,
        )
    else:
        workload = RolloutWorkload(
            phase="decode",
            n_tokens=request_count,
            sum_l2=0,
            ctxsum=x_value,
            request_count=request_count,
        )
    return float(model.predict(workload))


def _phase_features(phase: str) -> tuple[str, str]:
    if phase == "prefill":
        return "n_tokens", "sum_l2"
    return "request_count", "ctxsum"


def _predict_features(
    model: RolloutCostModel,
    phase: str,
    x_value: float,
    y_value: float,
) -> float:
    if phase == "prefill":
        workload = RolloutWorkload(
            phase="prefill",
            n_tokens=x_value,
            sum_l2=y_value,
            ctxsum=0,
            request_count=1,
        )
    else:
        workload = RolloutWorkload(
            phase="decode",
            n_tokens=x_value,
            sum_l2=0,
            ctxsum=y_value,
            request_count=x_value,
        )
    return float(model.predict(workload))


def _evaluate(
    model: RolloutCostModel,
    samples: Sequence[RolloutSample],
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


def _downsample(samples: Sequence[RolloutSample]) -> list[RolloutSample]:
    ordered = sorted(
        samples,
        key=lambda sample: (
            sample.payload.n_tokens,
            sample.payload.sum_l2,
            sample.payload.ctxsum,
            sample.payload.request_count,
            sample.payload.latency_s,
            sample.run_id,
        ),
    )
    if len(ordered) <= _MAX_SCATTER_POINTS:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, _MAX_SCATTER_POINTS, dtype=int)
    return [ordered[index] for index in indices]


def _metrics_line(label: str, metrics: FitMetrics) -> str:
    return (
        f"{label:10s} n={metrics.n_samples:<6d} R²={metrics.r_squared: .5f}  "
        f"RMSE={metrics.rmse:.3e}  MRE={metrics.mre_percent:6.2f}%  "
        f"max={metrics.max_error:.3e}"
    )


def _coefficient_text(model_type: str, entry: ArtifactEntry) -> str:
    if model_type == "psrl":
        formula = (
            "prefill = attn_b + attn_k*sum_l2 + max(other_floor, other_b + other_k*n_tokens)\n"
            "decode  = attn_b + attn_k*ctxsum + max(other_floor, other_b + other_k*request_count)"
        )
    elif model_type == "roofline":
        formula = "latency = max(F, W + G*n_tokens + A_p*sum_l2/b + A_d*ctxsum)"
    else:
        formula = (
            "latency = max(F, C3 + C4*gemm + C1*gemm*n_tokens "
            "+ C2*3h*sum_l2/b + C5*3h*ctxsum)"
        )
    coefficient_items = [
        f"{name}={value:.6g}" for name, value in sorted(entry.coefficients.items())
    ]
    coefficient_lines = [
        "  ".join(coefficient_items[index : index + 4])
        for index in range(0, len(coefficient_items), 4)
    ]
    coefficients = "\n              ".join(coefficient_lines)
    return f"{formula}\ncoefficients: {coefficients}"


def _plot_filename(model_type: str, parallelism: Parallelism) -> str:
    return (
        f"{model_type}_tp{parallelism.tp}_pp{parallelism.pp}_"
        f"dp{parallelism.dp}_cp{parallelism.cp}.png"
    )
