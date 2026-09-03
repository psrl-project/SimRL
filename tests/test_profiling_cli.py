"""CLI command-tree and dry-run command construction tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dryrun.cli import regress as regress_cli
from dryrun.cli.profile import build_parser as build_profile_parser
from dryrun.cli.profile import cmd_kernels_validate
from dryrun.cli.regress import build_parser as build_regress_parser
from dryrun.profiler.adapters.megatron import (
    MegatronBridgeConfig,
    prepare_bridge_candidate,
)
from dryrun.profiler.adapters.megatron.prepare import LAUNCH_MODULE
from dryrun.profiler.adapters.vllm import (
    VLLMSweepConfig,
    format_vllm_job_command,
    prepare_vllm_jobs,
    prepare_vllm_sweep,
)
from dryrun.profiler.schema import HardwareDescriptor, ModelDescriptor, Parallelism
from dryrun.profiler.search_space import TrainingCandidate


def test_profile_cli_exposes_generate_run_collect():
    parser = build_profile_parser()
    for backend in ("vllm", "megatron"):
        for workflow in ("generate", "run", "collect"):
            args = [backend, workflow, "--config", "config.yaml"]
            if backend == "megatron" and workflow == "run":
                args.extend(("--dry-run", "--index", "0"))
            parsed = parser.parse_args(args)
            assert parsed.backend_name == backend
            assert parsed.workflow == workflow


def test_regress_cli_exposes_model_owned_fitters():
    parser = build_regress_parser()
    rollout = parser.parse_args(
        [
            "rollout",
            "--model",
            "distserve",
            "--input",
            "profile.jsonl",
            "--output",
            "artifact.json",
        ]
    )
    training = parser.parse_args(
        [
            "training",
            "--input",
            "profile.jsonl",
            "--output",
            "artifact.json",
        ]
    )
    assert rollout.model == "distserve"
    assert rollout.validation_fraction == 0.0
    assert training.latency_model == "hydraulis"
    assert training.memory_model == "hydraulis"


@pytest.mark.parametrize(("backend", "plots_expected"), [("vllm", True), ("megatron", False)])
def test_rollout_regress_automatically_plots_only_vllm(
    tmp_path: Path,
    monkeypatch,
    backend: str,
    plots_expected: bool,
):
    records = [SimpleNamespace(software=SimpleNamespace(backend=backend))]
    saved_paths: list[Path] = []
    plotted: list[tuple] = []

    class FakeArtifact:
        entries = (object(),)

        def save(self, path):
            saved_paths.append(Path(path))

    artifact = FakeArtifact()
    monkeypatch.setattr(regress_cli, "read_records", lambda *_args, **_kwargs: records)
    monkeypatch.setattr(
        regress_cli.PSRLFitted,
        "fit",
        classmethod(lambda _cls, _records, *, validation_fraction: artifact),
    )
    monkeypatch.setattr(
        regress_cli,
        "plot_vllm_rollout_regression",
        lambda *args, **kwargs: plotted.append((args, kwargs)) or [],
    )
    output = tmp_path / "rollout_psrl.json"
    args = SimpleNamespace(
        model="psrl",
        input=tmp_path / "profile.jsonl",
        output=output,
        validation_fraction=0.2,
    )

    regress_cli.cmd_rollout(args)

    assert saved_paths == [output]
    assert bool(plotted) is plots_expected
    if plots_expected:
        plot_args, plot_kwargs = plotted[0]
        assert plot_args == (records, artifact, tmp_path / "rollout_psrl_plots")
        assert plot_kwargs == {"validation_fraction": 0.2}


def test_vllm_generated_command_uses_iteration_logs(tmp_path: Path):
    config = VLLMSweepConfig(
        model=ModelDescriptor(name="Qwen3-8B"),
        hardware=HardwareDescriptor(gpu_name="H100", gpu_count=1),
        output_dir=tmp_path,
        experiment_name="test",
        phase_parameters={"prefill": ({"random-output-len": 1},)},
    )
    command, _ = prepare_vllm_sweep(config, "prefill")
    assert command[:4] == ["vllm", "bench", "sweep", "serve"]
    bench = command[command.index("--bench-cmd") + 1]
    serve = command[command.index("--serve-cmd") + 1]
    assert "--save-detailed" in bench
    assert "--profile" not in bench
    assert "--enable-logging-iteration-details" in serve
    assert "--profiler-config" not in serve
    assert "random-range-ratio" in bench


def test_vllm_parallel_jobs_shard_points_across_eight_gpus(tmp_path: Path):
    config = VLLMSweepConfig(
        model=ModelDescriptor(name="Qwen3-8B"),
        hardware=HardwareDescriptor(gpu_name="H100", gpu_count=1),
        output_dir=tmp_path,
        experiment_name="test",
        phase_parameters={
            "prefill": tuple(
                {
                    "random-input-len": 128 * (index + 1),
                    "random-output-len": 1,
                }
                for index in range(8)
            )
        },
    )
    jobs = prepare_vllm_jobs(
        config,
        ["prefill"],
        visible_devices=[str(index) for index in range(8)],
    )

    assert len(jobs) == 8
    assert {job.shard_id for job in jobs} == {
        f"s{index:03d}" for index in range(8)
    }
    for wave in {job.wave for job in jobs}:
        wave_jobs = [job for job in jobs if job.wave == wave]
        assert len(wave_jobs) == 8
        assert len({job.gpu_ids for job in wave_jobs}) == 8
        assert len({job.port for job in wave_jobs}) == 8
    assert all(len(job.gpu_ids) == 1 for job in jobs)
    assert all(job.shard_id in job.experiment_dir.name for job in jobs)
    first_serve = jobs[0].command[jobs[0].command.index("--serve-cmd") + 1]
    first_bench = jobs[0].command[jobs[0].command.index("--bench-cmd") + 1]
    assert f"--port {jobs[0].port}" in first_serve
    assert f"--base-url http://127.0.0.1:{jobs[0].port}" in first_bench
    assert all(job.measurement == "timing" for job in jobs)
    assert len({job.log_path for job in jobs}) == 8
    assert "--dry-run" in format_vllm_job_command(jobs[0], dry_run=True)
    assert "CUDA_VISIBLE_DEVICES=" in format_vllm_job_command(jobs[0])


def test_vllm_parallel_jobs_pack_tp2_into_four_slots(tmp_path: Path):
    config = VLLMSweepConfig(
        model=ModelDescriptor(name="Qwen3-8B"),
        hardware=HardwareDescriptor(gpu_name="H100", gpu_count=2),
        output_dir=tmp_path,
        experiment_name="test",
        serve_parameters=(
            {
                "tensor-parallel-size": 2,
                "pipeline-parallel-size": 1,
            },
        ),
        phase_parameters={
            "prefill": tuple(
                {
                    "random-input-len": 128 * (index + 1),
                    "random-output-len": 1,
                }
                for index in range(8)
            )
        },
    )
    jobs = prepare_vllm_jobs(
        config,
        ["prefill"],
        visible_devices=[str(index) for index in range(8)],
    )

    assert len(jobs) == 4
    assert all(job.world_size == 2 for job in jobs)
    assert all(len(job.gpu_ids) == 2 for job in jobs)
    for wave in {job.wave for job in jobs}:
        wave_jobs = [job for job in jobs if job.wave == wave]
        assert len(wave_jobs) == 4
        assert len({gpu for job in wave_jobs for gpu in job.gpu_ids}) == 8


def test_vllm_max_parallel_one_preserves_legacy_sweeps(tmp_path: Path):
    config = VLLMSweepConfig(
        model=ModelDescriptor(name="Qwen3-8B"),
        hardware=HardwareDescriptor(gpu_name="H100", gpu_count=1),
        output_dir=tmp_path,
        experiment_name="test",
        phase_parameters={
            "prefill": (
                {"random-input-len": 128, "random-output-len": 1},
                {"random-input-len": 256, "random-output-len": 1},
            )
        },
    )
    jobs = prepare_vllm_jobs(
        config,
        ["prefill"],
        max_parallel=1,
        visible_devices=[str(index) for index in range(8)],
    )

    assert len(jobs) == 1
    assert all(job.shard_id is None for job in jobs)
    assert [job.wave for job in jobs] == [0]
    assert all(job.port is None for job in jobs)
    assert all("-s000" not in job.experiment_dir.name for job in jobs)


def test_vllm_run_cli_accepts_auto_or_bounded_parallelism():
    parser = build_profile_parser()
    automatic = parser.parse_args(
        [
            "vllm",
            "run",
            "--config",
            "config.yaml",
            "--max-parallel",
            "auto",
        ]
    )
    bounded = parser.parse_args(
        [
            "vllm",
            "run",
            "--config",
            "config.yaml",
            "--max-parallel",
            "4",
        ]
    )
    assert automatic.max_parallel == "auto"
    assert bounded.max_parallel == 4


def test_megatron_generated_command_targets_the_dryrun_launcher(tmp_path: Path):
    """
    The launcher is a DryRun module, not the vendored performance CLI.

    That CLI accepted only its own known arguments and discarded the rest, so the
    profiling and recompute overrides handed to it never took effect. It also
    validated `--gpu` against a fixed list that excludes this cluster's
    accelerator, which failed before training even started.
    """
    bridge_dir = tmp_path / "Megatron-Bridge"
    config = MegatronBridgeConfig(
        bridge_dir=bridge_dir,
        output_dir=tmp_path,
        experiment_name="test",
        model=ModelDescriptor(
            name="Qwen3-8B",
            parameter_count=8_000_000_000,
            extra={"vocab_size": 151936},
        ),
        hardware=HardwareDescriptor(gpu_name="H100", gpu_count=1),
    )
    candidate = TrainingCandidate(
        parallelism=Parallelism(),
        micro_batch_size=1,
        global_batch_size=1,
        num_microbatches=1,
        sequence_shape=(512,),
    )
    command, run_dir = prepare_bridge_candidate(config, candidate)
    assert command[0] == "torchrun"
    assert command[command.index("-m") + 1] == LAUNCH_MODULE
    assert command[command.index("--spec") + 1] == str(run_dir / "launch_spec.json")
    assert str(bridge_dir / "scripts" / "performance" / "run_recipe.py") not in command
    assert "--gpu" not in command
    # Every setting travels in the spec, so no key=value token should remain.
    assert not any("=" in item for item in command)


def test_megatron_run_no_longer_exposes_a_slurm_backend():
    """
    The Slurm path resolved a performance recipe that does not exist for this model.

    It also required a container mount and applied a different recompute method,
    which would have landed incomparable measurements in the same cost-model cell.
    """
    parser = build_profile_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["megatron", "run", "--config", "c.yaml", "--backend", "slurm"]
        )


def test_kernels_cli_no_longer_exposes_synthesize():
    parser = build_profile_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["kernels", "synthesize", "--config", "config.yaml"])


def test_kernels_module_no_longer_exports_the_synthesis_path():
    import dryrun.profiler.kernels as kernels

    for name in (
        "ComposedCostTerms",
        "fit_composed_terms",
        "synthesize_rollout_records",
        "SynthesisGrid",
        "resolve_scheduling_overhead",
    ):
        assert not hasattr(kernels, name), f"{name} should have been removed."


def test_kernels_cli_exposes_run_with_family_and_holdout():
    parser = build_profile_parser()
    run = parser.parse_args(["kernels", "run", "--config", "c.yaml", "--family", "gemm"])
    assert run.workflow == "run"
    assert run.family == "gemm"
    default_run = parser.parse_args(["kernels", "run", "--config", "c.yaml"])
    assert default_run.family == "all"
    holdout = parser.parse_args(["kernels", "holdout", "--config", "c.yaml"])
    assert holdout.workflow == "holdout"
    # Defaults to the full set: a sampled fraction once reported GEMM at 3.38% where all
    # 64 points gave 8.48%, so the default must not be a subset.
    assert holdout.fraction == pytest.approx(1.0)
    assert holdout.gate == pytest.approx(5.0)


def test_kernels_run_rejects_an_unmeasured_family():
    parser = build_profile_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["kernels", "run", "--config", "c.yaml", "--family", "moe"])


def test_kernels_run_accepts_every_registered_family():
    from dryrun.cost.kernel.ops.base import families

    parser = build_profile_parser()
    for family in families():
        args = parser.parse_args(["kernels", "run", "--config", "c.yaml", "--family", family])
        assert args.family == family


# --- Holdout CLI integration tests (CPU-only, torch-free) ---


def _write_gemm_parquet(table_root: Path, rows: list[dict]) -> Path:
    """Write a minimal GEMM perf parquet under the standard layout."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = table_root / "h20" / "gemm" / "vllm" / "2.6.1" / "gemm_perf.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(rows)
    data = {
        "framework": ["vllm"] * count,
        "version": ["2.6.1"] * count,
        "device": ["NVIDIA H20"] * count,
        "op_name": ["gemm"] * count,
        "kernel_source": ["torch.nn.functional.linear"] * count,
        "m": [r["m"] for r in rows],
        "n": [r["n"] for r in rows],
        "k": [r["k"] for r in rows],
        "gemm_dtype": [r.get("gemm_dtype", "bfloat16") for r in rows],
        "latency_s": [r["latency_s"] for r in rows],
    }
    pq.write_table(pa.table(data), path)
    return path


def _make_stub_config(table_root: Path):
    """Return a lightweight config stub exposing table_root() and gpu()."""
    return SimpleNamespace(
        table_root=lambda: table_root,
        gpu=lambda: "h20",
    )


def _holdout_args(table_root: Path, output: Path | None = None) -> SimpleNamespace:
    """Return a Namespace matching the arguments cmd_kernels_holdout expects."""
    return SimpleNamespace(
        config="<monkeypatched>",
        fraction=0.2,
        seed=0,
        gate=5.0,
        output=str(output) if output is not None else None,
    )


def test_holdout_zero_scorable_points_is_a_coverage_failure(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    """A 2-row GEMM table produces 0 scorable points; holdout must exit non-zero and log the coverage problem."""
    import logging

    from dryrun.cost.kernel.hardware import load_gpu_spec
    from dryrun.cost.kernel.sol import gemm_sol

    spec = load_gpu_spec("h20")
    target_utility = 0.6
    # Two rows only — the single slice has fewer than _MIN_SLICE_POINTS (4) rows, so it is skipped.
    sparse_rows = [
        {
            "m": m,
            "n": 4096,
            "k": 4096,
            "latency_s": gemm_sol(spec, m=m, n=4096, k=4096, dtype="bfloat16").latency_s / target_utility,
        }
        for m in (64, 128)
    ]
    table_root = tmp_path / "tables"
    _write_gemm_parquet(table_root, sparse_rows)

    import dryrun.cli.profile as profile_mod

    monkeypatch.setattr(profile_mod, "_kernel_config", lambda _path: _make_stub_config(table_root))

    args = _holdout_args(table_root)
    with caplog.at_level(logging.WARNING, logger="dryrun"):
        with pytest.raises(SystemExit) as exc_info:
            profile_mod.cmd_kernels_holdout(args)

    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    coverage_messages = [m for m in warning_messages if "coverage problem" in m]
    assert coverage_messages, (
        f"Expected a coverage-problem warning but got: {warning_messages!r}."
    )
    coverage_msg = coverage_messages[0]
    # Pin the distinguishing wording so an accuracy failure cannot masquerade as coverage.
    assert "0 held-out points scored" in coverage_msg, (
        f"Coverage warning must mention zero scored points, got: {coverage_msg!r}."
    )
    assert "coverage problem" in coverage_msg, (
        f"Coverage warning must name the problem as a coverage problem, got: {coverage_msg!r}."
    )
    assert "not an accuracy problem" in coverage_msg, (
        f"Coverage warning must distinguish from accuracy failure, got: {coverage_msg!r}."
    )

    exit_message = str(exc_info.value)
    assert "MRE exceeded" not in exit_message, (
        f"Coverage-failure exit message must not claim MRE exceeded the gate, got: {exit_message!r}."
    )
    assert "coverage problem" in exit_message, (
        f"Coverage-failure exit message must name the problem as a coverage problem, got: {exit_message!r}."
    )
    assert "scorable" in exit_message, (
        f"Coverage-failure exit message must state that nothing was scorable, got: {exit_message!r}."
    )


def test_holdout_accuracy_failure_names_mre_clause_not_coverage_clause(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A scorable table that fails a gate=0.0 must name MRE, not a coverage problem."""
    from dryrun.cost.kernel.hardware import load_gpu_spec
    from dryrun.cost.kernel.sol import gemm_sol

    spec = load_gpu_spec("h20")
    target_utility = 0.6
    # Ten rows — enough to produce scorable interior points.
    ms = [64, 128, 256, 512, 1024]
    nks = [(4096, 4096), (8192, 4096)]
    rows = [
        {
            "m": m,
            "n": n,
            "k": k,
            "latency_s": gemm_sol(spec, m=m, n=n, k=k, dtype="bfloat16").latency_s / target_utility,
        }
        for n, k in nks
        for m in ms
    ]
    table_root = tmp_path / "tables"
    _write_gemm_parquet(table_root, rows)

    import dryrun.cli.profile as profile_mod

    monkeypatch.setattr(profile_mod, "_kernel_config", lambda _path: _make_stub_config(table_root))

    # gate=0.0 forces any non-zero MRE to fail the accuracy gate.
    args = SimpleNamespace(
        config="<monkeypatched>",
        fraction=0.2,
        seed=0,
        gate=0.0,
        output=None,
    )
    with pytest.raises(SystemExit) as exc_info:
        profile_mod.cmd_kernels_holdout(args)

    exit_message = str(exc_info.value)
    assert "MRE exceeded" in exit_message, (
        f"Accuracy-failure exit message must claim MRE exceeded the gate, got: {exit_message!r}."
    )
    assert "coverage problem" not in exit_message, (
        f"Accuracy-failure exit message must not mention a coverage problem, got: {exit_message!r}."
    )


def test_holdout_healthy_table_passes_and_writes_report(tmp_path: Path, monkeypatch) -> None:
    """A healthy GEMM table with exact-uniform utility passes the gate and writes a JSON report."""
    import json

    from dryrun.cost.kernel.hardware import load_gpu_spec
    from dryrun.cost.kernel.sol import gemm_sol

    spec = load_gpu_spec("h20")
    target_utility = 0.6
    # Ten rows across two (n, k) pairs and five m values — produces bracketed interior points.
    ms = [64, 128, 256, 512, 1024]
    nks = [(4096, 4096), (8192, 4096)]
    rows = [
        {
            "m": m,
            "n": n,
            "k": k,
            "latency_s": gemm_sol(spec, m=m, n=n, k=k, dtype="bfloat16").latency_s / target_utility,
        }
        for n, k in nks
        for m in ms
    ]
    table_root = tmp_path / "tables"
    _write_gemm_parquet(table_root, rows)

    import dryrun.cli.profile as profile_mod

    monkeypatch.setattr(profile_mod, "_kernel_config", lambda _path: _make_stub_config(table_root))

    # Default output path: table_root / "holdout_report.json"
    expected_report = table_root / "holdout_report.json"
    args = _holdout_args(table_root)

    # Must not raise.
    profile_mod.cmd_kernels_holdout(args)

    assert expected_report.is_file(), "Holdout report JSON was not written."
    report = json.loads(expected_report.read_text(encoding="utf-8"))
    assert isinstance(report, list) and len(report) >= 1, (
        f"Expected a non-empty JSON list, got: {report!r}."
    )
    entry = report[0]
    assert entry["op_name"] == "gemm", f"Expected op_name 'gemm', got {entry['op_name']!r}."
    assert entry["backend"] == "vllm", f"Expected backend 'vllm', got {entry['backend']!r}."
    assert entry["point_count"] > 0, (
        f"Expected at least one scored point in the healthy table, got {entry['point_count']!r}."
    )


def test_holdout_empty_root_is_an_actionable_failure(tmp_path: Path, monkeypatch) -> None:
    """An empty table root must exit non-zero with a message telling the user to run kernels first."""
    table_root = tmp_path / "empty_tables"
    table_root.mkdir()

    import dryrun.cli.profile as profile_mod

    monkeypatch.setattr(profile_mod, "_kernel_config", lambda _path: _make_stub_config(table_root))

    args = _holdout_args(table_root)
    with pytest.raises(SystemExit) as exc_info:
        profile_mod.cmd_kernels_holdout(args)

    message = str(exc_info.value)
    assert "dryrun-profile kernels run" in message, (
        f"Exit message must tell the user to run kernels first, got: {message!r}."
    )


def test_fit_floor_empty_root_names_every_floor_taking_family_not_just_attention(
    tmp_path: Path, monkeypatch
) -> None:
    """`fit_floors_report` returns [] when no floor-taking family has tables, not just attention.

    `elementwise` and `collective` are floor-taking too, so a message naming only attention
    sends the user to re-measure a family that may already be present.
    """
    table_root = tmp_path / "empty_tables"
    table_root.mkdir()

    import dryrun.cli.profile as profile_mod

    monkeypatch.setattr(profile_mod, "_kernel_config", lambda _path: _make_stub_config(table_root))

    args = SimpleNamespace(config="<monkeypatched>", candidates_us=None)
    with pytest.raises(SystemExit) as exc_info:
        profile_mod.cmd_kernels_fit_floor(args)

    message = str(exc_info.value)
    assert "floor-taking" in message, f"Exit message must name the floor-taking set: {message!r}."
    assert "--family attention" not in message, (
        f"Exit message still names one family, but elementwise and collective take floors "
        f"too: {message!r}."
    )
    assert "dryrun-profile kernels run" in message


def test_kernels_validate_is_registered():
    parser = build_profile_parser()
    args = parser.parse_args(
        ["kernels", "validate", "--config", "c.yaml", "--baseline", "vllm.jsonl"]
    )
    assert args.func is cmd_kernels_validate
    assert args.surface == "pchip"
    assert args.min_samples == 20


def test_the_holdout_fraction_defaults_to_the_full_set():
    """A sampled fraction once reported GEMM at 3.38% where the full set was 8.48%."""
    parser = build_profile_parser()
    args = parser.parse_args(["kernels", "holdout", "--config", "c.yaml"])
    assert args.fraction == 1.0


def test_a_sampled_holdout_fraction_warns_that_it_understates(tmp_path, monkeypatch, caplog):
    """Scoring a subset must not be silent: the MRE it reports is not the surface's error."""
    import logging

    import dryrun.cli.profile as profile_mod

    rows = [
        {
            "framework": "vllm",
            "version": "0.0.0",
            "device": "NVIDIA H20",
            "op_name": "gemm",
            "kernel_source": "torch.nn.functional.linear",
            "gemm_dtype": "bfloat16",
            "m": m,
            "n": 4096,
            "k": 4096,
            "latency_s": 1.0e-4 * m,
        }
        for m in (16, 32, 64, 128, 256)
    ]
    table_root = tmp_path / "tables"
    _write_gemm_parquet(table_root, rows)
    monkeypatch.setattr(profile_mod, "_kernel_config", lambda _path: _make_stub_config(table_root))

    args = _holdout_args(table_root)
    args.fraction = 0.2
    args.gate = 100.0
    with caplog.at_level(logging.WARNING, logger="dryrun"):
        profile_mod.cmd_kernels_holdout(args)
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("understates" in message for message in warnings), warnings

    caplog.clear()
    args.fraction = 1.0
    with caplog.at_level(logging.WARNING, logger="dryrun"):
        profile_mod.cmd_kernels_holdout(args)
    assert not [r for r in caplog.records if "understates" in r.message]


def test_kernels_verify_shapes_is_registered():
    """--config is optional here so the gate runs on a model with no profiling config yet."""
    from dryrun.cli.profile import cmd_kernels_verify_shapes

    parser = build_profile_parser()
    args = parser.parse_args(["kernels", "verify-shapes", "--model-path", "/models/x"])
    assert args.func is cmd_kernels_verify_shapes
    assert args.model_path == "/models/x"
    assert args.config is None
