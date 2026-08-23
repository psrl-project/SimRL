"""CLI command-tree and dry-run command construction tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dryrun.cli import regress as regress_cli
from dryrun.cli.profile import build_parser as build_profile_parser
from dryrun.cli.regress import build_parser as build_regress_parser
from dryrun.profiler.adapters.megatron_bridge import (
    MegatronBridgeConfig,
    prepare_bridge_candidate,
)
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


def test_megatron_generated_command_uses_official_recipe(tmp_path: Path):
    bridge_dir = tmp_path / "Megatron-Bridge"
    config = MegatronBridgeConfig(
        bridge_dir=bridge_dir,
        output_dir=tmp_path,
        experiment_name="test",
        model=ModelDescriptor(name="Qwen3-8B", parameter_count=8_000_000_000),
        hardware=HardwareDescriptor(gpu_name="H100", gpu_count=1),
    )
    candidate = TrainingCandidate(
        parallelism=Parallelism(),
        micro_batch_size=1,
        global_batch_size=1,
        num_microbatches=1,
        sequence_shape=(512,),
    )
    command, _ = prepare_bridge_candidate(config, candidate)
    assert command[0] == "torchrun"
    assert str(bridge_dir / "scripts" / "performance" / "run_recipe.py") in command
    assert command[command.index("--model_family_name") + 1] == "qwen"
    assert "profiling.use_pytorch_profiler=true" in command
    assert "profiling.record_memory_history=true" in command
