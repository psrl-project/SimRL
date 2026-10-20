"""Fixture tests for official profiler normalization."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from srl.profiler.adapters.megatron import (
    MegatronBridgeConfig,
    collect_bridge_record,
    prepare_bridge_candidate,
)
from srl.profiler.adapters.vllm import VLLMSweepConfig, collect_vllm_records
from srl.profiler.adapters.vllm.execute import (
    _prefix_console_bytes,
    _run_vllm_job_pool,
    _sweep_runtime_environment,
)
from srl.profiler.adapters.vllm.parse import (
    _IterationDetail,
    parse_sweep_log,
    payload_from_iteration,
    trusted_iterations,
)
from srl.profiler.adapters.vllm.prepare import VLLMSweepJob
from srl.profiler.io import profile_digest, read_records, write_records
from srl.profiler.schema import (
    HardwareDescriptor,
    ModelDescriptor,
    Parallelism,
    ProfileValidationError,
    TrainingPayload,
)
from srl.profiler.search_space import (
    ModelConstraints,
    TrainingCandidate,
    enumerate_training_candidates,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _model() -> ModelDescriptor:
    return ModelDescriptor(
        name="Qwen3-8B",
        parameter_count=10_000_000,
        num_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,
        extra={"intermediate_size": 12288, "vocab_size": 151936},
    )


def _vllm_config(tmp_path: Path) -> VLLMSweepConfig:
    return VLLMSweepConfig(
        model=_model(),
        hardware=HardwareDescriptor(gpu_name="H100", gpu_count=1),
        output_dir=tmp_path,
        experiment_name="fixture",
        phase_parameters={
            "prefill": ({"random-output-len": 1},),
            "decode": ({"random-output-len": 3},),
        },
    )


@pytest.mark.parametrize(
    ("phase", "result_fixture", "expected_records"),
    [
        ("prefill", "prefill_run.json", 1),
        ("decode", "decode_run.json", 2),
    ],
)
def test_vllm_timing_formula_normalization(
    tmp_path: Path,
    phase: str,
    result_fixture: str,
    expected_records: int,
):
    config = _vllm_config(tmp_path)
    run_dir = (
        config.raw_experiment_dir(phase, "timing")
        / "BENCH--random-input-len=128"
    )
    run_dir.mkdir(parents=True)
    result_path = run_dir / "run=0.json"
    shutil.copy(FIXTURES / "vllm" / result_fixture, result_path)
    iteration_lines = {
        "prefill": [
            "Iteration(0): 2 context requests, 256 context tokens, "
            "0 generation requests, 0 generation tokens, "
            "iteration elapsed time: 10.00 ms, GPU KV cache usage: 1.0%",
        ],
        "decode": [
            "Iteration(0): 2 context requests, 256 context tokens, "
            "0 generation requests, 0 generation tokens, "
            "iteration elapsed time: 12.00 ms, GPU KV cache usage: 1.0%",
            "Iteration(1): 0 context requests, 0 context tokens, "
            "2 generation requests, 2 generation tokens, "
            "iteration elapsed time: 4.20 ms, GPU KV cache usage: 1.0%",
            "Iteration(2): 0 context requests, 0 context tokens, "
            "2 generation requests, 2 generation tokens, "
            "iteration elapsed time: 4.40 ms, GPU KV cache usage: 1.0%",
        ],
    }
    config.sweep_log_path(phase, "timing").parent.mkdir(parents=True, exist_ok=True)
    config.sweep_log_path(phase, "timing").write_text(
        "\n".join(
            [
                f"Output file: {result_path}",
                "Traffic request rate: inf",
                *iteration_lines[phase],
                "[END BENCHMARK]",
            ]
        ),
        encoding="utf-8",
    )

    records = collect_vllm_records(config, phase)
    assert len(records) == expected_records
    assert all(record.record_type == "rollout" for record in records)
    assert all(record.payload.phase == phase for record in records)
    assert all(record.payload.request_count == 2 for record in records)
    assert all(
        record.tags["feature_source"] == "equal-length-formula" for record in records
    )
    if phase == "prefill":
        assert records[0].payload.sum_l2 == 2 * 128**2
        assert records[0].payload.latency_s == 0.01
    else:
        assert records[0].payload.ctxsum == 2 * (128 + 1)
        assert records[1].payload.ctxsum == 2 * (128 + 2)
        assert records[0].payload.latency_s == pytest.approx(0.0042)

    output = tmp_path / f"{phase}.jsonl"
    write_records(output, records)
    assert read_records(output) == records
    assert profile_digest(output) == profile_digest(records)


def test_payload_from_iteration_prefill_and_chunked_sum_l2():
    full = payload_from_iteration(
        _IterationDetail(0, 4, 1024, 0, 0, 10.0),
        input_len=256,
        decode_ordinal=0,
        batch_step_index=0,
    )
    assert full is not None
    assert full.phase == "prefill"
    assert full.n_tokens == 1024
    assert full.sum_l2 == 4 * 256**2
    assert full.ctxsum == 0
    assert full.latency_s == 0.01

    chunked = payload_from_iteration(
        _IterationDetail(1, 2, 512, 0, 0, 8.0),
        input_len=256,
        decode_ordinal=0,
        batch_step_index=1,
    )
    assert chunked is not None
    assert chunked.sum_l2 == 2 * 256**2

    # Token budget that does not divide evenly uses as-equal-as-possible packing.
    uneven = payload_from_iteration(
        _IterationDetail(0, 3, 256, 0, 0, 1.0),
        input_len=256,
        decode_ordinal=0,
        batch_step_index=0,
    )
    assert uneven is not None
    assert uneven.sum_l2 == 1 * 86**2 + 2 * 85**2


def test_trusted_iterations_truncates_the_run_at_preemption_recompute():
    details = [
        _IterationDetail(0, 2, 2048, 0, 0, 40.0),
        _IterationDetail(1, 0, 0, 2, 2, 12.0),
        _IterationDetail(2, 0, 0, 2, 2, 12.0),
        # A request evicted when the KV cache filled re-prefills prompt plus
        # everything it already generated.
        _IterationDetail(3, 1, 3808, 1, 1, 900.0),
        _IterationDetail(4, 0, 0, 2, 2, 12.0),
    ]
    kept, trust = trusted_iterations(details, input_len=1024, prompt_token_budget=2048)
    assert [item.index for item in kept] == [0, 1, 2]
    assert trust.truncation_iteration == 3
    assert trust.truncation_reason == "context-longer-than-prompt"
    assert trust.dropped_after_preemption == 2

    # A resumed request whose chunk is short still exceeds the prompt budget.
    resumed = [
        _IterationDetail(0, 2, 2048, 0, 0, 40.0),
        _IterationDetail(1, 0, 0, 2, 2, 12.0),
        _IterationDetail(2, 1, 512, 1, 1, 30.0),
    ]
    kept, trust = trusted_iterations(resumed, input_len=1024, prompt_token_budget=2048)
    assert [item.index for item in kept] == [0, 1]
    assert trust.truncation_reason == "context-beyond-prompt-token-budget"


def test_trusted_iterations_drops_unmeasured_steps_without_shifting_the_rest():
    details = [
        _IterationDetail(0, 1, 1024, 0, 0, 0.03),
        _IterationDetail(1, 0, 0, 1, 1, 12.0),
        _IterationDetail(2, 0, 0, 1, 1, 0.02),
        _IterationDetail(3, 0, 0, 1, 1, 12.5),
    ]
    kept, trust = trusted_iterations(details, input_len=1024, prompt_token_budget=1024)
    assert [item.index for item in kept] == [1, 3]
    assert trust.dropped_unmeasured == 2
    assert trust.truncation_reason is None


def test_trusted_iterations_drops_a_large_step_at_an_impossible_token_rate():
    """
    A step above the absolute floor is still rejected when its token rate is impossible.

    The absolute `_MIN_MEASURED_ELAPSED_MS` floor is necessary but not sufficient: an
    async-scheduling overlap artifact on a *large* prefill lands just above 1 ms and so
    survived it. Qwen3.5-35B-A3B measured 16128 tokens in 1.080 ms (14.9M tok/s) and
    11317 in 1.030 ms. The psrl fit then scored them as a 39178% residual and dragged
    tp=1 prefill MRE to 813%.

    The 4096-token step at 400 ms (10k tok/s) is kept, which is the point: the bound sits
    two orders of magnitude above any real rate, so it cannot discard genuine data. A
    64-token decode step at the same 1.03 ms is only 62k tok/s and is kept too.

    Prompt lengths here are chosen so no step trips `context-longer-than-prompt`, which
    truncates the run and would return before the rate guard is ever consulted.
    """
    details = [
        _IterationDetail(0, 63, 16128, 0, 0, 1.080),
        _IterationDetail(1, 16, 4096, 0, 0, 400.0),
        _IterationDetail(2, 0, 0, 64, 64, 1.030),
        _IterationDetail(3, 44, 11264, 53, 53, 1.030),
    ]
    kept, trust = trusted_iterations(details, input_len=256, prompt_token_budget=31488)
    assert [item.index for item in kept] == [1, 2]
    assert trust.dropped_unmeasured == 2
    assert trust.truncation_reason is None


def test_payload_from_iteration_decode_ordinal_walk():
    first = payload_from_iteration(
        _IterationDetail(1, 0, 0, 8, 8, 4.2),
        input_len=128,
        decode_ordinal=0,
        batch_step_index=1,
    )
    second = payload_from_iteration(
        _IterationDetail(2, 0, 0, 8, 8, 4.4),
        input_len=128,
        decode_ordinal=1,
        batch_step_index=2,
    )
    assert first is not None and second is not None
    assert first.phase == "decode"
    assert first.n_tokens == 8
    assert first.ctxsum == 8 * (128 + 1)
    assert second.ctxsum == 8 * (128 + 2)
    assert first.sum_l2 == 0


def test_vllm_collect_merges_parallel_shards_by_combo_path(tmp_path: Path):
    config = VLLMSweepConfig(
        model=_model(),
        hardware=HardwareDescriptor(gpu_name="H100", gpu_count=1),
        output_dir=tmp_path,
        experiment_name="fixture",
        phase_parameters={
            "prefill": (
                {"random-input-len": 128, "random-output-len": 1},
                {"random-input-len": 256, "random-output-len": 1},
            )
        },
    )
    for shard_index in range(2):
        shard_id = f"s{shard_index:03d}"
        combo = f"BENCH--random-input-len={128 * (shard_index + 1)}"
        run_dir = config.raw_experiment_dir("prefill", "timing", shard_id) / combo
        run_dir.mkdir(parents=True)
        result_path = run_dir / "run=0.json"
        shutil.copy(FIXTURES / "vllm" / "prefill_run.json", result_path)
        log_path = config.sweep_log_path("prefill", "timing", shard_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "\n".join(
                [
                    f"Output file: {result_path}",
                    "Traffic request rate: inf",
                    "Iteration(0): 2 context requests, 256 context tokens, "
                    "0 generation requests, 0 generation tokens, "
                    "iteration elapsed time: 10.00 ms, "
                    "GPU KV cache usage: 1.0%",
                    "[END BENCHMARK]",
                ]
            ),
            encoding="utf-8",
        )

    records = collect_vllm_records(config, "prefill")
    assert len(records) == 2
    artifacts = sorted(record.run.raw_artifact for record in records)
    assert all("-s00" in artifact for artifact in artifacts)
    assert any("random-input-len=128" in artifact for artifact in artifacts)
    assert any("random-input-len=256" in artifact for artifact in artifacts)


def _bridge_config(tmp_path: Path) -> MegatronBridgeConfig:
    return MegatronBridgeConfig(
        bridge_dir=tmp_path / "Megatron-Bridge",
        output_dir=tmp_path,
        experiment_name="fixture",
        model=_model(),
        hardware=HardwareDescriptor(
            gpu_name="H100",
            gpu_count=1,
            gpu_memory_mib=81920,
        ),
        warmup_steps=0,
    )


def _candidate(repeat: int = 0) -> TrainingCandidate:
    return TrainingCandidate(
        parallelism=Parallelism(tp=1, pp=2, dp=1, cp=1),
        micro_batch_size=1,
        global_batch_size=4,
        num_microbatches=4,
        sequence_shape=(512,),
        repeat=repeat,
    )


def _prepare_bridge_fixture(
    tmp_path: Path,
    *,
    repeat: int,
    success: bool,
    oom_marker: bool = False,
) -> tuple[MegatronBridgeConfig, TrainingCandidate, Path]:
    config = _bridge_config(tmp_path)
    candidate = _candidate(repeat)
    _, run_dir = prepare_bridge_candidate(config, candidate)
    fixture_dir = FIXTURES / "megatron_bridge"
    shutil.copy(fixture_dir / "ConfigContainer.yaml", run_dir / "ConfigContainer.yaml")
    if success:
        shutil.copy(fixture_dir / "status_success.json", run_dir / "status.json")
        shutil.copy(fixture_dir / "srl_metrics.json", run_dir / "srl_metrics.json")
    else:
        shutil.copy(fixture_dir / "status_failed.json", run_dir / "status.json")
        # A log that merely mentions the phrase is not evidence of an OOM.
        shutil.copy(fixture_dir / "oom_output.txt", run_dir / "run.log")
        if oom_marker:
            shutil.copy(fixture_dir / "oom_rank0.json", run_dir / "oom_rank0.json")
    return config, candidate, run_dir


def test_megatron_bridge_timer_memory_and_resolved_config(tmp_path: Path):
    config, candidate, run_dir = _prepare_bridge_fixture(
        tmp_path,
        repeat=0,
        success=True,
    )
    record = collect_bridge_record(config, candidate, run_dir=run_dir)
    assert record.run.status == "success"
    assert isinstance(record.payload, TrainingPayload)
    assert record.payload.step_times_s == (0.101, 0.099, 0.1)
    assert record.payload.peak_allocated_bytes == 1_050_000_000
    assert record.payload.peak_reserved_bytes == 1_150_000_000
    assert record.payload.timers_s["optimizer"] == (0.01, 0.01, 0.01)
    assert record.tags["resolved_config"].endswith("ConfigContainer.yaml")


def test_megatron_bridge_success_without_measurements_is_rejected(tmp_path: Path):
    """A run cannot claim success with no measurements. That is a launcher bug."""
    config, candidate, run_dir = _prepare_bridge_fixture(
        tmp_path,
        repeat=0,
        success=True,
    )
    (run_dir / "srl_metrics.json").unlink()
    with pytest.raises(ProfileValidationError, match="No measured step time"):
        collect_bridge_record(config, candidate, run_dir=run_dir)


def test_megatron_bridge_oom_is_preserved(tmp_path: Path):
    config, candidate, run_dir = _prepare_bridge_fixture(
        tmp_path,
        repeat=1,
        success=False,
        oom_marker=True,
    )
    record = collect_bridge_record(config, candidate, run_dir=run_dir)
    assert record.run.status == "oom"
    assert isinstance(record.payload, TrainingPayload)
    assert record.payload.step_times_s == ()
    assert record.run.error == "CUDA out of memory on rank(s) 0."
    # The activation feature comes from the candidate, so the record still carries
    # everything the memory model needs to bound itself.
    assert record.payload.num_microbatches == candidate.num_microbatches


def test_megatron_bridge_oom_needs_a_marker_not_a_log_mention(tmp_path: Path):
    """
    A log mentioning the phrase must stay `failed`.

    An out-of-memory record contributes an inequality that can only raise the
    fitted memory floor, so a false positive silently biases the model.
    """
    config, candidate, run_dir = _prepare_bridge_fixture(
        tmp_path,
        repeat=1,
        success=False,
        oom_marker=False,
    )
    assert "out of memory" in (run_dir / "run.log").read_text().lower()
    record = collect_bridge_record(config, candidate, run_dir=run_dir)
    assert record.run.status == "failed"


def test_candidate_generator_enforces_parallel_constraints():
    candidates = enumerate_training_candidates(
        world_sizes=[4],
        tensor_parallel_sizes=[1, 3, 4],
        pipeline_parallel_sizes=[1, 5],
        micro_batch_sizes=[1],
        num_microbatches=[1, 2, 4],
        sequence_shapes=[[512]],
        constraints=ModelConstraints(
            num_layers=36,
            num_attention_heads=32,
            num_query_groups=8,
        ),
    )
    assert candidates
    assert {candidate.parallelism.tp for candidate in candidates} == {1, 4}
    assert {candidate.parallelism.pp for candidate in candidates} == {1}
    assert all(
        candidate.global_batch_size
        == candidate.micro_batch_size
        * candidate.parallelism.dp
        * candidate.num_microbatches
        for candidate in candidates
    )


def test_profile_schema_rejects_inconsistent_world_size():
    with pytest.raises(ProfileValidationError, match="world_size"):
        Parallelism(tp=2, pp=2, dp=1, cp=1, world_size=2)


def test_prefix_console_bytes_labels_newlines_and_carriage_returns():
    first, at_start = _prefix_console_bytes(
        b"load 10%\rload 20%\nready\n",
        b"[s000] ",
        at_line_start=True,
    )
    assert first == b"[s000] load 10%\r[s000] load 20%\n[s000] ready\n"
    assert at_start is True
    second, at_start = _prefix_console_bytes(
        b"partial",
        b"[s000] ",
        at_line_start=True,
    )
    assert second == b"[s000] partial"
    assert at_start is False


def test_sweep_runtime_environment_upgrades_warn_and_unbuffers():
    environment = _sweep_runtime_environment(
        {"VLLM_LOGGING_LEVEL": "WARN", "CUDA_VISIBLE_DEVICES": "0"}
    )
    assert environment["VLLM_LOGGING_LEVEL"] == "INFO"
    assert environment["PYTHONUNBUFFERED"] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"


def test_sweep_runtime_environment_preserves_debug():
    environment = _sweep_runtime_environment({"VLLM_LOGGING_LEVEL": "DEBUG"})
    assert environment["VLLM_LOGGING_LEVEL"] == "DEBUG"


def _fake_sweep_script(tmp_path: Path) -> Path:
    """A stand-in sweep that records its startup window and its ready marker."""
    path = tmp_path / "fake_sweep.py"
    path.write_text(
        "\n".join(
            [
                "import pathlib, sys, time",
                "events, label, exit_code = sys.argv[1], sys.argv[2], int(sys.argv[3])",
                "log = pathlib.Path(events)",
                "log.open('a').write(f'enter {label}\\n')",
                # Signs of life long before the server is ready, as the real sweep does.
                "print('[BEGIN SERVER]', flush=True)",
                "time.sleep(0.4)",
                "log.open('a').write(f'ready {label}\\n')",
                "print('[BEGIN BENCHMARK]', flush=True)",
                "time.sleep(0.2)",
                "sys.exit(exit_code)",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _fake_jobs(
    tmp_path: Path,
    script: Path,
    events: Path,
    *,
    count: int,
    failing: frozenset[str] = frozenset(),
) -> list[VLLMSweepJob]:
    return [
        VLLMSweepJob(
            phase="prefill",
            measurement="timing",
            shard_id=f"s{index:03d}",
            wave=0,
            gpu_ids=(str(index),),
            port=8000 + index,
            command=(
                sys.executable,
                str(script),
                str(events),
                f"s{index:03d}",
                "1" if f"s{index:03d}" in failing else "0",
            ),
            experiment_dir=tmp_path / "raw" / f"s{index:03d}",
            log_path=tmp_path / "control" / f"s{index:03d}.log",
            world_size=1,
            expected_runs=1,
        )
        for index in range(count)
    ]


def _max_concurrent_startups(events: Path) -> int:
    active = 0
    peak = 0
    for line in events.read_text(encoding="utf-8").splitlines():
        if line.startswith("enter "):
            active += 1
            peak = max(peak, active)
        elif line.startswith("ready "):
            active -= 1
    return peak


def test_vllm_pool_admits_one_startup_at_a_time(tmp_path: Path):
    events = tmp_path / "events.log"
    events.touch()
    config = VLLMSweepConfig(
        model=_model(),
        hardware=HardwareDescriptor(gpu_name="H100", gpu_count=1),
        output_dir=tmp_path,
        experiment_name="test",
        phase_parameters={"prefill": ({"random-output-len": 1},)},
    )
    jobs = _fake_jobs(tmp_path, _fake_sweep_script(tmp_path), events, count=4)

    _run_vllm_job_pool(config, jobs, resume=False)

    assert _max_concurrent_startups(events) == 1
    assert len(events.read_text(encoding="utf-8").splitlines()) == 8


def test_vllm_pool_runs_every_shard_before_reporting_failures(tmp_path: Path):
    events = tmp_path / "events.log"
    events.touch()
    config = VLLMSweepConfig(
        model=_model(),
        hardware=HardwareDescriptor(gpu_name="H100", gpu_count=1),
        output_dir=tmp_path,
        experiment_name="test",
        phase_parameters={"prefill": ({"random-output-len": 1},)},
        max_concurrent_startups=2,
    )
    jobs = _fake_jobs(
        tmp_path,
        _fake_sweep_script(tmp_path),
        events,
        count=4,
        failing=frozenset({"s001"}),
    )

    with pytest.raises(RuntimeError) as failure:
        _run_vllm_job_pool(config, jobs, resume=False)

    assert "prefill/timing/s001" in str(failure.value)
    assert "1 of 4" in str(failure.value)
    assert isinstance(failure.value.__cause__, subprocess.CalledProcessError)
    started = {
        line.removeprefix("enter ")
        for line in events.read_text(encoding="utf-8").splitlines()
        if line.startswith("enter ")
    }
    assert started == {f"s{index:03d}" for index in range(4)}


def test_parse_sweep_log_binds_namespace_and_skips_warmup(tmp_path: Path):
    result = tmp_path / "SERVE--tp=1" / "run=0.json"
    result.parent.mkdir()
    result.write_text("{}", encoding="utf-8")
    log_path = tmp_path / "timing_sweep.log"
    log_path.write_text(
        "\n".join(
            [
                "Starting initial single prompt test run...",
                "Iteration(9): 1 context requests, 8 context tokens, "
                "0 generation requests, 0 generation tokens, "
                "iteration elapsed time: 99.00 ms, GPU KV cache usage: 1.0%",
                f"Namespace(result_dir='{result.parent}', result_filename='run=0.json')",
                "Starting initial single prompt test run...",
                "Iteration(0): 1 context requests, 8 context tokens, "
                "0 generation requests, 0 generation tokens, "
                "iteration elapsed time: 1.00 ms, GPU KV cache usage: 1.0%",
                "Starting main benchmark run...",
                "Starting profiler...",
                "Traffic request rate: inf",
                "(EngineCore pid=1) INFO 08-17 12:00:00 [loggers.py:182] "
                "Engine 000: Iteration(1): 2 context requests, 256 context tokens, "
                "0 generation requests, 0 generation tokens, "
                "iteration elapsed time: 10.50 ms, GPU KV cache usage: 1.0%",
                "Stopping profiler...",
                "============ Serving Benchmark Result ============",
            ]
        ),
        encoding="utf-8",
    )
    observations = parse_sweep_log(log_path)
    details = observations[result.resolve()]
    assert len(details) == 1
    assert details[0].index == 1
    assert details[0].ctx_tokens == 256
    assert details[0].elapsed_ms == 10.5


def test_parse_sweep_log_extracts_concatenated_output_file(tmp_path: Path):
    result = tmp_path / "run=0.json"
    result.write_text("{}", encoding="utf-8")
    log_path = tmp_path / "timing_sweep.log"
    log_path.write_text(
        "\n".join(
            [
                f"Output file: {result}(APIServer pid=1) INFO: POST /tokenize 200 OK",
                "Traffic request rate: inf",
                "Iteration(0): 2 context requests, 256 context tokens, "
                "0 generation requests, 0 generation tokens, "
                "iteration elapsed time: 4.00 ms, GPU KV cache usage: 1.0%",
                "[END BENCHMARK]",
            ]
        ),
        encoding="utf-8",
    )
    observations = parse_sweep_log(log_path)
    details = observations[result.resolve()]
    assert len(details) == 1
    assert details[0].elapsed_ms == 4.0
