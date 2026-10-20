"""FSDP launch-spec application and measurement bookkeeping.

Anything needing a process group is covered by the smoke run in Task 4 Step 8, not
here: a unit test that spawns eight ranks is neither fast nor diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from srl.profiler.adapters.fsdp.launch import (
    build_model,
    read_spec,
    synthetic_batch,
)
from srl.profiler.adapters.fsdp.measure import (
    FsdpMeasurement,
    is_nccl_memory_exhaustion,
    write_oom_sidecar,
)


def _spec(tmp_path: Path, **overrides) -> dict:
    defaults = {
        "schema_version": 1,
        "model_path": "/models/Qwen3-8B",
        "mesh": {"replicate_size": 1, "shard_size": 8},
        "sequence_length": 128,
        "n_seqs_per_rank": 2,
        "num_microbatches": 1,
        "train_iters": 12,
        "warmup_steps": 4,
        "vocab_size": 151936,
        "gradient_checkpointing": True,
        "optimizer_offload": False,
        "reshard_after_forward": True,
        "compute_dtype": "bf16",
        "parameter_dtype": "bf16",
        "gradient_dtype": "fp32",
        "optimizer_state_dtype": "fp32",
        "optimizer": "adam",
        "attn_implementation": "flash_attention_2",
        "num_layers_override": 2,
        "run_dir": str(tmp_path),
    }
    return {**defaults, **overrides}


def test_read_spec_rejects_an_unknown_schema(tmp_path: Path):
    path = tmp_path / "launch_spec.json"
    path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported launch spec"):
        read_spec(path)


def test_read_spec_returns_the_document(tmp_path: Path):
    path = tmp_path / "launch_spec.json"
    path.write_text(json.dumps(_spec(tmp_path)), encoding="utf-8")
    assert read_spec(path)["sequence_length"] == 128


def test_synthetic_batch_has_the_requested_shape(tmp_path: Path):
    batch = synthetic_batch(_spec(tmp_path), device="cpu")
    assert batch["input_ids"].shape == (2, 128)
    assert batch["labels"].shape == (2, 128)
    assert batch["position_ids"].shape == (2, 128)
    assert batch["input_ids"].max() < 151936


def test_synthetic_batch_position_ids_do_not_restart(tmp_path: Path):
    """Profiling measures padded uniform shapes. Restarting ids would signal packing."""
    batch = synthetic_batch(_spec(tmp_path), device="cpu")
    assert torch.equal(batch["position_ids"][0], torch.arange(128))


def test_build_model_honours_the_layer_override(tmp_path: Path, monkeypatch):
    """A two-layer smoke model must actually have two layers, not thirty-six."""
    from transformers import AutoConfig

    def fake_from_pretrained(path, **kwargs):
        config = AutoConfig.for_model(
            "qwen3",
            num_hidden_layers=36,
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1024,
        )
        return config

    monkeypatch.setattr(AutoConfig, "from_pretrained", fake_from_pretrained)
    spec = _spec(tmp_path, num_layers_override=2, attn_implementation="sdpa")
    model = build_model(spec)
    assert model.config.num_hidden_layers == 2
    assert len(model.model.layers) == 2
    # The override must reach the built module, not just the config it was read from.
    assert model.config.use_cache is False


def test_measurement_reports_one_time_per_step(tmp_path: Path):
    measurement = FsdpMeasurement(run_dir=tmp_path, spec=_spec(tmp_path))
    for _ in range(3):
        measurement.step_begin()
        measurement.step_end()
    sidecar = measurement.sidecar([{"rank": 0, "allocated_bytes": 10, "reserved_bytes": 20}])
    assert len(sidecar["step_times_s"]) == 3
    assert all(value > 0 for value in sidecar["step_times_s"])


def test_measurement_keeps_warmup_steps_for_the_collector(tmp_path: Path):
    """The launcher is a pure measurer: the collector owns the warmup policy."""
    spec = _spec(tmp_path, train_iters=5, warmup_steps=2)
    measurement = FsdpMeasurement(run_dir=tmp_path, spec=spec)
    for _ in range(5):
        measurement.step_begin()
        measurement.step_end()
    assert len(measurement.sidecar([])["step_times_s"]) == 5


def test_step_end_without_a_begin_is_an_error(tmp_path: Path):
    measurement = FsdpMeasurement(run_dir=tmp_path, spec=_spec(tmp_path))
    with pytest.raises(RuntimeError, match="without a matching start"):
        measurement.step_end()


def test_sidecar_records_the_microbatch_shape(tmp_path: Path):
    spec = _spec(tmp_path, n_seqs_per_rank=3, sequence_length=64)
    measurement = FsdpMeasurement(run_dir=tmp_path, spec=spec)
    measurement.step_begin()
    measurement.step_end()
    assert measurement.sidecar([])["microbatch_sequence_lengths"] == [[64, 64, 64]]


def test_sidecar_reports_one_entry_per_accumulated_microbatch(tmp_path: Path):
    """The collector cross-checks this count against the candidate's `num_microbatches`."""
    spec = _spec(tmp_path, n_seqs_per_rank=2, sequence_length=64, num_microbatches=3)
    measurement = FsdpMeasurement(run_dir=tmp_path, spec=spec)
    measurement.step_begin()
    measurement.step_end()
    shapes = measurement.sidecar([])["microbatch_sequence_lengths"]
    assert shapes == [[64, 64], [64, 64], [64, 64]]


def test_a_step_runs_every_microbatch_before_one_optimizer_step(tmp_path: Path):
    """
    Gradient accumulation, not a discarded count: `nmb` forward/backward pairs per step.

    The microbatch count is a swept axis precisely because the latency design row is
    `[1, nmb*tokens, nmb]` and a constant `nmb` leaves it at rank two. If the launcher ran
    one microbatch regardless, the sweep would vary a number that changed nothing measured
    and the fit would see a rank-deficient design carrying rank-three labels.
    """
    from srl.profiler.adapters.fsdp.launch import run_accumulated_step

    calls: list[str] = []

    class _Loss:
        @staticmethod
        def backward():
            calls.append("backward")

    class _Output:
        loss = _Loss()

    class _Model:
        def __call__(self, **kwargs):
            calls.append("forward")
            return _Output()

    class _Optimizer:
        @staticmethod
        def step():
            calls.append("step")

    run_accumulated_step(_Model(), {}, _Optimizer(), 3)
    assert calls == [
        "forward",
        "backward",
        "forward",
        "backward",
        "forward",
        "backward",
        "step",
    ]


def test_a_single_microbatch_step_is_one_forward_backward_and_one_step(tmp_path: Path):
    from srl.profiler.adapters.fsdp.launch import run_accumulated_step

    calls: list[str] = []

    class _Output:
        class loss:  # noqa: N801 - mirrors the attribute name torch exposes
            @staticmethod
            def backward():
                calls.append("backward")

    class _Model:
        def __call__(self, **kwargs):
            calls.append("forward")
            return _Output()

    class _Optimizer:
        @staticmethod
        def step():
            calls.append("step")

    run_accumulated_step(_Model(), {}, _Optimizer(), 1)
    assert calls == ["forward", "backward", "step"]


def test_sidecar_sorts_rank_memory(tmp_path: Path):
    measurement = FsdpMeasurement(run_dir=tmp_path, spec=_spec(tmp_path))
    sidecar = measurement.sidecar(
        [
            {"rank": 2, "allocated_bytes": 3, "reserved_bytes": 4},
            {"rank": 0, "allocated_bytes": 1, "reserved_bytes": 2},
        ]
    )
    assert [item["rank"] for item in sidecar["rank_memory"]] == [0, 2]


def test_sidecar_carries_the_observed_mesh_for_cross_checking(tmp_path: Path):
    """A silently wrong mesh must fail in collect, not produce plausible numbers."""
    measurement = FsdpMeasurement(run_dir=tmp_path, spec=_spec(tmp_path))
    measurement.record_topology(replicate_size=1, shard_size=8, parameter_count=1234)
    sidecar = measurement.sidecar([])
    assert sidecar["observed_mesh"] == {"replicate_size": 1, "shard_size": 8}
    assert sidecar["parameter_count"] == 1234


def test_sidecar_records_the_comm_compute_split(tmp_path: Path):
    """
    Ground truth for a later kernel-summation model.

    The regression never reads these timers. They exist because measuring the
    all-gather and reduce-scatter share now is cheap and re-profiling later is not.
    """
    measurement = FsdpMeasurement(run_dir=tmp_path, spec=_spec(tmp_path))
    measurement.step_begin()
    measurement.record_collective_time(all_gather_s=0.02, reduce_scatter_s=0.03)
    measurement.step_end()
    timers = measurement.sidecar([])["timers_s"]
    assert timers["all_gather"] == (0.02,)
    assert timers["reduce_scatter"] == (0.03,)


def test_collectives_are_recorded_after_the_loop_ran_to_completion(tmp_path: Path):
    """
    The profiled step lives after the timed loop, so it adds no timing sample.

    torch.profiler inflates its own step by roughly an order of magnitude and leaves a
    ~2x tail on the next one or two steps: measured as 2.73x on retained[0] against a
    1.35x control on 8 ranks, and MAD dispersion is blind to a single outlier, so an
    in-loop profiled step passes the 5% smoke gate while the mean is inflated. Out of the
    loop the breakdown is still recorded and every retained sample stays clean.
    """
    spec = _spec(tmp_path, train_iters=12, warmup_steps=4)
    measurement = FsdpMeasurement(run_dir=tmp_path, spec=spec)
    for _ in range(spec["train_iters"]):
        measurement.step_begin()
        measurement.step_end()
    measurement.record_collective_time(all_gather_s=0.02, reduce_scatter_s=0.03)
    sidecar = measurement.sidecar([])
    assert len(sidecar["step_times_s"]) == spec["train_iters"]
    assert sidecar["timers_s"]["all_gather"] == (0.02,)
    assert sidecar["timers_s"]["reduce_scatter"] == (0.03,)


def test_the_root_module_is_never_resharded_after_forward(tmp_path: Path, monkeypatch):
    """
    The root's parameters are needed the instant backward begins, so resharding them is
    undone microseconds later: 66.21 ms against 64.15 ms with one extra all-gather and
    no memory saving. `reshard_after_forward` is a swept axis, so letting the root
    inherit it would load one side of the comparison with a fixed cost.
    """
    import torch.distributed.fsdp as fsdp_module

    from srl.profiler.adapters.fsdp.launch import apply_fsdp

    calls: list[tuple[object, bool]] = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs["reshard_after_forward"]))
        return module

    monkeypatch.setattr(fsdp_module, "fully_shard", fake_fully_shard)

    class _Block:
        pass

    class _Inner:
        layers = [_Block(), _Block()]

    class _Model:
        model = _Inner()

    model = _Model()
    apply_fsdp(model, _spec(tmp_path, reshard_after_forward=True), object())

    blocks, root = calls[:-1], calls[-1]
    assert [flag for _, flag in blocks] == [True, True]
    assert root[0] is model
    assert root[1] is False


def test_a_declared_optimizer_state_dtype_that_did_not_happen_is_an_error(tmp_path: Path):
    """
    The launcher cannot yet force the moment dtype, and the analytic inventory sizes
    optimizer state from the declared one. A silent disagreement is a wrong coefficient
    at fit time, so it is a loud failure at profile time instead.
    """
    from srl.profiler.adapters.fsdp.launch import check_optimizer_state_dtype

    class _Optimizer:
        state = {object(): {"exp_avg": torch.zeros(4, dtype=torch.bfloat16)}}

    with pytest.raises(ValueError, match="Optimizer state dtype mismatch"):
        check_optimizer_state_dtype(_Optimizer(), _spec(tmp_path, optimizer_state_dtype="fp32"))


def test_a_matching_optimizer_state_dtype_passes(tmp_path: Path):
    from srl.profiler.adapters.fsdp.launch import check_optimizer_state_dtype

    class _Optimizer:
        state = {object(): {"exp_avg": torch.zeros(4, dtype=torch.float32)}}

    check_optimizer_state_dtype(_Optimizer(), _spec(tmp_path, optimizer_state_dtype="fp32"))


def test_build_model_pins_train_mode(tmp_path: Path, monkeypatch):
    """HF gates recompute on `self.training`. Eval would silently disable checkpointing."""
    from transformers import AutoConfig

    def fake_from_pretrained(path, **kwargs):
        return AutoConfig.for_model(
            "qwen3",
            num_hidden_layers=1,
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1024,
        )

    monkeypatch.setattr(AutoConfig, "from_pretrained", fake_from_pretrained)
    model = build_model(_spec(tmp_path, num_layers_override=1, attn_implementation="sdpa"))
    assert model.training is True


def test_sidecar_omits_collective_timers_when_uninstrumented(tmp_path: Path):
    """Only one step is profiled, so an empty timer dict must stay empty."""
    measurement = FsdpMeasurement(run_dir=tmp_path, spec=_spec(tmp_path))
    measurement.step_begin()
    measurement.step_end()
    assert measurement.sidecar([])["timers_s"] == {}


def test_oom_sidecar_is_written_per_rank(tmp_path: Path):
    write_oom_sidecar(tmp_path, _spec(tmp_path), RuntimeError("CUDA out of memory"))
    markers = list(tmp_path.glob("oom_rank*.json"))
    assert len(markers) == 1
    payload = json.loads(markers[0].read_text())
    assert payload["torch_oom"] is True
    assert payload["sequence_length"] == 128
    assert payload["n_seqs_per_rank"] == 2


def test_oom_sidecar_records_torch_oom_source_by_default(tmp_path: Path):
    """The exact-type path must be distinguishable from the text-match path."""
    write_oom_sidecar(tmp_path, _spec(tmp_path), RuntimeError("CUDA out of memory"))
    markers = list(tmp_path.glob("oom_rank*.json"))
    payload = json.loads(markers[0].read_text())
    assert payload["oom_source"] == "torch_oom"


def test_oom_sidecar_records_nccl_alloc_source_when_requested(tmp_path: Path):
    """The text-match path must name its detection route so readers know the classification."""
    write_oom_sidecar(
        tmp_path,
        _spec(tmp_path),
        RuntimeError("Failed to CUDA calloc 10485760 bytes"),
        oom_source="nccl_alloc",
    )
    markers = list(tmp_path.glob("oom_rank*.json"))
    payload = json.loads(markers[0].read_text())
    assert payload["oom_source"] == "nccl_alloc"
    assert payload["torch_oom"] is True


def test_nccl_memory_exhaustion_detects_cuda_calloc_failure():
    """The real error text from the sweep must be classified as memory exhaustion."""
    error = RuntimeError(
        "NCCL error in: .../NCCLUtils.cpp:93, unhandled cuda error\n"
        "ncclUnhandledCudaError: Call to CUDA function failed.\n"
        "Last error: Failed to CUDA calloc 10485760 bytes"
    )
    assert is_nccl_memory_exhaustion(error) is True


def test_nccl_memory_exhaustion_detects_out_of_memory_phrase():
    """The common CUDA allocator phrasing must also match."""
    assert is_nccl_memory_exhaustion(RuntimeError("NCCL: out of memory")) is True


def test_nccl_memory_exhaustion_matching_is_case_insensitive():
    """Case must not matter: uppercase, lowercase, or mixed must all match."""
    assert is_nccl_memory_exhaustion(RuntimeError("FAILED TO CUDA CALLOC 1024 bytes")) is True
    assert is_nccl_memory_exhaustion(RuntimeError("failed to cuda calloc 1024 bytes")) is True
    assert is_nccl_memory_exhaustion(RuntimeError("Out Of Memory")) is True


def test_nccl_memory_exhaustion_rejects_bare_unhandled_cuda_error():
    """A bare `ncclUnhandledCudaError` with no memory phrase could be a driver fault."""
    error = RuntimeError(
        "NCCL error in: .../NCCLUtils.cpp:93, unhandled cuda error\n"
        "ncclUnhandledCudaError: Call to CUDA function failed."
    )
    assert is_nccl_memory_exhaustion(error) is False


def test_nccl_memory_exhaustion_rejects_timeout_error():
    """A timeout is a network fault, not memory exhaustion, and must stay `failed`."""
    assert is_nccl_memory_exhaustion(RuntimeError("NCCL timeout waiting for rendezvous")) is False


def test_nccl_memory_exhaustion_rejects_rendezvous_failure():
    """A rendezvous failure is a real network fault, not memory exhaustion."""
    assert (
        is_nccl_memory_exhaustion(RuntimeError("DistBackendError: Rendezvous failed"))
        is False
    )


def test_static_allocated_uses_live_not_peak(tmp_path: Path, monkeypatch):
    """
    `static_allocated_bytes_at_train_start` must reflect the live allocation, not the
    historical peak.

    A peak-based figure would include the pre-shard full-model residency that
    `fully_shard` releases before `on_train_start` is called, overstating the static
    pool and making the 16p/world_size inventory check read high. The live figure
    lands within ~6% of the analytic inventory, while the peak overshoots by ~23%.
    """
    LIVE_BYTES = 1_000_000
    PEAK_BYTES = 1_230_000  # ~23% higher, simulating the pre-shard spike

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def synchronize():
            pass

        @staticmethod
        def memory_allocated():
            return LIVE_BYTES

        @staticmethod
        def max_memory_allocated():
            return PEAK_BYTES

        @staticmethod
        def reset_peak_memory_stats():
            pass

    monkeypatch.setattr(torch, "cuda", _FakeCuda)

    measurement = FsdpMeasurement(run_dir=tmp_path, spec=_spec(tmp_path))
    measurement.on_train_start()
    # Captured after the first optimizer step rather than at train start: AdamW
    # allocates its moments lazily, so at train start the state does not exist yet.
    measurement.record_static_pool()
    sidecar = measurement.sidecar([])
    assert sidecar["static_allocated_bytes_at_train_start"] == LIVE_BYTES
