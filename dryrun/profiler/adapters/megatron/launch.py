"""Torchrun entry point that profiles one Megatron training candidate.

The vendored `scripts/performance/run_recipe.py` parses known arguments and
discards the rest, so recompute, profiling, and output-path settings never reached
the training config. This module takes the library recipe and applies the candidate
in Python instead, where a mistyped field is an error rather than a dropped token.

Imports torch and megatron at call time, so it must never be imported from the
adapter package namespace.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer

from .execute import OOM_EXIT_CODE


def _resolve_recipe(family: str, name: str, task: str):
    """Return the library recipe builder for one model family and size."""
    import importlib

    package = importlib.import_module(f"megatron.bridge.recipes.{family}")
    builder_name = f"{name}_{task}_config"
    builder = getattr(package, builder_name, None)
    if builder is None:
        raise ValueError(
            f"Megatron-Bridge has no library recipe {builder_name!r} in "
            f"megatron.bridge.recipes.{family}."
        )
    return builder


@contextlib.contextmanager
def _local_checkpoint(model_path: str):
    """
    Resolve the recipe's pinned hub identifier to a local checkpoint.

    Library recipes hardcode a hub id (`Qwen/Qwen3-8B`) and read its config while
    constructing the model provider. That lookup is a hard failure on an isolated
    host, and even when it succeeds it describes a checkpoint other than the one
    the rest of the pipeline profiles. Redirecting the lookup keeps the recipe's
    tuning intact: copying its fields here would silently drift from upstream --
    while pinning the architecture to `model_path`.
    """
    from megatron.bridge.models.conversion import auto_bridge

    original = auto_bridge.AutoBridge.from_hf_pretrained

    @classmethod
    def from_local(cls, path, **kwargs):
        return original.__func__(cls, model_path, **kwargs)

    auto_bridge.AutoBridge.from_hf_pretrained = from_local
    try:
        yield
    finally:
        auto_bridge.AutoBridge.from_hf_pretrained = original


def _normalize_flash_attn_version() -> None:
    """Make Transformer Engine accept a post-release flash-attn wheel.

    TE gates flash attention on `version_required <= version <= max_version` read
    from package metadata, and PEP 440 orders a `.postN` release *above* the plain
    version. A wheel matching the upper bound exactly is therefore rejected as too
    new, and TE falls back without complaint to `UnfusedDotProductAttention`, which
    materializes the full attention matrix and is several times slower. Since a
    post-release carries no functional change, truncating the suffix gives the bound
    the version it was written for.

    Must run before `transformer_engine` is first imported.
    """
    import importlib.metadata

    original = importlib.metadata.version

    def truncated(name: str) -> str:
        value = original(name)
        if name == "flash-attn" and ".post" in value:
            return value.split(".post", 1)[0]
        return value

    importlib.metadata.version = truncated


def _apex_wgrad_fusion_available() -> bool:
    """Report whether Apex's fused weight-gradient extension is importable."""
    try:
        import fused_weight_gradient_mlp_cuda  # noqa: F401
    except ImportError:
        return False
    return True


def _make_thd_batch_iterator(
    sequence_lengths: list[int],
    global_batch_size: int,
    num_microbatches: int,
    vocab_size: int,
    tensor_parallel_size: int = 1,
):
    """Yield synthetic THD-packed batches, matching verl's `use_remove_padding=True`.

    Token values are random: only the shapes and `cu_seqlens` drive the kernels
    being timed. The same batch is yielded forever so the loop never exhausts it.

    Args:
        sequence_lengths: Per-sequence lengths in one packed microbatch.
        global_batch_size: Unused; kept so callers can pass the spec verbatim.
        num_microbatches: Unused here, checked in `DryRunMeasurementCallback`.
        vocab_size: Upper bound for random token ids.
        tensor_parallel_size: Alignment multiple for each sequence.
    """
    import torch

    # Sequence parallelism splits the flattened token dimension across TP ranks, so
    # an unaligned length fails Megatron's own divisibility check. verl pads the same
    # way in `preprocess_thd_engine`, and the real trainer pays for these tokens too.
    align = max(1, tensor_parallel_size)
    sequence_lengths = [length + (-length % align) for length in sequence_lengths]
    n_seqs = len(sequence_lengths)
    total_tokens = sum(sequence_lengths)

    cu = [0]
    for length in sequence_lengths:
        cu.append(cu[-1] + length)
    # Bridge slices `cu_seqlens` at its first zero, so append one as a sentinel.
    cu_seqlens_padded = torch.cat(
        [torch.tensor(cu, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)]
    )
    argmin = torch.tensor([n_seqs + 1], dtype=torch.long)
    max_seqlen = torch.tensor([max(sequence_lengths)], dtype=torch.long)

    # Position ids restart per sequence; one arange over the flat buffer would
    # rotate later sequences as though they continued the first.
    position_ids_flat = torch.cat(
        [torch.arange(length, dtype=torch.long) for length in sequence_lengths]
    )

    # Built once and re-yielded. Rebuilding per microbatch would put a large `randint`
    # and a synchronous host-to-device copy on the critical path, with no dataloader
    # worker to overlap them against.
    #
    # The leading singleton is the batch dimension: attention splits the fused QKV
    # projection on `dim=3`, so a flat 1D buffer is one dimension short.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = {
        "tokens": torch.randint(0, vocab_size, (1, total_tokens), dtype=torch.long, device=device),
        "labels": torch.randint(0, vocab_size, (1, total_tokens), dtype=torch.long, device=device),
        "loss_mask": torch.ones(1, total_tokens, dtype=torch.float, device=device),
        "position_ids": position_ids_flat.unsqueeze(0).to(device),
        "cu_seqlens": cu_seqlens_padded.to(device),
        # Bridge moves these two to host itself, so they stay on the CPU.
        "cu_seqlens_argmin": argmin,
        "max_seqlen": max_seqlen,
    }

    while True:
        yield batch


def _patch_thd_batches(gpt_step_module, spec: dict[str, Any]) -> None:
    """Replace Bridge's batch reader with a synthetic THD-packed source.

    Supplying `cu_seqlens` is the whole switch: every downstream packed-sequence
    path keys off its presence, so `_forward_step_common` builds
    `PackedSeqParams(qkv_format="thd")` and TE dispatches the varlen kernel.
    A dataset cannot do this: `packed_sequence_specs` lives on
    `FinetuningDatasetConfig` and expects pre-built `.npy` packs on disk, and the
    patch is process-local anyway, since each candidate gets its own torchrun child.

    Args:
        gpt_step_module: The imported `megatron.bridge.training.gpt_step` module.
        spec: The launch spec, read for `sequence_lengths` and `vocab_size`.
    """
    from megatron.core.packed_seq_params import PackedSeqParams

    tp = int(spec["parallelism"]["tp"])
    # Aligned the same way the iterator aligns them, so `cu_seqlens` and the
    # recorded shape describe the same buffer.
    sequence_lengths = [
        length + (-length % max(1, tp))
        for length in (int(item) for item in spec["sequence_lengths"])
    ]
    vocab_size = int(spec["vocab_size"])
    iterator = _make_thd_batch_iterator(
        sequence_lengths,
        int(spec["global_batch_size"]),
        int(spec["num_microbatches"]),
        vocab_size,
        tensor_parallel_size=tp,
    )

    # Built directly rather than through `packed_seq_utils.get_packed_seq_params`,
    # which squeezes `max_seqlen` into a 0-dim tensor where a plain int is wanted.
    # These `cu_seqlens` are also already the right length, so the helper's argmin
    # slicing has nothing to do.
    max_seqlen_int = max(sequence_lengths)

    def packed_params_for(batch: dict[str, Any]) -> PackedSeqParams:
        cu = batch["cu_seqlens"][: len(sequence_lengths) + 1]
        return PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=max_seqlen_int,
            max_seqlen_kv=max_seqlen_int,
        )

    gpt_step_module.get_packed_seq_params = packed_params_for

    def patched_get_batch_from_iterator(
        data_iterator,
        use_mtp: bool = False,
        skip_getting_attention_mask_from_dataset: bool = True,
        *,
        is_first_pp_stage: bool,
        is_last_pp_stage: bool,
    ):
        """Return one synthetic THD batch, placed exactly as Bridge places its own."""
        batch = next(iterator)
        device_keys = {"cu_seqlens"}
        host_keys = {"cu_seqlens_argmin", "max_seqlen"}
        if is_first_pp_stage or use_mtp:
            device_keys.update(("tokens", "position_ids"))
        if is_last_pp_stage:
            device_keys.update(("labels", "loss_mask"))

        placed: dict[str, Any] = {}
        for key, value in batch.items():
            if key in device_keys:
                placed[key] = value if value.is_cuda else value.cuda(non_blocking=True)
            elif key in host_keys:
                placed[key] = value
            else:
                # Bridge nulls every key the rank does not need; middle pipeline
                # stages therefore receive no tokens.
                placed[key] = None
        return placed

    gpt_step_module.get_batch_from_iterator = patched_get_batch_from_iterator


def build_config(spec: dict[str, Any]) -> ConfigContainer:
    """
    Apply one candidate to the library recipe's configuration container.

    Pure with respect to the spec, so every field below is unit-testable without a
    process group. Reading the serialized result back in `collect` is the
    independent check that these mutations actually landed.
    """
    from megatron.bridge.training.config import (
        MockGPTDatasetConfig,
        ProfilingConfig,
        TokenizerConfig,
    )

    recipe = spec["recipe"]
    builder = _resolve_recipe(recipe["family"], recipe["name"], recipe["task"])
    with _local_checkpoint(spec["model_path"]):
        cfg = builder()

    parallelism = spec["parallelism"]
    cfg.model.tensor_model_parallel_size = int(parallelism["tp"])
    cfg.model.pipeline_model_parallel_size = int(parallelism["pp"])
    cfg.model.context_parallel_size = int(parallelism["cp"])
    cfg.model.virtual_pipeline_model_parallel_size = None
    # Follows TP, as Megatron and verl both do by default. Being a function of TP it
    # is not a free cell-key dimension, so one cell per TP still describes one
    # configuration. `DRYRUN_SEQUENCE_PARALLEL=0` forces it off for A/B diagnosis;
    # measured effect on step time is nil.
    _sp_override = os.environ.get("DRYRUN_SEQUENCE_PARALLEL")
    cfg.model.sequence_parallel = (
        int(parallelism["tp"]) > 1 if _sp_override is None else _sp_override == "1"
    )

    sequence_length = int(spec["seq_length"])
    cfg.model.seq_length = sequence_length
    cfg.train.micro_batch_size = int(spec["micro_batch_size"])
    cfg.train.global_batch_size = int(spec["global_batch_size"])
    cfg.train.train_iters = int(spec["train_iters"])

    if spec["recompute"] == "full":
        cfg.model.recompute_granularity = "full"
        cfg.model.recompute_method = str(spec["recompute_method"])
        cfg.model.recompute_num_layers = int(spec["recompute_num_layers"])
    elif spec["recompute"] in {"", "none", "off"}:
        cfg.model.recompute_granularity = None
        cfg.model.recompute_method = None
        cfg.model.recompute_num_layers = None
    else:
        raise ValueError(f"Unsupported recompute mode {spec['recompute']!r}.")
    cfg.model.recompute_modules = None

    if spec.get("num_layers_override") is not None:
        cfg.model.num_layers = int(spec["num_layers_override"])

    # The provider enables weight-gradient accumulation fusion whenever
    # Transformer Engine is importable, but the same flag also gates the non-TE
    # `ColumnParallelLinear` used for the output layer, which needs Apex's
    # `fused_weight_gradient_mlp_cuda`. Without Apex that layer raises during model
    # construction, so the fusion is only requested when both backends can honor it.
    if cfg.model.gradient_accumulation_fusion and not _apex_wgrad_fusion_available():
        cfg.model.gradient_accumulation_fusion = False

    # Mock data keeps the measurement about compute. The dataset and model sequence
    # lengths are validated against each other downstream, so they are set together.
    # Under THD this is constructed but never read: `_patch_thd_batches` supplies
    # the batches, since the mock dataset cannot emit `cu_seqlens`.
    cfg.dataset = MockGPTDatasetConfig(
        seq_length=sequence_length,
        random_seed=1234,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        num_dataset_builder_threads=1,
        split="99990,8,2",
        data_sharding=True,
        dataloader_type="single",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    # NullTokenizer skips loading the real tokenizer while keeping embedding and
    # LM-head shapes faithful, which is what makes the measured parameter count
    # match the real model.
    cfg.tokenizer = TokenizerConfig(
        tokenizer_type="NullTokenizer",
        vocab_size=int(spec["vocab_size"]),
    )

    cfg.checkpoint.save = None
    cfg.checkpoint.load = None
    cfg.checkpoint.pretrained_checkpoint = None
    cfg.checkpoint.save_interval = None

    # These live on the validation config; the identically named training fields
    # are deprecated and warn on every access. The recipe enables evaluation by
    # default, which a four-step profiling run must not pay for.
    cfg.validation.eval_iters = 0
    cfg.validation.eval_interval = None
    cfg.validation.skip_train = False

    run_dir = Path(spec["run_dir"])
    cfg.logger.tensorboard_dir = None
    cfg.logger.wandb_project = None
    cfg.logger.wandb_entity = None
    cfg.logger.wandb_exp_name = None
    cfg.logger.wandb_save_dir = None
    cfg.logger.log_interval = 1
    cfg.logger.log_timers_to_tensorboard = False
    # Level 0 by default: Megatron's timers synchronize on every start and stop,
    # and we take our own synchronized measurement, so a finer level only adds
    # overhead. DRYRUN_TIMING_LOG_LEVEL=2 turns on the per-phase breakdown when
    # diagnosing where a step's time actually goes.
    cfg.logger.timing_log_level = int(os.environ.get("DRYRUN_TIMING_LOG_LEVEL", "0"))
    cfg.logger.save_config_filepath = str(run_dir / "ConfigContainer.yaml")

    # Peaks are read straight from the allocator, so neither the PyTorch profiler
    # nor allocator history recording is needed. Recording history would also
    # perturb the very memory the model is fitted against.
    cfg.profiling = ProfilingConfig(
        use_pytorch_profiler=False,
        use_nsys_profiler=False,
        record_memory_history=False,
        profile_step_start=0,
        profile_step_end=0,
        profile_ranks=[],
    )

    # The static memory inventory is computed analytically from these, so an
    # upstream recipe change must not silently invalidate it.
    declared = bool(spec["distributed_optimizer"])
    if bool(cfg.ddp.use_distributed_optimizer) != declared:
        raise ValueError(
            f"Recipe use_distributed_optimizer={cfg.ddp.use_distributed_optimizer!r} "
            f"does not match the profiled declaration {declared!r}."
        )
    return cfg


def _read_spec(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        spec = json.load(handle)
    if spec.get("schema_version") != 1:
        raise ValueError(f"Unsupported launch spec schema in {path}.")
    return spec


def main() -> None:
    """Profile one candidate and write its measurements next to the spec."""
    parser = argparse.ArgumentParser(
        prog="dryrun-megatron-launch",
        description="Profile one Megatron training candidate under torchrun.",
    )
    parser.add_argument("--spec", required=True, help="Path to launch_spec.json.")
    args = parser.parse_args()

    # Before the first `transformer_engine` import, which happens via megatron
    # below and reads the flash-attn version exactly once at module scope.
    _normalize_flash_attn_version()

    spec = _read_spec(Path(args.spec))
    run_dir = Path(spec["run_dir"])

    import torch
    from megatron.bridge.training import gpt_step as gpt_step_module
    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.pretrain import pretrain

    from .measure import DryRunMeasurementCallback, write_oom_sidecar

    if str(spec.get("packing", "bshd")) == "thd":
        _patch_thd_batches(gpt_step_module, spec)

    callback = DryRunMeasurementCallback(run_dir=run_dir, spec=spec)
    try:
        pretrain(
            config=build_config(spec),
            forward_step_func=forward_step,
            callbacks=[callback],
        )
    except torch.OutOfMemoryError as error:
        # Caught by exact type. A driver-level CUDA failure or a NCCL error is a
        # different fault and must stay classified as a failure, because an
        # out-of-memory record can only raise the fitted memory floor.
        write_oom_sidecar(run_dir, spec, error)
        raise SystemExit(OOM_EXIT_CODE) from error
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
