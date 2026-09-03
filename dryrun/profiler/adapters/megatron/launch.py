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
    tuning intact -- copying its fields here would silently drift from upstream --
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


def _apex_wgrad_fusion_available() -> bool:
    """Report whether Apex's fused weight-gradient extension is importable."""
    try:
        import fused_weight_gradient_mlp_cuda  # noqa: F401
    except ImportError:
        return False
    return True


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
    # The cost-model cell key does not record sequence parallelism, so letting it
    # vary would mix incomparable measurements into one cell.
    cfg.model.sequence_parallel = False

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
        # Worker startup would land inside the first measured steps of every
        # candidate; generating mock batches inline is cheaper than paying it.
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
    # Megatron's timers synchronize on every start and stop. We take our own
    # synchronized measurement, so a finer level would only add overhead.
    cfg.logger.timing_log_level = 0
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

    spec = _read_spec(Path(args.spec))
    run_dir = Path(spec["run_dir"])

    import torch
    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.pretrain import pretrain

    from .measure import DryRunMeasurementCallback, write_oom_sidecar

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
