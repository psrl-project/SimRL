"""Fit versioned rollout and training cost artifacts from profile JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from .._logging import setup_logging, srl_logger
from ..cost.rollout import DistServe, PSRLFitted, UnifiedRoofline
from ..cost.rollout.plot import plot_vllm_rollout_regression
from ..cost.training import FsdpTrainingCost, HydraulisTrainingCost
from ..profiler.io import read_records


def cmd_rollout(args: argparse.Namespace) -> None:
    """Fit one rollout model family and save its artifact."""
    records = read_records(args.input, record_type="rollout")
    if args.model == "roofline":
        artifact = UnifiedRoofline.fit(
            records,
            block_size=args.block_size,
            validation_fraction=args.validation_fraction,
        )
    elif args.model == "psrl":
        artifact = PSRLFitted.fit(
            records,
            validation_fraction=args.validation_fraction,
        )
    elif args.model == "distserve":
        artifact = DistServe.fit(
            records,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            block_size=args.block_size,
            validation_fraction=args.validation_fraction,
        )
    else:
        raise ValueError(f"Unknown rollout regression model {args.model!r}.")
    artifact.save(args.output)
    srl_logger.info(
        f"Saved {args.model!r} rollout artifact with {len(artifact.entries)} entries "
        f"to {args.output}."
    )
    if {record.software.backend for record in records} == {"vllm"}:
        output = Path(args.output)
        plot_dir = output.parent / f"{output.stem}_plots"
        plot_paths = plot_vllm_rollout_regression(
            records,
            artifact,
            plot_dir,
            validation_fraction=args.validation_fraction,
        )
        for plot_path in plot_paths:
            srl_logger.info(f"Saved vLLM rollout regression plot to {plot_path}.")


def cmd_training(args: argparse.Namespace) -> None:
    """Fit training latency and memory entries for the selected model family."""
    records = read_records(args.input, record_type="training")
    if args.latency_model == "fsdp" or args.memory_model == "fsdp":
        if args.latency_model != args.memory_model:
            raise ValueError(
                "FSDP latency and memory must be fitted together. "
                "Set both --latency-model and --memory-model to fsdp."
            )
        artifact = FsdpTrainingCost.fit(
            records,
            validation_fraction=args.validation_fraction,
            require_full_rank=not args.allow_rank_deficient,
        )
        artifact.save(args.output)
        srl_logger.info(
            f"Saved FSDP training artifact with {len(artifact.entries)} entries "
            f"to {args.output}."
        )
        return
    if args.latency_model != "hydraulis" or args.memory_model != "hydraulis":
        raise ValueError("Training V1 only supports hydraulis latency and memory.")
    artifact = HydraulisTrainingCost.fit(
        records,
        validation_fraction=args.validation_fraction,
        require_full_rank=not args.allow_rank_deficient,
    )
    artifact.save(args.output)
    srl_logger.info(
        f"Saved Hydraulis training artifact with {len(artifact.entries)} entries "
        f"to {args.output}."
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the regression CLI for tests and the console entrypoint."""
    parser = argparse.ArgumentParser(
        prog="srl-regress",
        description="Fit cost models from versioned profiling JSONL.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    rollout = commands.add_parser("rollout")
    rollout.add_argument("--model", choices=["roofline", "psrl", "distserve"], required=True)
    rollout.add_argument("--input", required=True)
    rollout.add_argument("--output", required=True)
    rollout.add_argument("--block-size", type=int, default=16)
    rollout.add_argument("--validation-fraction", type=float, default=0.0)
    rollout.add_argument("--hidden-size", type=int)
    rollout.add_argument("--intermediate-size", type=int)
    rollout.set_defaults(func=cmd_rollout)

    training = commands.add_parser("training")
    training.add_argument("--latency-model", choices=["hydraulis", "fsdp"], default="hydraulis")
    training.add_argument("--memory-model", choices=["hydraulis", "fsdp"], default="hydraulis")
    training.add_argument("--input", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--validation-fraction", type=float, default=0.2)
    training.add_argument("--allow-rank-deficient", action="store_true")
    training.set_defaults(func=cmd_training)
    return parser


def main() -> None:
    """Run one regression workflow."""
    parser = build_parser()
    args = parser.parse_args()
    setup_logging()
    args.func(args)


if __name__ == "__main__":
    main()
