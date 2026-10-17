"""CLI for collecting and validating joint grouped workloads."""

from __future__ import annotations

import argparse
import sys

from .._logging import setup_logging, srl_logger


def cmd_collect_groups(args: argparse.Namespace) -> None:
    """Collect one worker's shard of a joint group trace."""
    from ..workload.inference import collect_group_trace  # noqa: PLC0415

    extra_serve_args = dict(item.split("=", 1) for item in args.serve_arg)
    path = collect_group_trace(
        dataset_path=args.dataset,
        tokenizer_path=args.tokenizer,
        model_path=args.model,
        output_path=args.output,
        worker_index=args.worker_index,
        n_workers=args.n_workers,
        sample_size=args.sample_size,
        n=args.n,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tp=args.tp,
        concurrency=args.concurrency,
        port=args.port,
        seed=args.seed,
        server_ready_timeout=args.server_ready_timeout,
        extra_serve_args=extra_serve_args,
        gpu=args.gpu,
    )
    srl_logger.info(f"Group trace shard written to {path}.")


def cmd_merge_shards(args: argparse.Namespace) -> None:
    """Merge worker shards into one joint group trace."""
    from ..workload.sharding import merge_shards  # noqa: PLC0415

    metadata: dict = {"kind": "joint_group_trace"}
    if args.max_output_tokens is not None:
        metadata["max_output_tokens"] = args.max_output_tokens
    path = merge_shards(
        shard_dir=args.shard_dir,
        n_workers=args.n_workers,
        output_path=args.output,
        metadata=metadata,
    )
    srl_logger.info(f"Merged group trace written to {path}.")


def cmd_list_workers(args: argparse.Namespace) -> None:
    """Print one machine-readable `index host gpu port` row per worker."""
    from ..workload.sharding import build_workers, parse_hosts_file  # noqa: PLC0415

    for worker in build_workers(parse_hosts_file(args.hosts_file), base_port=args.base_port):
        print(worker.index, worker.host, worker.gpu, worker.port)


def cmd_check_shard(args: argparse.Namespace) -> int:
    """Return success only when a shard body contains every requested group."""
    from ..workload.sharding import shard_is_complete  # noqa: PLC0415

    return 0 if shard_is_complete(args.path) else 1


def cmd_verify_collection(args: argparse.Namespace) -> int:
    """Verify a merged trace and print actionable short-shard diagnostics."""
    from ..workload.sharding import verify_collection  # noqa: PLC0415

    try:
        actual = verify_collection(
            args.trace,
            args.expected_groups,
            shard_dir=args.shard_dir,
            n_workers=args.n_workers,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Trace complete: {actual}/{args.expected_groups} groups collected.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the `srl-workload` argument parser."""
    parser = argparse.ArgumentParser(
        prog="srl-workload",
        description="Collect and validate joint grouped workload traces.",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    collect_groups = subs.add_parser(
        "collect-groups",
        help="Collect one worker's shard of a joint (prompt, n responses) trace.",
    )
    collect_groups.add_argument("--dataset", required=True, help="Parquet dataset path.")
    collect_groups.add_argument("--tokenizer", required=True, help="Tokenizer name or path.")
    collect_groups.add_argument("--model", required=True, help="Model passed to `vllm serve`.")
    collect_groups.add_argument("--output", required=True, help="Output shard JSONL path.")
    collect_groups.add_argument("--worker-index", type=int, default=0, help="This worker's index.")
    collect_groups.add_argument("--n-workers", type=int, default=1, help="Total worker count.")
    collect_groups.add_argument(
        "--sample-size", type=int, default=1024, help="Distinct prompts across all workers."
    )
    collect_groups.add_argument("--n", type=int, default=8, help="Responses sampled per prompt.")
    collect_groups.add_argument(
        "--max-tokens", type=int, default=10240, help="Max output tokens per response."
    )
    collect_groups.add_argument("--temperature", type=float, default=1.0)
    collect_groups.add_argument("--top-p", type=float, default=1.0)
    collect_groups.add_argument("--tp", type=int, default=1, help="Tensor-parallel size.")
    collect_groups.add_argument(
        "--concurrency", type=int, default=16, help="Max in-flight prompts per worker."
    )
    collect_groups.add_argument("--port", type=int, default=8700)
    collect_groups.add_argument("--seed", type=int, default=42)
    collect_groups.add_argument("--gpu", type=int, default=None, help="GPU to pin this worker to.")
    collect_groups.add_argument("--server-ready-timeout", type=int, default=1800)
    collect_groups.add_argument(
        "--serve-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra `vllm serve` flag. Repeatable.",
    )
    collect_groups.set_defaults(func=cmd_collect_groups)

    merge = subs.add_parser("merge-shards", help="Merge worker shards into one joint trace.")
    merge.add_argument("--shard-dir", required=True, help="Directory of worker-NN.jsonl shards.")
    merge.add_argument("--n-workers", type=int, required=True, help="Shards expected.")
    merge.add_argument("--output", required=True, help="Merged joint trace path.")
    merge.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help=(
            "If supplied, verify it against every shard header. "
            "Omit to let the shards be authoritative."
        ),
    )
    merge.set_defaults(func=cmd_merge_shards)

    list_workers = subs.add_parser(
        "list-workers", help="Expand a hosts file into `index host gpu port` rows."
    )
    list_workers.add_argument("--hosts-file", required=True, help="Path to host/GPU inventory.")
    list_workers.add_argument("--base-port", type=int, default=8700)
    list_workers.set_defaults(func=cmd_list_workers)

    check_shard = subs.add_parser(
        "check-shard", help="Exit successfully only when a worker shard is complete."
    )
    check_shard.add_argument("--path", required=True, help="Worker shard JSONL path.")
    check_shard.set_defaults(func=cmd_check_shard)

    verify = subs.add_parser(
        "verify-collection",
        help="Verify merged group count and diagnose short worker shards.",
    )
    verify.add_argument("--trace", required=True, help="Merged joint trace path.")
    verify.add_argument("--expected-groups", type=int, required=True)
    verify.add_argument("--shard-dir", required=True)
    verify.add_argument("--n-workers", type=int, required=True)
    verify.set_defaults(func=cmd_verify_collection)

    return parser


def main() -> None:
    """Run a workload collection command."""
    parser = build_parser()
    args = parser.parse_args()
    setup_logging()
    raise SystemExit(args.func(args) or 0)


if __name__ == "__main__":
    main()
