#!/usr/bin/env python
"""Deterministic digests of small simulator runs, to prove a refactor changed no behavior.

Each named configuration launches the simulator CLI against one code tree (`--root`) and
hashes the telemetry streams it writes. Those streams carry simulated time only, so two runs
of one tree agree bit for bit and a digest mismatch between two trees is a behavior change.

Usage:
    python examples/analysis/sim_digest.py list
    python examples/analysis/sim_digest.py run psrl_s1 --root /tmp/srl_ref --out /tmp/psrl_s1.json
    python examples/analysis/sim_digest.py table --root /tmp/srl_ref --out reference.json
    python examples/analysis/sim_digest.py check --reference reference.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from itertools import zip_longest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRATCH = Path("/tmp/sim_digest_runs")
MODEL_PATH = "/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B"
KERNEL_ROOT = REPO / "outputs" / "profile" / "qwen3_8b" / "kernel_tp1" / "kernel"
TRAINING_ARTIFACT = REPO / "outputs" / "cost" / "qwen3_8b_tp2_pp3_thd" / "training_hydraulis.json"

VERSIONS = 5
BATCH = 8
GROUP_SIZE = 8
INSTANCES = 2
TRACE_SHAPE = {"n-groups": 80, "mean": 6000, "cv": 1.0, "rho": 0.6, "cap": 16384}
SIM_TIMEOUT_S = 900


@dataclass(frozen=True)
class Config:
    note: str
    policy: str = "psrl"
    staleness: int = 2
    backend: str = "native"
    training: tuple[str, ...] = ("training_cost=fixed", "training_cost.latency_s=100")
    kv: tuple[str, ...] = ("rollout.kv_blocks=8000",)
    extra: tuple[str, ...] = ()
    deprecated: bool = False


_LENGTH_AWARE = (
    "rollout.router.strategy=length_aware",
    "rollout.router.queue_window=100000",
    "+rollout.router.strategy_args.order=lpt",
    "+rollout.router.strategy_args.balance=load",
    f"+rollout.router.strategy_args.window_groups={BATCH}",
    "+rollout.router.strategy_args.length_model=oracle",
)
_FITTED_TRAINING = (
    "training_cost=hydraulis",
    f"training_cost.artifact_path={TRAINING_ARTIFACT}",
    "training_cost.parallelism.tp=2",
    "training_cost.parallelism.pp=3",
    "training_cost.parallelism.dp=1",
    "training_cost.parallelism.cp=1",
    "training_cost.micro_batch_size=1",
    "training_cost.activation_recompute=full",
    "training_cost.max_tokens_per_microbatch=12800",
    "training_cost.extrapolate=[tokens,activation_bytes,pipeline_slots,sum_squared_tokens]",
)
_WEIGHT_LOAD = (
    "sync_cost.sync_time=3.0",
    "sync_cost.weight_load_time=4.5",
    "recompute_cost.name=fixed",
    "recompute_cost.time_per_token=2e-5",
)

CONFIGS: dict[str, Config] = {
    "psrl_s0": Config("psrl, staleness 0, no cap", staleness=0),
    "psrl_s1": Config("psrl, staleness 1, no cap", staleness=1),
    "psrl_s2": Config("psrl, staleness 2, no cap"),
    "psrl_s2_cap48": Config(
        "psrl, staleness 2, router cap of 48 per instance",
        extra=("rollout.router.max_concurrent_seqs_per_instance=48",),
    ),
    "psrl_s1_lpt": Config(
        "psrl, staleness 1, length_aware longest-first over one batch window (oracle lengths)",
        staleness=1,
        extra=_LENGTH_AWARE,
    ),
    "psrl_s2_longest_ctx": Config(
        "psrl, staleness 2, longest_ctx eviction", extra=("rollout.router.eviction=longest_ctx",)
    ),
    "verl_s2": Config("verl policy, staleness 2", policy="verl"),
    "roll_s2": Config("roll policy, staleness 2", policy="roll"),
    "areal_s2": Config("areal policy, staleness 2", policy="areal"),
    "slime": Config("slime policy", policy="slime"),
    "psrl_s1_cpp": Config("psrl, staleness 1, C++ engine, paired with psrl_s1", staleness=1, backend="cpp"),
    "psrl_s2_longest_ctx_cpp": Config(
        "psrl, staleness 2, longest_ctx eviction, C++ engine",
        backend="cpp",
        extra=("rollout.router.eviction=longest_ctx",),
    ),
    "psrl_s1_wload": Config(
        "psrl, staleness 1, partial rollout with sync, weight load and recompute charges",
        staleness=1,
        extra=_WEIGHT_LOAD,
    ),
    "psrl_s2_reclaim": Config(
        "psrl, staleness 2, preempted requests handed back to the router",
        extra=("rollout.router.max_waiting_after_preempt=4",),
    ),
    "psrl_s2_sticky": Config(
        "psrl, staleness 2, group sticky routing", extra=("rollout.router.enable_group_sticky=true",)
    ),
    "psrl_s2_throughput_balance": Config(
        "psrl, staleness 2, throughput_balance routing",
        extra=("rollout.router.strategy=throughput_balance",),
    ),
    "psrl_s2_kv_aware": Config(
        "psrl, staleness 2, kv_cache_aware routing", extra=("rollout.router.strategy=kv_cache_aware",)
    ),
    "psrl_s2_round_robin": Config(
        "psrl, staleness 2, round_robin routing", extra=("rollout.router.strategy=round_robin",)
    ),
    "psrl_s2_random": Config(
        "psrl, staleness 2, seeded random routing",
        extra=("rollout.router.strategy=random", "+rollout.router.strategy_args.seed=0"),
    ),
    "psrl_s1_fitted": Config(
        "psrl, staleness 1, fitted TP2/PP3 training model, training_forward recompute, arithmetic KV sizing",
        staleness=1,
        training=_FITTED_TRAINING,
        kv=(
            f"rollout.kv_cache_sizing.model_path={MODEL_PATH}",
            "rollout.kv_cache_sizing.probe=false",
        ),
        extra=("recompute_cost.name=training_forward",),
    ),
}


def ensure_trace() -> Path:
    """Generate the shared synthetic trace once, atomically."""
    tag = "_".join(f"{key}{value}" for key, value in TRACE_SHAPE.items())
    path = SCRATCH / "traces" / f"dispersed_{tag}_gs{GROUP_SIZE}.jsonl"
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    command = [
        sys.executable,
        str(REPO / "examples" / "workload" / "make_dispersed_trace.py"),
        "--out", str(partial),
        "--group-size", str(GROUP_SIZE),
    ]
    for key, value in TRACE_SHAPE.items():
        command += [f"--{key}", str(value)]
    subprocess.run(command, check=True, capture_output=True)
    os.replace(partial, path)
    return path


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def engine_extension(root: Path) -> Path:
    return next((root / "srl").glob("_engine_cpp*.so"))


def child_env(root: Path, backend: str, psrl_pin: Path | None) -> dict[str, str]:
    env = dict(os.environ)
    paths = [str(root)]
    if psrl_pin is not None:
        paths.append(str(psrl_pin))
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONHASHSEED"] = "0"
    env["SIMRL_BACKEND"] = backend
    env.pop("SIMRL_UNDOABLE_EVENTS", None)
    return env


@lru_cache(maxsize=None)
def verify_import(root: Path, backend: str, psrl_pin: Path | None) -> tuple[str, str]:
    """Return where `srl` and its engine extension import from, and require both under `root`."""
    probe = "import srl, srl._engine_cpp as e; print(srl.__file__); print(e.__file__)"
    if psrl_pin is not None:
        probe += "; import srl.policy.psrl as p; print(p._controller.__file__)"
    out = subprocess.run(
        [sys.executable, "-c", probe],
        env=child_env(root, backend, psrl_pin),
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    for path in out[:2]:
        assert Path(path).is_relative_to(root), f"Imported {path} from outside {root}."
    if psrl_pin is not None:
        assert Path(out[2]).is_relative_to(psrl_pin), f"Loaded the staleness controller from {out[2]}."
    return out[0], out[1]


def hydra_overrides(cfg: Config, run_dir: Path, trace: Path) -> list[str]:
    return [
        "workload=dapo_trace",
        f"workload.trace.path={trace}",
        f"policy={cfg.policy}",
        "rollout_cost=composed",
        f"rollout_cost.kernel_table_root={KERNEL_ROOT}",
        "rollout_cost.gpu=h20",
        f"rollout_cost.model_path={MODEL_PATH}",
        *(f"rollout_cost.parallelism.{axis}=1" for axis in ("tp", "pp", "dp", "cp")),
        *cfg.training,
        *cfg.kv,
        f"job.n_versions={VERSIONS}",
        f"job.batch_size={BATCH}",
        f"job.group_size={GROUP_SIZE}",
        f"policy.horizon={cfg.staleness}",
        "job.max_prompt_length=1024",
        f"job.max_output_length={TRACE_SHAPE['cap']}",
        f"rollout.n_instances={INSTANCES}",
        "rollout.token_budget=12800",
        "rollout.max_concurrency=1024",
        "rollout.router.max_waiting_after_preempt=1024",
        "rollout.router.reject_preempted_on_waiting=true",
        *cfg.extra,
        f"output_dir={run_dir}",
        f"hydra.run.dir={run_dir / 'hydra'}",
    ]


def stream_digests(logs: Path) -> dict[str, dict]:
    streams = {}
    for path in sorted(logs.rglob("*.jsonl")):
        digest, rows = hashlib.md5(), 0
        with path.open("rb") as handle:
            for line in handle:
                digest.update(line)
                rows += 1
        streams[path.relative_to(logs).with_suffix("").as_posix()] = {"md5": digest.hexdigest(), "rows": rows}
    return streams


def read_stream(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def headline(logs: Path) -> dict:
    """Result numbers recomputed from the telemetry alone."""
    train = read_stream(logs / "train.jsonl")
    requests = read_stream(logs / "request.jsonl")
    final = {row["rid"]: row for row in requests}
    consumed = [row for row in requests if row["event"] == "consume"]
    tokens = sum(row["generated"] for row in consumed)
    groups = len({row["group_id"] for row in consumed})
    sim_time = train[-1]["end"] if train else 0.0
    return {
        "sim_time": sim_time,
        "versions_done": len(train),
        "n_consumed": len(consumed),
        "consumed_groups": groups,
        "consumed_tokens": tokens,
        "tokens_per_group": tokens / groups if groups else 0.0,
        "n_dropped": sum(1 for row in requests if row["event"] == "drop"),
        "n_preemptions": sum(row["num_preemptions"] for row in final.values()),
        "n_aborts": sum(row["num_aborts"] for row in final.values()),
        "reprefill_tokens": sum(row["reprefill_tokens"] for row in final.values()),
        "max_staleness_true": max((row["staleness_true"] for row in consumed), default=0),
        "throughput": tokens / sim_time if sim_time else 0.0,
    }


def stdout_agrees(log: Path, head: dict) -> bool:
    """Check the CLI's own printed summary against the telemetry-derived numbers."""
    text = log.read_text()
    done = re.search(r"Simulation complete: (\d+) versions in ([\d.]+)s", text)
    consumed = re.search(r"Consumed: (\d+) requests", text)
    dropped = re.search(r"Dropped: (\d+) requests", text)
    if not (done and consumed and dropped):
        return False
    return (
        int(done[1]) == head["versions_done"]
        and done[2] == f"{head['sim_time']:.2f}"
        and int(consumed[1]) == head["n_consumed"]
        and int(dropped[1]) == head["n_dropped"]
    )


def run_config(
    name: str,
    root: Path,
    tag: str = "",
    psrl_pin: Path | None = None,
    keep_instance: bool = False,
) -> dict:
    cfg = CONFIGS[name]
    trace = ensure_trace()
    srl_file, engine_file = verify_import(root, cfg.backend, psrl_pin)
    run_dir = SCRATCH / (f"{name}.{tag}" if tag else name)
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    argv = [sys.executable, "-m", "srl.cli.main", *hydra_overrides(cfg, run_dir, trace)]
    started = time.perf_counter()
    with (run_dir / "run.log").open("w") as log:
        proc = subprocess.run(
            argv,
            env=child_env(root, cfg.backend, psrl_pin),
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=SIM_TIMEOUT_S,
        )
    wall = time.perf_counter() - started
    record = {
        "name": name,
        "root": str(root),
        "backend": cfg.backend,
        "srl_file": srl_file,
        "engine_file": engine_file,
        "engine_md5": file_md5(Path(engine_file)),
        "trace_md5": file_md5(trace),
        "psrl_pin": str(psrl_pin) if psrl_pin else None,
        "telemetry_dir": str(run_dir),
        "argv": argv,
        "returncode": proc.returncode,
        "wall_s": round(wall, 2),
        "deprecated": cfg.deprecated,
    }
    if proc.returncode != 0:
        record["failed"] = True
        return record
    logs = run_dir / "logs"
    streams = stream_digests(logs)
    record["streams"] = streams
    record["digest"] = hashlib.md5(
        "\n".join(f"{key} {value['md5']}" for key, value in streams.items()).encode()
    ).hexdigest()
    record["headline"] = headline(logs)
    record["stdout_agrees"] = stdout_agrees(run_dir / "run.log", record["headline"])
    shutil.rmtree(run_dir / "plots", ignore_errors=True)
    if not keep_instance:
        shutil.rmtree(logs / "instance", ignore_errors=True)
    return record


def first_divergence(reference: Path, current: Path) -> str:
    """Describe the first record where two stream files differ."""
    with reference.open() as left, current.open() as right:
        for row, (a, b) in enumerate(zip_longest(left, right)):
            if a == b:
                continue
            if a is None or b is None:
                return f"row {row}: one stream ends early"
            old, new = json.loads(a), json.loads(b)
            delta = {k: (old.get(k), new.get(k)) for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)}
            return f"row {row}: {str(delta)[:300]}"
    return "no divergence"


def map_runs(jobs: list[tuple], workers: int) -> list[dict]:
    def one(job: tuple) -> dict:
        record = run_config(*job)
        status = "FAILED" if record.get("failed") else record["digest"][:12]
        print(f"{record['name']}.{job[2]}: {status} {record['wall_s']}s", flush=True)
        return record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, jobs))


def build_table(args: argparse.Namespace) -> None:
    names = args.names or list(CONFIGS)
    root = Path(args.root).resolve()
    pin = Path(args.psrl_pin).resolve() if args.psrl_pin else None
    ensure_trace()
    engine_before = file_md5(engine_extension(root))
    runs = map_runs([(name, root, tag, pin, False) for name in names for tag in ("a", "b")], args.workers)
    table = {}
    for name in names:
        first, second = (r for r in runs if r["name"] == name)
        entry = dict(first)
        if first.get("failed") or second.get("failed"):
            entry["deterministic"] = False
            entry["failed"] = True
        else:
            entry["deterministic"] = first["digest"] == second["digest"]
            entry["unstable_streams"] = sorted(
                key for key in first["streams"] if first["streams"][key] != second["streams"].get(key)
            )
            entry["wall_s"] = round(statistics.mean((first["wall_s"], second["wall_s"])), 2)
        table[name] = entry
    engine_after = file_md5(engine_extension(root))
    by_digest: dict[str, list[str]] = {}
    for name, entry in table.items():
        by_digest.setdefault(entry.get("digest", name), []).append(name)
    payload = {
        "root": str(root),
        "engine_md5_before": engine_before,
        "engine_md5_after": engine_after,
        "identical_digest_groups": [sorted(group) for group in by_digest.values() if len(group) > 1],
        "configs": table,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"Wrote {args.out}")
    print(f"Identical digests across configs: {payload['identical_digest_groups']}")


def check(args: argparse.Namespace) -> None:
    reference = json.loads(Path(args.reference).read_text())["configs"]
    names = args.names or [n for n, e in reference.items() if args.include_deprecated or not e["deprecated"]]
    root = Path(args.root).resolve()
    pin = Path(args.psrl_pin).resolve() if args.psrl_pin else None
    runs = map_runs([(name, root, "check", pin, False) for name in names], args.workers)
    bad = 0
    for record in runs:
        ref = reference[record["name"]]
        if record.get("failed"):
            print(f"FAIL     {record['name']}: simulator exited {record['returncode']}")
            bad += 1
        elif record["digest"] != ref["digest"]:
            bad += 1
            print(f"MISMATCH {record['name']}: {ref['digest'][:12]} -> {record['digest'][:12]}")
            for stream, value in ref["streams"].items():
                if record["streams"].get(stream) == value:
                    continue
                before = Path(ref["telemetry_dir"]) / "logs" / f"{stream}.jsonl"
                after = Path(record["telemetry_dir"]) / "logs" / f"{stream}.jsonl"
                where = first_divergence(before, after) if before.is_file() and after.is_file() else "no telemetry"
                print(f"    {stream}: {where}")
        else:
            print(f"OK       {record['name']}: {record['digest'][:12]}")
    sys.exit(1 if bad else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    for command in ("run", "table", "check"):
        p = sub.add_parser(command)
        p.add_argument("--root", default=str(REPO), help="Repository root whose code is simulated.")
        p.add_argument("--psrl-pin", help="Directory prepended to PYTHONPATH to pin psrl's staleness controller.")
        p.add_argument("--workers", type=int, default=8)
        if command == "run":
            p.add_argument("name", choices=sorted(CONFIGS))
            p.add_argument("--out")
            p.add_argument("--tag", default="")
            p.add_argument("--keep-instance", action="store_true")
        else:
            p.add_argument("names", nargs="*")
        if command == "table":
            p.add_argument("--out", required=True)
        if command == "check":
            p.add_argument("--reference", required=True)
            p.add_argument("--include-deprecated", action="store_true")
    args = parser.parse_args()
    unknown = [name for name in getattr(args, "names", []) if name not in CONFIGS]
    if unknown:
        parser.error(f"Unknown configurations {unknown}. Known: {sorted(CONFIGS)}.")
    if args.command == "list":
        for name, cfg in CONFIGS.items():
            print(f"{name:<28}{cfg.backend:<8}{cfg.note}")
    elif args.command == "run":
        pin = Path(args.psrl_pin).resolve() if args.psrl_pin else None
        record = run_config(args.name, Path(args.root).resolve(), args.tag, pin, args.keep_instance)
        text = json.dumps(record, indent=1)
        if args.out:
            Path(args.out).write_text(text)
        print(text)
    elif args.command == "table":
        build_table(args)
    else:
        check(args)


if __name__ == "__main__":
    main()
