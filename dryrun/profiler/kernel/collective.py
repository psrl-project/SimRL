"""
Driver for the collective sweep: launches torchrun, locally or across nodes, and writes tables.

Five collectives are measured, not one. The original `all_reduce` covered rollout, where
vLLM issues an all-reduce after each attention and MLP block, but Megatron with sequence
parallelism on (which its profiler enables for every `tp > 1` run) issues reduce-scatter and
all-gather pairs instead, pipeline parallelism issues point-to-point sends, and the
data-parallel gradient reduction is a separate collective again.

Two tables come out of one sweep, because **which kernel a real server runs is not NCCL.**
`CudaCommunicator.all_reduce` dispatches through NCCL symmetric memory, then vLLM's custom
one-shot/two-shot P2P kernel, and only reaches `torch.distributed.all_reduce` as a last
resort. On H20 (sm_90) at two ranks the custom path covers everything under 8 MiB, while a
Qwen3-8B decode step's message is 8 KiB to 512 KiB, so every collective a `tp=2` server
issues is the custom kernel and none of them is NCCL. So the `vllm` table times the
dispatcher itself and the `nccl` table times `torch.distributed` as a control, letting the
bias of that choice be a number rather than an argument. `CommQuery.backend` selects between
them, defaulting to `vllm`.

Which backend actually fired is recorded per row in `kernel_source`, read off the live
communicator rather than re-derived from vLLM's size thresholds, which would drift silently
the next time vLLM retunes them.

Known gap. The custom kernel is timed outside a CUDA graph, where it pays a `cudaMemcpy`
that the server's graph replay skips (`custom_all_reduce.py:276-280` puts it at <=1% of
latency). The same trap in reverse cost the elementwise table 13.2x before it was timed as
a raw op.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..schema import ModelDescriptor
from .config import KernelProfileConfig
from .measurements import PerfTableWriter

dryrun_logger = logging.getLogger(__file__)

# A table whose rows disagree on which kernel produced them cannot be priced as one surface,
# so the writer takes a single `kernel_source`. Every measured world size lands here: the
# custom kernel covers the decode range and larger messages fall through to torch symmetric
# memory, so `mixed` is the normal outcome rather than a warning sign.
_MIXED = "mixed"

# Fabric settings are whatever the environment exported, never values this module chooses:
# they describe the cluster's RoCE configuration, and a sweep run under different ones would
# measure a fabric no real training job uses.
#
# Every NCCL_* variable is forwarded to remote nodes rather than a curated subset. A
# non-interactive ssh shell on this cluster inherits ~/.bashrc, which sets seven of them,
# while the project env script sets twenty-one, and the two disagree: the script orders
# NCCL_IB_HCA as bond_1,5,3,7,4,8,2,6 for rail alignment where .bashrc orders it 1..8, and
# the script additionally sets NCCL_ALGO and NCCL_LL_THRESHOLD. Both of those change the
# schedule NCCL picks. Letting each node resolve its own would mean the two ends of a
# cross-node ring ran different configurations, which is a measurement of nothing.
_FABRIC_PREFIX = "NCCL_"

# Logged explicitly because these are the ones that change what is measured rather than how
# it is reported, so they belong in the run record even though all NCCL_* vars are forwarded.
_FABRIC_LOGGED = (
    "NCCL_IB_HCA",
    "NCCL_IB_DISABLE",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_TC",
    "NCCL_IB_SL",
    "NCCL_SOCKET_IFNAME",
    "NCCL_ALGO",
    "NCCL_LL_THRESHOLD",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_PXN_DISABLE",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _log_fabric() -> dict[str, str]:
    """Return and log the inherited NCCL fabric settings, so rows trace to a configuration."""
    resolved = {name: os.environ.get(name, "") for name in _FABRIC_LOGGED}
    for name, value in resolved.items():
        dryrun_logger.info(f"Collective sweep inherits {name}={value!r}.")
    return resolved


def _fabric_exports() -> list[str]:
    """
    Return `NAME=value` assignments for every NCCL variable in this process's environment.

    Prepended to the remote command so both ends of a cross-node ring run the same fabric
    configuration. Without this each node resolves its own from `~/.bashrc`, which on this
    cluster disagrees with the project env script on `NCCL_IB_HCA`'s ordering and omits
    `NCCL_ALGO` and `NCCL_LL_THRESHOLD` entirely.
    """
    return [
        f"{name}={shlex.quote(value)}"
        for name, value in sorted(os.environ.items())
        if name.startswith(_FABRIC_PREFIX)
    ]


def _torchrun_command(
    *,
    nproc: int,
    nnodes: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
    spec_path: Path,
    output_path: Path,
) -> list[str]:
    """Build the torchrun argv that launches one topology's ranks.

    The child module is named as a **string**, which is why `collective_launch.py` stays at
    the package root rather than moving under `runner/` with the in-process measurement loops:
    no import graph and no grep for a symbol would follow it, so a move breaks this only at
    sweep time, on a cluster, after the rendezvous. `tests/test_kernel_collective_launch.py`
    reads the module through `inspect.getsource`, which does fail here in CI.
    """
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc-per-node={nproc}",
        f"--nnodes={nnodes}",
        f"--node-rank={node_rank}",
        f"--master-addr={master_addr}",
        f"--master-port={master_port}",
        "-m",
        "dryrun.profiler.kernel.collective_launch",
        "--spec",
        str(spec_path),
        "--output",
        str(output_path),
    ]


def _run_topology(
    *,
    world_size: int,
    nodes: int,
    hostnames: tuple[str, ...],
    spec: dict[str, Any],
    work_dir: Path,
) -> list[dict[str, Any]]:
    """
    Run one `(world_size, nodes)` point and return rank 0's rows.

    Args:
        world_size (int): Total ranks.
        nodes (int): Nodes the ranks are spread across.
        hostnames (tuple[str, ...]): Hosts, rank-0 host first. Unused at `nodes == 1`.
        spec (dict[str, Any]): Measurement spec handed to the child.
        work_dir (Path): Directory for the spec and result files. Must be visible to every
            node, which on this cluster means the shared filesystem rather than /tmp.

    Returns:
        list[dict[str, Any]]: Raw measurement rows.

    Raises:
        RuntimeError: If any node exits non-zero, or if the result file is absent. A
            partially-failed multi-node launch must not yield a table.
    """
    nproc = world_size // nodes
    spec_path = work_dir / f"spec_ws{world_size}_n{nodes}.json"
    output_path = work_dir / f"rows_ws{world_size}_n{nodes}.json"
    spec_path.write_text(json.dumps({**spec, "nodes": nodes}, indent=2), encoding="utf-8")
    master_port = _free_port()

    if nodes == 1:
        command = _torchrun_command(
            nproc=nproc,
            nnodes=1,
            node_rank=0,
            master_addr="127.0.0.1",
            master_port=master_port,
            spec_path=spec_path,
            output_path=output_path,
        )
        dryrun_logger.info(f"Launching world_size {world_size} on one node: {' '.join(command)}.")
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Collective sweep at world_size {world_size} on one node exited "
                f"{completed.returncode}."
            )
    else:
        if os.environ.get("NCCL_IB_DISABLE") == "1":
            # Without IB the ranks fall back to TCP sockets, and the resulting rows would
            # look like a slow NIC rather than an absent one: a wrong number that still
            # interpolates smoothly is worse than a failure.
            raise RuntimeError(
                "NCCL_IB_DISABLE=1 with a cross-node topology. The sweep would measure TCP "
                "sockets and label the rows as fabric traffic. Unset it and re-run."
            )
        master_addr = hostnames[0]
        processes = []
        for node_rank, host in enumerate(hostnames[:nodes]):
            command = _torchrun_command(
                nproc=nproc,
                nnodes=nodes,
                node_rank=node_rank,
                master_addr=master_addr,
                master_port=master_port,
                spec_path=spec_path,
                output_path=output_path,
            )
            # `env` rather than a login shell: sourcing a profile would let the remote
            # node's own NCCL settings win, which is the thing being prevented. The
            # interpreter is addressed by absolute path (sys.executable on the shared
            # filesystem), so no PATH setup is needed and none is done.
            remote = " ".join(
                ("cd", shlex.quote(str(Path.cwd())), "&&", "env", *_fabric_exports(), *command)
            )
            ssh_command = [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
                host,
                remote,
            ]
            dryrun_logger.info(
                f"Launching node_rank {node_rank} of {nodes} on {host} for world_size "
                f"{world_size}."
            )
            processes.append((host, subprocess.Popen(ssh_command)))
        failures = [(host, proc.wait()) for host, proc in processes]
        bad = [(host, code) for host, code in failures if code != 0]
        if bad:
            raise RuntimeError(
                f"Collective sweep at world_size {world_size} across {nodes} nodes had "
                f"failing nodes: {bad}. A partial launch cannot produce a usable table."
            )

    if not output_path.is_file():
        raise RuntimeError(
            f"Collective sweep at world_size {world_size} across {nodes} nodes produced no "
            f"result file at {output_path}. Rank 0 did not complete."
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _rows_for(raw: list[dict[str, Any]], op: str, latency_key: str) -> list[dict[str, Any]]:
    """Project raw measurements into one op's table columns, dropping unmeasured points."""
    return [
        {
            "dtype": row["dtype"],
            "world_size": int(row["world_size"]),
            "nodes": int(row["nodes"]),
            "message_bytes": int(row["message_bytes"]),
            "latency_s": float(row[latency_key]),
        }
        for row in raw
        if row["op"] == op and row.get(latency_key) is not None
    ]


def _kernel_source(raw: list[dict[str, Any]], op: str) -> str:
    """Return the single dispatched backend for one op, or `mixed` when a sweep spans several."""
    labels = {
        row["dispatched_backend"]
        for row in raw
        if row["op"] == op and row.get("dispatched_backend")
    }
    if len(labels) == 1:
        return labels.pop()
    if labels:
        dryrun_logger.warning(
            f"The {op} sweep spans more than one backend ({sorted(labels)}), so kernel_source "
            f"is {_MIXED!r}. The per-size dispatch is in the rows."
        )
        return _MIXED
    return "unknown"


def profile_collective(
    model: ModelDescriptor,
    config: KernelProfileConfig,
    *,
    writer_factory: Callable[[str, str, str], PerfTableWriter],
) -> tuple[Path, ...]:
    """
    Time every registered collective at every swept topology and write the perf tables.

    Called once for the whole family. Process-group setup and the NCCL handshake dominate a
    small-message measurement, and a cross-node rendezvous costs seconds, so all five ops
    are timed inside one group per topology. That also guarantees every row describes the
    same communicator.

    Args:
        model (ModelDescriptor): Unused; message sizes are swept directly rather than
            derived from the architecture, because the five collectives do not carry the
            same size at the same token count. Kept for the `KernelOp.measure` signature.
        config (KernelProfileConfig): `collective.topologies`, `collective.message_bytes`,
            `collective.dtypes` and `collective.hostnames` define the sweep.
        writer_factory (Callable[[str, str, str], PerfTableWriter]): Builds a writer from
            `(framework, version, kernel_source)`.

    Returns:
        tuple[Path, ...]: The written parquet paths, one per `(op, framework)` with data.

    Raises:
        RuntimeError: If a topology's launch fails, or if `ssh` is unavailable for a
            cross-node topology.
    """
    import torch

    from ...cost.kernel.ops.base import ops_for_family

    sweep = config.sweep("all_reduce")
    ops = tuple(op.name for op in ops_for_family("collective"))
    _log_fabric()

    needs_ssh = any(nodes > 1 for _, nodes in sweep.topologies)
    if needs_ssh and shutil.which("ssh") is None:
        raise RuntimeError(
            "A cross-node topology is configured but ssh is not on PATH, so the remote "
            "ranks cannot be launched."
        )

    spec = {
        "ops": list(ops),
        "dtypes": list(sweep.dtypes),
        "message_bytes": list(sweep.message_bytes),
        "warmup_iterations": config.measurement.warmup_iterations,
        "active_iterations": config.measurement.active_iterations,
    }

    raw: list[dict[str, Any]] = []
    # The spec and result files must be readable from every node, so they go under the
    # output directory on the shared filesystem rather than a local /tmp that a remote
    # torchrun would not find.
    work_root = Path(config.output_dir) / "collective_runs"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_root) as work_name:
        work_dir = Path(work_name)
        for world_size, nodes in sweep.topologies:
            raw.extend(
                _run_topology(
                    world_size=world_size,
                    nodes=nodes,
                    hostnames=sweep.hostnames,
                    spec=spec,
                    work_dir=work_dir,
                )
            )

    written: list[Path] = []
    for op in ops:
        vllm_rows = _rows_for(raw, op, "vllm_latency_s")
        if vllm_rows:
            # Imported here rather than at the top of the loop: a cross-node-only sweep
            # produces no dispatched rows at all, because vLLM refuses a group spanning
            # nodes, and paying its import to write nothing is waste.
            import vllm

            writer = writer_factory("vllm", str(vllm.__version__), _kernel_source(raw, op))
            written.append(writer.write(op, vllm_rows))
        nccl_rows = _rows_for(raw, op, "nccl_latency_s")
        if nccl_rows:
            writer = writer_factory(
                "nccl", str(torch.version.cuda or "unknown"), f"torch.distributed.{op}"
            )
            written.append(writer.write(op, nccl_rows))
        if not vllm_rows:
            # Two distinct reasons, both stated rather than silently skipped. Within a node,
            # not every collective has a server-level entry point: vLLM's pipeline send/recv
            # uses a separate group with its own buffers, so a `vllm` row for p2p would
            # misattribute the kernel. Across nodes, vLLM disables its custom all-reduce and
            # symmetric memory outright, so no op has a dispatched path.
            dryrun_logger.info(
                f"No vllm-dispatched rows for {op}; only the nccl control table is written."
            )
    return tuple(written)
