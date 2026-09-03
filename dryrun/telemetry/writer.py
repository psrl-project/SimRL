"""High-level, schema-owning telemetry writer for one simulation run.

`Simulator` should never format a JSONL record or pick a stream name itself;
it just calls the methods below with the values it already computed. This
class owns:

- the four stream schemas (`instance/*`, `rollout_step`, `train`, `request`),
- the small amount of bookkeeping needed to log *committed* state only (a
  de-dup key per instance, and a flush cursor into
  `NativeInstance.step_stats`), and
- the on/off switch: when disabled (`store=None` or `enabled=False`), every
  method below is a no-op and nothing touches disk.

See `dryrun.simulator.simulator.Simulator` for the call sites, and
`dryrun.telemetry.store.JsonlStore` for the underlying file format.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from .store import JsonlStore

if TYPE_CHECKING:
    from ..component.rollout.engine import NativeInstance, Request


class SimTelemetry:
    """Commit-only JSONL telemetry sink for a single `Simulator` run."""

    def __init__(self, store: JsonlStore | None, enabled: bool = True):
        self.store = store if enabled else None
        self._last_instance_key: dict[int, tuple] = {}
        self._stats_cursor: dict[int, int] = {}

    @property
    def enabled(self) -> bool:
        return self.store is not None

    # --- instance occupancy ("instance/{id}" stream) --------------------

    @staticmethod
    def _req_len_snapshot(reqs: Iterable[Request]) -> dict[str, dict[str, int]]:
        """`rid -> {generated, target_len, ctx, prefilled}` for one queue."""
        return {
            str(r.rid): {
                "generated": r.generated,
                "target_len": r.target_len,
                "ctx": r.ctx,
                "prefilled": r.prefilled,
            }
            for r in reqs
        }

    def emit_instance(self, inst: NativeInstance) -> None:
        """Log one occupancy sample for `inst`, skipping exact duplicates."""
        if self.store is None:
            return
        kv_used = inst.blocks_used()
        running_len = self._req_len_snapshot(inst.running)
        waiting_len = self._req_len_snapshot(inst.waiting)
        # Include per-req generated lengths so two samples with the same
        # occupancy counts but different progress are not collapsed.
        gen_fp = tuple(
            (r.rid, r.generated, r.prefilled) for r in (*inst.running, *inst.waiting)
        )
        key = (inst.t_local, len(inst.running), len(inst.waiting), kv_used, gen_fp)
        if self._last_instance_key.get(inst.instance_id) == key:
            return
        self._last_instance_key[inst.instance_id] = key
        self.store.write(
            f"instance/{inst.instance_id}",
            t=inst.t_local,
            instance_id=inst.instance_id,
            num_running_reqs=len(inst.running),
            num_waiting_reqs=len(inst.waiting),
            kv_cache_usage=kv_used / max(1, inst.kv_blocks),
            kv_blocks_used=kv_used,
            total_ctx=sum(r.ctx for r in inst.running),
            running_len=running_len,
            waiting_len=waiting_len,
        )

    def emit_instance_all(self, instances: Iterable[NativeInstance]) -> None:
        for inst in instances:
            self.emit_instance(inst)

    # --- rollout steps ("rollout_step" stream) --------------------------

    def flush_rollout_step(self, inst: NativeInstance) -> None:
        """Append any `inst.step_stats` entries committed since the last flush."""
        if self.store is None:
            return
        cursor = self._stats_cursor.get(inst.instance_id, 0)
        stats = inst.step_stats
        for stat in stats[cursor:]:
            self.store.write(
                "rollout_step",
                t=stat.t,
                instance_id=inst.instance_id,
                phase=stat.phase,
                steps=stat.steps,
                n_prefill=stat.n_prefill,
                n_decode=stat.n_decode,
                n_tokens=stat.n_tokens,
                dt=stat.dt,
                saturated=stat.saturated,
                saturated_steps=stat.saturated_steps,
            )
        self._stats_cursor[inst.instance_id] = len(stats)

    def sync(self, inst: NativeInstance) -> None:
        """Emit committed occupancy + step telemetry for one instance."""
        self.emit_instance(inst)
        self.flush_rollout_step(inst)

    def sync_all(self, instances: Iterable[NativeInstance]) -> None:
        for inst in instances:
            self.sync(inst)

    # --- request lifecycle ("request" stream) ---------------------------

    def write_request(self, event: str, t: float, req: Request) -> None:
        if self.store is None:
            return
        self.store.write(
            "request",
            t=t,
            event=event,
            rid=req.rid,
            group_id=req.group_id,
            instance_id=req.instance_id,
            admit_time=req.admit_time,
            dispatch_time=req.dispatch_time,
            complete_time=req.complete_time,
            consume_version=req.consume_version,
            prompt_len=req.prompt_len,
            target_len=req.target_len,
            generated=req.generated,
            ctx=req.ctx,
            v_traj=req.v_traj,
            v_min=req.v_min,
            v_max=req.v_max,
            staleness_true=(
                req.consume_version - req.v_min if req.consume_version is not None else None
            ),
            staleness_reported=(
                req.consume_version - req.v_max if req.consume_version is not None else None
            ),
            num_preemptions=req.num_preemptions,
            num_aborts=req.num_aborts,
        )

    # --- train steps ("train" stream) -----------------------------------

    def write_train(
        self,
        t: float,
        start: float,
        end: float,
        version: int,
        batch_size: int,
        n_reqs: int,
        total_tokens: int,
        train_time: float,
        peak_memory_bytes: int,
        training_tp: int,
        training_pp: int,
        training_dp: int,
        training_cp: int,
        micro_batch_size: int,
        schedule: str,
        activation_recompute: str,
        sync_time: float,
        recompute_time: float,
        n_inflight: int,
        n_ready: int,
        n_dropped_step: int,
        extrapolation_notes: list[str] | None = None,
    ) -> None:
        if self.store is None:
            return
        payload: dict = dict(
            t=t,
            start=start,
            end=end,
            dt=end - start,
            version=version,
            batch_size=batch_size,
            n_reqs=n_reqs,
            total_tokens=total_tokens,
            train_time=train_time,
            peak_memory_bytes=peak_memory_bytes,
            training_tp=training_tp,
            training_pp=training_pp,
            training_dp=training_dp,
            training_cp=training_cp,
            micro_batch_size=micro_batch_size,
            schedule=schedule,
            activation_recompute=activation_recompute,
            sync_time=sync_time,
            recompute_time=recompute_time,
            n_inflight=n_inflight,
            n_ready=n_ready,
            n_dropped_step=n_dropped_step,
        )
        if extrapolation_notes:
            payload["extrapolation_notes"] = extrapolation_notes
        self.store.write("train", **payload)

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
