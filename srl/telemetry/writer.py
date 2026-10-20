"""High-level, schema-owning telemetry writer for one simulation run.

`Simulator` should never format a JSONL record or pick a stream name itself.
It just calls the methods below with the values it already computed. This
class owns:

- the four stream schemas (`instance/*`, `rollout_step`, `train`, `request`),
- the small amount of bookkeeping needed to log *committed* state only (a
  de-dup key per instance, and a flush cursor into
  `NativeInstance.step_stats`), and
- the on/off switch: when disabled (`store=None` or `enabled=False`), every
  method below is a no-op and nothing touches disk.

See `srl.simulator.simulator.Simulator` for the call sites, and
`srl.telemetry.store.JsonlStore` for the underlying file format.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from .store import JsonlStore

if TYPE_CHECKING:
    from ..component.rollout.engine import NativeInstance, Request
    from ..component.rollout.router import Router


def _n_running(inst: "NativeInstance") -> int:
    """
    Count running requests without converting the engine's container.

    On the C++ backend `inst.running` builds a whole Python list per read, so its size
    query `n_running` is used instead. A native instance has no such property.
    """
    count = getattr(inst, "n_running", None)
    return len(inst.running) if count is None else int(count)


def _n_waiting(inst: "NativeInstance") -> int:
    """Count waiting requests without converting the engine's container."""
    count = getattr(inst, "n_waiting", None)
    return len(inst.waiting) if count is None else int(count)


class SimTelemetry:
    """Commit-only JSONL telemetry sink for a single `Simulator` run."""

    def __init__(self, store: JsonlStore | None, enabled: bool = True):
        self.store = store if enabled else None
        self._last_instance_key: dict[int, tuple] = {}
        self._stats_cursor: dict[int, int] = {}
        self._last_gateway_key: tuple | None = None

    @property
    def enabled(self) -> bool:
        return self.store is not None

    # --- instance occupancy ("instance/{id}" stream) --------------------

    @staticmethod
    def _req_len_snapshot(reqs: Iterable[Request]) -> dict[str, dict[str, int]]:
        """`rid -> {generated, target_len, ctx, prefilled, num_aborts}` for one queue.

        `num_aborts` is here because `generated` alone can no longer identify an
        interrupted request: a weight update discards the generated tokens, so a
        just-aborted waiter looks exactly like a freshly admitted one.
        """
        return {
            str(r.rid): {
                "generated": r.generated,
                "target_len": r.target_len,
                "ctx": r.ctx,
                "prefilled": r.prefilled,
                "num_aborts": r.num_aborts,
            }
            for r in reqs
        }

    def _suppressed(self, stream: str) -> bool:
        """
        Return whether `stream` is switched off, so building its record is waste.

        Callers check this before assembling a record, not only before writing it. On the
        C++ backend the record's inputs are engine vectors that pybind converts to a new
        list on every read, so a suppressed stream that still read them would pay O(n) per
        sync for output nobody keeps.
        """
        return self.store is None or self.store.excluded(stream)

    def emit_instance(self, inst: NativeInstance) -> None:
        """Log one occupancy sample for `inst`, skipping exact duplicates."""
        if self._suppressed(f"instance/{inst.instance_id}"):
            return
        kv_used = inst.blocks_used()
        # Read each queue once, since every read converts a whole vector on the C++ backend.
        running = inst.running
        waiting = inst.waiting
        running_len = self._req_len_snapshot(running)
        waiting_len = self._req_len_snapshot(waiting)
        # Include per-req generated lengths so two samples with the same
        # occupancy counts but different progress are not collapsed.
        gen_fp = tuple((r.rid, r.generated, r.prefilled) for r in (*running, *waiting))
        key = (inst.t_local, len(running), len(waiting), kv_used, gen_fp)
        if self._last_instance_key.get(inst.instance_id) == key:
            return
        self._last_instance_key[inst.instance_id] = key
        self.store.write(
            f"instance/{inst.instance_id}",
            t=inst.t_local,
            instance_id=inst.instance_id,
            # Per sample rather than once, because `DynamicParallelFleet` may
            # re-shape the fleet at a weight update.
            tp=inst.tp,
            num_running_reqs=len(running),
            num_waiting_reqs=len(waiting),
            kv_cache_usage=kv_used / max(1, inst.kv_blocks),
            kv_blocks_used=kv_used,
            total_ctx=sum(r.ctx for r in running),
            running_len=running_len,
            waiting_len=waiting_len,
        )

    def emit_instance_all(self, instances: Iterable[NativeInstance]) -> None:
        for inst in instances:
            self.emit_instance(inst)

    # --- router queue ("gateway" stream) --------------------------------

    def emit_gateway(
        self,
        t: float,
        router: "Router",
        instances: Iterable[NativeInstance],
    ) -> None:
        """Log one router queue-depth sample, skipping exact duplicates.

        Deliberately the *same* schema the real-run converter writes from
        `smg_routing_status.jsonl` (`telemetry/psrl_log.py`), so one
        `plot_instance_load` draws both and `srl-compare` puts like against
        like. A separate `router` stream would fork the figure in two.

        `running_request_num` counts what the instances hold, matching the
        gateway's own meaning of requests committed to an instance and not yet
        finished, rather than the queue depth repeated.
        """
        if self._suppressed("gateway"):
            return
        queue_len = router.queue_len
        queue_by_version = router.queue_depths
        running = sum(_n_running(inst) + _n_waiting(inst) for inst in instances)
        key = (t, queue_len, tuple(queue_by_version.items()), running)
        if self._last_gateway_key == key:
            return
        self._last_gateway_key = key
        self.store.write(
            "gateway",
            t=t,
            queue_len=queue_len,
            pending_request_num=queue_len,
            running_request_num=running,
            queue_by_version={
                str(version): depth for version, depth in queue_by_version.items()
            },
            n_routed=router.n_routed,
            n_reroutes=router.n_reroutes,
            n_cross_instance_reroutes=router.n_cross_instance_reroutes,
        )

    # --- rollout steps ("rollout_step" stream) --------------------------

    def flush_rollout_step(self, inst: NativeInstance, upto: int | None = None) -> None:
        """
        Append any `inst.step_stats` entries committed since the last flush.

        Args:
            inst (NativeInstance): The instance to flush.
            upto (int | None): Flush only rows before this index. After a batch of committed
                events, the trailing row belongs to a still speculative event that may yet
                be undone.
        """
        if self._suppressed("rollout_step"):
            return
        cursor = self._stats_cursor.get(inst.instance_id, 0)
        # Fetch only the new rows. Reading all of `step_stats` on every sync is O(steps^2)
        # on the C++ backend, which converts the whole vector per read.
        since = getattr(inst, "step_stats_since", None)
        if since is not None:
            new_stats = since(cursor)
            total = inst.n_step_stats
        else:
            stats = inst.step_stats
            new_stats = stats[cursor:]
            total = len(stats)
        if upto is not None and upto < total:
            new_stats = new_stats[: max(0, upto - cursor)]
            total = max(cursor, upto)
        for stat in new_stats:
            self.store.write(
                "rollout_step",
                t=stat.t,
                instance_id=inst.instance_id,
                phase=stat.phase,
                steps=stat.steps,
                n_prefill=stat.n_prefill,
                n_prefill_hit=stat.n_prefill_hit,
                n_prefill_computed=stat.n_prefill_computed,
                n_decode=stat.n_decode,
                n_tokens=stat.n_tokens,
                dt=stat.dt,
                saturated=stat.saturated,
                saturated_steps=stat.saturated_steps,
            )
        self._stats_cursor[inst.instance_id] = total

    def sync(self, inst: NativeInstance, stats_upto: int | None = None) -> None:
        """Emit committed occupancy + step telemetry for one instance."""
        self.emit_instance(inst)
        self.flush_rollout_step(inst, upto=stats_upto)

    def instance_stream_suppressed(self, instance_id: int) -> bool:
        """Return whether per-instance occupancy samples are switched off for `instance_id`."""
        return self._suppressed(f"instance/{instance_id}")

    def sync_all(self, instances: Iterable[NativeInstance]) -> None:
        for inst in instances:
            self.sync(inst)

    # --- request lifecycle ("request" stream) ---------------------------

    def write_request(
        self,
        event: str,
        t: float,
        req: Request,
        drop_reason: str | None = None,
        retry: bool = False,
    ) -> None:
        if self.store is None:
            return
        self.store.write(
            "request",
            t=t,
            event=event,
            rid=req.rid,
            group_id=req.group_id,
            group_index=req.group_index,
            source_group_id=req.source_group_id,
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
            num_reroutes=req.num_reroutes,
            # Both were already maintained on Request but never written, which
            # left the cost of a loopback inferable but not measurable.
            reprefill_tokens=req.reprefill_tokens,
            prefix_hit_tokens=req.prefix_hit_tokens,
            drop_reason=drop_reason,
            retry=retry,
        )

    # --- train steps ("train" stream) -----------------------------------

    def write_train(
        self,
        t: float,
        start: float,
        end: float,
        version: int,
        n_groups: int,
        group_size: int,
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
        n_pending: int,
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
            n_groups=n_groups,
            group_size=group_size,
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
            n_pending=n_pending,
            n_dropped_step=n_dropped_step,
        )
        if extrapolation_notes:
            payload["extrapolation_notes"] = extrapolation_notes
        self.store.write("train", **payload)

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
