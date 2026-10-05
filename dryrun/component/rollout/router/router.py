"""The router: where a request waits, and how it is placed.

The simulator admits into the router rather than onto an instance, so a request
that no instance can take dwells here instead of being rolled back. That is
what the real gateway does, and it is what makes the router queue depth a
signal rather than a constant zero.

**When dispatch runs.** A dispatch attempt can only succeed if KV capacity was
released, because `NativeInstance.add_request`'s binding gate is a block count
and nothing else in the attempt varies with time. Capacity is released in
exactly three places, all of them already event boundaries: a request retiring
(`REQUEST_COMPLETE`), a preemption freeing its victim, and a weight update
aborting everything. So `dispatch` is called after those commit, and nowhere
else. A decode segment only *grows* blocks, so it triggers nothing -- which
matters, because one closed-form segment covers a median 122 engine steps.

This is not an approximation of the real gateway's polling loop. A segment ends
at the first completion or KV exhaustion across the batch, which is precisely
the first instant capacity can change, so the trigger set is exact given the
engine's event granularity. There is deliberately no `check_interval_ms`
analogue.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..engine import InstanceView, NativeInstance, Request
from .request_queue import RouterQueue, SortKey
from .strategy import RoutingStrategy


@dataclass
class AdmissionGate:
    """Which instances may be offered a request at all.

    Mirrors SMG's `admission_reject_reason`. This decides *eligibility* only.
    `NativeInstance.add_request` stays the authority on whether the request
    actually fits, and is not duplicated here: two places deciding admission is
    how the two drift apart.
    """

    max_concurrent_seqs_per_instance: int = 0
    """Cap on requests one instance may hold. 0 disables the limb."""
    reject_on_waiting: bool = False
    """Refuse an instance that already has a queue of its own."""

    def reject_reason(self, view: InstanceView) -> str | None:
        """Why `view` cannot be offered work, or None if it can."""
        cap = self.max_concurrent_seqs_per_instance
        if cap > 0 and view.n_active >= cap:
            return "count_cap"
        if self.reject_on_waiting and view.n_waiting > 0:
            return "waiting_queue"
        return None


class Router:
    """Holds unplaced requests and decides where each one goes.

    Usage:
        router = Router(strategy, instances)
        router.enqueue(reqs, t)
        placed = router.dispatch(t)     # after a committed capacity release
    """

    def __init__(
        self,
        strategy: RoutingStrategy,
        instances: list[NativeInstance],
        gate: AdmissionGate | None = None,
        sort_key: SortKey = SortKey.SHORT_LENGTH,
        enable_group_sticky: bool = False,
        max_waiting_after_preempt: int | None = None,
    ):
        self.strategy = strategy
        self.instances = instances
        self.gate = gate or AdmissionGate()
        self.queue = RouterQueue(sort_key)
        self.enable_group_sticky = enable_group_sticky
        self.max_waiting_after_preempt = max_waiting_after_preempt

        self._group_pin: dict[int, int] = {}
        """Group id -> instance id, written once at the group's first placement."""

        self.n_routed = 0
        self.n_reroutes = 0
        self.n_cross_instance_reroutes = 0

    # --- queue side ------------------------------------------------------

    def __len__(self) -> int:
        return len(self.queue)

    @property
    def queue_len(self) -> int:
        return len(self.queue)

    def enqueue(self, reqs: list[Request], t: float) -> None:
        for req in reqs:
            self.queue.push(req, t)

    def remove(self, reqs: list[Request]) -> list[Request]:
        """Drop `reqs` from the queue, returning those it held.

        A discarded request must leave, or a later dispatch would place work
        the simulator has already written off.
        """
        return self.queue.remove(reqs)

    def release_group_pins(self, group_ids: set[int]) -> None:
        """Forget the sticky pin for finished or dropped groups.

        Without this the map grows for the whole run and a repeating workload
        would be pinned by a group that no longer exists.
        """
        for group_id in group_ids:
            self._group_pin.pop(group_id, None)

    # --- dispatch --------------------------------------------------------

    def _view(self, inst: NativeInstance) -> InstanceView:
        load = inst.get_load()
        return InstanceView(
            instance_id=load.instance_id,
            n_waiting=load.n_waiting,
            n_running=load.n_running,
            kv_utilization=load.kv_utilization,
            total_ctx=load.total_ctx,
            free_blocks=inst.free_blocks(),
            committed_blocks=inst.committed_blocks(),
            prefix_hit_tokens=inst.kv.peek_hit_tokens,
            step_time_delta=lambda req, i=inst: i.step_time_delta(req),
        )

    def _candidates(self, req: Request) -> list[tuple[int, InstanceView]]:
        """Eligible instances for `req`, as `(instance_id, view)` pairs.

        A group-sticky pin narrows the set but never empties it: SMG treats the
        pin as soft, falling through when the pinned instance cannot help, and
        a hard pin here would dwell a request behind one busy instance while
        others idle.
        """
        views = [
            (inst.instance_id, self._view(inst))
            for inst in self.instances
            if self.gate.reject_reason(self._view(inst)) is None
        ]
        if self.enable_group_sticky:
            pinned = self._group_pin.get(req.group_id)
            if pinned is not None:
                narrowed = [pair for pair in views if pair[0] == pinned]
                if narrowed:
                    return narrowed
        return views

    def _place(self, req: Request, t: float) -> int | None:
        """Try to put `req` on an instance now. Returns the instance id or None."""
        candidates = self._candidates(req)
        if not candidates:
            return None
        index = self.strategy.select(req, [view for _, view in candidates])
        if index is None:
            return None
        instance_id = candidates[index][0]
        previous = req.instance_id
        if not self.instances[instance_id].add_request(req, t):
            return None

        if self.enable_group_sticky:
            self._group_pin.setdefault(req.group_id, instance_id)
        self.n_routed += 1
        if previous is not None:
            req.num_reroutes += 1
            self.n_reroutes += 1
            if previous != instance_id:
                self.n_cross_instance_reroutes += 1
        return instance_id

    def dispatch(self, t: float) -> list[tuple[Request, int]]:
        """Place as many queued requests as the instances will take.

        Walks the queue in priority order and stops at the first request
        nothing can take, rather than skipping it to try the next. That is
        SMG's peek-then-pop shape, and it is what stops a long request being
        starved by a stream of short ones that keep fitting into the gap it
        needs.

        Returns:
            `(request, instance_id)` for each placement, in dispatch order.
        """
        assert all(inst._snapshot is None for inst in self.instances), (
            "Router dispatch ran inside a speculative engine event. Dispatch must "
            "only happen after an event commits, or the placement could be rolled "
            "back while the router still believes it happened."
        )
        placed: list[tuple[Request, int]] = []
        while self.queue:
            req = self.queue.peek()
            instance_id = self._place(req, t)
            if instance_id is None:
                break
            self.queue.pop()
            placed.append((req, instance_id))
        return placed

    def has_dispatchable(self) -> bool:
        """Whether a dispatch pass could place anything right now.

        The engine loop breaks out to refill only when this or the policy's
        admit quota says there is something to do. Without it, a non-empty
        queue would sit untouched whenever the policy is saturated, even though
        an instance just freed capacity.
        """
        return bool(self.queue)

    # --- loopback --------------------------------------------------------

    def reclaim_preempted(self, t: float) -> list[Request]:
        """Take back waiters from instances whose queue has run too deep.

        This is vLLM's `preemption_notification_threshold`
        (`rollout_scheduler.py:289`): a preempted request is requeued locally
        first, and the engine only hands it to the gateway once the local queue
        is already congested, because local requeueing would make a congested
        instance worse.

        Only preempted requests are reclaimed, never ones mid-prefill: those
        hold blocks without decoding, and moving them would throw away prefill
        the instance has already paid for.
        """
        if self.max_waiting_after_preempt is None:
            return []
        reclaimed: list[Request] = []
        for inst in self.instances:
            overflow = len(inst.waiting) - self.max_waiting_after_preempt
            if overflow <= 0:
                continue
            victims = [
                req for req in reversed(inst.waiting)
                if req.num_preemptions > 0 and req.prefilled == 0
            ][:overflow]
            if victims:
                reclaimed.extend(inst.cancel(victims, keep_tokens=True, requeue=False))
        for req in reclaimed:
            self.queue.push(req, t)
        return reclaimed
