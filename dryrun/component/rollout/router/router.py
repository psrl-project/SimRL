"""The router: where a request waits, and how it is placed.

The simulator stages work in the router and admits a whole group when its first
response reaches an instance. A request that cannot be placed dwells here
without consuming policy inventory. That is what makes router queue depth a
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
from typing import Callable

from ..engine import InstanceView, NativeInstance, Request, blocks_for, traj_view
from .request_queue import RouterQueue, SortKey
from .strategy import (
    RoutingContext,
    RoutingStrategy,
    allows_version_bypass,
    narrow_candidates,
    select_with_context,
)


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
    reject_preempted_on_waiting: bool = False
    """Apply the waiting-queue refusal only to preemption loopbacks."""

    def reject_reason(
        self,
        view: InstanceView,
        *,
        is_preempt_loopback: bool = False,
    ) -> str | None:
        """Why `view` cannot be offered work, or None if it can."""
        cap = self.max_concurrent_seqs_per_instance
        if cap > 0 and view.n_active >= cap:
            return "count_cap"
        reject_waiting = self.reject_on_waiting or (
            self.reject_preempted_on_waiting and is_preempt_loopback
        )
        if reject_waiting and view.n_waiting > 0:
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
        sort_key: SortKey = SortKey.SMALL_ID,
        enable_group_sticky: bool = False,
        max_waiting_after_preempt: int | None = None,
        cost_model=None,
        queue_window: int = 32,
    ):
        self.strategy = strategy
        self.instances = instances
        self.gate = gate or AdmissionGate()
        self.cost_model = cost_model
        """Shared rollout cost model, for a strategy that prices a hypothetical batch.

        Passed in rather than read off an instance because the C++ backend holds
        its own translated model and exposes no handle to it. Taking it from the
        caller keeps both backends pricing through the same Python object.
        """

        self.queue = RouterQueue(sort_key)
        self.queue_window = max(1, int(queue_window))
        """Queued requests a ranking strategy may choose among per placement.

        Bounds the cost of ranking, which is per placement and so would otherwise
        scale with a backlog that reaches several hundred after a weight update.
        """
        self.enable_group_sticky = enable_group_sticky
        self.max_waiting_after_preempt = max_waiting_after_preempt

        self._group_pin: dict[int, int] = {}
        """Group id -> instance id, written once at the group's first placement."""
        self._preempt_loopbacks: set[int] = set()
        """Object ids whose current router dwell came from preemption overflow."""

        self.n_routed = 0
        self.n_reroutes = 0
        self.n_cross_instance_reroutes = 0

    # --- queue side ------------------------------------------------------

    def __len__(self) -> int:
        return len(self.queue)

    @property
    def queue_len(self) -> int:
        return len(self.queue)

    @property
    def queue_depths(self) -> dict[int, int]:
        """Return queued requests grouped by trajectory version."""
        return self.queue.depths()

    def enqueue(self, reqs: list[Request], t: float) -> None:
        """Queue `reqs`, a whole prompt group when called from staging.

        A strategy may implement `observe_enqueue` to see the group at the one
        moment all its members are known to be queued together. Admission is
        per group but placement is per request, so a rule keyed on the group
        (its slowest member, say) has no other point at which to compute it.
        """
        observe = getattr(self.strategy, "observe_enqueue", None)
        if callable(observe):
            observe(reqs)
        for req in reqs:
            self.queue.push(req, t)

    def remove(self, reqs: list[Request]) -> list[Request]:
        """Drop `reqs` from the queue, returning those it held.

        A discarded request must leave, or a later dispatch would place work
        the simulator has already written off.
        """
        removed = self.queue.remove(reqs)
        self._preempt_loopbacks.difference_update(id(req) for req in removed)
        return removed

    def rekey(self, reqs: list[Request]) -> None:
        """Move queued requests to the layer matching their current version."""
        self.queue.rekey(reqs)

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
            tp=load.tp,
            free_blocks=inst.free_blocks(),
            committed_blocks=inst.committed_blocks(),
            prefix_hit_tokens=inst.peek_hit_tokens,
            step_time_delta=lambda req, i=inst: i.step_time_delta(req),
            # Waiting and running both, since `add_request` only queues and `committed_blocks` counts the same set.
            residents=lambda i=inst: [traj_view(r) for r in (*i.waiting, *i.running)],
            cost_model=self.cost_model,
            kv_blocks=inst.kv_blocks,
            block_size=inst.block_size,
        )

    def _candidates(
        self,
        req: Request,
        context: RoutingContext,
    ) -> list[tuple[int, InstanceView]]:
        """Eligible instances for `req`, as `(instance_id, view)` pairs.

        A group-sticky pin narrows the set but never empties it: SMG treats the
        pin as soft, falling through when the pinned instance cannot help, and
        a hard pin here would dwell a request behind one busy instance while
        others idle.
        """
        all_views = [(inst.instance_id, self._view(inst)) for inst in self.instances]
        is_preempt_loopback = id(req) in self._preempt_loopbacks
        views = [
            pair
            for pair in all_views
            if self.gate.reject_reason(
                pair[1],
                is_preempt_loopback=is_preempt_loopback,
            )
            is None
        ]
        allowed = narrow_candidates(
            self.strategy,
            req,
            [view for _, view in views],
            context,
        )
        views = [views[index] for index in allowed]
        if self.enable_group_sticky:
            pinned = self._group_pin.get(req.group_id)
            if pinned is not None:
                narrowed = [pair for pair in views if pair[0] == pinned]
                if narrowed:
                    return narrowed
        return views

    def _place(
        self,
        req: Request,
        t: float,
        context: RoutingContext,
    ) -> int | None:
        """Try to put `req` on an instance now. Returns the instance id or None."""
        candidates = self._candidates(req, context)
        if not candidates:
            return None
        index = select_with_context(
            self.strategy,
            req,
            [view for _, view in candidates],
            context,
        )
        if index is None:
            return None
        instance_id = candidates[index][0]
        previous = req.instance_id
        if not self.instances[instance_id].add_request(req, t):
            return None

        self._preempt_loopbacks.discard(id(req))
        if self.enable_group_sticky:
            self._group_pin.setdefault(req.group_id, instance_id)
        self.n_routed += 1
        if previous is not None:
            req.num_reroutes += 1
            self.n_reroutes += 1
            if previous != instance_id:
                self.n_cross_instance_reroutes += 1
        return instance_id

    def dispatch(
        self,
        t: float,
        *,
        can_admit: Callable[[Request], bool] | None = None,
        on_placed: Callable[[Request, int], None] | None = None,
    ) -> list[tuple[Request, int]]:
        """Place as many queued requests as the instances will take.

        Tries version queues from oldest to newest. After a blocked head, the
        strategy decides whether dispatch stops or continues to the next
        version. A successful placement restarts from the oldest version, so
        the router supplies ordering while the strategy owns bypass policy.

        Returns:
            `(request, instance_id)` for each placement, in dispatch order.
        """
        assert all(not inst.has_pending_event for inst in self.instances), (
            "Router dispatch ran inside a speculative engine event. Dispatch must "
            "only happen after an event commits, or the placement could be rolled "
            "back while the router still believes it happened."
        )
        begin = getattr(self.strategy, "begin_dispatch", None)
        if callable(begin):
            begin(t)
        placed: list[tuple[Request, int]] = []
        while self.queue:
            blocked_versions: list[int] = []
            placed_one = False
            for version in self.queue.versions():
                req = self._next_from_version(version, t)
                context = self._routing_context(blocked_versions, t)
                if req is None:
                    # A strategy that scans the whole layer found nothing any tier can take.
                    if not allows_version_bypass(self.strategy, None, context):
                        return placed
                    blocked_versions.append(version)
                    continue
                if can_admit is not None and not can_admit(req):
                    return placed
                instance_id = self._place(req, t, context)
                if instance_id is not None:
                    self.queue.remove_one(version, req)
                    placed.append((req, instance_id))
                    if on_placed is not None:
                        on_placed(req, instance_id)
                    placed_one = True
                    break
                if not allows_version_bypass(self.strategy, req, context):
                    return placed
                blocked_versions.append(version)
            if not placed_one:
                break
        return placed

    def _routing_context(self, blocked_versions: list[int], t: float) -> RoutingContext:
        """Build the strategy context for one version-queue attempt."""
        tps = [inst.tp for inst in self.instances]
        return RoutingContext(
            blocked_versions=tuple(blocked_versions),
            fleet_min_tp=min(tps),
            fleet_max_tp=max(tps),
            now=t,
        )

    def _next_from_version(self, version: int, t: float) -> Request | None:
        """Return the request this version should offer next.

        The heap key is fixed at push time, so a strategy ranking on anything
        that moves with the clock cannot be expressed by it. One that implements
        `rank_queued` is shown a window of heap-ordered candidates and picks
        within it. Everything else sees the head, unchanged.
        """
        pick = getattr(self.strategy, "pick_queued", None)
        if callable(pick):
            # The whole layer, so a blocked head cannot hide work another tier could take.
            window = self.queue.candidates_version(version, 1 << 30)
            return pick(window, [self._view(inst) for inst in self.instances])
        rank = getattr(self.strategy, "rank_queued", None)
        if not callable(rank):
            return self.queue.peek_version(version)
        window = self.queue.candidates_version(version, self.queue_window)
        if not window:
            return None
        chosen = rank(window, t)
        return chosen if chosen is not None else window[0]

    def has_dispatchable(self) -> bool:
        """Whether a dispatch pass could place anything right now.

        The engine loop breaks out to refill only when this or the policy's
        admit quota says there is something to do. Without it, a non-empty
        queue would sit untouched whenever the policy is saturated, even though
        an instance just freed capacity.
        """
        return bool(self.queue)

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
            self._preempt_loopbacks.add(id(req))
            self.queue.push(req, t)
        return reclaimed
