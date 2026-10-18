"""AReaL token bucket staleness policy."""

from __future__ import annotations

from ..component.rollout.engine import Request
from .base import CompleteAction, PolicyContext, SimState, StalenessPolicy
from .rules import FIFO, ProgressQuota


class ArealPolicy(StalenessPolicy):
    """
    AReaL's cumulative token bucket admission.

    Bounds total unconsumed work but NOT individual sample age.
    The bucket is indexed by `train_version`, which the simulator advances
    when a training step is published.
    """

    name = "areal"

    def __init__(self, max_inflight: int | None = None):
        self.max_inflight = max_inflight
        """Cap on concurrently generating prompt groups, or None."""
        self.accepted = 0
        self.running = 0

    def admit_quota(self, st: SimState) -> int:
        # The bucket counts prompt groups, since a group is what training consumes.
        ctx = PolicyContext(state=st, accepted=self.accepted, running=self.running)
        staleness_capacity = ProgressQuota(ofp=st.max_staleness).quota(ctx)
        if self.max_inflight is not None:
            staleness_capacity = min(staleness_capacity, self.max_inflight - self.running)
        return max(0, staleness_capacity) * st.group_size

    def on_admit(self, group: list[Request], st: SimState) -> None:
        self.running += 1

    def on_group_complete(self, group: list[Request], st: SimState) -> CompleteAction:
        self.running -= 1
        self.accepted += 1
        return CompleteAction.KEEP

    def peek_batch(self, st: SimState) -> list[Request] | None:
        return FIFO().select(st.ready_groups, st.batch_size)

    def check_invariants(self, st: SimState) -> None:
        unconsumed = st.n_admitted - st.n_consumed
        cap = (st.max_staleness + 1) * st.responses_per_batch
        assert unconsumed <= cap, f"Token bucket violated: {unconsumed} > {cap}."
