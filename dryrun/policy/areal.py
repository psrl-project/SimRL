"""AReaL token bucket staleness policy."""

from __future__ import annotations

from ..component.rollout.engine import Request
from .base import CompleteAction, SimState, StalenessPolicy


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
        self.accepted = 0
        self.running = 0

    def admit_quota(self, st: SimState) -> int:
        ofp, B = st.max_staleness, st.batch_size
        staleness_capacity = (ofp + st.train_version + 1) * B - (self.accepted + self.running)
        if self.max_inflight is not None:
            staleness_capacity = min(staleness_capacity, self.max_inflight - self.running)
        return max(0, staleness_capacity)

    def on_admit(self, req: Request, st: SimState) -> None:
        self.running += 1

    def on_complete(self, req: Request, st: SimState) -> CompleteAction:
        self.running -= 1
        self.accepted += 1
        return CompleteAction.KEEP

    def peek_batch(self, st: SimState) -> list[Request] | None:
        if len(st.ready) < st.batch_size:
            return None
        ordered = sorted(st.ready, key=lambda r: (r.dispatch_time, r.rid))
        return ordered[:st.batch_size]

    def check_invariants(self, st: SimState) -> None:
        unconsumed = st.n_admitted - st.n_consumed
        cap = (st.max_staleness + 1) * st.batch_size
        assert unconsumed <= cap, f"Token bucket violated: {unconsumed} > {cap}."
