"""Route and admit by forecasting the nearest unready training buffer's finish time.

Every decision -- place an admitted group's queued member, or admit a staged group -- is priced
as a whole `Pipeline`: one predicted training-finish time `E_k` per pending buffer. Options
compare lexicographically, buffer by buffer, with `compare`'s tolerance, so a decision that
barely moves the nearest buffer looks past it to the next one. This is what makes the admitted
cap emergent rather than a configured constant: an option that would delay a critical buffer
loses to DEFER, and one that would not is free to go wherever `PlanSession` finds it feasible.

`PlanSession` makes one dispatch burst incremental: `begin_dispatch` builds one session from the
live `SchedulingRuntime`, and every placement this strategy commits updates it in place so the
next decision in the same burst sees the last one's effect, without replaying the whole fleet.

Both decisions go longest-predicted-first. Among an admitted group's queued members, that is
`PlanSession`'s own tail-fold order (reserved deadline buffer, then longest remaining); among
staged groups not yet admitted, it is the group's own slowest predicted member, since that
straggler is what the admission decision is actually about -- but only within a `window_groups`
block of arrival order (same convention as `length_aware`'s knob of the same name). A deadline
is real urgency and is never window-bounded; a length-only tiebreak ranking the whole backlog
would starve whichever end of the length distribution it ranks last, exactly the failure mode
`length_aware`'s own docstring warns against.
"""

from __future__ import annotations

from .....forecast import PlanSession, build_length_predictor, compare
from ...engine import InstanceView, Request, traj_view
from ..strategy import register_strategy


@register_strategy("inventory_deadline")
class InventoryDeadline:
    """Admit and route by minimizing the forecast delay to every pending training buffer."""

    name = "inventory_deadline"

    def __init__(
        self,
        length_model: str = "oracle",
        length_model_args: dict | None = None,
        horizon_buffers: int = 2,
        eps_frac: float = 0.01,
        defer: bool = True,
        window_groups: int = 32,
        kv_watermark: float = 0.0,
    ) -> None:
        """
        Args:
            length_model (str): Name in `LENGTH_PREDICTORS`.
            length_model_args (dict | None): Constructor keywords for that predictor.
            horizon_buffers (int): Pending buffers each decision's `Pipeline` projects.
            eps_frac (float): Fraction of the priced training cycle two options must differ by
                before one is preferred over the other, on each buffer `compare` checks.
            defer (bool): Whether DEFER is an option at all. False forces every feasible request
                onto the best-scoring instance, the ablation that isolates what deferral buys.
            window_groups (int): Staged groups per reordering block, in arrival order (same
                convention as `length_aware`'s knob of the same name). Zero reorders admission
                priority across the whole backlog, which starves whichever end of the length
                distribution the slowest-first rule ranks last.
            kv_watermark (float): Fraction of an instance's forecast KV peak a staged group's
                admission may reach before that instance is refused as an admission candidate,
                same convention as `length_aware`'s knob of the same name. Zero disables the check,
                leaving literal `kv_overflow` as the only admission-feasibility bar -- this is the
                design's own "admission candidates = instances whose forecast future KV peak fits"
                limb, applied only to admitting a new group, never to routing an already-admitted
                member (which already holds a buffer reservation and must still go somewhere).
        """
        self.predictor = build_length_predictor(length_model, **(length_model_args or {}))
        if horizon_buffers < 1:
            raise ValueError(f"horizon_buffers must be >= 1, got {horizon_buffers!r}.")
        self.horizon_buffers = int(horizon_buffers)
        if eps_frac < 0.0:
            raise ValueError(f"eps_frac must be >= 0, got {eps_frac!r}.")
        self.eps_frac = float(eps_frac)
        self.defer = bool(defer)
        if window_groups < 0:
            raise ValueError(f"window_groups must be >= 0, got {window_groups!r}.")
        self.window_groups = int(window_groups)
        if not 0.0 <= kv_watermark <= 1.0:
            raise ValueError(f"kv_watermark must be in [0, 1], got {kv_watermark!r}.")
        self.kv_watermark = float(kv_watermark)
        self._runtime = None
        self._session: PlanSession | None = None
        self._arrival: dict[int, int] = {}

    def bind_runtime(self, runtime) -> None:
        self._runtime = runtime

    def observe_enqueue(self, reqs: list[Request]) -> None:
        """Record each group's arrival order, the first time any of its members is queued."""
        for req in reqs:
            if req.group_id not in self._arrival:
                self._arrival[req.group_id] = len(self._arrival)

    def begin_dispatch(self, _t: float) -> None:
        """Snapshot a fresh `PlanSession` for this dispatch burst."""
        self._session = PlanSession.build(
            self._runtime, self.predictor, self.horizon_buffers, kv_watermark=self.kv_watermark
        )

    def pick_queued(self, window: list[Request], _views: list[InstanceView]) -> Request | None:
        """Offer the highest-priority request in this version's layer.

        An admitted group's queued member always goes ahead of a staged group's: it already
        holds a reservation a buffer is waiting on, where a staged group holds nothing yet.
        """
        session = self._session
        if session is None or not window:
            return None
        remaining_of = self.predictor.remaining
        admitted: list[tuple[int, float, int, Request]] = []
        staged: list[tuple[int, float, int, Request]] = []
        staged_seen: set[int] = set()
        for req in window:
            if req.group_id in session.members:
                deadline = self._deadline(session, req.group_id)
                admitted.append((deadline, -remaining_of(traj_view(req)), req.rid, req))
            elif req.group_id in session.staged_groups and req.group_id not in staged_seen:
                staged_seen.add(req.group_id)
                group = session.staged_groups[req.group_id]
                slowest = max(remaining_of(traj_view(member)) for member in group)
                block = self._arrival.get(req.group_id, 0) // self.window_groups if self.window_groups else 0
                staged.append((block, -slowest, group[0].rid, group[0]))
        if admitted:
            return min(admitted)[3]
        if staged:
            return min(staged)[3]
        return None

    def select_context(self, req: Request, candidates: list[InstanceView], _context) -> int | None:
        """Price every feasible option for `req` and commit the lexicographically best one."""
        session = self._session
        if session is None:
            return None
        if req.group_id in session.members:
            options = session.evaluate_member(req)
        elif req.group_id in session.staged_groups:
            group = session.staged_groups[req.group_id]
            if req.rid != group[0].rid:
                return None
            options = session.evaluate_admission(group)
        else:
            return None
        if not self.defer:
            options = {option: pipeline for option, pipeline in options.items() if option is not None}

        candidate_ids = {view.instance_id for view in candidates}
        eps = self._eps(session)
        best_option: int | None = None
        best_pipeline = None
        for option, pipeline in options.items():
            if pipeline is None:
                continue
            if option is not None and option not in candidate_ids:
                continue
            if best_pipeline is None or compare(pipeline, best_pipeline, eps) < 0:
                best_option, best_pipeline = option, pipeline
        if best_pipeline is None:
            return None
        best_option = self._break_tie(best_option, best_pipeline, options, candidates, candidate_ids, eps)

        session.commit(req, best_option)
        if best_option is None:
            return None
        return next(i for i, view in enumerate(candidates) if view.instance_id == best_option)

    def allow_version_bypass(self, _req: Request | None, _context) -> bool:
        """Never: letting a newer version jump a blocked or deferred head would break EDF."""
        return False

    def _break_tie(
        self,
        best_option: int | None,
        best_pipeline,
        options: dict,
        candidates: list[InstanceView],
        candidate_ids: set[int],
        eps: float,
    ) -> int | None:
        """Prefer the least-loaded instance among options the buffer forecast cannot tell apart.

        Below the decode knee, most placements tie the lexicographic pipeline within `eps` because
        nothing is buffer-critical yet; without a secondary signal the loop above keeps whichever
        instance happens to iterate first, concentrating every admission there.
        """
        if best_option is None:
            return None
        tied = [
            option
            for option, pipeline in options.items()
            if option is not None
            and option in candidate_ids
            and pipeline is not None
            and compare(pipeline, best_pipeline, eps) == 0
        ]
        if len(tied) <= 1:
            return best_option
        load = {view.instance_id: view.n_active for view in candidates}
        return min(tied, key=lambda option: (load[option], option))

    def _deadline(self, session: PlanSession, group_id: int) -> int:
        """A group admitted earlier this burst has no snapshot reservation yet.

        Its real `reserve_data` call has not happened, so `session.snapshot` -- taken before this
        burst -- does not know it. psrl's own rule is `deadline = version + staleness`, and a
        group `_commit_admission` just reserved was reserved at this session's rollout version.
        """
        try:
            return session.deadline(group_id)
        except KeyError:
            return session.rollout_version + session.runtime.max_staleness

    def _eps(self, session: PlanSession) -> float:
        if not session.train_cache:
            return self.eps_frac
        cycle = next(iter(session.train_cache.values())).cycle
        return self.eps_frac * cycle
