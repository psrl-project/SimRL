"""Route by a lane plan rebuilt at every dispatch burst.

Where `length_aware` balances load and `inventory_deadline` prices one placement at a time,
this strategy plans the whole fleet at once: `plan_lanes` decides which instance every
unplaced request goes to, and how many requests share it. Planning the fleet as a unit is
what lets a straggler end up alone on an instance, which no per-placement rule can arrange,
and a weight publish is what makes it affordable: psrl aborts every in-flight request and
hands them all back at once, so the plan is rebuilt on an empty fleet at no extra cost.

Requests already decoding when a mid-burst dispatch happens are pinned to their instance.
Moving one outside a publish would discard KV the plan has no way to pay back.
"""

from __future__ import annotations

from .....forecast import LaneJob, LaneSlot, build_length_predictor, plan_lanes
from .....forecast.fleet import members_of, sequence_hint
from ...engine import InstanceView, Request, traj_view
from ..strategy import register_strategy


@register_strategy("slack_lane")
class SlackLane:
    """Plan the whole fleet's placement at each burst, thinning instances for late stragglers."""

    name = "slack_lane"
    python_engine_only = True

    def __init__(
        self,
        length_model: str = "oracle",
        length_model_args: dict | None = None,
        horizon_buffers: int = 2,
        c_max: int = 4,
        eps_frac: float = 0.01,
        max_moves: int = 8,
        enable_upgrade: bool = True,
    ) -> None:
        """
        Args:
            length_model (str): Name in `LENGTH_PREDICTORS`.
            length_model_args (dict | None): Constructor keywords for that predictor.
            horizon_buffers (int): Pending buffers the objective sums over.
            c_max (int): Requests a fast lane may hold.
            eps_frac (float): Fraction of one training cycle an upgrade must save to be kept.
            max_moves (int): Upgrades one plan may accept.
            enable_upgrade (bool): Whether to search for fast lanes at all. False leaves the
                balanced assignment untouched, which isolates what the lanes themselves buy
                from what planning the fleet as a unit buys.

        Raises:
            ValueError: If `horizon_buffers` is below 1.
        """
        if horizon_buffers < 1:
            raise ValueError(f"horizon_buffers must be >= 1, got {horizon_buffers!r}.")
        self.predictor = build_length_predictor(length_model, **(length_model_args or {}))
        self.horizon_buffers = int(horizon_buffers)
        self.c_max = int(c_max)
        self.eps_frac = float(eps_frac)
        self.max_moves = int(max_moves)
        self.enable_upgrade = bool(enable_upgrade)
        self._runtime = None
        self._plan = None

    def bind_runtime(self, runtime) -> None:
        self._runtime = runtime

    def begin_dispatch(self, t: float) -> None:
        """Rebuild the plan for this burst, or drop it when the fleet has nothing to decide."""
        self._plan = self._build_plan(t)

    def narrow_candidates(self, req: Request, candidates: list[InstanceView], _context) -> list[int]:
        """Confine a planned request to its instance, and leave an unplanned one unconstrained."""
        plan = self._plan
        if plan is None:
            return list(range(len(candidates)))
        wanted = plan.assignment.get(req.rid)
        if wanted is None:
            # Enqueued after `begin_dispatch` priced the burst. It may go anywhere the
            # plan did not reserve for a fast lane.
            return [i for i, view in enumerate(candidates) if view.instance_id not in plan.fast_instances] or list(
                range(len(candidates))
            )
        matches = [i for i, view in enumerate(candidates) if view.instance_id == wanted]
        return matches or list(range(len(candidates)))

    def select_context(self, req: Request, candidates: list[InstanceView], _context) -> int | None:
        """Take the planned instance, else the one with the fewest predicted tokens left."""
        plan = self._plan
        if plan is not None:
            wanted = plan.assignment.get(req.rid)
            for index, view in enumerate(candidates):
                if view.instance_id == wanted:
                    return index
        return min(range(len(candidates)), key=lambda i: (self._remaining_load(candidates[i]), candidates[i].n_active))

    def _remaining_load(self, view: InstanceView) -> float:
        total = 0.0
        for traj in view.residents():
            total += max(0.0, self.predictor.predict_total(traj) - traj.generated)
        return total

    def _build_plan(self, t: float):
        runtime = self._runtime
        if runtime is None:
            return None
        instances = runtime.instances
        if len(instances) < 2:
            return None
        members = members_of(runtime)
        if not members:
            return None

        pinned: dict[int, int] = {}
        jobs: list[LaneJob] = []
        seen: set[int] = set()
        for inst in instances:
            for req in (*inst.waiting, *inst.running):
                jobs.append(self._job(req))
                pinned[req.rid] = inst.instance_id
                seen.add(req.rid)
        for group in runtime.queued().values():
            for req in group:
                if req.rid not in seen:
                    jobs.append(self._job(req))
                    seen.add(req.rid)
        if not jobs:
            return None

        finished = {rid: done for group in runtime.pending.values() for rid, done in group.items()}
        lengths = {job.rid: job.prompt_len + job.generated + job.remaining for job in jobs}
        hint = sequence_hint(
            {group_id: 0.0 for group_id in members},
            members,
            lengths,
            runtime.batch_size,
            runtime.group_size,
        )
        slots = [
            LaneSlot(
                instance_id=inst.instance_id,
                cost=runtime.cost_model(inst.tp),
                block_size=inst.block_size,
                kv_blocks=inst.kv_blocks,
                token_budget=runtime.token_budget,
                max_concurrency=None,
                tp=inst.tp,
            )
            for inst in instances
        ]
        return plan_lanes(
            jobs,
            slots,
            snapshot=runtime.inventory(),
            train=runtime.train_state(hint),
            members=members,
            finished=finished,
            now=t,
            pinned=pinned,
            horizon_buffers=self.horizon_buffers,
            c_max=self.c_max,
            eps_frac=self.eps_frac,
            max_moves=self.max_moves if self.enable_upgrade else 0,
        )

    def _job(self, req: Request) -> LaneJob:
        traj = traj_view(req)
        return LaneJob(
            rid=req.rid,
            group_id=req.group_id,
            prompt_key=traj.prompt_key,
            prompt_len=req.prompt_len,
            generated=req.generated,
            remaining=max(1, round(self.predictor.remaining(traj))),
        )
