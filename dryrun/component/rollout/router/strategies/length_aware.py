"""Order and place requests by how long they are predicted to run.

Two independent decisions use the same prediction:

* **Which request goes next.** A prompt group is trainable only when its slowest
  member finishes, so the unit that is early or late is the group, keyed on its
  longest predicted member. `lpt` starts the longest group first so a straggler
  never begins last; `spt` starts the shortest first so the earliest batch fills
  sooner. Members of a group that has already started always go ahead of a new
  group, since each one left in the queue holds that group back.

  Longest-first has a cost as well as a benefit. It keeps the fleet at its
  concurrency limit to the end of the batch, but it fills that limit with the
  deepest contexts, and a decode step reads every resident's context, so the same
  number of residents then produces fewer tokens per second.

  Reordering must not change which trajectories get trained on. Ranking the
  whole backlog by length starves the groups it ranks last, so the consumed set
  drifts toward one end of the length distribution and a faster run is partly a
  different run. `window_groups` bounds this: groups are served in consecutive
  blocks of that many in arrival order and only reordered inside a block.
* **Where it goes.** `load` sends it to the instance with the least predicted
  remaining work rather than the fewest requests, which is the better balance
  when lengths differ.
* **Whether it goes.** `kv_watermark` refuses an instance whose committed KV, plus
  the newcomer's current context, would pass that fraction of the pool. It looks at
  what residents hold now and reserves nothing for their growth, so it needs no
  length prediction. A count cap cannot say this: the same count is half a pool of
  short requests or a full pool of long ones.

Both read a `LengthPredictor`, so the same rule runs against the oracle (the bound on
what any predictor buys) and against a noisy one.
"""

from __future__ import annotations

from .....forecast.length import build_length_predictor
from ...engine import InstanceView, Request, traj_view
from ..strategy import register_strategy

_ORDERS = ("fifo", "lpt", "spt")
_BALANCES = ("count", "load")


@register_strategy("length_aware")
class LengthAware:
    """Group-level longest- or shortest-first ordering with load-aware placement."""

    name = "length_aware"

    def __init__(
        self,
        order: str = "lpt",
        balance: str = "load",
        window_groups: int = 0,
        kv_watermark: float = 0.0,
        length_model: str = "oracle",
        length_model_args: dict | None = None,
    ) -> None:
        """
        Args:
            order (str): `fifo` leaves the queue order alone, `lpt` starts the
                group with the longest predicted member first, `spt` the shortest.
            balance (str): `count` picks the instance holding the fewest requests,
                `load` the one with the least predicted remaining tokens.
            window_groups (int): Groups per reordering block, in arrival order.
                Zero reorders across the whole backlog.
            kv_watermark (float): Fraction of an instance's KV pool that committed
                context may reach before it is refused. Zero disables the check.
            length_model (str): Name in `LENGTH_PREDICTORS`.
            length_model_args (dict | None): Constructor keywords for that predictor.
        """
        if order not in _ORDERS:
            raise ValueError(f"order must be one of {_ORDERS}, got {order!r}.")
        if balance not in _BALANCES:
            raise ValueError(f"balance must be one of {_BALANCES}, got {balance!r}.")
        self.order = order
        self.balance = balance
        if window_groups < 0:
            raise ValueError(f"window_groups must be >= 0, got {window_groups!r}.")
        self.window_groups = int(window_groups)
        if not 0.0 <= kv_watermark <= 1.0:
            raise ValueError(f"kv_watermark must be in [0, 1], got {kv_watermark!r}.")
        self.kv_watermark = float(kv_watermark)
        self.predictor = build_length_predictor(length_model, **(length_model_args or {}))
        self._group_length: dict[int, float] = {}
        self._arrival: dict[int, int] = {}
        self._started: set[int] = set()

    def observe_enqueue(self, reqs: list[Request]) -> None:
        """Record each group's predicted slowest member when it is first queued."""
        slowest: dict[int, float] = {}
        for req in reqs:
            length = self.predictor.predict_total(traj_view(req))
            if length > slowest.get(req.group_id, 0.0):
                slowest[req.group_id] = length
        for group_id, length in slowest.items():
            if group_id not in self._group_length:
                self._group_length[group_id] = length
                self._arrival[group_id] = len(self._arrival)

    def rank_queued(self, window: list[Request], _now: float) -> Request | None:
        if self.order == "fifo":
            return None
        pool = [req for req in window if req.group_id in self._started] or window
        sign = -1.0 if self.order == "lpt" else 1.0

        def key(req: Request) -> tuple[int, float, float, int]:
            group = self._group_length.get(req.group_id, 0.0)
            own = self.predictor.predict_total(traj_view(req))
            block = (
                self._arrival.get(req.group_id, 0) // self.window_groups
                if self.window_groups
                else 0
            )
            return (block, sign * group, sign * own, req.rid)

        return min(pool, key=key)

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        indices = [i for i in range(len(candidates)) if self._below_watermark(candidates[i], req)]
        if not indices:
            return None
        if self.balance == "count":
            index = min(indices, key=lambda i: candidates[i].n_active)
        else:
            index = min(
                indices,
                key=lambda i: (self._remaining_load(candidates[i]), candidates[i].n_active),
            )
        self._started.add(req.group_id)
        return index

    def _below_watermark(self, view: InstanceView, req: Request) -> bool:
        if self.kv_watermark <= 0.0:
            return True
        blocks = -(-req.ctx // view.block_size)
        return view.committed_blocks + blocks <= self.kv_watermark * view.kv_blocks

    def _remaining_load(self, view: InstanceView) -> float:
        """Predicted tokens the instance's residents still have to generate."""
        total = 0.0
        for traj in view.residents():
            total += max(0.0, self.predictor.predict_total(traj) - traj.generated)
        return total
