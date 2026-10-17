"""Prefix-cache-aware routing with load-balance safeguards."""

from __future__ import annotations

from ...engine import InstanceView, Request
from ..strategy import register_strategy
from .request_num_balance import RequestNumBalance


@register_strategy("prefix_aware")
class PrefixAware:
    """Prefer a useful prefix hit unless the fleet is already imbalanced."""

    name = "prefix_aware"

    def __init__(
        self,
        cache_threshold: float = 0.3,
        balance_abs_threshold: int = 64,
        balance_rel_threshold: float = 1.5,
    ) -> None:
        self.cache_threshold = cache_threshold
        self.balance_abs_threshold = balance_abs_threshold
        self.balance_rel_threshold = balance_rel_threshold
        self._balance = RequestNumBalance()

    def _imbalanced(self, candidates: list[InstanceView]) -> bool:
        loads = [view.n_active for view in candidates]
        low, high = min(loads), max(loads)
        return (
            high - low > self.balance_abs_threshold
            and high > low * self.balance_rel_threshold
        )

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        if self._imbalanced(candidates):
            return self._balance.select(req, candidates)
        context = max(1, req.prompt_len + req.generated)
        best = max(
            range(len(candidates)),
            key=lambda i: candidates[i].prefix_hit_tokens(req),
        )
        if candidates[best].prefix_hit_tokens(req) / context <= self.cache_threshold:
            return self._balance.select(req, candidates)
        return best

