"""Price one request as if it had an instance to itself.

The bounds in `bounds.py` need a counterfactual: how long this request's work
would take with no co-residents. Both methods call the same cost model the
engine called, so the difference between a priced solo time and the observed
time is interference and nothing else.
"""

from __future__ import annotations


class SoloPricer:
    """Solo decode and prefill times, by tensor-parallel degree.

    Args:
        cost_by_tp (dict[int, object]): Rollout cost model per TP degree. A
            homogeneous fleet passes one entry.
        block_size (int): KV block size, used only to report the unit the
            caller priced against.

    Raises:
        ValueError: When `cost_by_tp` is empty.
    """

    def __init__(self, cost_by_tp: dict[int, object], block_size: int):
        if not cost_by_tp:
            raise ValueError("cost_by_tp must hold at least one tensor-parallel degree.")
        self._cost_by_tp = dict(cost_by_tp)
        self.block_size = block_size

    def _cost(self, tp: int):
        cost = self._cost_by_tp.get(tp)
        if cost is None:
            raise KeyError(
                f"No cost model for tensor-parallel degree {tp!r}, have {sorted(self._cost_by_tp)}."
            )
        return cost

    def decode_seconds(self, ctx0: int, tokens: int, tp: int = 1) -> float:
        """
        Return the seconds to decode `tokens` steps alone, starting at context `ctx0`.

        Args:
            ctx0 (int): Context tokens held when the segment starts.
            tokens (int): Decode steps in the segment.
            tp (int): Tensor-parallel degree of the instance.

        Returns:
            float: Seconds for a batch of one over the whole segment.
        """
        if tokens <= 0:
            return 0.0
        return self._cost(tp).decode_segment_time(1, float(ctx0), int(tokens))

    def prefill_seconds(self, ctx_tokens: int, tp: int = 1) -> float:
        """
        Return the seconds to prefill `ctx_tokens` context tokens alone.

        Args:
            ctx_tokens (int): Context tokens to compute.
            tp (int): Tensor-parallel degree of the instance.

        Returns:
            float: Seconds for a step that prefills this context and decodes nothing.
        """
        if ctx_tokens <= 0:
            return 0.0
        return self._cost(tp).step_time(
            int(ctx_tokens), 0.0, 0.0, n_reqs=1, prefill_ctx=float(ctx_tokens)
        )
