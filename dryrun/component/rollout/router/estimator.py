"""Deployable remaining-length estimates for the planner and the router.

Three signals, none of which needs the trace's true lengths:

- Prompt history. A prompt that produced a long response in one round tends to
  produce a long one in the next, so the geometric mean of a prompt's past
  response lengths is its estimate.
- Survival. A request that has already produced `g` tokens is not a fresh
  draw. Its expected remaining length is the mean excess of the completed
  lengths above `g`.
- Population quantiles. A new request with no history is unknown. Planning
  fills such requests with the quantiles of the completed-length distribution,
  so the planner can size a tail tier without labelling any request.
"""

from __future__ import annotations

import numpy as np


class LengthEstimator:
    """Estimates remaining tokens from completed lengths and prompt history."""

    def __init__(
        self,
        mode: str = "online",
        prior_total: float = 4096.0,
        min_tail: int = 3,
        use_history: bool = True,
        noise_sigma: float = 0.0,
        miss_rate: float = 0.0,
        seed: int = 0,
    ) -> None:
        """
        Args:
            mode: `oracle` reads the true remaining length. `online` uses only what
                a scheduler could observe. `noisy` is a controlled predictor: the true
                total times lognormal noise, corrected by survival once a request
                outruns its prediction.
            prior_total: Expected total length before any completion has been seen.
            min_tail: Completed lengths needed above `g` before the survival
                estimate is trusted.
            use_history: Whether prompt history is used. Off isolates the survival
                and population signals.
            noise_sigma: Log-sd of the `noisy` predictor's multiplicative error.
            miss_rate: Fraction of requests the `noisy` predictor knows nothing about,
                which then look average.
            seed: Seed of the per-request noise.

        Raises:
            ValueError: If `mode` is unknown.
        """
        if mode not in ("oracle", "online", "noisy"):
            raise ValueError(f"Unknown estimator mode {mode!r}, expected oracle, online or noisy.")
        self.mode = mode
        self.prior_total = float(prior_total)
        self.min_tail = int(min_tail)
        self.use_history = bool(use_history)
        self.noise_sigma = float(noise_sigma)
        self.miss_rate = float(miss_rate)
        self.seed = int(seed)
        self._predicted: dict[int, float] = {}
        self._completed: list[int] = []
        self._history: dict[int, list[int]] = {}
        self._sorted = np.zeros(0)
        self._suffix = np.zeros(1)
        self._dirty = False

    def observe(self, req) -> None:
        """Record a finished request's total generated length."""
        self._completed.append(int(req.generated))
        self._history.setdefault(int(req.source_group_id), []).append(int(req.generated))
        self._dirty = True

    def _refresh(self) -> None:
        if not self._dirty:
            return
        self._sorted = np.sort(np.asarray(self._completed, dtype=float))
        self._suffix = np.concatenate((np.cumsum(self._sorted[::-1])[::-1], [0.0]))
        self._dirty = False

    @property
    def n_completed(self) -> int:
        return len(self._completed)

    def mean_excess(self, generated: float) -> float | None:
        """Expected remaining tokens given `generated` already, or None if too few samples."""
        self._refresh()
        n = len(self._sorted)
        i = int(np.searchsorted(self._sorted, generated, side="right"))
        k = n - i
        if k < self.min_tail:
            return None
        return float(self._suffix[i] / k - generated)

    def history_total(self, req) -> float | None:
        """Geometric mean of the prompt's past response lengths, if it has any."""
        if not self.use_history:
            return None
        past = self._history.get(int(req.source_group_id))
        if not past:
            return None
        return float(np.exp(np.mean(np.log(np.maximum(past, 1)))))

    def predicted_total(self, req) -> float:
        """The `noisy` predictor's fixed guess of a request's total length."""
        rid = int(req.rid)
        cached = self._predicted.get(rid)
        if cached is not None:
            return cached
        rng = np.random.default_rng([self.seed, rid])
        z, u = float(rng.standard_normal()), float(rng.random())
        if u < self.miss_rate:
            value = self.prior_total
        else:
            value = float(req.target_len) * float(np.exp(self.noise_sigma * z))
        self._predicted[rid] = value
        return value

    def remaining(self, req) -> float:
        """Estimated tokens `req` still has to generate."""
        if self.mode == "oracle":
            return float(req.remaining)
        g = float(req.generated)
        if self.mode == "noisy":
            guess = self.predicted_total(req)
            if g < 0.9 * guess:
                return max(guess - g, 0.1 * guess)
            excess = self.mean_excess(g)
            return max(excess, 1.0) if excess is not None else max(g, self.prior_total)
        total = self.history_total(req)
        if total is not None and g < 0.8 * total:
            return max(total - g, 0.25 * total)
        excess = self.mean_excess(g)
        if excess is not None:
            return max(excess, 1.0)
        return max(g, self.prior_total) if g > 0 else self.prior_total

    def is_unknown(self, req) -> bool:
        """Whether `req` has neither history nor progress to estimate from."""
        return self.mode == "online" and req.generated == 0 and self.history_total(req) is None

    def population_quantiles(self, n: int) -> np.ndarray | None:
        """`n` quantiles of the completed-length distribution, or None before it exists."""
        self._refresh()
        if len(self._sorted) < max(self.min_tail, 16) or n <= 0:
            return None
        return np.quantile(self._sorted, (np.arange(n) + 0.5) / n)

    def plan_remaining(self, reqs) -> np.ndarray:
        """Estimates for planning. Unknown requests are filled with population quantiles."""
        est = np.array([self.remaining(r) for r in reqs], dtype=float)
        if self.mode != "online":
            return est
        unknown = [i for i, r in enumerate(reqs) if self.is_unknown(r)]
        fill = self.population_quantiles(len(unknown))
        if fill is not None:
            est[np.array(unknown, dtype=int)] = fill
        return est
