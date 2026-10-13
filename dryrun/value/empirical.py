"""The conditional length CDF, learned online from trajectories as they finish.

One histogram of final lengths, read as

    V = P(L <= l + B | L > l) = [F(l + B) - F(l)] / [1 - F(l)]

**Why a histogram and not the raw samples.** `G` is a derivative of `V`, and the
empirical CDF of a sample set is a staircase: its derivative is zero almost
everywhere and unbounded on the steps. Binning with linear interpolation inside a
bin makes `F` piecewise linear, so `G` is piecewise constant and finite. The bin
width is therefore a smoothing parameter, not a discretisation of the model.

**Per-prompt conditioning.** A prompt's own finished responses are the sharpest
evidence about its siblings, and GRPO supplies `group_size` of them at once. The
per-prompt CDF is mixed toward the global one by `m / (m + prior_strength)`, so a
prompt with no history contributes nothing and one with a full group dominates. No
branch on sample count: the weight does it.
"""

from __future__ import annotations

from .length import TrajView


class EmpiricalLength:
    """A binned conditional CDF over observed final lengths.

    Usage:
        model = EmpiricalLength(max_length=32768, bin_width=256)
        model.observe(traj, final_length=4096)      # per finished trajectory
        v = model.hit_prob(traj, budget=6000.0)
    """

    name = "empirical"

    def __init__(
        self,
        max_length: int = 65536,
        bin_width: int = 256,
        min_samples: int = 32,
        per_prompt: bool = True,
        prior_strength: float = 4.0,
    ) -> None:
        """
        Args:
            max_length (int): Upper edge of the histogram. Lengths at or above it
                land in the final bin, which is correct here because the simulator
                clips every target to the context budget before admitting it.
            bin_width (int): Histogram resolution in tokens, the smoothing
                parameter described in the module docstring.
            min_samples (int): Finished trajectories required before `ready` turns
                True. Below it the caller admits freely rather than ordering on
                noise.
            per_prompt (bool): Whether to keep a per-prompt histogram and mix it
                with the global one.
            prior_strength (float): Samples at which a prompt's own evidence gets
                half the weight.
        """
        if max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length!r}.")
        if bin_width < 1:
            raise ValueError(f"bin_width must be >= 1, got {bin_width!r}.")
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples!r}.")
        if prior_strength < 0.0:
            raise ValueError(f"prior_strength must be >= 0, got {prior_strength!r}.")
        self.max_length = int(max_length)
        self.bin_width = int(bin_width)
        self.min_samples = int(min_samples)
        self.per_prompt = bool(per_prompt)
        self.prior_strength = float(prior_strength)

        self._n_bins = self.max_length // self.bin_width + 1
        # Cumulative counts rather than per-bin: a read is a lookup, and a write is
        # a suffix increment. Reads outnumber writes by orders of magnitude here,
        # since every placement attempt prices every candidate.
        self._global: list[float] = [0.0] * (self._n_bins + 1)
        self._global_n = 0.0
        self._prompt: dict[int, list[float]] = {}
        self._prompt_n: dict[int, float] = {}

    # --- observation ------------------------------------------------------

    def observe(self, traj: TrajView, final_length: int) -> None:
        """Record one finished trajectory's total generated length."""
        index = self._bin_of(final_length)
        for position in range(index + 1, self._n_bins + 1):
            self._global[position] += 1.0
        self._global_n += 1.0
        if not self.per_prompt:
            return
        counts = self._prompt.get(traj.prompt_key)
        if counts is None:
            counts = [0.0] * (self._n_bins + 1)
            self._prompt[traj.prompt_key] = counts
        for position in range(index + 1, self._n_bins + 1):
            counts[position] += 1.0
        self._prompt_n[traj.prompt_key] = self._prompt_n.get(traj.prompt_key, 0.0) + 1.0

    def decay(self, factor: float) -> None:
        """Discount every count, so the model tracks the policy as it drifts.

        RL training lengthens outputs, so an undiscounted history mixes a model that
        no longer exists with the current one. The per-prompt tables are left alone:
        relative difficulty between prompts is stable, and the overall length level
        is carried by the global table.
        """
        if not 0.0 < factor <= 1.0:
            raise ValueError(f"Decay factor must lie in (0, 1], got {factor!r}.")
        if factor == 1.0:
            return
        self._global = [value * factor for value in self._global]
        self._global_n *= factor

    # --- query ------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._global_n >= self.min_samples

    def hit_prob(self, traj: TrajView, budget: float) -> float:
        """Return `P(L <= generated + budget | L > generated)`."""
        if budget <= 0.0:
            return 0.0
        if not self.ready:
            return 0.0
        start = float(traj.generated)
        survive_start = 1.0 - self._cdf(traj.prompt_key, start)
        if survive_start <= 1e-9:
            # Every observed trajectory was shorter than this one already is, so
            # the evidence says nothing about when it stops. Claiming certainty
            # either way would be invention.
            return 0.0
        survive_end = 1.0 - self._cdf(traj.prompt_key, start + budget)
        return min(1.0, max(0.0, (survive_start - survive_end) / survive_start))

    def expected_final_length(self, traj: TrajView) -> float:
        """Return the conditional mean final length, for sizing a KV footprint.

        `E[L | L > generated]`, integrated from the survival function so it uses
        the same conditional CDF every other reading does. Falls back to the
        trajectory's current depth before the model is ready, which is the
        no-information answer rather than an invented one.
        """
        start = float(traj.generated)
        if not self.ready:
            return start
        survive_start = 1.0 - self._cdf(traj.prompt_key, start)
        if survive_start <= 1e-9:
            return max(start, float(self.max_length))
        # E[L | L > s] = s + integral over x > s of P(L > x | L > s).
        total = 0.0
        step = float(self.bin_width)
        position = start
        while position < self.max_length:
            survive = 1.0 - self._cdf(traj.prompt_key, position + step)
            total += (survive / survive_start) * step
            position += step
        return start + total

    def _cdf(self, prompt_key: int, length: float) -> float:
        """Return `F(length)`, mixing the prompt's own evidence with the global."""
        global_value = self._interpolate(self._global, self._global_n, length)
        if not self.per_prompt:
            return global_value
        counts = self._prompt.get(prompt_key)
        if counts is None:
            return global_value
        n = self._prompt_n.get(prompt_key, 0.0)
        if n <= 0.0:
            return global_value
        weight = n / (n + self.prior_strength)
        return (1.0 - weight) * global_value + weight * self._interpolate(counts, n, length)

    def _interpolate(self, cumulative: list[float], total: float, length: float) -> float:
        """Return a piecewise-linear `F(length)` from cumulative bin counts."""
        if total <= 0.0:
            return 0.0
        if length <= 0.0:
            return 0.0
        if length >= self.max_length:
            return cumulative[self._n_bins] / total
        position = length / self.bin_width
        index = int(position)
        fraction = position - index
        low = cumulative[index]
        high = cumulative[index + 1]
        return (low + fraction * (high - low)) / total

    def _bin_of(self, length: int) -> int:
        """Return the bin a final length falls in, clamped to the last one."""
        if length <= 0:
            return 0
        return min(length // self.bin_width, self._n_bins - 1)
