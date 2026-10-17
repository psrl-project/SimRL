"""When a trajectory's data stops being worth anything, and how much it can still generate.

SCG counts a trajectory only if it is consumed within `S` weight versions of its
admission. That makes staleness a deadline, denominated in training cycles rather
than seconds, and a token budget is what is left of it:

    cycles_left = v_traj + S + 1 - next_train_version
    seconds     = cycles_left * cycle_s - elapsed_this_cycle
    budget      = rate * seconds

The cycle length is observed rather than configured. It depends on the batch that
was actually taken, so a configured constant would mis-date every deadline in the
same direction.
"""

from __future__ import annotations


class TrainCycle:
    """Tracks how long a training cycle takes, and when the current one started.

    Usage:
        cycle = TrainCycle(seed_s=0.0)
        cycle.observe(train_version=3, now=120.0)   # at each weight publish
        seconds = cycle.seconds_left(v_traj=1, max_staleness=1, now=130.0)
    """

    def __init__(self, seed_s: float = 0.0, smoothing: float = 0.3) -> None:
        """
        Args:
            seed_s (float): Starting estimate in seconds. Zero means "wait for a
                real measurement", which keeps every budget unknown until one
                cycle has been seen.
            smoothing (float): EWMA weight on the newest observation. A single
                long step should not redate every deadline at once.
        """
        if not 0.0 < smoothing <= 1.0:
            raise ValueError(f"Smoothing must lie in (0, 1], got {smoothing!r}.")
        self.cycle_s = float(seed_s)
        self.smoothing = float(smoothing)
        self.train_version = 0
        self.cycle_start = 0.0

    @property
    def known(self) -> bool:
        """Whether a cycle duration has been established."""
        return self.cycle_s > 0.0

    def observe(self, train_version: int, now: float) -> None:
        """Record a weight publish, learning the interval since the last one.

        The very first publish counts. The interval from run start to it is a
        real cycle, and waiting for a second one costs a whole version of a short
        run, during which the caller has no deadline and cannot price anything.
        """
        interval = now - self.cycle_start
        if interval > 0.0:
            self.cycle_s = (
                interval
                if self.cycle_s <= 0.0
                else (1.0 - self.smoothing) * self.cycle_s + self.smoothing * interval
            )
        self.train_version = int(train_version)
        self.cycle_start = float(now)

    def cycles_left(self, v_traj: int, max_staleness: int) -> int:
        """Return the training cycles before a trajectory admitted at `v_traj` ages out.

        A cohort admitted at `v` is collected once the published version reaches
        `v + max_staleness + 1`, so that version minus the one being trained now is
        what remains. Matches `RollPolicy.expire`'s `max_gc_step` arithmetic.
        """
        return v_traj + max_staleness + 1 - (self.train_version + 1)

    def seconds_left(self, v_traj: int, max_staleness: int, now: float) -> float:
        """Return the wall-clock seconds a trajectory has before its deadline.

        Returns:
            Seconds remaining, floored at zero. Zero when no cycle has been
            observed yet, which callers read as "no deadline known".
        """
        if not self.known:
            return 0.0
        whole = self.cycles_left(v_traj, max_staleness)
        if whole <= 0:
            return 0.0
        elapsed = max(0.0, now - self.cycle_start)
        return max(0.0, whole * self.cycle_s - elapsed)


def budget_tokens(rate: float, seconds: float) -> float:
    """Return the tokens a trajectory can still generate before its deadline."""
    if rate <= 0.0 or seconds <= 0.0:
        return 0.0
    return rate * seconds
