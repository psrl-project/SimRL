"""PSRL Reserve/Occupy/Consume staleness policy.

Delegates buffer management to `psrl.workers.ps.staleness_controller.StalenessInventory`
so that the simulation faithfully reproduces the real system's reserve-at-worst-case,
occupy-greedy-earliest, and cross-buffer entry-movement logic.

The psrl package must be importable.  Install it with:
    pip install --no-deps -e /path/to/psrl
and ensure the psrl source root is on sys.path so the lightweight import succeeds
(the staleness_controller module has no ray/torch dependency after the isolation fix).
"""

from __future__ import annotations

import sys

from ..component.rollout.engine import Request
from .base import CompleteAction, SimState, StalenessPolicy

_PSRL_ROOT = "/apdcephfs_zwfy10_303541817/share_303541817/lhy/psrl"

try:
    from psrl.workers.ps.staleness_controller import (
        BufferStatus,
        EntryCategory,
        EntryInfo,
        StalenessInventory,
    )

    _HAS_PSRL = True
except ImportError:
    if _PSRL_ROOT not in sys.path:
        sys.path.insert(0, _PSRL_ROOT)
    try:
        from psrl.workers.ps.staleness_controller import (  # type: ignore[no-redef]
            BufferStatus,
            EntryCategory,
            EntryInfo,
            StalenessInventory,
        )

        _HAS_PSRL = True
    except ImportError:
        _HAS_PSRL = False


class PSRLPolicy(StalenessPolicy):
    """
    PSRL Reserve/Occupy/Consume policy.

    Wraps `psrl.workers.ps.staleness_controller.StalenessInventory` to reproduce
    the real system's buffer management:

    - Reserve (on_admit): pessimistic placement in the highest pending buffer,
      filling from the end backwards (`get_last_non_reserved`).
    - Occupy (on_complete): greedy re-placement in the earliest buffer, with
      cross-buffer reserved-entry movement to avoid spurious STUCK states.
    - Consume (peek_batch/take_batch): waits for the current-version buffer to
      be READY, then deletes it only when the batch is formally taken.
    """

    name = "psrl"

    def __init__(
        self,
        proactive_filter: bool = False,
        proactive_threshold: int = 0,
    ):
        if not _HAS_PSRL:
            raise ImportError(
                "psrl package not found. "
                f"Add {_PSRL_ROOT!r} to sys.path or install psrl with: "
                f"pip install --no-deps -e {_PSRL_ROOT}"
            )
        self.proactive_filter = proactive_filter
        self.proactive_threshold = proactive_threshold

        self._inventory: StalenessInventory | None = None
        self._batch_size: int | None = None
        self.stuck_since: float | None = None
        self.stuck_time: float = 0.0
        self.filtered: int = 0

    # --- Lazy initialisation ---

    def _lazy_init(self, st: SimState) -> None:
        self._batch_size = st.batch_size
        self._group_size = st.group_size
        # An entry holds one prompt group, and `rollout_n` is how many responses
        # share it, which is what the real controller means by these fields.
        self._inventory = StalenessInventory(
            num_entries=st.batch_size,
            ready_num_entries=st.batch_size,
            staleness=st.max_staleness,
            rollout_n=st.group_size,
        )
        self._inventory.ensure_buffer_exists(st.train_version + st.max_staleness)

    # --- StalenessPolicy interface ---

    def admit_quota(self, st: SimState) -> int:
        if self._inventory is None:
            self._lazy_init(st)
        # New requests are generated at `rollout_version`, so only count slots
        # those trajectories can actually occupy. Entries are prompt groups.
        max_bid = st.rollout_version + st.max_staleness
        self._inventory.ensure_buffer_exists(max(st.train_version, st.rollout_version) + st.max_staleness)
        free_groups = self._inventory.get_empty_entries_total_num(max_bid)
        return free_groups * st.group_size

    def on_admit(self, group: list[Request], st: SimState) -> None:
        assert self._inventory is not None, "on_admit called before admit_quota."
        for req in group:
            max_bid = req.v_traj + st.max_staleness
            entry_info = EntryInfo(
                rollout_instance_id=("sim", req.instance_id or 0),
                prompt_id=req.group_id,
                request_idx=req.group_index,
                model_version=req.v_traj,
            )
            buffer_id, entry_id = self._inventory.reserve_data(entry_info, max_bid)
            assert buffer_id is not None, (
                f"reserve_data returned None for group {req.group_id} "
                f"(v_traj={req.v_traj}, max_bid={max_bid}); admit_quota should have gated this."
            )

    def on_group_complete(self, group: list[Request], st: SimState) -> CompleteAction:
        assert self._inventory is not None, "on_group_complete called before any admit."
        group_id = group[0].group_id
        if group_id not in self._inventory.data_tracker:
            return CompleteAction.DROP
        buffer_id, entry_id, occupy_num = self._inventory.occupy_data_with_reserve(group_id)
        if buffer_id is None:
            return CompleteAction.DROP
        return CompleteAction.KEEP

    def peek_batch(self, st: SimState) -> list[Request] | None:
        assert self._inventory is not None
        bid = st.train_version
        if bid not in self._inventory.buffers:
            return None

        status = self._inventory.get_buffer_status(bid)

        if status in (BufferStatus.READY, BufferStatus.READY_WITH_CAPACITY):
            buffer = self._inventory.buffers[bid]
            ready_n = buffer.ready_num_entries
            group_ids = [
                buffer.entries[i].entry_info.prompt_id
                for i in range(ready_n)
                if buffer.entries[i].entry_info is not None
            ]
            by_group = {group[0].group_id: group for group in st.ready_groups}
            chosen = [by_group[gid] for gid in group_ids if gid in by_group]
            if len(chosen) < st.batch_size:
                return None

            return [req for group in chosen[: st.batch_size] for req in group]

        return None

    def take_batch(self, st: SimState) -> list[Request] | None:
        chosen = self.peek_batch(st)
        if chosen is None:
            return None
        if self.stuck_since is not None:
            self.stuck_time += st.now - self.stuck_since
            self.stuck_since = None
        assert self._inventory is not None
        self._inventory.delete_buffer(st.train_version)
        return chosen

    def on_batch_unavailable(self, st: SimState) -> None:
        assert self._inventory is not None
        bid = st.train_version
        if bid not in self._inventory.buffers:
            return
        status = self._inventory.get_buffer_status(bid)
        if status == BufferStatus.STUCK:
            if self.stuck_since is None:
                self.stuck_since = st.now
            self._maybe_proactive_filter(bid, st)

    def on_train_version_advance(self, train_version: int, st: SimState) -> None:
        assert self._inventory is not None
        self._inventory.ensure_buffer_exists(train_version + st.max_staleness)

    def on_interrupt(self, reqs: list[Request], st: SimState) -> None:
        # Partial rollout only resets engine-side prefill/KV state. Requests
        # loop back and keep the reservation made from their original v_traj.
        return

    def on_discard(self, reqs: list[Request], st: SimState) -> None:
        if self._inventory is None:
            return
        reserved: list[int] = []
        occupied: list[int] = []
        for prompt_id in dict.fromkeys(r.group_id for r in reqs):
            location = self._inventory.data_tracker.get(prompt_id)
            if location is None:
                continue
            buffer_id, entry_id = location
            category = self._inventory.buffers[buffer_id].entries[entry_id].category
            if category == EntryCategory.RESERVED:
                reserved.append(prompt_id)
            elif category == EntryCategory.OCCUPIED:
                occupied.append(prompt_id)
        if reserved:
            self._inventory.clear_reserved_entries(reserved)
        if occupied:
            self._inventory.clear_occupied_entries(occupied)

    def check_invariants(self, st: SimState) -> None:
        if self._inventory is None:
            return
        for prompt_id, (buffer_id, entry_id) in self._inventory.data_tracker.items():
            entry = self._inventory.buffers[buffer_id].entries[entry_id]
            if entry.entry_info is None:
                continue
            v_gen = entry.entry_info.get_entry_version()
            assert buffer_id - v_gen <= st.max_staleness, (
                f"Prompt {prompt_id} placed in buffer {buffer_id} but generated at "
                f"version {v_gen} (max_staleness={st.max_staleness})."
            )

    # --- Internal helpers ---

    def _maybe_proactive_filter(self, bid: int, st: SimState) -> None:
        if not self.proactive_filter:
            return
        if bid not in self._inventory.buffers:
            return
        buffer = self._inventory.buffers[bid]
        reserved_pids = [
            buffer.entries[i].entry_info.prompt_id
            for i in range(buffer.num_entries)
            if (
                buffer.entries[i].category == EntryCategory.RESERVED
                and buffer.entries[i].entry_info is not None
            )
        ]
        if not (0 < len(reserved_pids) <= self.proactive_threshold):
            return
        self._inventory.clear_reserved_entries(reserved_pids)
        self.filtered += len(reserved_pids)
