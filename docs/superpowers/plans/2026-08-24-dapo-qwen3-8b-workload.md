# Group-Aware DAPO Workload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect a real joint (prompt, n=8 responses) workload trace for Qwen3-8B on dapo-math-17k across 16 GPUs, and simulate it end to end at rollout tp=1.

**Architecture:** A joint trace records one line per DAPO group, so group boundaries live in the data instead of being derived from `rid // 8` arithmetic. `GroupedWorkload` becomes the simulator's single workload source, replacing the two independently-modulo-indexed length lists. Collection fans out over `(host, gpu)` workers that write per-worker shards to the shared filesystem, then merges.

**Tech Stack:** Python 3, vLLM 0.26.1.dev0 (`/v1/completions` with `n` + `return_token_ids`), pyarrow, Hydra/OmegaConf, pytest, ruff.

## Global Constraints

- Conda env: `source /apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh` before running anything.
- Model: `/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B`. Dataset: `/apdcephfs_zwfy/share_303760348/lhy/data/dapo/dapo-math-17k.parquet`.
- Hosts: `29.162.247.148` (local), `28.49.16.220` (remote, passwordless SSH). 8 H20 each.
- **No backward compatibility.** Remove obsolete paths rather than adding fallbacks (AGENTS.md).
- **Per-response lengths MUST come from `len(choice["token_ids"])`.** `usage.completion_tokens` is the sum over all `n` choices.
- **Dedup dataset prompts on `extra_info.index`** before subsampling: only 17,917 of 1,791,700 rows are unique.
- Style: `.claude/coding-style.md`. Identifier strings never take a trailing period; log/assert messages do. Docstrings use `Args:`/`Returns:`.
- Every task ends `pytest` green and `ruff check` clean.

---

### Task 1: Joint group trace I/O

**Files:**
- Modify: `dryrun/workload/trace.py`
- Test: `tests/test_workload_trace_groups.py`

**Interfaces:**
- Produces:
  - `RequestGroup(group_id: int, prompt_len: int, output_lens: tuple[int, ...])`: frozen dataclass in `dryrun/workload/trace.py`
  - `write_group_trace(groups: list[RequestGroup], output_path: str | Path, metadata: dict[str, Any], n_bins: int = 20) -> Path`
  - `read_group_trace(path: str | Path, n_groups: int = 0) -> list[RequestGroup]`
  - `compute_group_stats(groups: list[RequestGroup], max_output_tokens: int, n_bins: int = 20) -> dict[str, Any]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_workload_trace_groups.py`:

```python
"""Tests for joint group trace artifacts."""

from __future__ import annotations

import json

import pytest

from dryrun.workload.trace import (
    RequestGroup,
    compute_group_stats,
    read_group_trace,
    write_group_trace,
)


def _groups() -> list[RequestGroup]:
    return [
        RequestGroup(group_id=0, prompt_len=411, output_lens=(6300, 10240)),
        RequestGroup(group_id=1, prompt_len=90, output_lens=(9179, 10240)),
    ]


class TestWriteGroupTrace:
    def test_roundtrip_preserves_groups(self, tmp_path):
        path = write_group_trace(_groups(), tmp_path / "joint.jsonl", metadata={"tp": 1})
        assert read_group_trace(path) == _groups()

    def test_header_records_group_size_and_metadata(self, tmp_path):
        path = write_group_trace(_groups(), tmp_path / "joint.jsonl", metadata={"tp": 1})
        with path.open() as fh:
            header = json.loads(fh.readline())
        assert header["_meta"] is True
        assert header["n_groups"] == 2
        assert header["group_size"] == 2
        assert header["tp"] == 1
        assert "stats" in header

    def test_ragged_group_sizes_are_rejected(self, tmp_path):
        ragged = [
            RequestGroup(group_id=0, prompt_len=10, output_lens=(1, 2)),
            RequestGroup(group_id=1, prompt_len=20, output_lens=(3,)),
        ]
        with pytest.raises(ValueError, match="same number of responses"):
            write_group_trace(ragged, tmp_path / "joint.jsonl", metadata={})

    def test_empty_groups_are_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="at least one group"):
            write_group_trace([], tmp_path / "joint.jsonl", metadata={})


class TestReadGroupTrace:
    def test_n_groups_caps_the_read(self, tmp_path):
        path = write_group_trace(_groups(), tmp_path / "joint.jsonl", metadata={})
        assert read_group_trace(path, n_groups=1) == _groups()[:1]

    def test_zero_n_groups_reads_all(self, tmp_path):
        path = write_group_trace(_groups(), tmp_path / "joint.jsonl", metadata={})
        assert len(read_group_trace(path, n_groups=0)) == 2


class TestComputeGroupStats:
    def test_reports_prompt_output_and_truncation(self):
        stats = compute_group_stats(_groups(), max_output_tokens=10240, n_bins=4)
        assert stats["n_groups"] == 2
        assert stats["group_size"] == 2
        assert stats["prompt"]["min"] == 90
        assert stats["output"]["max"] == 10240
        assert stats["output"]["truncated_fraction"] == pytest.approx(0.5)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
source /apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh
cd /apdcephfs_zwfy/share_303760348/lhy/DryRun-RL
pytest tests/test_workload_trace_groups.py -q
```

Expected: FAIL: `ImportError: cannot import name 'RequestGroup'`.

- [ ] **Step 3: Implement**

Append to `dryrun/workload/trace.py` (keep the existing single-dimension functions; they serve the synthetic/marginal path):

```python
@dataclass(frozen=True)
class RequestGroup:
    """One DAPO group: a prompt and the response lengths sampled for it.

    `output_lens` holds one entry per response (`n` in the DAPO recipe), so a
    group's boundary is explicit in the data rather than inferred from
    request-index arithmetic.
    """

    group_id: int
    prompt_len: int
    output_lens: tuple[int, ...]


def compute_group_stats(
    groups: list[RequestGroup],
    max_output_tokens: int,
    n_bins: int = 20,
) -> dict[str, Any]:
    """
    Summarize prompt lengths, response lengths, and truncation for a group set.

    Args:
        groups (list[RequestGroup]): Groups to summarize.
        max_output_tokens (int): Generation cap the responses were collected
            under. Responses at this length are counted as truncated.
        n_bins (int): Number of histogram bins per dimension.

    Returns:
        dict[str, Any]: `n_groups`, `group_size`, and `prompt`/`output` blocks.
        The `output` block carries an extra `truncated_fraction`.
    """
    prompt_lengths = [group.prompt_len for group in groups]
    output_lengths = [length for group in groups for length in group.output_lens]
    truncated = sum(1 for length in output_lengths if length >= max_output_tokens)
    output_stats = compute_length_stats(output_lengths, n_bins=n_bins)
    output_stats["truncated_fraction"] = truncated / len(output_lengths)
    return {
        "n_groups": len(groups),
        "group_size": len(groups[0].output_lens),
        "max_output_tokens": max_output_tokens,
        "prompt": compute_length_stats(prompt_lengths, n_bins=n_bins),
        "output": output_stats,
    }


def write_group_trace(
    groups: list[RequestGroup],
    output_path: str | Path,
    metadata: dict[str, Any],
    n_bins: int = 20,
) -> Path:
    """
    Write a joint group trace artifact.

    Args:
        groups (list[RequestGroup]): Groups to write, one JSONL line each.
        output_path (str | Path): Destination JSONL path. Parents are created.
        metadata (dict[str, Any]): Provenance merged into the `_meta` header.
            A `max_output_tokens` entry is used for the truncation statistic.
        n_bins (int): Histogram bins for the summary stats.

    Returns:
        Path: The path written to.
    """
    if not groups:
        raise ValueError("A group trace needs at least one group.")
    group_size = len(groups[0].output_lens)
    if any(len(group.output_lens) != group_size for group in groups):
        raise ValueError(
            f"Every group must hold the same number of responses, expected {group_size}."
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    max_output_tokens = int(metadata.get("max_output_tokens", 0)) or max(
        length for group in groups for length in group.output_lens
    )
    header = {
        "_meta": True,
        "n_groups": len(groups),
        "group_size": group_size,
        "stats": compute_group_stats(groups, max_output_tokens, n_bins=n_bins),
        **metadata,
    }
    with output.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        for group in groups:
            fh.write(
                json.dumps(
                    {
                        "group_id": group.group_id,
                        "prompt_len": group.prompt_len,
                        "output_lens": list(group.output_lens),
                    }
                )
                + "\n"
            )
    return output


def read_group_trace(path: str | Path, n_groups: int = 0) -> list[RequestGroup]:
    """
    Read a joint group trace artifact.

    Args:
        path (str | Path): JSONL trace written by `write_group_trace`.
        n_groups (int): Cap on groups read. 0 reads every group.

    Returns:
        list[RequestGroup]: Groups in file order.
    """
    groups: list[RequestGroup] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "output_lens" not in record:
                continue
            groups.append(
                RequestGroup(
                    group_id=int(record["group_id"]),
                    prompt_len=int(record["prompt_len"]),
                    output_lens=tuple(int(v) for v in record["output_lens"]),
                )
            )
            if n_groups and len(groups) >= n_groups:
                break
    return groups
```

Add `from dataclasses import dataclass` to the imports at the top of the file.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_workload_trace_groups.py -q && ruff check dryrun/workload/trace.py
```

Expected: 8 passed, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add dryrun/workload/trace.py tests/test_workload_trace_groups.py
git commit -m "feat: add joint group trace artifact I/O"
```

---

### Task 2: GroupedWorkload as the simulator's workload source

**Files:**
- Create: `dryrun/workload/grouped.py`
- Modify: `dryrun/workload/__init__.py`
- Modify: `dryrun/component/rollout/engine/request.py` (add `group_id`)
- Modify: `dryrun/simulator/config.py` (replace `prompt_lengths` with `workload`)
- Modify: `dryrun/simulator/simulator.py:39-70, 120-127`
- Modify: `dryrun/cli/main.py:266, 291`
- Modify: `tests/test_e2e.py:201-210, 375-385`
- Test: `tests/test_workload_grouped.py`

**Interfaces:**
- Consumes: `RequestGroup`, `read_group_trace` from Task 1.
- Produces:
  - `GroupedWorkload` with `from_groups(groups)`, `from_lengths(prompt_lengths, output_lengths)`, `from_trace(path, n_groups=0)`, `n_requests` / `n_groups` properties, `mean_prompt_len` property, and `request(index: int) -> tuple[int, int, int]` returning `(group_id, prompt_len, output_len)`.
  - `SimConfig.workload: GroupedWorkload` replaces `SimConfig.prompt_lengths`.
  - `Simulator.__init__(rollout_cost, policy, cfg, ...)`: the `lengths` parameter is gone; the workload comes from `cfg.workload`.
  - `Request.group_id: int = 0`.

An ungrouped workload is genuinely a workload of groups of size 1, so `from_lengths` is the universal representation rather than a compatibility shim.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_workload_grouped.py`:

```python
"""Tests for the grouped workload source."""

from __future__ import annotations

import pytest

from dryrun.workload.grouped import GroupedWorkload
from dryrun.workload.trace import RequestGroup, write_group_trace


def _groups() -> list[RequestGroup]:
    return [
        RequestGroup(group_id=0, prompt_len=100, output_lens=(11, 12)),
        RequestGroup(group_id=1, prompt_len=200, output_lens=(21, 22)),
    ]


class TestFromGroups:
    def test_counts(self):
        workload = GroupedWorkload.from_groups(_groups())
        assert workload.n_groups == 2
        assert workload.n_requests == 4

    def test_requests_are_group_ordered(self):
        workload = GroupedWorkload.from_groups(_groups())
        assert [workload.request(i) for i in range(4)] == [
            (0, 100, 11),
            (0, 100, 12),
            (1, 200, 21),
            (1, 200, 22),
        ]

    def test_prompt_length_is_shared_within_a_group(self):
        workload = GroupedWorkload.from_groups(_groups())
        assert workload.request(0)[1] == workload.request(1)[1]

    def test_wraparound_restarts_at_a_group_boundary(self):
        workload = GroupedWorkload.from_groups(_groups())
        assert workload.request(4) == workload.request(0)
        assert workload.request(5) == workload.request(1)

    def test_mean_prompt_len_weights_every_request(self):
        workload = GroupedWorkload.from_groups(_groups())
        assert workload.mean_prompt_len == pytest.approx(150.0)

    def test_empty_workload_is_rejected(self):
        with pytest.raises(ValueError, match="at least one group"):
            GroupedWorkload.from_groups([])


class TestFromLengths:
    def test_each_response_is_its_own_group(self):
        workload = GroupedWorkload.from_lengths([10, 20], [1, 2])
        assert workload.n_groups == 2
        assert workload.n_requests == 2
        assert [workload.request(i) for i in range(2)] == [(0, 10, 1), (1, 20, 2)]

    def test_uneven_lists_cycle_independently(self):
        workload = GroupedWorkload.from_lengths([10], [1, 2, 3])
        assert workload.n_requests == 3
        assert [workload.request(i)[1] for i in range(3)] == [10, 10, 10]
        assert [workload.request(i)[2] for i in range(3)] == [1, 2, 3]


class TestFromTrace:
    def test_reads_a_written_trace(self, tmp_path):
        path = write_group_trace(_groups(), tmp_path / "joint.jsonl", metadata={})
        workload = GroupedWorkload.from_trace(path)
        assert workload.n_requests == 4
        assert workload.request(2) == (1, 200, 21)

    def test_n_groups_caps_the_read(self, tmp_path):
        path = write_group_trace(_groups(), tmp_path / "joint.jsonl", metadata={})
        assert GroupedWorkload.from_trace(path, n_groups=1).n_requests == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_workload_grouped.py -q
```

Expected: FAIL: `ModuleNotFoundError: No module named 'dryrun.workload.grouped'`.

- [ ] **Step 3: Create `dryrun/workload/grouped.py`**

```python
"""The simulator's workload source: groups of requests sharing one prompt.

DAPO samples `n` responses per prompt and computes group-relative advantages,
so a group is a structural unit of the workload. Holding the workload as an
ordered list of groups keeps group boundaries in the data: request wraparound
lands on a boundary instead of splitting a group, which is what lets whole-group
batch consumption and shared-prefix prefill build on this later.
"""

from __future__ import annotations

from pathlib import Path

from .trace import RequestGroup, read_group_trace


class GroupedWorkload:
    """An ordered, cycling sequence of requests grouped by shared prompt."""

    def __init__(self, groups: list[RequestGroup]):
        if not groups:
            raise ValueError("A grouped workload needs at least one group.")
        self.groups = groups
        # Flatten once so `request()` is a list index rather than a scan, and so
        # wraparound is modulo the whole group-ordered sequence.
        self._requests: list[tuple[int, int, int]] = [
            (group.group_id, group.prompt_len, output_len)
            for group in groups
            for output_len in group.output_lens
        ]

    @classmethod
    def from_groups(cls, groups: list[RequestGroup]) -> "GroupedWorkload":
        """Build from explicit groups."""
        return cls(list(groups))

    @classmethod
    def from_lengths(
        cls,
        prompt_lengths: list[int],
        output_lengths: list[int],
    ) -> "GroupedWorkload":
        """
        Build a workload of single-response groups from two length lists.

        An ungrouped workload is a workload whose groups hold one response each.
        The two lists cycle independently, so lists of different lengths pair as
        `prompt_lengths[i % len(prompt_lengths)]` with
        `output_lengths[i % len(output_lengths)]`.

        Args:
            prompt_lengths (list[int]): Prompt lengths to cycle.
            output_lengths (list[int]): Response lengths to cycle.

        Returns:
            GroupedWorkload: One group per request.
        """
        if not prompt_lengths or not output_lengths:
            raise ValueError("Both prompt_lengths and output_lengths must be non-empty.")
        n = max(len(prompt_lengths), len(output_lengths))
        return cls(
            [
                RequestGroup(
                    group_id=i,
                    prompt_len=int(prompt_lengths[i % len(prompt_lengths)]),
                    output_lens=(int(output_lengths[i % len(output_lengths)]),),
                )
                for i in range(n)
            ]
        )

    @classmethod
    def from_trace(cls, path: str | Path, n_groups: int = 0) -> "GroupedWorkload":
        """Build from a joint group trace artifact."""
        return cls(read_group_trace(path, n_groups=n_groups))

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def n_requests(self) -> int:
        return len(self._requests)

    @property
    def group_size(self) -> int:
        return len(self.groups[0].output_lens)

    @property
    def mean_prompt_len(self) -> float:
        """Mean prompt length per request, so larger groups weigh more."""
        return sum(prompt_len for _, prompt_len, _ in self._requests) / len(self._requests)

    def request(self, index: int) -> tuple[int, int, int]:
        """
        Return `(group_id, prompt_len, output_len)` for a request index.

        Indices beyond the workload wrap around. Because the sequence is
        group-ordered and complete, wraparound restarts at a group boundary.
        """
        return self._requests[index % len(self._requests)]
```

- [ ] **Step 4: Export it**

In `dryrun/workload/__init__.py`, add the import and `__all__` entry:

```python
from .grouped import GroupedWorkload
```

Add `"GroupedWorkload",` to `__all__` (keep the list alphabetical).

- [ ] **Step 5: Add `group_id` to `Request`**

In `dryrun/component/rollout/engine/request.py`, add the field directly after `v_traj: int`:

```python
    group_id: int = 0
```

- [ ] **Step 6: Replace `prompt_lengths` in `SimConfig`**

In `dryrun/simulator/config.py`, add the import and swap the field. Delete the `prompt_lengths` line entirely: no alias.

```python
from ..workload.grouped import GroupedWorkload
```

Replace `prompt_lengths: list[int] = field(default_factory=lambda: [512])` with:

```python
    workload: GroupedWorkload = field(
        default_factory=lambda: GroupedWorkload.from_lengths([512], [100])
    )
```

- [ ] **Step 7: Point the simulator at `cfg.workload`**

In `dryrun/simulator/simulator.py`, delete the `lengths: list[int],` parameter from `__init__`, and replace these three assignments:

```python
        self.lengths = list(lengths)
        self.prompt_lengths = list(cfg.prompt_lengths)
```

with:

```python
        self.workload = cfg.workload
```

Replace the `mean_prompt_len` line:

```python
        mean_prompt_len = sum(self.prompt_lengths) / len(self.prompt_lengths)
```

with:

```python
        mean_prompt_len = self.workload.mean_prompt_len
```

Replace `_next_length` and `_next_prompt_length` (lines 120-127) with one group-aware accessor:

```python
    def _next_request_shape(self) -> tuple[int, int, int]:
        """Return `(group_id, prompt_len, target_len)` for the next request.

        Both lengths come from the same workload group, so the prompt/response
        pairing is the one that was observed rather than a product of two
        independent modulo lookups.
        """
        group_id, prompt_len, output_len = self.workload.request(self.next_rid)
        return (
            group_id,
            clip_length(prompt_len, self.cfg.max_prompt_length),
            clip_length(output_len, self.cfg.max_output_length),
        )
```

In `_admit`, replace the `Request(...)` construction with:

```python
            group_id, prompt_len, target_len = self._next_request_shape()
            req = Request(
                rid=self.next_rid,
                group_id=group_id,
                prompt_len=prompt_len,
                target_len=target_len,
                v_traj=self.rollout_version,
                admit_time=self.t,
            )
```

- [ ] **Step 8: Update the CLI**

In `dryrun/cli/main.py`, `build_workload` now returns a `GroupedWorkload`. Change its body and the call site. Replace the function (line ~168):

```python
def build_workload(cfg: DictConfig) -> GroupedWorkload:
    """Build the simulator's workload source from config."""
    return GroupedWorkload.from_lengths(
        build_distribution(cfg.prompt),
        build_distribution(cfg.output),
    )
```

Add the import `from ..workload.grouped import GroupedWorkload`. At line 266 replace:

```python
    prompt_lengths, output_lengths = build_workload(cfg.workload)
```

with:

```python
    workload = build_workload(cfg.workload)
```

At line 291 replace `prompt_lengths=prompt_lengths,` with `workload=workload,`. Then in the `Simulator(...)` construction remove the `lengths=output_lengths,` argument.

- [ ] **Step 9: Update the two `prompt_lengths` test sites**

In `tests/test_e2e.py`, `_run_sim` no longer passes `lengths`. Change the helper:

```python
def _run_sim(policy, lengths, cfg):
    cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7)
    training_cost = TrainingCostModel(
        FixedTrainingLatency(0.5),
        FixedTrainingMemory(0),
    )
    if lengths is not None:
        cfg.workload = GroupedWorkload.from_lengths(
            [group.prompt_len for group in cfg.workload.groups], lengths
        )
    sim = Simulator(
        rollout_cost=cost,
        training_cost=training_cost,
        policy=policy,
        cfg=cfg,
    )
```

Add `from dryrun.workload.grouped import GroupedWorkload` to the test imports.

At line ~201, replace the `prompt_lengths=prompt_lengths,` argument with:

```python
            workload=GroupedWorkload.from_lengths(prompt_lengths, uniform(100, 50, 100, seed=0)),
```

and pass `None` for `lengths` in that test's `_run_sim` call.

At line ~379, replace `prompt_lengths=[100, 4000, 50],` with:

```python
            workload=GroupedWorkload.from_lengths([100, 4000, 50], [500, 400, 10]),
```

and pass `None` for `lengths` in that `_run_sim` call.

- [ ] **Step 9b: Update the `build_workload` test for the new return type**

`build_workload` now returns one `GroupedWorkload` instead of a tuple, so
`tests/test_config.py:test_build_workload_prompt_and_output` (line ~189) no
longer unpacks. Replace it with:

```python
    def test_build_workload_prompt_and_output(self):
        cfg = OmegaConf.create({
            "output": {"name": "uniform", "n": 20, "seed": 1, "lo": 50, "hi": 100},
            "prompt": {"name": "fixed", "n": 20, "seed": 1, "lo": 512},
        })
        workload = build_workload(cfg)
        assert workload.n_requests == 20
        assert {workload.request(i)[1] for i in range(20)} == {512}
```

- [ ] **Step 10: Run the full suite**

```bash
pytest tests/test_workload_grouped.py tests/test_e2e.py tests/test_config.py -q
```

Expected: all pass. Fix any other `SimConfig(prompt_lengths=...)` or `Simulator(lengths=...)` call site the failures reveal: `grep -rn "prompt_lengths\|lengths=" tests/ dryrun/` should return no stale uses.

- [ ] **Step 11: Commit**

```bash
git add dryrun/workload/grouped.py dryrun/workload/__init__.py \
        dryrun/component/rollout/engine/request.py dryrun/simulator/config.py \
        dryrun/simulator/simulator.py dryrun/cli/main.py \
        tests/test_workload_grouped.py tests/test_e2e.py
git commit -m "feat: make a grouped workload the simulator's request source"
```

---

### Task 3: Per-response length collection with n

**Files:**
- Modify: `dryrun/workload/inference.py`
- Test: `tests/test_workload_group_collection.py`

**Interfaces:**
- Consumes: `RequestGroup`, `write_group_trace` (Task 1).
- Produces:
  - `load_unique_prompt_messages(dataset_path, sample_size, seed, index_key="index") -> list[list[dict[str, Any]]]` in `dryrun/workload/fitting.py`
  - `parse_group_response(payload: dict[str, Any], group_id: int) -> RequestGroup` in `dryrun/workload/inference.py`
  - `build_completion_payload(model, prompt, max_tokens, n, temperature, top_p) -> dict[str, Any]` in `dryrun/workload/inference.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_workload_group_collection.py`:

```python
"""Tests for per-response group collection from a vLLM completions payload."""

from __future__ import annotations

import pytest

from dryrun.workload.inference import build_completion_payload, parse_group_response
from dryrun.workload.trace import RequestGroup


def _payload(lengths: list[int], prompt_len: int = 5) -> dict:
    return {
        "choices": [
            {
                "token_ids": list(range(length)),
                "prompt_token_ids": list(range(prompt_len)),
                "finish_reason": "length",
            }
            for length in lengths
        ],
        "usage": {"completion_tokens": sum(lengths), "prompt_tokens": prompt_len},
    }


class TestParseGroupResponse:
    def test_uses_per_choice_token_ids_not_usage(self):
        # usage.completion_tokens is the SUM over choices (60 here). Reading it
        # instead of per-choice token_ids would record one 60-token response.
        group = parse_group_response(_payload([10, 20, 30]), group_id=7)
        assert group == RequestGroup(group_id=7, prompt_len=5, output_lens=(10, 20, 30))

    def test_prompt_len_comes_from_prompt_token_ids(self):
        group = parse_group_response(_payload([1, 2], prompt_len=196), group_id=0)
        assert group.prompt_len == 196

    def test_missing_token_ids_is_an_error(self):
        payload = {"choices": [{"finish_reason": "length"}], "usage": {}}
        with pytest.raises(ValueError, match="return_token_ids"):
            parse_group_response(payload, group_id=0)

    def test_empty_choices_is_an_error(self):
        with pytest.raises(ValueError, match="no choices"):
            parse_group_response({"choices": [], "usage": {}}, group_id=0)


class TestBuildCompletionPayload:
    def test_requests_token_ids_and_n(self):
        payload = build_completion_payload(
            model="/models/Qwen3-8B",
            prompt="hi",
            max_tokens=10240,
            n=8,
            temperature=1.0,
            top_p=1.0,
        )
        assert payload["n"] == 8
        assert payload["return_token_ids"] is True
        assert payload["max_tokens"] == 10240
        assert payload["temperature"] == 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_workload_group_collection.py -q
```

Expected: FAIL: `ImportError: cannot import name 'build_completion_payload'`.

- [ ] **Step 3: Add the deduplicating loader**

In `dryrun/workload/fitting.py`, add alongside `load_prompt_messages`:

```python
def load_unique_prompt_messages(
    dataset_path: str | Path,
    sample_size: int | None,
    seed: int,
    index_key: str = "index",
) -> list[list[dict[str, Any]]]:
    """
    Read distinct prompts from a parquet dataset, optionally subsampled.

    dapo-math-17k repeats every prompt ~100 times (1.79M rows over 17,917
    distinct problems), so sampling rows draws mostly duplicates. Prompts are
    deduplicated on `extra_info[index_key]` before subsampling, which makes one
    sampled prompt one distinct problem.

    Args:
        dataset_path (str | Path): Parquet dataset with `prompt` and
            `extra_info` columns.
        sample_size (int | None): If set, sample this many distinct prompts.
        seed (int): RNG seed for subsampling.
        index_key (str): Field inside `extra_info` holding the unique id.

    Returns:
        list[list[dict[str, Any]]]: Chat-message lists, one per distinct prompt.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pq.read_table(dataset_path, columns=["prompt", "extra_info"])
    prompts = table.column("prompt").to_pylist()
    extra_info = table.column("extra_info").to_pylist()

    unique: dict[str, list[dict[str, Any]]] = {}
    for messages, info in zip(prompts, extra_info):
        key = str(info[index_key])
        if key not in unique:
            unique[key] = messages
    messages_list = [unique[key] for key in sorted(unique)]
    dryrun_logger.info(
        f"Found {len(messages_list)} distinct prompts among {len(prompts)} rows."
    )

    if sample_size is not None and sample_size < len(messages_list):
        rng = random.Random(seed)
        indices = rng.sample(range(len(messages_list)), sample_size)
        messages_list = [messages_list[i] for i in sorted(indices)]
    return messages_list
```

- [ ] **Step 4: Add the payload builder and parser**

In `dryrun/workload/inference.py`, add:

```python
def build_completion_payload(
    model: str,
    prompt: str,
    max_tokens: int,
    n: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    """
    Build a `/v1/completions` payload that reports per-response token ids.

    `return_token_ids` is what makes per-response lengths available:
    `usage.completion_tokens` is the sum across all `n` choices.
    """
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "n": n,
        "temperature": temperature,
        "top_p": top_p,
        "return_token_ids": True,
    }


def parse_group_response(payload: dict[str, Any], group_id: int) -> RequestGroup:
    """
    Turn one completions response into a `RequestGroup`.

    Args:
        payload (dict[str, Any]): Decoded `/v1/completions` response.
        group_id (int): Identifier to assign to this group.

    Returns:
        RequestGroup: Prompt length and one response length per choice.
    """
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError(f"Completions response for group {group_id} had no choices.")
    for choice in choices:
        if choice.get("token_ids") is None:
            raise ValueError(
                "Completions response is missing token_ids; "
                "the request must set return_token_ids to true."
            )
    prompt_token_ids = choices[0].get("prompt_token_ids")
    if prompt_token_ids is None:
        raise ValueError(
            "Completions response is missing prompt_token_ids; "
            "the request must set return_token_ids to true."
        )
    return RequestGroup(
        group_id=group_id,
        prompt_len=len(prompt_token_ids),
        output_lens=tuple(len(choice["token_ids"]) for choice in choices),
    )
```

Add `from .trace import RequestGroup, write_group_trace` to the imports (the module already imports `write_length_trace`; keep it, the single-dimension `collect_output_lengths` path still uses it).

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest tests/test_workload_group_collection.py -q && ruff check dryrun/workload/
```

Expected: 6 passed, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add dryrun/workload/inference.py dryrun/workload/fitting.py \
        tests/test_workload_group_collection.py
git commit -m "feat: parse per-response group lengths from completions responses"
```

---

### Task 4: Deterministic worker sharding and merge

**Files:**
- Create: `dryrun/workload/sharding.py`
- Test: `tests/test_workload_sharding.py`

**Interfaces:**
- Consumes: `RequestGroup`, `read_group_trace`, `write_group_trace` (Task 1).
- Produces:
  - `Worker(index: int, host: str, gpu: int, port: int)` frozen dataclass
  - `parse_hosts_file(path: str | Path) -> list[tuple[str, int]]`
  - `build_workers(hosts: list[tuple[str, int]], base_port: int = 8700) -> list[Worker]`
  - `shard_indices(n_items: int, worker_index: int, n_workers: int) -> list[int]`
  - `shard_path(shard_dir: str | Path, worker_index: int) -> Path`
  - `merge_shards(shard_dir: str | Path, n_workers: int, output_path: str | Path, metadata: dict[str, Any]) -> Path`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_workload_sharding.py`:

```python
"""Tests for distributed collection sharding and shard merging."""

from __future__ import annotations

import pytest

from dryrun.workload.sharding import (
    Worker,
    build_workers,
    merge_shards,
    parse_hosts_file,
    shard_indices,
    shard_path,
)
from dryrun.workload.trace import RequestGroup, read_group_trace, write_group_trace


class TestParseHostsFile:
    def test_parses_host_and_gpu_count(self, tmp_path):
        path = tmp_path / "hosts.txt"
        path.write_text("# host [gpus]\n29.162.247.148 8\n28.49.16.220 4\n")
        assert parse_hosts_file(path) == [("29.162.247.148", 8), ("28.49.16.220", 4)]

    def test_gpu_count_defaults_to_one(self, tmp_path):
        path = tmp_path / "hosts.txt"
        path.write_text("10.0.0.1\n")
        assert parse_hosts_file(path) == [("10.0.0.1", 1)]

    def test_blank_lines_and_comments_are_skipped(self, tmp_path):
        path = tmp_path / "hosts.txt"
        path.write_text("\n# comment\n10.0.0.1 2\n\n")
        assert parse_hosts_file(path) == [("10.0.0.1", 2)]

    def test_empty_file_is_an_error(self, tmp_path):
        path = tmp_path / "hosts.txt"
        path.write_text("# only a comment\n")
        with pytest.raises(ValueError, match="no hosts"):
            parse_hosts_file(path)


class TestBuildWorkers:
    def test_one_worker_per_gpu_with_distinct_ports_per_host(self):
        workers = build_workers([("a", 2), ("b", 2)], base_port=8700)
        assert workers == [
            Worker(index=0, host="a", gpu=0, port=8700),
            Worker(index=1, host="a", gpu=1, port=8701),
            Worker(index=2, host="b", gpu=0, port=8700),
            Worker(index=3, host="b", gpu=1, port=8701),
        ]


class TestShardIndices:
    def test_shards_partition_every_item_exactly_once(self):
        n_items, n_workers = 10, 4
        seen = [i for w in range(n_workers) for i in shard_indices(n_items, w, n_workers)]
        assert sorted(seen) == list(range(n_items))

    def test_shard_is_deterministic(self):
        assert shard_indices(10, 1, 4) == shard_indices(10, 1, 4) == [1, 5, 9]

    def test_more_workers_than_items_yields_empty_shards(self):
        assert shard_indices(2, 3, 4) == []


class TestMergeShards:
    def _write_shard(self, shard_dir, worker_index, groups):
        return write_group_trace(
            groups, shard_path(shard_dir, worker_index), metadata={"max_output_tokens": 100}
        )

    def test_merges_in_group_id_order(self, tmp_path):
        self._write_shard(tmp_path, 0, [RequestGroup(0, 10, (1, 2)), RequestGroup(2, 30, (5, 6))])
        self._write_shard(tmp_path, 1, [RequestGroup(1, 20, (3, 4))])
        out = merge_shards(tmp_path, 2, tmp_path / "joint.jsonl", {"max_output_tokens": 100})
        assert [g.group_id for g in read_group_trace(out)] == [0, 1, 2]

    def test_missing_shard_is_named(self, tmp_path):
        self._write_shard(tmp_path, 0, [RequestGroup(0, 10, (1, 2))])
        with pytest.raises(FileNotFoundError, match="worker-01"):
            merge_shards(tmp_path, 2, tmp_path / "joint.jsonl", {})
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_workload_sharding.py -q
```

Expected: FAIL: `ModuleNotFoundError: No module named 'dryrun.workload.sharding'`.

- [ ] **Step 3: Create `dryrun/workload/sharding.py`**

```python
"""Sharding for distributed workload collection.

A worker is one `(host, gpu)` pair addressed by a flat index. Prompts are split
as `prompts[index:n_workers]`, so a worker's shard is a function of its index
alone: rerunning worker 7 reproduces exactly worker 7's prompts, and one failed
shard retries without disturbing the others.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._logging import dryrun_logger
from .trace import RequestGroup, read_group_trace, write_group_trace


@dataclass(frozen=True)
class Worker:
    """One collection worker: a GPU on a host, serving on its own port."""

    index: int
    host: str
    gpu: int
    port: int


def parse_hosts_file(path: str | Path) -> list[tuple[str, int]]:
    """
    Read a hosts file of `host [gpus]` lines.

    Blank lines and `#` comments are ignored. A line without a GPU count is
    treated as a single-GPU host.

    Args:
        path (str | Path): Hosts file to read.

    Returns:
        list[tuple[str, int]]: `(host, gpu_count)` in file order.
    """
    hosts: list[tuple[str, int]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        hosts.append((parts[0], int(parts[1]) if len(parts) > 1 else 1))
    if not hosts:
        raise ValueError(f"Hosts file {str(path)!r} lists no hosts.")
    return hosts


def build_workers(hosts: list[tuple[str, int]], base_port: int = 8700) -> list[Worker]:
    """
    Expand `(host, gpu_count)` pairs into one worker per GPU.

    Ports restart at `base_port` on each host, since each host has its own
    port space.

    Args:
        hosts (list[tuple[str, int]]): `(host, gpu_count)` pairs.
        base_port (int): First port used on each host.

    Returns:
        list[Worker]: Workers in host-then-GPU order, indexed from zero.
    """
    workers: list[Worker] = []
    for host, gpu_count in hosts:
        for gpu in range(gpu_count):
            workers.append(
                Worker(index=len(workers), host=host, gpu=gpu, port=base_port + gpu)
            )
    return workers


def shard_indices(n_items: int, worker_index: int, n_workers: int) -> list[int]:
    """Return the item indices assigned to `worker_index` by strided split."""
    return list(range(worker_index, n_items, n_workers))


def shard_path(shard_dir: str | Path, worker_index: int) -> Path:
    """Return the shard artifact path for one worker."""
    return Path(shard_dir) / f"worker-{worker_index:02d}.jsonl"


def merge_shards(
    shard_dir: str | Path,
    n_workers: int,
    output_path: str | Path,
    metadata: dict[str, Any],
) -> Path:
    """
    Merge every worker shard into one joint trace ordered by `group_id`.

    Every expected shard must exist: a missing shard is named rather than
    silently producing a short trace.

    Args:
        shard_dir (str | Path): Directory holding `worker-NN.jsonl` shards.
        n_workers (int): Number of shards expected.
        output_path (str | Path): Destination joint trace path.
        metadata (dict[str, Any]): Provenance for the merged `_meta` header.

    Returns:
        Path: The merged trace path.
    """
    groups: list[RequestGroup] = []
    for worker_index in range(n_workers):
        path = shard_path(shard_dir, worker_index)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing shard for worker-{worker_index:02d} at {str(path)!r}. "
                f"Rerun that worker before merging."
            )
        groups.extend(read_group_trace(path))
    groups.sort(key=lambda group: group.group_id)
    dryrun_logger.info(f"Merged {len(groups)} groups from {n_workers} shards.")
    return write_group_trace(groups, output_path, metadata=metadata)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_workload_sharding.py -q && ruff check dryrun/workload/sharding.py
```

Expected: 11 passed, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add dryrun/workload/sharding.py tests/test_workload_sharding.py
git commit -m "feat: add deterministic worker sharding and shard merging"
```

---

### Task 5: Group collection driver and CLI

**Files:**
- Modify: `dryrun/workload/inference.py` (add `collect_group_trace`)
- Modify: `dryrun/cli/workload.py` (add `collect-groups` and `merge-shards`)
- Test: `tests/test_workload_group_driver.py`

**Interfaces:**
- Consumes: Tasks 1, 3, 4.
- Produces:
  - `collect_group_trace(dataset_path, tokenizer_path, model_path, output_path, worker_index, n_workers, sample_size, n, max_tokens, temperature, top_p, tp, concurrency, port, seed, server_ready_timeout, extra_serve_args=None, gpu=None) -> Path`
  - CLI: `dryrun-workload collect-groups`, `dryrun-workload merge-shards`

- [ ] **Step 1: Write the failing test**

Create `tests/test_workload_group_driver.py`:

```python
"""Tests for the group collection driver's prompt selection and CLI wiring."""

from __future__ import annotations

from dryrun.cli.workload import build_parser


class TestCollectGroupsCLI:
    def test_parses_worker_and_group_options(self):
        args = build_parser().parse_args(
            [
                "collect-groups",
                "--dataset", "d.parquet",
                "--tokenizer", "/tok",
                "--model", "/model",
                "--output", "shard.jsonl",
                "--worker-index", "3",
                "--n-workers", "16",
                "--sample-size", "1024",
                "--n", "8",
                "--max-tokens", "10240",
                "--gpu", "3",
            ]
        )
        assert args.worker_index == 3
        assert args.n_workers == 16
        assert args.n == 8
        assert args.max_tokens == 10240
        assert args.gpu == 3

    def test_defaults_match_the_dapo_recipe(self):
        args = build_parser().parse_args(
            [
                "collect-groups",
                "--dataset", "d.parquet",
                "--tokenizer", "/tok",
                "--model", "/model",
                "--output", "shard.jsonl",
            ]
        )
        assert args.n == 8
        assert args.temperature == 1.0
        assert args.top_p == 1.0
        assert args.worker_index == 0
        assert args.n_workers == 1


class TestMergeShardsCLI:
    def test_parses_merge_options(self):
        args = build_parser().parse_args(
            [
                "merge-shards",
                "--shard-dir", "shards",
                "--n-workers", "16",
                "--output", "joint.jsonl",
                "--max-output-tokens", "10240",
            ]
        )
        assert args.n_workers == 16
        assert args.max_output_tokens == 10240
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_workload_group_driver.py -q
```

Expected: FAIL: `argument command: invalid choice: 'collect-groups'`.

- [ ] **Step 3: Add the driver**

In `dryrun/workload/inference.py`, add. Note `_VLLMServer` gains a `gpu` argument so a worker can pin `CUDA_VISIBLE_DEVICES`; add `gpu: int | None = None` to its `__init__`, store it, and in `__enter__` pass an environment to `subprocess.Popen`:

```python
        env = None
        if self.gpu is not None:
            import os  # noqa: PLC0415

            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(self.gpu)}
        self.process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
```

Then the driver itself:

```python
async def _send_group_requests(
    base_url: str,
    served_model_name: str,
    indexed_prompts: list[tuple[int, str]],
    max_tokens: int,
    n: int,
    temperature: float,
    top_p: float,
    concurrency: int,
) -> list[RequestGroup]:
    """Send prompts with bounded concurrency, one group per prompt."""
    import aiohttp  # noqa: PLC0415

    semaphore = asyncio.Semaphore(concurrency)
    url = f"{base_url}/v1/completions"

    async def _one_group(
        session: "aiohttp.ClientSession",
        group_id: int,
        prompt: str,
    ) -> RequestGroup:
        payload = build_completion_payload(
            model=served_model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            n=n,
            temperature=temperature,
            top_p=top_p,
        )
        async with semaphore, session.post(url, json=payload) as response:
            data = await response.json()
        return parse_group_response(data, group_id=group_id)

    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        return list(
            await asyncio.gather(
                *(_one_group(session, gid, prompt) for gid, prompt in indexed_prompts)
            )
        )


def collect_group_trace(
    dataset_path: str | Path,
    tokenizer_path: str,
    model_path: str,
    output_path: str | Path,
    worker_index: int = 0,
    n_workers: int = 1,
    sample_size: int = 1024,
    n: int = 8,
    max_tokens: int = 10240,
    temperature: float = 1.0,
    top_p: float = 1.0,
    tp: int = 1,
    concurrency: int = 16,
    port: int = 8700,
    seed: int = 42,
    server_ready_timeout: int = 1800,
    extra_serve_args: dict[str, Any] | None = None,
    gpu: int | None = None,
    log_path: str | Path | None = None,
) -> Path:
    """
    Collect this worker's share of a joint group trace.

    Samples `sample_size` distinct prompts, keeps the strided shard belonging to
    `worker_index`, serves the model on one GPU, and records one group of `n`
    response lengths per prompt. `group_id` is the prompt's index in the full
    sampled list, so merged shards reassemble in a stable order.

    Args:
        dataset_path (str | Path): Parquet dataset with `prompt` and `extra_info`.
        tokenizer_path (str): Tokenizer used to render the chat template.
        model_path (str): Model passed to `vllm serve` and the request body.
        output_path (str | Path): Destination shard artifact.
        worker_index (int): This worker's flat index.
        n_workers (int): Total number of workers.
        sample_size (int): Distinct prompts sampled across all workers.
        n (int): Responses sampled per prompt.
        max_tokens (int): Generation cap per response.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling parameter.
        tp (int): Tensor-parallel size for the served model.
        concurrency (int): Maximum in-flight prompts for this worker.
        port (int): Local port for this worker's server.
        seed (int): RNG seed for prompt subsampling.
        server_ready_timeout (int): Seconds to wait for `/health`.
        extra_serve_args (dict[str, Any] | None): Extra `vllm serve` flags.
        gpu (int | None): GPU pinned via `CUDA_VISIBLE_DEVICES`.
        log_path (str | Path | None): vLLM server log path.

    Returns:
        Path: Path to this worker's shard artifact.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    from .fitting import load_unique_prompt_messages  # noqa: PLC0415
    from .sharding import shard_indices  # noqa: PLC0415

    output = Path(output_path)
    server_log = Path(log_path) if log_path is not None else output.with_suffix(".server.log")

    dryrun_logger.info(f"Loading tokenizer from {tokenizer_path!r}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    messages_list = load_unique_prompt_messages(dataset_path, sample_size, seed)
    indices = shard_indices(len(messages_list), worker_index, n_workers)
    if not indices:
        raise ValueError(
            f"Worker {worker_index} of {n_workers} has no prompts to collect "
            f"from {len(messages_list)} sampled prompts."
        )
    indexed_prompts = [
        (index, render_prompt(tokenizer, messages_list[index])) for index in indices
    ]
    dryrun_logger.info(
        f"Worker {worker_index}/{n_workers} collecting {len(indexed_prompts)} groups "
        f"of n={n} on GPU {gpu!r}."
    )

    with _VLLMServer(
        model_path=model_path,
        port=port,
        tp=tp,
        extra_serve_args=extra_serve_args or {},
        ready_timeout=server_ready_timeout,
        log_path=server_log,
        gpu=gpu,
    ) as server:
        groups = asyncio.run(
            _send_group_requests(
                server.base_url,
                model_path,
                indexed_prompts,
                max_tokens,
                n,
                temperature,
                top_p,
                concurrency,
            )
        )

    written = write_group_trace(
        groups,
        output,
        metadata={
            "kind": "joint_group_trace",
            "dataset": str(dataset_path),
            "tokenizer": str(tokenizer_path),
            "model": str(model_path),
            "tp": tp,
            "n": n,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "worker_index": worker_index,
            "n_workers": n_workers,
        },
    )
    dryrun_logger.info(f"Worker {worker_index} wrote {len(groups)} groups to {written}.")
    return written
```

- [ ] **Step 4: Add the CLI subcommands**

In `dryrun/cli/workload.py`, add two handlers and register them in `build_parser`:

```python
def cmd_collect_groups(args: argparse.Namespace) -> None:
    """Collect one worker's shard of a joint group trace."""
    from ..workload.inference import collect_group_trace  # noqa: PLC0415

    extra_serve_args = dict(item.split("=", 1) for item in args.serve_arg)
    path = collect_group_trace(
        dataset_path=args.dataset,
        tokenizer_path=args.tokenizer,
        model_path=args.model,
        output_path=args.output,
        worker_index=args.worker_index,
        n_workers=args.n_workers,
        sample_size=args.sample_size,
        n=args.n,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tp=args.tp,
        concurrency=args.concurrency,
        port=args.port,
        seed=args.seed,
        server_ready_timeout=args.server_ready_timeout,
        extra_serve_args=extra_serve_args,
        gpu=args.gpu,
    )
    dryrun_logger.info(f"Group trace shard written to {path}.")


def cmd_merge_shards(args: argparse.Namespace) -> None:
    """Merge worker shards into one joint group trace."""
    from ..workload.sharding import merge_shards  # noqa: PLC0415

    path = merge_shards(
        shard_dir=args.shard_dir,
        n_workers=args.n_workers,
        output_path=args.output,
        metadata={"kind": "joint_group_trace", "max_output_tokens": args.max_output_tokens},
    )
    dryrun_logger.info(f"Merged group trace written to {path}.")
```

In `build_parser`, after the `fit_output` block:

```python
    collect_groups = subs.add_parser(
        "collect-groups",
        help="Collect one worker's shard of a joint (prompt, n responses) trace.",
    )
    collect_groups.add_argument("--dataset", required=True, help="Parquet dataset path.")
    collect_groups.add_argument("--tokenizer", required=True, help="Tokenizer name or path.")
    collect_groups.add_argument("--model", required=True, help="Model passed to `vllm serve`.")
    collect_groups.add_argument("--output", required=True, help="Output shard JSONL path.")
    collect_groups.add_argument("--worker-index", type=int, default=0, help="This worker's index.")
    collect_groups.add_argument("--n-workers", type=int, default=1, help="Total worker count.")
    collect_groups.add_argument(
        "--sample-size", type=int, default=1024, help="Distinct prompts across all workers."
    )
    collect_groups.add_argument("--n", type=int, default=8, help="Responses sampled per prompt.")
    collect_groups.add_argument("--max-tokens", type=int, default=10240)
    collect_groups.add_argument("--temperature", type=float, default=1.0)
    collect_groups.add_argument("--top-p", type=float, default=1.0)
    collect_groups.add_argument("--tp", type=int, default=1, help="Tensor-parallel size.")
    collect_groups.add_argument("--concurrency", type=int, default=16)
    collect_groups.add_argument("--port", type=int, default=8700)
    collect_groups.add_argument("--seed", type=int, default=42)
    collect_groups.add_argument("--gpu", type=int, default=None, help="GPU to pin this worker to.")
    collect_groups.add_argument("--server-ready-timeout", type=int, default=1800)
    collect_groups.add_argument(
        "--serve-arg", action="append", default=[], metavar="KEY=VALUE",
        help="Extra `vllm serve` flag. Repeatable.",
    )
    collect_groups.set_defaults(func=cmd_collect_groups)

    merge = subs.add_parser("merge-shards", help="Merge worker shards into one joint trace.")
    merge.add_argument("--shard-dir", required=True, help="Directory of worker-NN.jsonl shards.")
    merge.add_argument("--n-workers", type=int, required=True, help="Shards expected.")
    merge.add_argument("--output", required=True, help="Merged trace path.")
    merge.add_argument("--max-output-tokens", type=int, default=10240)
    merge.set_defaults(func=cmd_merge_shards)
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
pytest tests/test_workload_group_driver.py -q && ruff check dryrun/cli/workload.py dryrun/workload/inference.py
```

Expected: 3 passed, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add dryrun/workload/inference.py dryrun/cli/workload.py tests/test_workload_group_driver.py
git commit -m "feat: add group trace collection driver and CLI"
```

---

### Task 6: `workload.source` config switch

**Files:**
- Modify: `dryrun/config/schema.py:54-86`
- Create: `dryrun/config/defaults/workload/dapo_trace.yaml`
- Modify: `dryrun/config/defaults/workload/{normal,uniform,bimodal,lognormal,powerlaw,from_trace}.yaml`
- Modify: `dryrun/cli/main.py` (`build_workload`)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `GroupedWorkload` (Task 2).
- Produces: `WorkloadConfig.source: str`, `WorkloadConfig.trace: JointTraceConfig`; `build_workload(cfg) -> GroupedWorkload` dispatching on `source`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
class TestWorkloadSource:
    def test_distributions_source_builds_single_response_groups(self):
        cfg = OmegaConf.create({
            "source": "distributions",
            "output": {"name": "uniform", "n": 20, "seed": 1, "lo": 50, "hi": 100},
            "prompt": {"name": "fixed", "n": 20, "seed": 1, "lo": 512},
        })
        workload = build_workload(cfg)
        assert workload.group_size == 1
        assert workload.n_requests == 20

    def test_source_defaults_to_distributions(self):
        cfg = OmegaConf.create({
            "output": {"name": "uniform", "n": 4, "seed": 1, "lo": 50, "hi": 100},
            "prompt": {"name": "fixed", "n": 4, "seed": 1, "lo": 512},
        })
        assert build_workload(cfg).group_size == 1

    def test_trace_source_reads_groups(self, tmp_path):
        from dryrun.workload.trace import RequestGroup, write_group_trace

        path = write_group_trace(
            [RequestGroup(0, 100, (7, 8)), RequestGroup(1, 200, (9, 10))],
            tmp_path / "joint.jsonl",
            metadata={"max_output_tokens": 10},
        )
        cfg = OmegaConf.create({"source": "trace", "trace": {"path": str(path), "n_groups": 0}})
        workload = build_workload(cfg)
        assert workload.group_size == 2
        assert workload.n_requests == 4

    def test_trace_source_requires_a_path(self):
        cfg = OmegaConf.create({"source": "trace", "trace": {"path": "", "n_groups": 0}})
        with pytest.raises(ValueError, match="workload.trace.path is required"):
            build_workload(cfg)

    def test_unknown_source_is_rejected(self):
        cfg = OmegaConf.create({"source": "nonsense"})
        with pytest.raises(ValueError, match="Unknown workload source"):
            build_workload(cfg)
```

Also update the existing `test_build_workload_prompt_and_output` if Task 2 Step 9b
has not already done so; if it has, keep that version and only add the new
classes here.

A separate config-group smoke check belongs with the YAML, since these unit tests
never touch Hydra composition. Add to `tests/test_config.py`:

```python
class TestWorkloadConfigGroups:
    def test_dapo_trace_group_selects_the_trace_source(self):
        from hydra import compose, initialize_config_dir

        from dryrun.config import register_configs

        register_configs()
        config_dir = str(Path(dryrun.config.__file__).parent / "defaults")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="simulate", overrides=["workload=dapo_trace"])
        assert cfg.workload.source == "trace"
```

with `from pathlib import Path` and `import dryrun.config` added to the imports.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_config.py -k WorkloadSource -q
```

Expected: FAIL: `dapo_trace` config group not found.

- [ ] **Step 3: Extend the schema**

In `dryrun/config/schema.py`, add before `WorkloadConfig`:

```python
@dataclass
class JointTraceConfig:
    """Joint group trace source (see `dryrun-workload collect-groups`)."""

    path: str = ""
    n_groups: int = 0
```

Replace `WorkloadConfig` with:

```python
@dataclass
class WorkloadConfig:
    """Workload source: synthetic marginal distributions or a joint group trace."""

    source: str = "distributions"
    output: DistributionConfig = field(default_factory=DistributionConfig)
    prompt: DistributionConfig = field(
        default_factory=lambda: DistributionConfig(name="fixed", lo=512, hi=512)
    )
    trace: JointTraceConfig = field(default_factory=JointTraceConfig)
```

Export `JointTraceConfig` from `dryrun/config/__init__.py` alongside `WorkloadConfig` (import it and add to `__all__`).

- [ ] **Step 4: Create the trace config group**

Create `dryrun/config/defaults/workload/dapo_trace.yaml`:

```yaml
# Joint (prompt, n responses) workload read from a collected group trace.
# Collect one with `examples/workload/dapo_qwen3_8b/collect_joint_trace.sh`.
source: trace

trace:
  # Path to a joint group trace artifact.
  path: ""

  # Cap on groups read from the trace. 0 = all groups.
  n_groups: 0
```

Add `source: distributions` as the first line of the body of each existing workload YAML (`normal.yaml`, `uniform.yaml`, `bimodal.yaml`, `lognormal.yaml`, `powerlaw.yaml`, `from_trace.yaml`), so the source is always explicit.

- [ ] **Step 5: Dispatch in `build_workload`**

In `dryrun/cli/main.py`, replace the `build_workload` written in Task 2 with:

```python
def build_workload(cfg: DictConfig) -> GroupedWorkload:
    """Build the simulator's workload source from config."""
    source = cfg.get("source", "distributions")
    if source == "distributions":
        return GroupedWorkload.from_lengths(
            build_distribution(cfg.prompt),
            build_distribution(cfg.output),
        )
    if source == "trace":
        trace = cfg.trace
        path = trace.get("path", "")
        if not path:
            raise ValueError("workload.trace.path is required when workload.source is trace.")
        return GroupedWorkload.from_trace(path, n_groups=trace.get("n_groups", 0))
    raise ValueError(f"Unknown workload source: {source!r}.")
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
pytest tests/test_config.py -q && ruff check dryrun/config/ dryrun/cli/main.py
```

Expected: all pass, ruff clean.

- [ ] **Step 7: Commit**

```bash
git add dryrun/config/ dryrun/cli/main.py tests/test_config.py
git commit -m "feat: select the workload source explicitly in config"
```

---

### Task 7: Case scripts and the simulate wiring fix

**Files:**
- Create: `examples/workload/dapo_qwen3_8b/hosts.txt`
- Create: `examples/workload/dapo_qwen3_8b/collect_joint_trace.sh`
- Create: `examples/workload/dapo_qwen3_8b/README.md`
- Create: `examples/simulate/dapo_qwen3_8b.sh`
- Modify: `examples/simulate/simulate_with_fitted_cost.sh:68`
- Modify: `examples/workload/README.md`
- Modify: `examples/workload/fit_prompt_distribution.sh`, `fit_output_distribution.sh` (fix stale paths)

- [ ] **Step 1: Create `hosts.txt`**

```
# host [gpus]
29.162.247.148 8
28.49.16.220 8
```

- [ ] **Step 2: Create `collect_joint_trace.sh`**

```bash
#!/usr/bin/env bash
# Collect a joint (prompt, n responses) workload trace for Qwen3-8B on
# dapo-math-17k across every GPU listed in hosts.txt.
set -euo pipefail

ROOT_DIR="$(cd ("$(dirname) "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd ("$(dirname) "${BASH_SOURCE[0]}")" && pwd)"

HOSTS_FILE="${HOSTS_FILE:-${SCRIPT_DIR}/hosts.txt}"
DATASET="${DATASET:-/apdcephfs_zwfy/share_303760348/lhy/data/dapo/dapo-math-17k.parquet}"
MODEL="${MODEL:-/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B}"
TOKENIZER="${TOKENIZER:-${MODEL}}"
ENV_SCRIPT="${ENV_SCRIPT:-/apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh}"

OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/workload/qwen3_8b_dapo_tp1}"
SHARD_DIR="${SHARD_DIR:-${OUT_DIR}/shards}"
TRACE="${TRACE:-${OUT_DIR}/joint.jsonl}"

SAMPLE_SIZE="${SAMPLE_SIZE:-1024}"
GROUP_N="${GROUP_N:-8}"
MAX_TOKENS="${MAX_TOKENS:-10240}"
TP="${TP:-1}"
CONCURRENCY="${CONCURRENCY:-16}"
BASE_PORT="${BASE_PORT:-8700}"
SEED="${SEED:-42}"
# Serialize server boots per host: concurrent loads of a 5-shard checkpoint from
# the shared filesystem are slow enough to trip the readiness timeout.
STARTUP_STAGGER_S="${STARTUP_STAGGER_S:-25}"

mkdir -p "${SHARD_DIR}"

mapfile -t WORKERS < <(
    cd "${ROOT_DIR}" && python -c "
from dryrun.workload.sharding import build_workers, parse_hosts_file
for w in build_workers(parse_hosts_file('${HOSTS_FILE}'), base_port=${BASE_PORT}):
    print(w.index, w.host, w.gpu, w.port)
"
)
N_WORKERS="${#WORKERS[@]}"
echo "Collecting ${SAMPLE_SIZE} groups of n=${GROUP_N} across ${N_WORKERS} workers."

LOCAL_IPS="$(hostname -I)"
PIDS=()
for entry in "${WORKERS[@]}"; do
    read -r INDEX HOST GPU PORT <<<"${entry}"
    SHARD="$(printf '%s/worker-%02d.jsonl' "${SHARD_DIR}" "${INDEX}")"
    if [[ -f "${SHARD}" ]]; then
        echo "worker-${INDEX} on ${HOST} gpu ${GPU}: shard exists, skipping."
        continue
    fi
    CMD="source ${ENV_SCRIPT} >/dev/null 2>&1; cd ${ROOT_DIR} && dryrun-workload collect-groups"
    CMD+=" --dataset ${DATASET} --tokenizer ${TOKENIZER} --model ${MODEL}"
    CMD+=" --output ${SHARD} --worker-index ${INDEX} --n-workers ${N_WORKERS}"
    CMD+=" --sample-size ${SAMPLE_SIZE} --n ${GROUP_N} --max-tokens ${MAX_TOKENS}"
    CMD+=" --tp ${TP} --concurrency ${CONCURRENCY} --port ${PORT} --gpu ${GPU} --seed ${SEED}"
    CMD+=" --serve-arg gpu_memory_utilization=0.85"
    CMD+=" --serve-arg max_model_len=$((MAX_TOKENS + 2048))"

    LOG="${SHARD_DIR}/worker-$(printf '%02d' "${INDEX}").log"
    if [[ " ${LOCAL_IPS} " == *" ${HOST} "* ]]; then
        bash -lc "${CMD}" >"${LOG}" 2>&1 &
    else
        ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${HOST}" "${CMD}" >"${LOG}" 2>&1 &
    fi
    PIDS+=("$!")
    echo "launched worker-${INDEX} on ${HOST} gpu ${GPU} (log ${LOG})"
    sleep "${STARTUP_STAGGER_S}"
done

FAILED=0
for pid in "${PIDS[@]}"; do
    wait "${pid}" || FAILED=$((FAILED + 1))
done
if (( FAILED > 0 )); then
    echo "${FAILED} worker(s) failed. Inspect ${SHARD_DIR}/worker-*.log, then rerun" >&2
    echo "this script: completed shards are skipped and only failures retry." >&2
    exit 1
fi

cd "${ROOT_DIR}"
dryrun-workload merge-shards \
    --shard-dir "${SHARD_DIR}" \
    --n-workers "${N_WORKERS}" \
    --output "${TRACE}" \
    --max-output-tokens "${MAX_TOKENS}"
echo "Joint trace: ${TRACE}"
```

`chmod +x` it.

- [ ] **Step 3: Fix the hardcoded workload group**

In `examples/simulate/simulate_with_fitted_cost.sh`, add near the other defaults:

```bash
WORKLOAD="${WORKLOAD:-normal}"
```

and replace line 68's `workload=normal \` with `workload="${WORKLOAD}" \`.

The `DIST_OVERRIDES` block only makes sense for synthetic distributions, so guard it. Replace the unconditional emit block with:

```bash
DIST_OVERRIDES=()
if [[ "${WORKLOAD}" == "normal" ]]; then
    if [[ ! -f "${DIST_FILE}" ]]; then
        echo "Missing distribution config: ${DIST_FILE}" >&2
        exit 1
    fi
    while IFS= read -r line; do
        DIST_OVERRIDES+=("${line}")
    done < <(python "${SCRIPT_DIR}/emit_workload_overrides.py" "${DIST_FILE}")
    if [[ ${#DIST_OVERRIDES[@]} -eq 0 ]]; then
        echo "No workload overrides emitted from ${DIST_FILE}." >&2
        exit 1
    fi
fi
```

and delete the earlier standalone `if [[ ! -f "${DIST_FILE}" ]]` check. Because `DIST_OVERRIDES` can now be empty, change its expansion on the `python -m dryrun.cli.main` line to `"${DIST_OVERRIDES[@]+"${DIST_OVERRIDES[@]}"}"` so `set -u` does not abort on an empty array.

- [ ] **Step 4: Create `examples/simulate/dapo_qwen3_8b.sh`**

```bash
#!/usr/bin/env bash
# Simulate the DAPO Qwen3-8B tp=1 workload from a collected joint group trace.
set -euo pipefail

ROOT_DIR="$(cd ("$(dirname) "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd ("$(dirname) "${BASH_SOURCE[0]}")" && pwd)"

TRACE="${TRACE:-${ROOT_DIR}/outputs/workload/qwen3_8b_dapo_tp1/joint.jsonl}"
if [[ ! -f "${TRACE}" ]]; then
    echo "Missing joint workload trace: ${TRACE}" >&2
    echo "Collect one first: bash examples/workload/dapo_qwen3_8b/collect_joint_trace.sh" >&2
    exit 1
fi

# PSRL DAPO recipe (psrl/examples/dapo_trainer/qwen3_8b_megatron.sh):
# 64 prompts x n=8 per step, prompt cap 1024, response cap 10240, staleness 2,
# gen tp=1, 16 rollout instances. batch_size counts responses, so 64*8 = 512.
export WORKLOAD=dapo_trace
export N_VERSIONS="${N_VERSIONS:-16}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export N_INSTANCES="${N_INSTANCES:-16}"
export MAX_STALENESS="${MAX_STALENESS:-2}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_OUTPUT_LENGTH="${MAX_OUTPUT_LENGTH:-10240}"
export TOKEN_BUDGET="${TOKEN_BUDGET:-40960}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/simulate/qwen3_8b_dapo_tp1}"
export EXTRA_OVERRIDES="workload.trace.path=${TRACE}"

bash "${SCRIPT_DIR}/simulate_with_fitted_cost.sh"
```

For `EXTRA_OVERRIDES` to reach Hydra, add to `simulate_with_fitted_cost.sh` near the other defaults:

```bash
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
```

and append `${EXTRA_OVERRIDES:+${EXTRA_OVERRIDES}} \` as the last argument of the `python -m dryrun.cli.main` invocation.

`chmod +x` the new script.

- [ ] **Step 5: Fix the stale paths in the two older scripts**

In `examples/workload/fit_prompt_distribution.sh` and `fit_output_distribution.sh`, replace the stale defaults with the live paths:

- `DATASET` → `/apdcephfs_zwfy/share_303760348/lhy/data/dapo/dapo-math-17k.parquet` (prompt script) and `.../aime-2024.parquet` (output script)
- `TOKENIZER` / `MODEL` → `/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B`

- [ ] **Step 6: Write `examples/workload/dapo_qwen3_8b/README.md`**

Document, in prose: what the case is; the collection command; measured provenance (17,917 distinct prompts among 1.79M rows, dedup on `extra_info.index`; output mean ~8218, median 8642, 31% truncation at 10240; ~1886 tok/s per H20); the ~35-40 min runtime over 16 GPUs; retry behaviour (rerun skips finished shards); and the two caveats verbatim:

> `job.batch_size` counts **responses**, not prompts: this case uses 512 = 64 prompts × 8 responses. Whole-group batch consumption is a follow-up, so a training batch can currently be assembled from a partial group.
>
> Prefill is charged **once per response** rather than once per group, so rollout time is pessimistic by roughly one prompt prefill per extra group member. With mean prompt 154 tokens against mean output 8218, the error is small here, but it is real.
>
> Training cost is `fixed`, not fitted: no Qwen3-8B training artifact exists under `outputs/cost/qwen3_8b/`. The training latency is an assumption, not a measurement.

Also add a pointer to this directory from `examples/workload/README.md`.

- [ ] **Step 7: Verify the scripts parse and the wiring resolves**

```bash
bash -n examples/workload/dapo_qwen3_8b/collect_joint_trace.sh
bash -n examples/simulate/dapo_qwen3_8b.sh
bash -n examples/simulate/simulate_with_fitted_cost.sh
```

Expected: no output. Then dry-run the config resolution with a tiny synthetic trace:

```bash
source /apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh
cd /apdcephfs_zwfy/share_303760348/lhy/DryRun-RL
python - <<'EOF'
from dryrun.workload.trace import RequestGroup, write_group_trace
write_group_trace(
    [RequestGroup(i, 150 + i, tuple(2000 + 100 * j for j in range(8))) for i in range(64)],
    "/tmp/smoke_joint.jsonl", metadata={"max_output_tokens": 10240},
)
EOF
TRACE=/tmp/smoke_joint.jsonl N_VERSIONS=2 BATCH_SIZE=64 N_INSTANCES=2 \
  OUTPUT_DIR=/tmp/smoke_sim bash examples/simulate/dapo_qwen3_8b.sh
```

Expected: the simulation completes and prints `Simulation complete: 2 versions`.

- [ ] **Step 8: Commit**

```bash
git add examples/ && git commit -m "feat: add the DAPO Qwen3-8B tp=1 workload and simulate case"
```

---

### Task 8: Full collection and simulation run

**Files:**
- Modify: `examples/workload/dapo_qwen3_8b/README.md` (record observed results)

- [ ] **Step 1: Run the full collection**

```bash
source /apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh
cd /apdcephfs_zwfy/share_303760348/lhy/DryRun-RL
bash examples/workload/dapo_qwen3_8b/collect_joint_trace.sh
```

Expected: 16 workers launch (staggered), roughly 35-40 minutes, ending with `Joint trace: .../joint.jsonl`. If a worker fails, read its log under `outputs/workload/qwen3_8b_dapo_tp1/shards/` and rerun the same script: finished shards are skipped.

- [ ] **Step 2: Check the collected distribution against the probe**

```bash
python -c "
import json
with open('outputs/workload/qwen3_8b_dapo_tp1/joint.jsonl') as fh:
    meta = json.loads(fh.readline())
s = meta['stats']
print('groups', meta['n_groups'], 'group_size', meta['group_size'])
print('prompt', {k: s['prompt'][k] for k in ('mean','median','p95','max')})
print('output', {k: s['output'][k] for k in ('mean','median','p95','max')})
print('truncated', s['output']['truncated_fraction'])
"
```

Expected: 1024 groups, group_size 8, prompt mean near 154, output mean near 8200, truncated fraction near 0.31. A large deviation means the sampling or the cap differs from the probe: investigate before simulating rather than reporting numbers built on a bad trace.

- [ ] **Step 3: Run the full simulation**

```bash
bash examples/simulate/dapo_qwen3_8b.sh
```

Expected: `Simulation complete: 16 versions in <T>s`, a throughput line, and a staleness distribution with every entry `<= 2`.

- [ ] **Step 4: Record the observed results**

Add a short "Observed results" section to the case README with the collected trace stats and the simulation's versions/throughput/staleness output. Report what actually happened, including anything that surprised you.

- [ ] **Step 5: Full suite and lint**

```bash
pytest -q && ruff check dryrun/ tests/
```

Expected: all pass, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add examples/workload/dapo_qwen3_8b/README.md
git commit -m "docs: record the DAPO Qwen3-8B workload collection and simulation results"
```

---

## Self-Review

**Spec coverage:** joint trace format → Task 1; distributed collection (hosts file, deterministic shards, merge validation, staggered startup, local-vs-SSH) → Tasks 4, 5, 7; `workload.source` switch → Task 6; `Request.group_id` + `GroupedWorkload` → Task 2; case scripts → Task 7; `workload=normal` fix → Task 7 Step 3; documented caveats (batch_size, prefix, training cost) → Task 7 Step 6; verification items → Tasks 1-8.

**Types:** `RequestGroup(group_id, prompt_len, output_lens)` is used identically in Tasks 1-5. `GroupedWorkload.request()` returns `(group_id, prompt_len, output_len)` in Tasks 2 and 6. `shard_path`/`merge_shards` names match between Tasks 4, 5, and 7.

**Known follow-ups (out of scope, per the spec):** whole-group batch consumption; shared-prefix prefill; DAPO dynamic-sampling filter.
