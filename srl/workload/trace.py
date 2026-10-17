"""Trace artifact I/O for joint grouped workloads.

The JSONL file contains a `_meta` header followed by
one `{"group_id": ..., "prompt_len": ..., "output_lens": [...]}` record per
`RequestGroup`.  This format captures a joint prompt-and-response distribution
with explicit group boundaries, as produced by the DAPO recipe where each
prompt is paired with `n` independently sampled responses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def compute_length_stats(lengths: list[int], n_bins: int = 20) -> dict[str, Any]:
    """
    Compute summary statistics and a histogram for a list of lengths.

    Args:
        lengths (list[int]): Token lengths to summarize.
        n_bins (int): Number of histogram bins.

    Returns:
        dict[str, Any]: `mean`, `median`, `p95`, `min`, `max`, and a
        `histogram` with `bin_edges`/`counts`.
    """
    array = np.asarray(lengths, dtype=np.int64)
    counts, bin_edges = np.histogram(array, bins=n_bins)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "min": int(array.min()),
        "max": int(array.max()),
        "histogram": {
            "bin_edges": [float(edge) for edge in bin_edges],
            "counts": [int(count) for count in counts],
        },
    }


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
    output_stats = {
        **compute_length_stats(output_lengths, n_bins=n_bins),
        "truncated_fraction": truncated / len(output_lengths),
    }
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
            Must contain a positive `max_output_tokens` entry (the generation
            cap used during data collection); this value is required to
            correctly count truncated responses in the summary stats.
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
    max_output_tokens = int(metadata.get("max_output_tokens", 0))
    if max_output_tokens < 1:
        raise ValueError(
            "write_group_trace needs a positive max_output_tokens in metadata to "
            "count truncated responses, got "
            f"{metadata.get('max_output_tokens')!r}."
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
