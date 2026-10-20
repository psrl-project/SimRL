"""Block hashes derived from workload provenance.

vLLM hashes a block as `hash(parent_block_hash, block_token_ids)`
(`third_party/vllm/vllm/v1/core/kv_cache_utils.py:617-623`). Chaining the
parent in is what makes a block hash mean *every token from position 0 to this
block's end*, so that

    A: [hello] [world]
    B: [bye]   [world]

do not collide on block 1: reuse is only valid when the whole prefix agrees.

We hold no token ids. We hold something that answers the same question: two
contexts agree on a prefix exactly when they agree on the provenance that
produced it. So the hash is written rather than computed, from two integers
already on `Request`.

This is where we diverge from Frontier, which must be handed a
`block_hash_ids` column on its trace because it replays real requests and
never sees their tokens. We generate the workload, so the provenance is known
and the trace format stays untouched.

**Known one-directional bias.** Two *different* prompts that happen to share
leading full blocks hit in the real engine and miss here. On RL workloads
distinct problems almost certainly differ at token 0, and every weight update
clears the hash map anyway, so this is second order. It only ever
under-reports hits, never over-reports. See `docs/reference/limitations.md`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Hashable

if TYPE_CHECKING:
    from ..request import Request


def block_count(req: "Request", block_size: int) -> int:
    """How many full blocks this request's current context covers.

    The length `block_hashes` would return for the whole context, without
    building the list. Callers that only need the extent, a bound on which
    blocks are cacheable, ask for this instead.
    """
    return (req.prompt_len + req.generated) // block_size


def block_hashes(
    req: "Request",
    block_size: int,
    start: int = 0,
    stop: int | None = None,
) -> list[Hashable]:
    """Hash ids for full blocks `[start, stop)` of this request's context.

    Args:
        req: The request. Its context is `prompt_len + generated`, which is the
            token count the engine knows about so far.
        block_size: Tokens per KV block.
        start: First block index to emit. Defaults to the start of the context.
        stop: One past the last block index to emit, clamped to the context's
            own block count. Defaults to that count, i.e. the whole context.

    Returns:
        One id per full block, in block order. A block lying wholly inside the
        shared prompt keys on `source_group_id`, so siblings of one prompt
        agree on it. Every other block keys on `rid` and is private.

        `source_group_id` rather than `group_id` is deliberate: `group_id` is
        monotonic by design, so a repeating workload gives the same prompt a
        new id on wraparound and reuse the real engine achieves would be
        missed. `source_group_id` records which workload group the lengths came
        from, which is the prompt's actual identity.

        A hash depends only on `j` and on identity fields that never change, so
        the list is a pure function of its index range: block `j`'s id is the
        same whatever the context length was when it was asked for. That is
        what makes a range request equivalent to slicing the whole list, and
        why `allocate` can ask only for the blocks it is about to fill rather
        than rebuilding every hash of a 30k-token context per decode step.

    Raises:
        ValueError: If `block_size` is not positive.
    """
    if block_size <= 0:
        raise ValueError(f"Expected block_size > 0, got {block_size!r}.")

    n_blocks = (req.prompt_len + req.generated) // block_size
    if stop is None or stop > n_blocks:
        stop = n_blocks
    if start < 0:
        start = 0
    if start >= stop:
        return []

    # Only full blocks are cacheable, which is vLLM's own rule. The block
    # straddling the end of the prompt is therefore private even though most of
    # its tokens are shared.
    #
    # The prompt/response split is a threshold on `j`, not a per-element test:
    # block `j` is shared exactly while `(j + 1) * block_size <= prompt_len`.
    # Computing the boundary once turns a per-block branch into two slices.
    shared_blocks = req.prompt_len // block_size
    group_id = req.source_group_id
    rid = req.rid
    if stop <= shared_blocks:
        return [("p", group_id, j) for j in range(start, stop)]
    if start >= shared_blocks:
        return [("r", rid, j) for j in range(start, stop)]
    return [("p", group_id, j) for j in range(start, shared_blocks)] + [
        ("r", rid, j) for j in range(shared_blocks, stop)
    ]
