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
clears the hash map anyway, so this is second order -- but it only ever
under-reports hits, never over-reports. See `docs/reference/limitations.md`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Hashable

if TYPE_CHECKING:
    from ..request import Request


def block_hashes(req: "Request", block_size: int) -> list[Hashable]:
    """Hash ids for every full block of this request's current context.

    Args:
        req: The request. Its context is `prompt_len + generated`, which is the
            token count the engine knows about so far.
        block_size: Tokens per KV block.

    Returns:
        One id per full block, in block order. A block lying wholly inside the
        shared prompt keys on `source_group_id`, so siblings of one prompt
        agree on it. Every other block keys on `rid` and is private.

        `source_group_id` rather than `group_id` is deliberate: `group_id` is
        monotonic by design, so a repeating workload gives the same prompt a
        new id on wraparound and reuse the real engine achieves would be
        missed. `source_group_id` records which workload group the lengths came
        from, which is the prompt's actual identity.

    Raises:
        ValueError: If `block_size` is not positive.
    """
    if block_size <= 0:
        raise ValueError(f"Expected block_size > 0, got {block_size!r}.")

    context = req.prompt_len + req.generated
    # Only full blocks are cacheable, which is vLLM's own rule. The block
    # straddling the end of the prompt is therefore private even though most of
    # its tokens are shared.
    return [
        ("p", req.source_group_id, j)
        if (j + 1) * block_size <= req.prompt_len
        else ("r", req.rid, j)
        for j in range(context // block_size)
    ]
