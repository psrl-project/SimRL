"""Load and render distinct dataset prompts for joint workload collection."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .._logging import srl_logger

_CONTENT_KEY = "content"


def render_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    """
    Render chat messages to a single prompt string.

    Falls back to concatenating message contents if the tokenizer has no
    chat template configured.

    Args:
        tokenizer (Any): A HuggingFace tokenizer instance.
        messages (list[dict[str, Any]]): Chat messages with `content`/`role` keys.

    Returns:
        str: The rendered prompt text.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:  # noqa: BLE001
        return "\n".join(message[_CONTENT_KEY] for message in messages)


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
    srl_logger.info(
        f"Found {len(messages_list)} distinct prompts among {len(prompts)} rows."
    )

    if sample_size is not None and sample_size < len(messages_list):
        rng = random.Random(seed)
        indices = rng.sample(range(len(messages_list)), sample_size)
        messages_list = [messages_list[i] for i in sorted(indices)]
    return messages_list
