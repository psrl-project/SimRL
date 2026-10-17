"""Measure vLLM's prefix-cache behaviour in four controlled scenarios.

Each scenario's expected hit count is arithmetic over `prompt_len` and
`block_size`, registered in the config before anything is measured. Nothing
here is fitted: a disagreement means the simulated pool's semantics are wrong
and the logic has to change, not that a coefficient needs tuning.

The four scenarios exist because each exercises a different path through
`srl/component/rollout/engine/kv/`:

- `within_group_shared_prompt` -- one prompt at `n=8`, the GRPO shape. Hits on
  blocks another live request holds.
- `continued_prefix` -- round two's prompt is round one's prompt plus its
  output, the shape a resumed partial rollout and an agentic next round both
  take. Hits on blocks whose `ref_cnt` has reached zero.
- `hit_rate_after_reset` -- `reset_prefix_cache`, then resend. The weight-sync
  path.
- `lru_eviction` -- squeeze the pool until reuse must evict. The only scenario
  that reaches `_maybe_evict`, because KV never saturates under the project's
  main workload.

Run with `examples/profile/vllm/prefix_cache_probe.sh`.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from .._logging import setup_logging, srl_logger
from .inference import _VLLMServer
from .prefix_probe import PrefixCacheCounters, read_prefix_cache_counters

_RESET_PATH = "/reset_prefix_cache"
# `/reset_prefix_cache` lives on vLLM's dev router
# (`vllm/entrypoints/serve/dev/cache/api_router.py`), which `api_server.py` only
# mounts when this is set. Without it the endpoint 404s and the weight-sync
# scenario cannot be measured at all.
_DEV_MODE_ENV = "VLLM_SERVER_DEV_MODE"


def _post(base_url: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """POST JSON and decode the JSON response.

    Raises:
        urllib.error.HTTPError: Propagated, so a scenario fails loudly rather
            than recording a reading taken against a server that refused.
    """
    data = json.dumps(payload or {}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body.strip() else {}


def _completions(
    base_url: str,
    model: str,
    prompt: Any,
    max_tokens: int,
    n: int,
) -> dict[str, Any]:
    """One `/v1/completions` call with `return_token_ids` on.

    `prompt` may be a string or a list of token ids. Token ids are what the
    continued-prefix scenario needs: appending the first round's output ids to
    its prompt ids reproduces exactly the context a resumed request carries,
    without a tokeniser round-trip that could alter the boundary.

    Sampling is greedy only at `n == 1`: vLLM rejects `n > 1` with
    `temperature=0` ("n must be 1 when using greedy sampling"). Prefix reuse
    depends on the prompt, not on what is sampled from it, so a non-zero
    temperature at `n > 1` changes nothing this measures.
    """
    return _post(
        base_url,
        "/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "n": n,
            "temperature": 0.0 if n == 1 else 1.0,
            "return_token_ids": True,
        },
    )


def _prompt_ids(response: dict[str, Any]) -> list[int]:
    """Prompt token ids echoed back by `return_token_ids`."""
    choice = response["choices"][0]
    ids = choice.get("prompt_token_ids")
    if ids is None:
        raise ValueError(
            "No prompt_token_ids in the completions response, so the continued "
            "prefix cannot be built exactly. Check return_token_ids support."
        )
    return list(ids)


def _output_ids(response: dict[str, Any], index: int = 0) -> list[int]:
    choice = response["choices"][index]
    ids = choice.get("token_ids")
    if ids is None:
        raise ValueError("No token_ids in the completions response.")
    return list(ids)


def _synthetic_prompt(n_tokens: int) -> str:
    """A prompt of roughly `n_tokens` tokens, with no repeated structure.

    Distinct integers rather than repeated filler: repeated text would let
    different block positions hash alike in the real engine, which would
    confuse a hit count that is supposed to be arithmetic.

    The count is approximate by construction, which is why every expectation in
    the fixture is a hit *rate* rather than a block count. A multi-digit integer
    costs more than one token, so emitting `n_tokens` words would overshoot the
    context limit; the divisor below is calibrated against Qwen3's tokeniser
    (about two tokens per "<number><space>" pair at these magnitudes) and the
    exact prompt length is read back from `prompt_token_ids` anyway.
    """
    return " ".join(str(i) for i in range(max(1, n_tokens // 2)))


def _measure(base_url: str, before: PrefixCacheCounters) -> PrefixCacheCounters:
    """Counters accumulated since `before`."""
    return read_prefix_cache_counters(base_url) - before


def _run_within_group(base_url: str, model: str, scenario: dict[str, Any]) -> dict[str, Any]:
    """One prompt, `n` samples. The shared prefix is paid for once."""
    prompt = _synthetic_prompt(scenario["prompt_len"])
    n = int(scenario["n"])
    before = read_prefix_cache_counters(base_url)
    response = _completions(base_url, model, prompt, scenario["max_tokens"], n)
    delta = _measure(base_url, before)
    return {
        "queries": delta.queries,
        "hits": delta.hits,
        "hit_rate": delta.hit_rate,
        "prompt_tokens": len(_prompt_ids(response)),
        "n": n,
    }


def _run_continued_prefix(
    base_url: str, model: str, scenario: dict[str, Any]
) -> dict[str, Any]:
    """Round two's prompt is round one's prompt plus its output."""
    prompt = _synthetic_prompt(scenario["prompt_len"])
    first = _completions(base_url, model, prompt, scenario["max_tokens"], 1)
    continued = _prompt_ids(first) + _output_ids(first)

    before = read_prefix_cache_counters(base_url)
    _completions(base_url, model, continued, scenario["max_tokens"], 1)
    delta = _measure(base_url, before)
    return {
        "queries": delta.queries,
        "hits": delta.hits,
        "hit_rate": delta.hit_rate,
        "round1_context_tokens": len(continued),
    }


def _run_after_reset(base_url: str, model: str, scenario: dict[str, Any]) -> dict[str, Any]:
    """Warm the cache, reset it, then resend the same request."""
    prompt = _synthetic_prompt(scenario["prompt_len"])
    n = int(scenario["n"])
    _completions(base_url, model, prompt, scenario["max_tokens"], n)

    _post(base_url, _RESET_PATH)

    before = read_prefix_cache_counters(base_url)
    _completions(base_url, model, prompt, scenario["max_tokens"], n)
    delta = _measure(base_url, before)
    return {
        "queries": delta.queries,
        "hits": delta.hits,
        "hit_rate": delta.hit_rate,
        "n": n,
    }


def _run_lru_eviction(base_url: str, model: str, scenario: dict[str, Any]) -> dict[str, Any]:
    """Issue more distinct prompts than the pool can hold, then revisit them.

    With capacity squeezed by a low `gpu_memory_utilization`, the early prompts'
    blocks must be evicted before the revisit, so the second pass hits less than
    a pass with capacity to spare would.
    """
    n_prompts = int(scenario["n_prompts"])
    # The differentiator goes at the FRONT. Appending it would leave every
    # prompt sharing an identical prefix, so they would all hit and nothing
    # would ever be evicted -- the scenario would pass while testing nothing.
    prompts = [
        f"case {i}: " + _synthetic_prompt(scenario["prompt_len"])
        for i in range(n_prompts)
    ]
    for prompt in prompts:
        _completions(base_url, model, prompt, scenario["max_tokens"], 1)

    before = read_prefix_cache_counters(base_url)
    for prompt in prompts:
        _completions(base_url, model, prompt, scenario["max_tokens"], 1)
    delta = _measure(base_url, before)
    return {
        "queries": delta.queries,
        "hits": delta.hits,
        "hit_rate": delta.hit_rate,
        "n_prompts": n_prompts,
    }


_RUNNERS = {
    "within_group_shared_prompt": _run_within_group,
    "continued_prefix": _run_continued_prefix,
    "hit_rate_after_reset": _run_after_reset,
    "lru_eviction": _run_lru_eviction,
}


def run_scenarios(config_path: Path, model: str, port: int, output_path: Path) -> Path:
    """Run every scenario in `config_path` and write the measurements.

    Each scenario gets its own server, because `gpu_memory_utilization` differs
    between them and that is fixed at startup. Servers run one at a time:
    parallel vLLM servers contaminate each other's measurements.

    Args:
        config_path: The scenario YAML.
        model: Checkpoint directory to serve.
        port: Port for the server.
        output_path: Destination JSON.

    Returns:
        The path written.
    """
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    block_size = int(config["block_size"])
    # Inherited by every server launched below.
    os.environ[_DEV_MODE_ENV] = "1"
    results = []

    for scenario in config["scenarios"]:
        name = scenario["name"]
        runner = _RUNNERS.get(name)
        if runner is None:
            raise ValueError(
                f"No runner for scenario {name!r}. Known: {sorted(_RUNNERS)}."
            )
        srl_logger.info("Scenario %s: starting a server.", name)
        server = _VLLMServer(
            model_path=model,
            port=port,
            tp=1,
            extra_serve_args={
                "gpu_memory_utilization": scenario["gpu_memory_utilization"],
                "block_size": block_size,
                "enable_prefix_caching": True,
                "max_model_len": scenario.get("max_model_len", 8192),
            },
            ready_timeout=int(config.get("ready_timeout", 900)),
            log_path=output_path.parent / f"server_{name}.log",
        )
        with server:
            measured = runner(server.base_url, model, scenario)

        expected = {k: v for k, v in scenario.items() if k.startswith("expect")}
        row = {"name": name, "expected": expected, "measured": measured}
        srl_logger.info("Scenario %s: %s", name, json.dumps(row["measured"]))
        results.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    srl_logger.info("Wrote %d scenario results to %s.", len(results), output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    setup_logging(level="INFO")
    run_scenarios(args.config, args.model, args.port, args.output)


if __name__ == "__main__":
    main()
