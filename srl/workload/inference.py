"""Collect joint prompt-and-response workload traces through live inference."""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .._logging import srl_logger
from .fitting import render_prompt
from .trace import RequestGroup, write_group_trace

_POLL_INTERVAL_S = 2.0
_TERMINATE_TIMEOUT_S = 30.0


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

    Args:
        model (str): The served model name.
        prompt (str): The input prompt text.
        max_tokens (int): Maximum completion tokens to generate.
        n (int): Number of independent completions to generate.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling parameter.

    Returns:
        dict[str, Any]: A `/v1/completions` request payload with `return_token_ids` set to true.
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
    # All choices of a request share the same prompt, and vLLM echoes the identical
    # `prompt_token_ids` on each choice, so reading from the first choice is sufficient.
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


class _VLLMServer:
    """Manages the lifecycle of one `vllm serve` subprocess."""

    def __init__(
        self,
        model_path: str,
        port: int,
        tp: int,
        extra_serve_args: dict[str, Any],
        ready_timeout: int,
        log_path: Path,
        gpu: int | None = None,
    ):
        self.model_path = model_path
        self.port = port
        self.tp = tp
        self.extra_serve_args = extra_serve_args
        self.ready_timeout = ready_timeout
        self.log_path = log_path
        self.gpu = gpu
        self.process: subprocess.Popen[bytes] | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _command(self) -> list[str]:
        command = [
            "vllm",
            "serve",
            self.model_path,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(self.tp),
        ]
        for key, value in self.extra_serve_args.items():
            option = f"--{str(key).replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    command.append(option)
                continue
            command.extend((option, str(value)))
        return command

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.ready_timeout
        health_url = f"{self.base_url}/health"
        while time.monotonic() < deadline:
            assert self.process is not None, "vLLM server process was not started."
            return_code = self.process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"vLLM server exited early with code {return_code}. See log at {self.log_path}."
                )
            try:
                with urllib.request.urlopen(health_url, timeout=5):  # noqa: S310
                    srl_logger.info(f"vLLM server is ready at {self.base_url!r}.")
                    return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                time.sleep(_POLL_INTERVAL_S)
        raise TimeoutError(f"vLLM server did not become ready within {self.ready_timeout}s.")

    def __enter__(self) -> "_VLLMServer":
        command = self._command()
        srl_logger.info(f"Launching vLLM server: {' '.join(command)}")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = self.log_path.open("w", encoding="utf-8")
        env = None
        if self.gpu is not None:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(self.gpu)}
        self.process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        self._log_handle = log_handle
        self._wait_until_ready()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self.process is None:
            return
        srl_logger.info("Shutting down vLLM server...")
        self.process.terminate()
        try:
            self.process.wait(timeout=_TERMINATE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            srl_logger.warning("vLLM server did not exit in time, killing it.")
            self.process.kill()
            self.process.wait()
        self._log_handle.close()


def _partition_group_results(
    results: list[RequestGroup | BaseException],
    group_ids: list[int],
) -> list[RequestGroup]:
    """
    Split `asyncio.gather(return_exceptions=True)` results into successes and failures.

    Failures are logged at warning level. If every result is a failure, a
    `RuntimeError` is raised so the caller knows nothing was collected.

    Args:
        results (list[RequestGroup | BaseException]): Parallel results from
            `asyncio.gather` with `return_exceptions=True`.
        group_ids (list[int]): Group id for each result position, used in log messages.

    Returns:
        list[RequestGroup]: Successful results only.

    Raises:
        RuntimeError: If every result failed.
    """
    successes: list[RequestGroup] = []
    for result, gid in zip(results, group_ids):
        if isinstance(result, BaseException):
            srl_logger.warning(
                f"Group {gid} failed with {type(result).__name__}: {result}."
            )
        else:
            successes.append(result)
    if not successes:
        raise RuntimeError(
            f"All {len(results)} group requests failed; no output to write."
        )
    return successes


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
    """
    Send prompts with bounded concurrency, one group per prompt.

    Collects responses with best-effort semantics: individual request failures
    are logged and skipped rather than discarding the entire batch. If every
    request fails, a `RuntimeError` is raised. Otherwise, only successful
    `RequestGroup` objects are returned along with a summary log line.

    Args:
        base_url (str): Base URL of the vLLM server, e.g. `http://127.0.0.1:8700`.
        served_model_name (str): Model name passed in the request `model` field.
        indexed_prompts (list[tuple[int, str]]): `(group_id, prompt)` pairs to send.
        max_tokens (int): Maximum completion tokens per response.
        n (int): Number of independent completions per prompt.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling parameter.
        concurrency (int): Maximum number of in-flight requests.

    Returns:
        list[RequestGroup]: Successful groups (may be fewer than requested if
            some requests failed).

    Raises:
        RuntimeError: If every request in the batch failed.
    """
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
            if response.status // 100 != 2:
                body = await response.text()
                raise RuntimeError(
                    f"Group {group_id} got HTTP {response.status} from server: {body}"
                )
            data = await response.json()
        return parse_group_response(data, group_id=group_id)

    group_ids = [gid for gid, _ in indexed_prompts]
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        raw = await asyncio.gather(
            *(_one_group(session, gid, prompt) for gid, prompt in indexed_prompts),
            return_exceptions=True,
        )
    successes = _partition_group_results(list(raw), group_ids)
    if len(successes) < len(indexed_prompts):
        srl_logger.warning(
            f"{len(successes)}/{len(indexed_prompts)} groups succeeded; "
            f"{len(indexed_prompts) - len(successes)} failed and were skipped."
        )
    return successes


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

    The shard file is always rewritten from scratch, so reruns are safe: a
    partial or failed shard from a previous attempt is cleanly overwritten.

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

    Raises:
        ValueError: If `n_workers` is less than 1 or `worker_index` is out of range.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    from .fitting import load_unique_prompt_messages  # noqa: PLC0415
    from .sharding import shard_indices  # noqa: PLC0415

    if n_workers < 1:
        raise ValueError(
            f"n_workers must be at least 1, got {n_workers!r}."
        )
    if not (0 <= worker_index < n_workers):
        raise ValueError(
            f"worker_index must be in [0, n_workers), "
            f"got worker_index={worker_index!r} with n_workers={n_workers!r}."
        )

    output = Path(output_path)
    server_log = Path(log_path) if log_path is not None else output.with_suffix(".server.log")

    srl_logger.info(f"Loading tokenizer from {tokenizer_path!r}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    messages_list = load_unique_prompt_messages(dataset_path, sample_size, seed)
    indices = shard_indices(len(messages_list), worker_index, n_workers)
    if not indices:
        raise ValueError(
            f"Worker {worker_index} of {n_workers} has no prompts to collect "
            f"from {len(messages_list)} sampled prompts."
        )
    n_groups_requested = len(indices)
    indexed_prompts = [
        (index, render_prompt(tokenizer, messages_list[index])) for index in indices
    ]
    srl_logger.info(
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
            "n_groups_requested": n_groups_requested,
        },
    )
    srl_logger.info(f"Worker {worker_index} wrote {len(groups)} groups to {written}.")
    return written
