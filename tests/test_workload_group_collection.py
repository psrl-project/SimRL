"""Tests for per-response group collection from a vLLM completions payload."""

from __future__ import annotations

import pytest

from srl.workload.inference import build_completion_payload, parse_group_response
from srl.workload.trace import RequestGroup


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
