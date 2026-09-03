"""The introspection gate: declared kernel shapes versus a real module graph.

This is the only check that can falsify `families_dense.py`'s claim that a dense layer fuses into
four projections, so it needs to be able to fail. The mismatch test below is what establishes
that, by declaring a deliberately wrong shape and asserting the gate catches it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from dryrun.profiler.kernels.introspect import _fuse, verify_shapes
from dryrun.profiler.schema import ModelDescriptor, ProfileValidationError

_QWEN3 = Path("/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B")
_QWEN2 = Path("/apdcephfs_zwfy/share_303760348/lhy/models/SWE-agent-LM-7B")

pytestmark = pytest.mark.skipif(
    not _QWEN3.is_dir(), reason="The checkpoints are machine-local."
)


def _descriptor(path: Path) -> ModelDescriptor:
    from dryrun.profiler.hf import load_hf_model_descriptor

    return load_hf_model_descriptor(path)


def test_the_fusion_rules_concatenate_rows_and_share_columns():
    """A pure-arithmetic check of the rule, independent of any checkpoint."""
    fused = _fuse(
        {
            "self_attn.q_proj": (4096, 4096),
            "self_attn.k_proj": (1024, 4096),
            "self_attn.v_proj": (1024, 4096),
            "self_attn.o_proj": (4096, 4096),
            "mlp.gate_proj": (12288, 4096),
            "mlp.up_proj": (12288, 4096),
            "mlp.down_proj": (4096, 12288),
        }
    )
    assert fused == {
        "qkv_proj": (6144, 4096),
        "o_proj": (4096, 4096),
        "gate_up_proj": (24576, 4096),
        "down_proj": (4096, 12288),
    }


def test_a_fusion_rule_declines_when_its_parts_disagree_on_input_width():
    """Concatenation is along rows, so mismatched columns mean these are not those modules."""
    fused = _fuse(
        {
            "self_attn.q_proj": (4096, 4096),
            "self_attn.k_proj": (1024, 2048),
            "self_attn.v_proj": (1024, 4096),
        }
    )
    assert "qkv_proj" not in fused


def test_a_fusion_rule_declines_when_a_part_is_missing():
    fused = _fuse({"mlp.gate_proj": (12288, 4096)})
    assert "gate_up_proj" not in fused


@pytest.mark.parametrize("path", [_QWEN3, _QWEN2], ids=["qwen3-8b", "qwen2-7b"])
def test_the_declared_shapes_match_a_real_module_graph(path):
    """The claim under test: our four fused shapes are what the engine runs."""
    if not path.is_dir():
        pytest.skip(f"{path} is machine-local.")
    report = verify_shapes(path, _descriptor(path))
    assert report.ok, [
        (item.name, item.declared, item.observed) for item in report.comparisons if not item.matches
    ]
    assert report.family == "dense_llama"
    assert {item.name for item in report.comparisons} == {
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
        "lm_head",
    }


def test_every_layer_is_reported_and_homogeneous():
    report = verify_shapes(_QWEN3, _descriptor(_QWEN3))
    assert report.layer_count == 36
    assert report.heterogeneous_layers == ()


def test_a_wrong_declared_shape_is_caught():
    """Without this the gate could pass vacuously, which is its only real failure mode."""

    @dataclass(frozen=True)
    class _WrongFamily:
        name: str = "deliberately_wrong"

        def gemm_shapes(self, model, *, tensor_parallel):
            from dryrun.profiler.kernels.shape import GemmShape

            # qkv_proj is off by one row; everything else is right.
            return (
                GemmShape("qkv_proj", 6145, 4096),
                GemmShape("o_proj", 4096, 4096),
                GemmShape("gate_up_proj", 24576, 4096),
                GemmShape("down_proj", 4096, 12288),
            )

        def logits_shape(self, model, *, tensor_parallel):
            from dryrun.profiler.kernels.shape import GemmShape

            return GemmShape("lm_head", 151936, 4096)

        def head_counts(self, model, *, tensor_parallel):
            return 32, 8

    from dryrun.profiler.kernels import families

    saved = families.FAMILY_REGISTRY["Qwen3ForCausalLM"]
    families.FAMILY_REGISTRY["Qwen3ForCausalLM"] = _WrongFamily()
    try:
        report = verify_shapes(_QWEN3, _descriptor(_QWEN3))
    finally:
        families.FAMILY_REGISTRY["Qwen3ForCausalLM"] = saved
    assert not report.ok
    wrong = [item for item in report.comparisons if not item.matches]
    assert [item.name for item in wrong] == ["qkv_proj"]
    assert wrong[0].declared == (6145, 4096)
    assert wrong[0].observed == (6144, 4096)


def test_a_sharded_comparison_is_refused():
    """The module graph is unsharded, so a tp>1 comparison would fail on every entry."""
    with pytest.raises(ProfileValidationError, match="tensor_parallel"):
        verify_shapes(_QWEN3, _descriptor(_QWEN3), tensor_parallel=2)
