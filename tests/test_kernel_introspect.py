"""The introspection gate: declared kernel shapes versus a real module graph.

This is the only check that can falsify `families_dense.py`'s claim that a dense layer fuses into
four projections, so it needs to be able to fail. The mismatch test below is what establishes
that, by declaring a deliberately wrong shape and asserting the gate catches it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from dryrun.profiler.kernel.introspect import _fuse, verify_shapes
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
            from dryrun.profiler.kernel.grid.shape import GemmShape

            # qkv_proj is off by one row; everything else is right.
            return (
                GemmShape("qkv_proj", 6145, 4096),
                GemmShape("o_proj", 4096, 4096),
                GemmShape("gate_up_proj", 24576, 4096),
                GemmShape("down_proj", 4096, 12288),
            )

        def logits_shape(self, model, *, tensor_parallel):
            from dryrun.profiler.kernel.grid.shape import GemmShape

            return GemmShape("lm_head", 151936, 4096)

        def head_counts(self, model, *, tensor_parallel):
            return 32, 8

    from dryrun.profiler.kernel.grid import families

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


def test_fuse_scales_the_two_expert_projections_on_different_axes():
    """
    A per-token expert shape multiplies rows for gate_up and the contraction for down.

    The module graph holds ONE expert's slice; the family declares the work of all
    `num_experts_per_tok` of them as a single GEMM. Which dimension carries the factor
    differs between the two projections, and getting it backwards reports a mismatch for a
    family that is right:

      gate_up: each expert produces its own rows from the same hidden vector -> stack rows.
      down:    each expert maps its own activations onto the same hidden rows -> widen k.

    Both declared shapes carry the right work: gate_up is 16.8M MACs per token per layer
    (it fuses gate AND up, so 2x) and down is 8.4M. Scaling the wrong axis would change
    those numbers, so this is a costing check and not only a bookkeeping one.
    """
    from dryrun.profiler.kernel.introspect import _fuse

    layer = {
        "mlp.experts.gate_up_proj": (1024, 2048),
        "mlp.experts.down_proj": (2048, 512),
    }
    fused = _fuse(layer, experts_per_tok=8)

    assert fused["expert_gate_up"] == (8192, 2048)
    assert fused["expert_down"] == (2048, 4096)
    # topk * 2 * moe_inter * hidden, and topk * hidden * moe_inter: the real per-token work.
    assert 8192 * 2048 == 8 * (2 * 512) * 2048
    assert 2048 * 4096 == 8 * 2048 * 512


def test_fuse_reads_expert_stacks_and_the_router_from_a_moe_layer():
    """
    `_linear_shapes` must see 3-D expert stacks and a bare-module router weight.

    256 experts are 93% of Qwen3.5-35B-A3B's parameters and they are stored as plain 3-D
    `nn.Parameter` stacks, not `nn.Linear`. A verifier that only walked 2-D module weights
    could not see them at all, so `verify-shapes` would pass a family that priced them
    wrongly -- which is exactly how a 1.8x error in `qkv_proj` survived until this gate
    could reach a MoE layer.
    """
    import torch

    from dryrun.profiler.kernel.introspect import _linear_shapes

    class _Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = torch.nn.Parameter(torch.empty(256, 1024, 2048, device="meta"))
            self.down_proj = torch.nn.Parameter(torch.empty(256, 2048, 512, device="meta"))

    class _Mlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = torch.nn.Linear(2048, 256, bias=False, device="meta")
            self.experts = _Experts()

    shapes = _linear_shapes(_Mlp())

    # The router is an nn.Linear, keyed by module name with no `.weight`.
    assert shapes["gate"] == (256, 2048)
    # Expert stacks report the per-expert slice, with the expert count stripped.
    assert shapes["experts.gate_up_proj"] == (1024, 2048)
    assert shapes["experts.down_proj"] == (2048, 512)
