"""Building ComposedRollout from config, and the errors that guard it."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from srl.cli.main import build_rollout_cost
from srl.config.schema import RolloutCostConfig
from srl.cost.rollout.composed import ComposedRollout

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CHECKPOINT = Path("/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B")
_TABLES = _REPO_ROOT / "outputs/profile/qwen3_8b/kernel_tp1/kernel"

pytestmark = pytest.mark.skipif(
    not (_CHECKPOINT / "config.json").exists() or not _TABLES.is_dir(),
    reason="Needs the real Qwen3-8B checkpoint and the profiled kernel tables.",
)


def _config(**overrides):
    value = {
        "name": "composed",
        "kernel_table_root": str(_TABLES),
        "gpu": "h20",
        "model_path": str(_CHECKPOINT),
        "attention_backend": "",
        "utility_surface": "pchip",
        "decode_integration_stride": 1024,
        "saturation_step_threshold": 1 << 62,
        "parallelism": {"tp": 1, "pp": 1, "dp": 1, "cp": 1},
    }
    value.update(overrides)
    return OmegaConf.create(value)


def test_the_schema_carries_the_composed_fields():
    """Hydra validates against the dataclass, so an absent field cannot be overridden."""
    config = RolloutCostConfig()
    assert config.kernel_table_root == ""
    assert config.gpu == ""
    assert config.model_path == ""
    assert config.attention_backend == ""
    assert config.utility_surface == "pchip"
    assert config.decode_integration_stride == 1024
    # Disabled by default: the fast path measured 1.20x for 4.62% deviation end to end.
    assert config.saturation_step_threshold == 1 << 62


def test_a_composed_model_is_built_from_config():
    model = build_rollout_cost(_config())
    assert isinstance(model, ComposedRollout)


def test_the_built_model_predicts_a_positive_decode_step():
    from srl.cost.rollout import RolloutWorkload

    model = build_rollout_cost(_config())
    workload = RolloutWorkload(
        phase="decode", n_tokens=16, sum_l2=0.0, ctxsum=16 * 2048.0, request_count=16
    )
    assert model.predict(workload) > 0.0


def test_the_surface_choice_reaches_the_prediction():
    """Two surfaces must give different predictions, or the setting is not wired."""
    from srl.cost.rollout import RolloutWorkload

    workload = RolloutWorkload(
        phase="decode", n_tokens=48, sum_l2=0.0, ctxsum=48 * 3000.0, request_count=48
    )
    pchip = build_rollout_cost(_config(utility_surface="pchip")).predict(workload)
    shepard = build_rollout_cost(_config(utility_surface="shepard")).predict(workload)
    assert pchip != pytest.approx(shepard, rel=1e-6)


def test_tp_above_one_builds_now_that_the_all_reduce_op_exists():
    """Replaces a test asserting tp>1 was rejected. The collective table removed that refusal.

    Construction only. This config points at the tp=1 tables, which cover neither the tp=2
    sharded GEMM shapes nor a collective, so a prediction raises a coverage error naming what
    was never measured rather than silently returning a number. That refusal moving from
    construction to the query is the point: it now names the specific missing table.
    """
    from srl.cost.kernel.interp import KernelCoverageError
    from srl.cost.rollout import RolloutWorkload

    model = build_rollout_cost(_config(parallelism={"tp": 2, "pp": 1, "dp": 1, "cp": 1}))
    assert isinstance(model, ComposedRollout)
    workload = RolloutWorkload(
        phase="decode", n_tokens=16, sum_l2=0.0, ctxsum=16 * 2048.0, request_count=16
    )
    with pytest.raises(KernelCoverageError):
        model.predict(workload)


def test_tp_two_against_the_tp2_tree_prices_a_collective():
    """The degree in config must reach `comm_queries`, and a tp=2 step must pay for collectives.

    The preceding test pins the failure when the tree is wrong. This pins the success when it is
    right, which is what `ROLLOUT_TP` in simulate.sh exists to reach. Without this, a config whose
    tp never propagated would look identical to a working one: the prediction would still be a
    plausible number, just missing 72 all-reduces.
    """
    from srl.cost.kernel.model_ops import ModelOps
    from srl.cost.rollout import RolloutWorkload

    tables = _REPO_ROOT / "outputs/profile/qwen3_8b/kernel_tp2/kernel"
    if not (tables / "h20" / "collective").is_dir():
        pytest.skip("Needs the tp=2 collective sweep.")

    model = build_rollout_cost(
        _config(
            kernel_table_root=str(tables),
            parallelism={"tp": 2, "pp": 1, "dp": 1, "cp": 1},
        )
    )
    assert model._ops.tensor_parallel == 2
    workload = RolloutWorkload(
        phase="decode", n_tokens=16, sum_l2=0.0, ctxsum=16 * 2048.0, request_count=16
    )
    total = model.predict(workload)
    assert total > 0.0

    ops = ModelOps.from_checkpoint(str(_CHECKPOINT), tensor_parallel=2)
    comm_queries = ops.comm_queries(16)
    assert len(comm_queries) == 1, "A tp=2 step must emit exactly one collective query shape."
    # Two per layer, and the multiplier is the thing a plausible-looking layer-count bug hides in.
    assert comm_queries[0][1] == 2 * ops.num_layers
    comm = sum(mult * model._database.query_comm(q) for q, mult in comm_queries)
    assert 0.0 < comm < total, f"Collective {comm} must be a real, bounded share of {total}."


def test_tp_one_pays_for_no_collective():
    """The mirror of the above: at one rank there is no collective to pay for."""
    from srl.cost.kernel.model_ops import ModelOps

    ops = ModelOps.from_checkpoint(str(_CHECKPOINT), tensor_parallel=1)
    assert ops.comm_queries(16) == ()


def test_a_missing_table_root_is_rejected():
    with pytest.raises(ValueError, match="kernel_table_root"):
        build_rollout_cost(_config(kernel_table_root=""))


def test_a_missing_model_path_is_rejected():
    with pytest.raises(ValueError, match="model_path"):
        build_rollout_cost(_config(model_path=""))


def test_a_missing_gpu_is_rejected():
    with pytest.raises(ValueError, match="gpu"):
        build_rollout_cost(_config(gpu=""))


def test_an_empty_attention_backend_means_the_vllm_selected_one():
    """A prediction must reflect the backend the server runs, which on H20 is FA3, not FlashInfer."""
    model = build_rollout_cost(_config(attention_backend=""))
    explicit = build_rollout_cost(_config(attention_backend="vllm"))
    from srl.cost.rollout import RolloutWorkload

    workload = RolloutWorkload(
        phase="decode", n_tokens=16, sum_l2=0.0, ctxsum=16 * 2048.0, request_count=16
    )
    assert model.predict(workload) == pytest.approx(explicit.predict(workload), rel=1e-12)


def test_the_other_models_are_unaffected():
    """Adding a branch must not perturb the existing ones."""
    from srl.cost.rollout import UnifiedRoofline

    model = build_rollout_cost(OmegaConf.create({"name": "roofline", "F": 0.005}))
    assert isinstance(model, UnifiedRoofline)


def test_saturation_step_threshold_is_wired_through_config():
    """The saturation_step_threshold config field must reach the model's routing logic."""
    # At ctxsum/n_d = 512 (below 8192), the default model uses the lattice.
    # With threshold=0, every segment uses closed form instead.
    n_d, ctxsum = 16, 16 * 512.0
    default_model = build_rollout_cost(_config())
    always_closed = build_rollout_cost(_config(saturation_step_threshold=0))
    assert default_model._use_closed_form(n_d, ctxsum) is False
    assert always_closed._use_closed_form(n_d, ctxsum) is True
