"""ComposedRollout's routed-MoE term, against a real profiled `moe` table.

Needs a real `moe` perf table under the kernel tree, which only exists after
`srl-profile kernels run --family moe --config examples/profile/kernel/qwen3_30b_a3b_h20_tp1.yaml`
has been run on a GPU. Absent that, this file skips entirely, the same convention
`tests/test_rollout_composed.py` uses for Qwen3-8B's real checkpoint and tables.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from srl.cost.kernel.database import AttentionQuery, GemmQuery, KernelDatabase
from srl.cost.kernel.model_ops import ModelOps
from srl.cost.rollout import RolloutWorkload
from srl.cost.rollout.composed import ComposedRollout

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CHECKPOINT = Path("/apdcephfs_zwfy10/share_303541817/lhy/models/Qwen3-30B-A3B")
_TABLES = _REPO_ROOT / "outputs/profile/qwen3_30b_a3b/kernel_tp1/kernel"
_MOE_TABLE_GLOB = "h20/moe/vllm/*/moe_perf.parquet"

pytestmark = pytest.mark.skipif(
    not (_CHECKPOINT / "config.json").exists() or not any(_TABLES.glob(_MOE_TABLE_GLOB)),
    reason="Needs the real Qwen3-30B-A3B checkpoint and a profiled `moe` kernel table.",
)


@pytest.fixture(scope="module")
def ops():
    return ModelOps.from_checkpoint(str(_CHECKPOINT), tensor_parallel=1)


@pytest.fixture(scope="module")
def database():
    return KernelDatabase.from_root(str(_TABLES), gpu="h20", utility_surface="pchip")


@pytest.fixture
def model(database, ops):
    return ComposedRollout(database, ops, tensor_parallel=1)


def test_moe_queries_is_nonempty_for_this_architecture(ops):
    assert ops.moe_layers == ops.num_layers
    assert len(ops.moe_queries(token_count=64)) == 1


def test_a_decode_step_equals_the_hand_composed_sum_including_moe(model, database, ops):
    """Nothing may be dropped or double-counted relative to an explicit sum."""
    n_d, ctx = 16, 2048
    moe_query, moe_multiplier = ops.moe_queries(token_count=n_d)[0]
    # `router` stays on the dense table; `expert_gate_up`/`expert_down` are priced by
    # `moe_query` instead, off the real measured table -- summing them here too would
    # double-count the routed-expert cost.
    non_expert_ops = [op for op in ops.projection_ops if op.name not in ("expert_gate_up", "expert_down")]
    expected = (
        ops.num_layers
        * sum(database.query_gemm(GemmQuery(m=n_d, n=op.n, k=op.k)) for op in non_expert_ops)
        + ops.attention_layers
        * database.query_attention(
            AttentionQuery(
                phase="generation",
                batch_size=n_d,
                num_heads=ops.num_heads,
                num_kv_heads=ops.num_kv_heads,
                head_dim=ops.head_dim,
                step=ctx,
            )
        )
        + moe_multiplier * database.query_moe(moe_query)
        + database.query_gemm(GemmQuery(m=n_d, n=ops.logits_op.n, k=ops.logits_op.k))
    )
    workload = RolloutWorkload(
        phase="decode", n_tokens=n_d, sum_l2=0.0, ctxsum=float(n_d * ctx), request_count=n_d
    )
    assert model.predict(workload) == pytest.approx(expected, rel=1e-9)


def test_cpp_term_spec_still_refuses_a_routed_moe_model(model):
    """No regression: the fast C++ backend never assembled MoE, before or after this change."""
    assert model.cpp_term_spec() is None
