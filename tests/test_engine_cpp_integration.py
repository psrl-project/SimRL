"""End-to-end: C++ backend produces identical results to Python backend."""

from __future__ import annotations

import pytest

try:
    from srl._engine_cpp import CppInstance, CppRequest  # noqa: F401

    HAS_CPP = True
except ImportError:
    HAS_CPP = False

from srl.cost.rollout import UnifiedRoofline
from srl.cost.training import (
    FixedTrainingLatency,
    FixedTrainingMemory,
    TrainingCostModel,
)
from srl.policy.areal import ArealPolicy
from srl.policy.psrl import PSRLPolicy
from srl.policy.roll import RollPolicy
from srl.policy.verl import VerlPolicy
from srl.simulator import SimConfig, Simulator
from srl.telemetry.store import JsonlStore
from srl.workload.distributions import bimodal, uniform
from srl.workload.grouped import GroupedWorkload

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="C++ engine not compiled")


def _make_sim(policy, cfg, backend, store=None):
    cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7)
    training_cost = TrainingCostModel(
        FixedTrainingLatency(0.5),
        FixedTrainingMemory(0),
    )
    return Simulator(
        rollout_cost=cost,
        training_cost=training_cost,
        policy=policy,
        cfg=cfg,
        store=store,
        backend=backend,
    )


def _compare_results(native_result, cpp_result, *, rtol=1e-9):
    """Assert that two SimResults agree on all observable quantities."""
    assert native_result.versions_done == cpp_result.versions_done, (
        f"versions_done: native={native_result.versions_done}, "
        f"cpp={cpp_result.versions_done}"
    )
    assert native_result.livelocked == cpp_result.livelocked
    assert len(native_result.consumed) == len(cpp_result.consumed), (
        f"consumed count: native={len(native_result.consumed)}, "
        f"cpp={len(cpp_result.consumed)}"
    )
    assert len(native_result.dropped) == len(cpp_result.dropped), (
        f"dropped count: native={len(native_result.dropped)}, "
        f"cpp={len(cpp_result.dropped)}"
    )
    assert native_result.sim_time == pytest.approx(
        cpp_result.sim_time, rel=rtol
    ), (
        f"sim_time: native={native_result.sim_time}, "
        f"cpp={cpp_result.sim_time}"
    )

    # Compare per-request data, sorted by rid.
    native_reqs = sorted(native_result.consumed, key=lambda r: r.rid)
    cpp_reqs = sorted(cpp_result.consumed, key=lambda r: r.rid)
    for n_req, c_req in zip(native_reqs, cpp_reqs):
        assert n_req.rid == c_req.rid
        assert n_req.prompt_len == c_req.prompt_len
        assert n_req.target_len == c_req.target_len
        assert n_req.generated == c_req.generated, (
            f"rid {n_req.rid}: generated native={n_req.generated}, "
            f"cpp={c_req.generated}"
        )
        assert n_req.v_traj == c_req.v_traj
        assert n_req.group_id == c_req.group_id
        assert n_req.staleness_true() == c_req.staleness_true()


class TestCppBackendMatchesNative:
    """Run the same simulation under both backends and compare results."""

    def test_psrl_uniform(self):
        cfg = SimConfig(
            batch_size=4,
            max_staleness=2,
            n_versions=8,
            log_telemetry=False,
        )
        lengths = uniform(80, 50, 200, seed=42)
        cfg.workload = GroupedWorkload.from_lengths([512], lengths)
        native = _make_sim(PSRLPolicy(), cfg, "native").run()
        cpp = _make_sim(PSRLPolicy(), cfg, "cpp").run()
        _compare_results(native, cpp)

    def test_areal_uniform(self):
        cfg = SimConfig(
            batch_size=4,
            max_staleness=2,
            n_versions=6,
            log_telemetry=False,
        )
        lengths = uniform(60, 50, 100, seed=0)
        cfg.workload = GroupedWorkload.from_lengths([256], lengths)
        native = _make_sim(ArealPolicy(), cfg, "native").run()
        cpp = _make_sim(ArealPolicy(), cfg, "cpp").run()
        _compare_results(native, cpp)

    def test_roll_uniform(self):
        cfg = SimConfig(
            batch_size=4,
            max_staleness=2,
            n_versions=6,
            log_telemetry=False,
        )
        lengths = uniform(60, 50, 100, seed=0)
        cfg.workload = GroupedWorkload.from_lengths([256], lengths)
        native = _make_sim(RollPolicy(), cfg, "native").run()
        cpp = _make_sim(RollPolicy(), cfg, "cpp").run()
        _compare_results(native, cpp)

    def test_verl_uniform(self):
        cfg = SimConfig(
            batch_size=4,
            max_staleness=1,
            n_versions=6,
            log_telemetry=False,
        )
        lengths = uniform(60, 50, 100, seed=0)
        cfg.workload = GroupedWorkload.from_lengths([256], lengths)
        policy_n = VerlPolicy(staleness_threshold=0.1)
        policy_c = VerlPolicy(staleness_threshold=0.1)
        native = _make_sim(policy_n, cfg, "native").run()
        cpp = _make_sim(policy_c, cfg, "cpp").run()
        _compare_results(native, cpp)

    def test_two_instances(self):
        cfg = SimConfig(
            batch_size=4,
            max_staleness=2,
            n_versions=6,
            n_instances=2,
            log_telemetry=False,
        )
        lengths = uniform(60, 50, 100, seed=0)
        cfg.workload = GroupedWorkload.from_lengths([256], lengths)
        native = _make_sim(PSRLPolicy(), cfg, "native").run()
        cpp = _make_sim(PSRLPolicy(), cfg, "cpp").run()
        _compare_results(native, cpp)

    def test_partial_rollout(self):
        cfg = SimConfig(
            batch_size=4,
            max_staleness=2,
            n_versions=4,
            n_instances=1,
            partial_rollout=True,
            sync_time=0.25,
            log_telemetry=False,
        )
        cfg.workload = GroupedWorkload.from_lengths(
            [512], uniform(80, 200, 400, seed=0)
        )
        native = _make_sim(ArealPolicy(), cfg, "native").run()
        cpp = _make_sim(ArealPolicy(), cfg, "cpp").run()
        _compare_results(native, cpp)

    def test_bimodal_lengths(self):
        cfg = SimConfig(
            batch_size=4,
            max_staleness=2,
            n_versions=6,
            log_telemetry=False,
        )
        lengths = bimodal(80, short_len=50, long_len=500, long_frac=0.25, seed=42)
        cfg.workload = GroupedWorkload.from_lengths([512], lengths)
        native = _make_sim(PSRLPolicy(), cfg, "native").run()
        cpp = _make_sim(PSRLPolicy(), cfg, "cpp").run()
        # Bimodal with long sequences accumulates more floating-point drift
        # between the Python and C++ cost model arithmetic. Use a looser
        # tolerance but still tight enough to catch real bugs.
        _compare_results(native, cpp, rtol=1e-3)


class TestCppBackendTelemetry:
    """Verify that the C++ backend produces valid telemetry."""

    def test_telemetry_streams_populated(self, tmp_path):
        store_native = JsonlStore(tmp_path / "native")
        store_cpp = JsonlStore(tmp_path / "cpp")
        cfg = SimConfig(
            batch_size=4,
            max_staleness=2,
            n_versions=4,
            n_instances=1,
        )
        cfg.workload = GroupedWorkload.from_lengths(
            [512], uniform(50, 50, 100, seed=0)
        )
        native_result = _make_sim(
            PSRLPolicy(), cfg, "native", store=store_native,
        ).run()
        cpp_result = _make_sim(
            PSRLPolicy(), cfg, "cpp", store=store_cpp,
        ).run()
        store_native.close()
        store_cpp.close()

        _compare_results(native_result, cpp_result)

        # Both backends must produce the same streams.
        native_trains = list(store_native.iter("train"))
        cpp_trains = list(store_cpp.iter("train"))
        assert len(native_trains) == len(cpp_trains), (
            f"train stream length: native={len(native_trains)}, "
            f"cpp={len(cpp_trains)}"
        )

        native_reqs = list(store_native.iter("request"))
        cpp_reqs = list(store_cpp.iter("request"))
        assert len(native_reqs) == len(cpp_reqs), (
            f"request stream length: native={len(native_reqs)}, "
            f"cpp={len(cpp_reqs)}"
        )

        # Compare train records field by field.
        for i, (n_train, c_train) in enumerate(zip(native_trains, cpp_trains)):
            assert n_train["version"] == c_train["version"], (
                f"train[{i}] version mismatch."
            )
            assert n_train["n_reqs"] == c_train["n_reqs"], (
                f"train[{i}] n_reqs mismatch."
            )
            assert n_train["start"] == pytest.approx(
                c_train["start"], rel=1e-9
            ), f"train[{i}] start mismatch."
            assert n_train["end"] == pytest.approx(
                c_train["end"], rel=1e-9
            ), f"train[{i}] end mismatch."


class TestCppBackendGrouping:
    """Verify that grouping works correctly with the C++ backend."""

    def test_grouped_workload(self):
        from srl.workload.trace import RequestGroup

        groups = [
            RequestGroup(group_id=i, prompt_len=128, output_lens=(50, 60, 70, 80))
            for i in range(6)
        ]
        cfg = SimConfig(
            batch_size=2,
            group_size=4,
            max_staleness=2,
            n_versions=4,
            workload=GroupedWorkload.from_groups(groups),
            log_telemetry=False,
        )
        native = _make_sim(PSRLPolicy(), cfg, "native").run()
        cpp = _make_sim(PSRLPolicy(), cfg, "cpp").run()
        _compare_results(native, cpp)


class TestCppBackendAutoDetection:
    """Test the auto backend selection logic."""

    def test_auto_selects_cpp_when_available(self):
        cfg = SimConfig(
            batch_size=4, max_staleness=2, n_versions=2,
            log_telemetry=False,
        )
        cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7)
        sim = Simulator(
            rollout_cost=cost,
            policy=PSRLPolicy(),
            cfg=cfg,
            backend="auto",
        )
        assert sim._backend == "cpp"
