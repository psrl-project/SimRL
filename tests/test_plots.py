"""Tests that SimPlotter renders correctly from a JsonlStore, with no SimResult involved."""

from __future__ import annotations

from dryrun.cost.rollout import UnifiedRoofline
from dryrun.policy.psrl import PSRLPolicy
from dryrun.simulator import SimConfig, Simulator
from dryrun.telemetry.store import JsonlStore
from dryrun.visualizer.plots import SimPlotter
from dryrun.workload.distributions import uniform
from dryrun.workload.grouped import GroupedWorkload


def _simulate_into_store(tmp_path, n_instances=2):
    store = JsonlStore(tmp_path / "logs")
    cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7)
    cfg = SimConfig(
        batch_size=4,
        max_staleness=2,
        n_versions=5,
        n_instances=n_instances,
        workload=GroupedWorkload.from_lengths([512], uniform(80, 50, 120, seed=1)),
    )
    sim = Simulator(
        rollout_cost=cost,
        policy=PSRLPolicy(),
        cfg=cfg,
        store=store,
    )
    sim.run()
    store.close()
    return store


class TestPlotAllFromRealRun:
    def test_plot_all_produces_six_pngs(self, tmp_path):
        store = _simulate_into_store(tmp_path)
        plotter = SimPlotter(store, tmp_path / "plots")
        paths = plotter.plot_all()
        assert len(paths) == 6
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0

    def test_instance_load_has_one_subplot_per_instance(self, tmp_path):
        store = _simulate_into_store(tmp_path, n_instances=3)
        plotter = SimPlotter(store, tmp_path / "plots")
        plotter.plot_instance_load()
        assert plotter._instance_ids() == [0, 1, 2]

    def test_reopening_store_readonly_still_plots(self, tmp_path):
        """A `dryrun-plot`-style re-render against a fresh JsonlStore over the same logs dir."""
        _simulate_into_store(tmp_path)
        reopened = JsonlStore(tmp_path / "logs")
        plotter = SimPlotter(reopened, tmp_path / "plots2")
        paths = plotter.plot_all()
        for p in paths:
            assert p.exists()


class TestPlotAllFromFixtureStore:
    """Hand-built store exercising the plot code without running a simulation."""

    def _build_fixture(self, tmp_path):
        store = JsonlStore(tmp_path / "logs")
        store.write("train", t=1.0, start=0.0, end=1.0, dt=1.0, version=1, batch_size=2)
        store.write("train", t=3.0, start=2.0, end=3.0, dt=1.0, version=2, batch_size=2)

        store.write(
            "instance/0", t=0.0, instance_id=0, num_running_reqs=0, num_waiting_reqs=2,
            kv_cache_usage=0.0, kv_blocks_used=0, total_ctx=0,
        )
        store.write(
            "instance/0", t=0.5, instance_id=0, num_running_reqs=2, num_waiting_reqs=0,
            kv_cache_usage=0.2, kv_blocks_used=20, total_ctx=1024,
        )
        store.write(
            "instance/0", t=1.5, instance_id=0, num_running_reqs=1, num_waiting_reqs=0,
            kv_cache_usage=0.15, kv_blocks_used=15, total_ctx=600,
        )

        store.write(
            "rollout_step", t=0.5, instance_id=0, phase="prefill", steps=1,
            n_prefill=1024, n_decode=0, n_tokens=1024, dt=0.5, saturated=True, saturated_steps=1,
        )
        store.write(
            "rollout_step", t=1.5, instance_id=0, phase="decode", steps=10,
            n_prefill=0, n_decode=20, n_tokens=20, dt=0.1, saturated=False, saturated_steps=0,
        )

        store.write(
            "request", t=0.0, event="admit", rid=0, instance_id=0, admit_time=0.0,
            dispatch_time=0.0, complete_time=None, consume_version=None,
            prompt_len=512, target_len=50, generated=0, ctx=512, v_traj=0, v_min=0, v_max=0,
            num_preemptions=0, num_aborts=0,
        )
        store.write(
            "request", t=1.5, event="complete", rid=0, instance_id=0, admit_time=0.0,
            dispatch_time=0.0, complete_time=1.5, consume_version=None,
            prompt_len=512, target_len=50, generated=50, ctx=562, v_traj=0, v_min=0, v_max=0,
            num_preemptions=0, num_aborts=0,
        )
        store.write(
            "request", t=3.0, event="consume", rid=0, instance_id=0, admit_time=0.0,
            dispatch_time=0.0, complete_time=1.5, consume_version=2,
            prompt_len=512, target_len=50, generated=50, ctx=562, v_traj=0, v_min=0, v_max=1,
            num_preemptions=0, num_aborts=0,
        )
        store.close()
        return store

    def test_all_plots_render(self, tmp_path):
        store = self._build_fixture(tmp_path)
        plotter = SimPlotter(store, tmp_path / "plots")
        paths = plotter.plot_all()
        assert len(paths) == 6
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0

    def test_staleness_distribution_reflects_one_consume_row(self, tmp_path):
        store = self._build_fixture(tmp_path)
        plotter = SimPlotter(store, tmp_path / "plots")
        # consume_version=2, v_min=0, v_max=1 -> true staleness 2, reported staleness 1.
        consume_rows = plotter._consume_rows()
        assert len(consume_rows) == 1
        r = consume_rows[0]
        assert r["consume_version"] - r["v_min"] == 2
        assert r["consume_version"] - r["v_max"] == 1


class TestEmptyStore:
    """No telemetry logged at all (e.g. Simulator constructed without a store)."""

    def test_plot_all_does_not_raise_and_still_writes_placeholders(self, tmp_path):
        store = JsonlStore(tmp_path / "logs")
        plotter = SimPlotter(store, tmp_path / "plots")
        paths = plotter.plot_all()
        assert len(paths) == 6
        for p in paths:
            assert p.exists()


class TestTrainPhaseBand:
    """The band splits into recompute/train/sync, shared by simulated and real runs."""

    def _store_with_train(self, tmp_path, **phase_times):
        store = JsonlStore(tmp_path / "logs", overwrite=True)
        store.write("train", t=10.0, start=0.0, end=10.0, dt=10.0, version=1, **phase_times)
        store.write(
            "instance/0", t=0.0, instance_id=0, num_running_reqs=1, num_waiting_reqs=0,
            kv_cache_usage=0.1,
        )
        store.close()
        return JsonlStore(tmp_path / "logs")

    def test_all_three_phases_are_drawn_in_order(self, tmp_path):
        store = self._store_with_train(
            tmp_path, recompute_time=2.0, train_time=7.0, sync_time=1.0
        )
        spans = SimPlotter(store, tmp_path / "plots")._train_phase_spans()
        assert [(label, start, end) for start, end, label, _ in spans[0]] == [
            ("recompute", 0.0, 2.0),
            ("train", 2.0, 9.0),
            ("sync", 9.0, 10.0),
        ]

    def test_a_zero_width_phase_is_omitted_rather_than_widening_its_neighbour(self, tmp_path):
        """The simulator does not model recompute, and that gap must stay visible."""
        store = self._store_with_train(
            tmp_path, recompute_time=0.0, train_time=10.0, sync_time=0.0
        )
        spans = SimPlotter(store, tmp_path / "plots")._train_phase_spans()
        assert [label for _, _, label, _ in spans[0]] == ["train"]

    def test_unattributed_time_is_charged_to_train(self, tmp_path):
        """A train row predating the phase fields still covers its whole block."""
        store = self._store_with_train(tmp_path)
        spans = SimPlotter(store, tmp_path / "plots")._train_phase_spans()
        assert spans[0] == [(0.0, 10.0, "train", "steelblue")]

    def test_the_figure_renders_with_a_split_band(self, tmp_path):
        store = self._store_with_train(
            tmp_path, recompute_time=2.0, train_time=7.0, sync_time=1.0
        )
        path = SimPlotter(store, tmp_path / "plots").plot_instance_load()
        assert path.exists() and path.stat().st_size > 0


class TestGatewayRow:
    """The router queue row appears only for a run that recorded one."""

    def _store(self, tmp_path, with_gateway: bool):
        store = JsonlStore(tmp_path / "logs", overwrite=True)
        store.write("train", t=10.0, start=0.0, end=10.0, dt=10.0, version=1, train_time=10.0)
        store.write(
            "instance/0", t=0.0, instance_id=0, num_running_reqs=1, num_waiting_reqs=0,
            kv_cache_usage=0.1,
        )
        if with_gateway:
            store.write("gateway", t=0.0, queue_len=5, pending_request_num=5, running_request_num=8)
        store.close()
        return JsonlStore(tmp_path / "logs")

    def test_absent_for_a_simulation(self, tmp_path):
        plotter = SimPlotter(self._store(tmp_path, with_gateway=False), tmp_path / "plots")
        assert plotter._gateway_rows() == []
        assert plotter.plot_instance_load().exists()

    def test_present_for_a_converted_real_run(self, tmp_path):
        plotter = SimPlotter(self._store(tmp_path, with_gateway=True), tmp_path / "plots")
        assert len(plotter._gateway_rows()) == 1
        assert plotter.plot_instance_load().exists()


class TestSCGCurve:
    """SCG(S) curve: exact values, monotonicity, and graceful empty-data handling."""

    def _store_with_staleness(self, tmp_path, staleness_list: list[int]) -> JsonlStore:
        """Build a store with consume rows at the given staleness_true values."""
        store = JsonlStore(tmp_path / "logs")
        for i, s in enumerate(staleness_list):
            store.write(
                "request",
                t=float(i + 1),
                event="consume",
                rid=i,
                instance_id=0,
                admit_time=0.0,
                dispatch_time=0.0,
                complete_time=float(i + 1),
                consume_version=s,
                prompt_len=512,
                target_len=50,
                generated=50,
                ctx=562,
                v_traj=0,
                v_min=0,
                v_max=0,
                staleness_true=s,
                staleness_reported=s,
                num_preemptions=0,
                num_aborts=0,
            )
        store.close()
        return store

    def test_curve_values_are_exact(self, tmp_path):
        # staleness = [0, 0, 1, 2], max complete_time = 4.0
        # SCG(0) = 2/4 = 0.5, SCG(1) = 3/4 = 0.75, SCG(2) = 4/4 = 1.0
        xs, ys = SimPlotter._scg_curve_data([0, 0, 1, 2], span=4.0)
        assert xs == [0, 1, 2]
        assert ys == [0.5, 0.75, 1.0]

    def test_curve_is_non_decreasing_on_mixed_staleness(self, tmp_path):
        store = self._store_with_staleness(tmp_path, [0, 2, 1, 0, 3, 1])
        plotter = SimPlotter(store, tmp_path / "plots")
        path = plotter.plot_scg_curve()
        assert path.exists() and path.stat().st_size > 0

        # Verify monotonicity directly on the helper.
        staleness = [0, 2, 1, 0, 3, 1]
        xs, ys = plotter._scg_curve_data(staleness, span=float(len(staleness)))
        assert xs == list(range(max(staleness) + 1))
        for a, b in zip(ys, ys[1:]):
            assert b >= a, f"SCG curve not non-decreasing: {ys}"

    def test_no_consume_rows_produces_placeholder_not_exception(self, tmp_path):
        store = JsonlStore(tmp_path / "logs")
        store.close()
        plotter = SimPlotter(store, tmp_path / "plots")
        path = plotter.plot_scg_curve()
        # _empty_placeholder writes a file; the important thing is no exception.
        assert path.exists()
        assert path.name == "scg_curve.png"

    def test_consume_rows_without_staleness_true_produce_placeholder(self, tmp_path):
        """Rows missing staleness_true are skipped; if none survive, a placeholder is returned."""
        store = JsonlStore(tmp_path / "logs")
        store.write(
            "request",
            t=1.0,
            event="consume",
            rid=0,
            instance_id=0,
            admit_time=0.0,
            dispatch_time=0.0,
            complete_time=1.0,
            consume_version=2,
            prompt_len=512,
            target_len=50,
            generated=50,
            ctx=562,
            v_traj=0,
            v_min=0,
            v_max=1,
            num_preemptions=0,
            num_aborts=0,
            # staleness_true intentionally absent — simulates a pre-feature log
        )
        store.close()
        plotter = SimPlotter(store, tmp_path / "plots")
        path = plotter.plot_scg_curve()
        assert path.exists()
        assert path.name == "scg_curve.png"

