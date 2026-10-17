# deprecated/

Code, tests, scripts, and notes for the SCG scheduling line, moved out of the live tree with their
repository-relative paths preserved. Nothing here runs. It is kept so the measurements and the
reasoning behind them stay recoverable.

## What moved

| Group | Paths under `deprecated/` |
|---|---|
| Value pricing | `srl/value/` (length models, deadline budget, marginal valuator, tool-time model) |
| Routing strategies | `srl/component/rollout/router/strategies/{scg_admission,planned_tier_tp,version_aware_tp,version_capped_tp}.py`, `srl/component/rollout/router/estimator.py` |
| Eviction | `srl/component/rollout/engine/eviction/scg_value.py` |
| Fleet solvers | `srl/simulator/dynamic_parallel/{solver,scg_value,planner}.py` (`ParallelismSolver`, the SCG plan scorer, `TierPlanner`) |
| Policy | `srl/policy/roll_doom.py`, `srl/config/defaults/policy/roll_doom.yaml` |
| Tests | `tests/test_scg_admission.py`, `test_tier_planner.py`, `test_value_length.py`, `test_value_marginal.py`, `test_policy_roll_doom.py`, `test_dynamic_parallel.py` (routing preference and solver halves only), `test_simulator_dynamic_parallel.py` |
| Examples | `examples/simulate/{scg_sweep.sh,scg_14b_fleet_shapes.sh,scg_14b_sweep_conditions.sh,scg_admission_sweep.sh}`, three `README_*` notes beside them, `examples/workload/make_heavy_tail_trace.py` |
| Figures | `outputs/simulate/scg_sweep/compare/*.png` |
| Notes | `docs/design/scg-scheduling-*.md`, `docs/superpowers/plans/2026-10-06-heterogeneous-tp-scg.md`, `docs/superpowers/specs/2026-10-06-heterogeneous-tp-scg-design.md` |

## Why

The line priced each placement, eviction, or fleet shape by its marginal contribution to SCG(S),
using a learned or oracle length model. It was abandoned for two reasons.

- Under `policy=psrl` the staleness deadline is a commitment made at admission, so SCG(S) reduces
  to throughput.
- The tier planner's gain was measured with oracle lengths. With deployable length predictors it
  fell below the homogeneous baseline.

The replacement is the inventory-deadline scheduler in
`docs/superpowers/specs/2026-10-13-inventory-deadline-scheduling-design.md`. It forecasts when each
staleness buffer becomes ready and admits, routes, and reshapes the fleet against that forecast.

## What stayed in the live tree

- The SCG(S) metric: `SimResult.scg`, `scg_count`, `scg_curve`, `scg_w`, the plots, and `tests/test_scg.py`.
- `TrajView` (now in the engine) and `srl/forecast/` for length prediction.
- `srl/simulator/dynamic_parallel/{plan,fleet}.py` with `tests/test_dynamic_parallel_fleet.py`.
  Nothing in `Simulator` constructs the fleet today.
- The eviction framework, `roll.py`, and the `length_aware` router strategy.

## This directory is inert

- **Not packaged.** `pyproject.toml` sets `[tool.setuptools.packages.find] include = ["srl*"]`.
  `deprecated` does not match, and `deprecated/srl` has the dotted name `deprecated.srl`, which
  does not match either.
- **Not collected.** `[tool.pytest.ini_options] testpaths = ["tests"]`, so a bare `pytest` never
  descends here.
- **Not importable.** The modules use relative imports that resolve against the live package layout
  (`from ...engine import InstanceView`, `from .....value import ...`). `deprecated/` has no
  `__init__.py` and does not mirror the engine, so those imports have nothing to land on, and
  `import deprecated` resolves to an unrelated third-party package of that name.

## Where the measurements live

- Result tables and reproduction commands: the three `README_*` notes under
  `deprecated/examples/simulate/`.
- Design history and the reasoning that ended each version: `deprecated/docs/design/`.
- Figures: `deprecated/outputs/simulate/scg_sweep/compare/`.
- Raw sweep outputs: `outputs/simulate/scg_sweep/{search,tp1_x4,tp2_x2}` stay in place. They are
  untracked and not copied here.
- Higher-level conclusions: the project memory notes on SCG, TierPlanner, heterogeneous TP, and
  doomed-trajectory abandonment.
- The comparison against Heddle and Libra:
  `docs/design/heterogeneous-rollout-practice-and-papers.md`.

To revive any of this, revert the commit that created this directory instead of copying files back.
