# Development

SimRL has six extension points. Each is a small protocol with one registry, so adding one does not
require touching the simulator or the CLI.

| Extension point | Protocol | Registry | Page |
|---|---|---|---|
| Kernel op | `KernelOp` | `OP_REGISTRY` in `cost/kernel/ops/base.py` | {doc}`kernel-ops` |
| Model family | `ModelFamily` | `FAMILY_REGISTRY` in `profiler/kernel/grid/families.py` | {doc}`model-families` |
| Profiler adapter | `generate`, `run`, `collect` | none, selected by CLI backend name | {doc}`adapters-and-cost-models` |
| Rollout cost model | `RolloutCostModel` | `ROLLOUT_MODEL_REGISTRY` | {doc}`adapters-and-cost-models` |
| Training cost model | `TrainingCostModel` | `TRAINING_MODEL_REGISTRY` | {doc}`adapters-and-cost-models` |
| Staleness policy | `StalenessPolicy` | `POLICY_REGISTRY` in `policy/registry.py` | {doc}`../simulation/policies` |

There is also a routing strategy protocol (`RoutingStrategy`, `STRATEGY_REGISTRY`) described in
{doc}`../simulation/routing`.

{doc}`architecture` describes how the packages fit together: the three layers, the package tree,
the dependency direction, the event loop and the two engine backends.

---

## Before writing code

Read `.claude/coding-style.md`. Two of its rules prevent silent bugs rather than style complaints:

- **Identifier strings never gain a period or a case change.** A log message ends with `.`, a dict
  key, a status value or a registry name does not. Appending a period to an identifier changes what
  the program looks up.
- **A comment says what the code cannot.** No history, no measurements that justified a choice, no
  restatement of the line below it. Those belong in a commit message.

Two structural rules recur in this codebase:

- **The cost path stays free of `torch`.** A simulation prices kernels from parquet. The modules on
  that path are gated by tests that import them in fresh subprocesses.
- **A registry has one population path.** `ops/__init__.py`, the tail of `families.py` and the
  router's package `__init__` each import every implementation for its side effect. Registration
  order must not decide what is registered.

---

## What to run

{doc}`testing` covers the commands, the tests that skip without local data, the pinned accuracy
figures, and why a re-sweep invalidates them. In short:

```bash
source /path/to/env/simrl.sh      # the environment that holds the dependencies
pytest -q
ruff check .
make -C docs strict
```

{doc}`docs` covers this site: how to build it, where each fact lives, and the conventions the
strict build enforces.

```{toctree}
:maxdepth: 2

architecture
kernel-ops
model-families
adapters-and-cost-models
testing
docs
```
