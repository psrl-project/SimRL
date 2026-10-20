# Engine backends

The rollout engine exists twice. The Python engine is the reference and the only place
every feature is implemented. The C++ engine is a port of the same event loop, request
model and KV pool, built as a pybind11 extension for speed. Everything above the engine
(simulator, router, policy, forecast, telemetry) is Python and shared, so a backend is a
choice of how `NativeInstance` and `Request` are implemented, not a different simulator.

| | Python engine | C++ engine |
|---|---|---|
| Location | `srl/component/rollout/engine/` | `srl/component/rollout/engine_cpp/` |
| Instance class | `NativeInstance` | `CppInstance` (`srl._engine_cpp`) |
| Request class | `Request` dataclass | `CppRequest` |
| KV pool, free list, undo | `kv/`, `undo/` | `kv_manager.h`, `pool.h`, `free_list.h`, `undo.h` and their sources |
| Cost model | called directly | called through `PyWrappedCostModel`, see below |
| Eviction strategy | Python object | the same Python object, called back from C++ |

## Selecting a backend

`srl/cli/main.py` reads `SIMRL_BACKEND` and passes it to `Simulator(backend=...)`.

| `SIMRL_BACKEND` | Behaviour |
|---|---|
| `cpp` (default) | require the C++ engine. An `ImportError` is raised if the extension is not built, a `ValueError` if a configured strategy or eviction is `python_engine_only` |
| `auto` | use the C++ engine if it imports and nothing configured is `python_engine_only`, otherwise the Python engine. Never raises on its own account |
| `native` | the Python engine |

Any other value raises `ValueError`. The `Simulator` constructor itself defaults to `native`,
since it is the reference implementation used directly by tests and scripts. The CLI defaults
to `cpp`.

The run log does not print which engine was chosen. To check, run
`python -c "import srl._engine_cpp"`, or read `Simulator._backend` after construction, which
`auto` resolves to `cpp` or `native`.

```bash
srl policy=psrl ...                        # cpp by default
SIMRL_BACKEND=native srl policy=psrl ...   # the reference engine
SIMRL_BACKEND=auto   srl policy=verl ...   # degrade to native if the extension is missing
```

`inventory_deadline` and the `latest_deadline` eviction strategy stamp `deadline_buffer` on
requests, which a `CppRequest` cannot carry. Components marked `python_engine_only` make `auto`
resolve to the Python engine, and the default `cpp` raise `ValueError` rather than silently
degrade. Use the Python engine also when you want to step through engine code, or when the
extension is stale.

## Building the extension

The extension is the single module `_engine_cpp`, loaded as `srl._engine_cpp`. Its
interpreter-specific file lives next to the package:

```
srl/_engine_cpp.cpython-312-x86_64-linux-gnu.so
```

The file is tracked by git and is **not** built by `pip install`: the package metadata
(`pyproject.toml`) is pure setuptools. The suffix must match the running interpreter, so a
different Python version needs its own build. The C++ sources compile with C++17 and need
CMake 3.15 or later and pybind11.

```bash
source /apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh
cmake -S srl/component/rollout/engine_cpp -B /tmp/engine_cpp_build \
    -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build /tmp/engine_cpp_build -j4
# Install by writing a temporary name in srl/ and renaming it over the old file.
cp /tmp/engine_cpp_build/_engine_cpp.cpython-312-x86_64-linux-gnu.so \
    srl/_engine_cpp.cpython-312-x86_64-linux-gnu.so.tmp
mv srl/_engine_cpp.cpython-312-x86_64-linux-gnu.so.tmp \
    srl/_engine_cpp.cpython-312-x86_64-linux-gnu.so
```

Build out of the source tree: the repository's `.gitignore` excludes `build/`, and a
`CMakeCache.txt` records the absolute source path it was configured from, so a stale cache
refuses to be reused from another checkout. Keep the parallelism low on a memory-constrained
host.

Two cautions about replacing the file:

- Install by rename, not by copying over the old file. A running simulation has the extension
  mapped, and overwriting it in place can crash that process. Do not rebuild while sweeps are
  running. The sweep scripts under `examples/analysis/` record the file's checksum before and
  after to catch this.
- After any change under `engine_cpp/`, rebuild and rerun `tests/test_engine_cpp_*`, because
  the Python and C++ engines must be changed together.

`tests/test_engine_cpp_*` skip when the extension cannot be imported.

## Cost models across the boundary

The C++ engine prices every step through a cost model. The simulator wraps whatever Python
`RolloutCostModel` it was given in `PyWrappedCostModel` (`_make_cpp_cost`). The wrapper is a
trampoline: each of `step_time`, `decode_segment_time`, `decode_steps_within`,
`decode_saturated_steps`, `decode_coeffs` and `saturated` calls the Python object's method of
the same name. Forwarding all of them matters. A model such as `ComposedRollout` integrates
decode segments exactly, while the C++ defaults rebuild each segment query from
`decode_coeffs`, which is a secant approximation. A model overriding only some methods would
otherwise be priced differently on the two backends.

**Term memos.** A model may publish `cpp_term_spec()`. `ComposedRollout` does so for dense
models, and returns `None` for a model with gated delta-net layers or routed experts. When
the spec exists, `PyWrappedCostModel.step_time` is assembled inside C++ from two memos: the
GEMM and communication term keyed by `(tokens, requests)`, and one layer's attention term
keyed by the integer shapes `ComposedRollout` itself would query. Python is called only on
a memo miss. The keys are derived with CPython's `//`, `round` (half to even) and `int`
semantics, and the terms are summed in the order `predict` sums them, so the result is
bit-identical to `predict`. Inputs Python would reject go to Python so it raises. Each memo
is cleared when it reaches 2^18 entries. Entries are exact, so clearing costs only
recomputation. `PyWrappedCostModel.term_fast_path` reports whether the memo path is active.

A model without a spec is called through Python on every step. That is slower and exact.
The native `UnifiedRooflineCostModel` and `PsrlCostModel` classes in C++ are used by tests
that build a `CppInstance` directly. The simulator always goes through the trampoline.

The router and the forecast package see the Python cost model, not the wrapper, so a strategy
that prices a hypothetical batch uses the same coefficients on both backends.

## Selective undo

Instances share one global clock by speculation. `Simulator._run_engine` asks every active
instance for its next event, commits the earliest, and rolls back the rest, and it rolls back
everything past a training deadline. Every mutation of engine state is therefore reversible:

- The Python engine records old values into an `UndoLog` only for what an event touches
  (`record_once` / `record_attrs_once`, unwound newest-first). Commit clears the log.
  Cost is proportional to the work done by the event, never to the size of the pool.
- The C++ engine keeps a journal of typed inverse operations (`undo.h`): field snapshots,
  list restores and truncations, per-request snapshots, free-list relinks, reference-count
  and hash journals. The same discipline applies.

`undo_last_event` after `advance_one_event`, then advancing again, must be invisible. The
test `test_engine_cpp_undo_invisible.py` runs two instances in lockstep, one with
undo-and-replay before every commit, and compares full block-level state at every step on
pools tight enough to force preemption.

**Which events are speculative.** `SimConfig.undoable_events`, set from
`SIMRL_UNDOABLE_EVENTS` (comma separated, default `request_complete,preempt,iteration_done`),
chooses the event kinds whose *successor* is produced with an open undo log. `request_complete`
must be present, because completions decide the order batches are collected in.

| Setting | Behaviour |
|---|---|
| default, all three | every event can be undone. Results are exact |
| `request_complete` only | the successor of a `preempt` or `iteration_done` event is produced with `advance_one_event_committed`, with no snapshot. The dominant bookkeeping cost disappears, but a committed advance cannot be rolled back, so an instance can end up ahead of the global clock. `_align_instances` then leaves it where it is and widens the clock, so results can differ from the exact mode |

**Iteration batching.** When the `instance/<id>` telemetry stream is suppressed
(`+log_exclude='instance'`), the simulator lets the engine commit consecutive
`ITERATION_DONE` events itself (`advance_iterations`), stopping at the first event that is
anything else, passes the deadline, or would not be popped next. The order of steps is the
same. Every committed state and written row is reproduced except the per-event occupancy
sample, which is why it is only enabled when that stream is not written.

## What is exact and what is not

The engines are deterministic, and the C++ engine is meant to reproduce the Python one
exactly, not approximately. Equivalence is asserted by tests rather than assumed. For the same
configuration:

| Quantity | Relationship | Test |
|---|---|---|
| Event-by-event state of one instance (queues, counters, per-request progress, KV counters) | equal | `test_engine_cpp_instance.py` |
| Undo then replay, including block-level KV state | equal | `test_engine_cpp_undo_invisible.py` |
| `versions_done`, `livelocked`, consumed and dropped counts, per-request lengths, versions and staleness | equal | `test_engine_cpp_integration.py` |
| `sim_time` | relative 1e-9, relative 1e-3 for the long bimodal case | same |
| `request.jsonl` rows and `sim_time` under `vllm_lru` and `longest_ctx` eviction on a pool that preempts | identical | `test_engine_cpp_eviction.py` |
| `ComposedRollout.step_time` through the term memos | bit-identical to `predict` (needs the measured kernel tables) | `test_composed_term_memo.py` |
| Every cost-model method overridden in Python | reached from C++ | `test_engine_cpp_cost_dispatch.py` |

The tolerances in the integration test are looser than the behaviour. A default two-instance
`psrl` run on the roofline model, and the test's own long bimodal case, gave identical
`sim_time` on both backends when checked for this page, and the default run produced
byte-identical `request`, `train`, `gateway`, `rollout_step` and `instance/*` streams. Those
are observations, not guarantees: the table is the guarantee.

What is **not** the same:

- **Fast undo mode** is an approximation on either backend, see above.
- **Requests have a fixed schema in C++.** `CppRequest` rejects new attributes. This is why
  the simulator keeps drop reasons in `SimResult.dropped_reasons` rather than on the request,
  and why `deadline_buffer` is unavailable.
- **Reads that convert.** A C++ `waiting`, `running`, `completed` or `step_stats` read builds a
  new Python list each time. The simulator and telemetry avoid this with `n_active`,
  `n_running`, `n_waiting`, `n_step_stats` and `step_stats_since`. A strategy that reads
  `inst.waiting` per placement pays for it.
- **Cost-model speed.** A model with no `cpp_term_spec` pays a Python call per step. The
  results are the same.

## Verifying a change

`examples/analysis/diff_backends.py` runs both backends in-process on a set of cases that
reach preemption, chunked prefill, prefix caching on and off, partial-rollout aborts,
concurrency caps and several instances, and compares every telemetry stream row by row. Its
`--mode batching` compares the iteration-batched engines against event-by-event stepping.
`examples/analysis/bench_backends.sh` runs four arms (native and C++, exact and fast undo) on
one configuration and reports an arm as DIVERGED if its request telemetry is not
byte-identical to native exact undo. For a new cost model or a new engine feature,
run the same configuration under `SIMRL_BACKEND=native` and `SIMRL_BACKEND=cpp` and compare
`logs/request.jsonl` and `logs/train.jsonl`.
