# Execution checklist

Tick each item in order. An unticked item means the result is not reportable yet.

## Before running

- [ ] The change is behind a flag whose default reproduces the baseline bit for bit.
- [ ] The control arm and every treatment arm are declared in one experiment script.
- [ ] Each arm runs the same workload: same model, TP, sequence lengths, batch, group size,
      staleness, and seed.
- [ ] Static baselines are swept at every point, not carried over from another point.
- [ ] When sweeping a resource count (instances, GPUs) to test a capacity-dependent hypothesis,
      state whether the workload size is held fixed (tests dilution: more resources, same total
      load) or scaled proportionally (tests the hypothesis at constant per-resource pressure, e.g.
      `batch ∝ n_instances`). Holding the workload fixed while sweeping resources answers a
      different question than the one usually being asked, and the two are easy to conflate.
- [ ] `SIMRL_WORKSPACE` is exported and `scripts/preflight.py` exits 0. Its warnings are carried
      into the report.
- [ ] `source "$SIMRL_WORKSPACE/env/simrl.sh"` ran.
- [ ] `SIMRL_KV_CACHE=$HOME/.cache/srl/kv_capacity.json` is set when any run re-sizes KV.
- [ ] The md5 of the simulator binary used is recorded, before and after the sweep.

## After each run

- [ ] The run finished: `versions_done == n_versions`, no livelock.
- [ ] `consumed_groups == batch_size * n_versions`.
- [ ] `tokens_per_group` is within 1% of the control arm at the same point.
- [ ] The treatment is not byte-identical to the control. If it is, the flag is not wired.
- [ ] The experiment script exposes `build(spec, output_dir=None) -> (Simulator, n_instances)`, and
      every summary row carries `row_id` and `replay` via `scripts/replayspec.attach`. A row that
      cannot be replayed cannot be diagnosed later.

## Before reporting

- [ ] Every point in the table passed all gates above.
- [ ] Points that failed a gate are dropped, not annotated and kept.
- [ ] The headline is the geometric mean over gated points, with the number of points stated.
- [ ] The treatment's own concurrency is reported next to the best static arm's.
- [ ] Any knob tuned on these points is named, and generalization is claimed only on points that
      played no role in tuning.

## Reporting failures

- [ ] Each dropped point appears in a short "failed gates" list with the gate that failed.
- [ ] No failing point appears in a speedup table or geometric mean.
