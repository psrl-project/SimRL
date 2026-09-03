---
sd_hide_title: true
---

# DryRun-RL Documentation

## DryRun-RL

**DryRun-RL** is a discrete-event simulator for asynchronous, disaggregated
reinforcement learning post-training of large language models. You profile a real
system once, fit cost models from those measurements, then simulate configurations you
never ran — different parallelism, different staleness policies, different rollout-to-
training ratios — in seconds rather than GPU-hours.

Its design premise is that a simulator is only useful if it knows what it does not
know. Every cost model records the range of inputs it was fitted over and **refuses a
query outside that range** rather than extrapolating silently.

---

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.5em` Install & first simulation
:link: getting-started/quickstart
:link-type: doc

Run a simulation in ten minutes on a synthetic workload and a fixed training cost — no
GPU required.
:::

:::{grid-item-card} {octicon}`book;1.5em` Concepts
:link: getting-started/concepts
:link-type: doc

Cells, slices, utility, staleness: the vocabulary every other page assumes.
:::

:::{grid-item-card} {octicon}`pulse;1.5em` Profile a real system
:link: profiling/index
:link-type: doc

Four pipelines, three verbs, and how long each sweep actually takes.
:::

:::{grid-item-card} {octicon}`graph;1.5em` Cost models
:link: cost-models/index
:link-type: doc

Which model fits which measurement, and the formula each one owns.
:::

:::{grid-item-card} {octicon}`database;1.5em` Workload
:link: workload/index
:link-type: doc

Synthetic length distributions, or a collected joint (prompt, n responses) trace.
:::

:::{grid-item-card} {octicon}`play;1.5em` Simulate
:link: simulation/index
:link-type: doc

Staleness policies, domain gates, telemetry streams, and plots.
:::

:::{grid-item-card} {octicon}`beaker;1.5em` Accuracy & limitations
:link: reference/validation
:link-type: doc

Every measured error figure, its sample count, and what is not validated.
:::

:::{grid-item-card} {octicon}`terminal;1.5em` CLI reference
:link: reference/cli
:link-type: doc

`dryrun`, `dryrun-profile`, `dryrun-regress`, `dryrun-workload`, `dryrun-plot`.
:::

:::{grid-item-card} {octicon}`tools;1.5em` Extend it
:link: development/index
:link-type: doc

Add a kernel op, a model family, a cost model, or a staleness policy.
:::

::::

---

## How the pieces connect

There are two kinds of cost model, and they reach the simulator by different routes.

```{mermaid}
graph TD
    A["Real system<br/>vLLM · Megatron-Bridge · FSDP2"] -->|"dryrun-profile … generate / run / collect"| B["ProfileRecord JSONL"]
    B -->|"dryrun-regress rollout / training"| C["CostArtifact JSON"]
    C --> E["dryrun (simulate)"]
    F["Kernels<br/>GEMM · attention · elementwise · collective"] -->|"dryrun-profile kernels run"| G["Perf tables (parquet)"]
    G -->|"read directly — no regress step"| E
    H["Workload<br/>distributions or joint trace"] --> E
    E --> I["Telemetry JSONL + plots"]
```

Kernels are the exception worth remembering: `ComposedRollout` reads the measured perf
tables and sums kernel latencies per step, so there is **no regression step** on that
path.

| Pipeline | Measures | Produces | Regress? |
|---|---|---|---|
| [vLLM](profiling/vllm.md) | end-to-end scheduler iterations | `rollout_{roofline,psrl,distserve}.json` | yes |
| [Kernels](profiling/kernels.md) | individual GEMM / attention / elementwise / collective kernels | perf tables (parquet) | **no** |
| [Megatron](profiling/megatron.md) | training steps under Megatron-Bridge | `training_hydraulis.json` | yes |
| [FSDP](profiling/fsdp.md) | training steps under the DryRun FSDP2 launcher | `training_fsdp.json` | yes |

---

```{toctree}
:maxdepth: 2
:hidden:

overview/index
getting-started/index
workload/index
profiling/index
cost-models/index
simulation/index
reference/index
development/index
cookbook
```
