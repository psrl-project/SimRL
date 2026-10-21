---
sd_hide_title: true
---

# SimRL Documentation

:::{image} /_static/img/SimRL_logo_merge.png
:alt: SimRL
:width: 480px
:align: center
:::

SimRL models the asynchronous rollout/training loop. You can simulate how a framework behaves, design
a deployment before buying it, predict at runtime, and run auto research on scheduling policies.

:::{image} /_static/img/simrl-capabilities-plain.png
:alt: What you can do with SimRL
:width: 100%
:align: center
:::

## What you can do with SimRL

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} Simulate frameworks
:link: simulation/policies

More than ten typed presets, each reimplementing one real system's admission, runtime and filtering
logic. Does PSRL stay latency-bound while verl's overage queue fills up?
:::

:::{grid-item-card} Design a deployment
:link: cost-models/index

GPU splits between rollout and training, instance counts, TP/PP/DP/EP. On 32 GPUs, is 20/12 with
TP2 rollout better than 16/16 with TP4?
:::

:::{grid-item-card} Predict at runtime
:link: simulation/routing

Placement and finish times from a snapshot of the live system. Rollout load spikes, so how many
engines can move to training without stalling the next batch?
:::

:::{grid-item-card} Auto research
:link: simulation/index

A proposal agent designs a policy, a validation agent simulates it, an analysis agent writes the
lesson back. Is `kv_cache_aware` routing better than `throughput_balance` on this workload?
:::

::::

## What SimRL is

SimRL answers questions of the form *"I have Qwen3-8B on eight H20s, rollout and
training splitting the cluster, staleness bounded at 2. What does that system look like
when it runs? What throughput, how much time spent waiting, does training memory fit?"*
It answers them without running the job.

It does this by separating three concerns that are usually tangled together:

Measurement
: A **profiler adapter** drives a real system (vLLM, Megatron-Bridge, an FSDP2
  launcher, or individual CUDA kernels) and normalises what it observes into versioned
  JSONL. Adapters never model anything. They record.

Modelling
: A **cost model** is fitted from that JSONL and reduced to a small set of coefficients
  plus the input range they are valid over. Models never measure anything. They
  interpolate.

Simulation
: A **discrete-event simulator** drives a rollout/training loop under a staleness
  policy, asking the cost models for latencies and peak memory as it goes.

The boundary between the first two is a documented JSONL schema. That is what lets a new
backend be added without touching the simulator, and a new cost model be fitted without
re-profiling.

## Choose an entry point

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Getting started
:link: getting-started/index

Install, learn the vocabulary, then follow the walkthrough from the first profiling
command to a simulated run.
:::

:::{grid-item-card} Workload
:link: workload/index

Parametric distributions or a replayed trace, and the group semantics GRPO/DAPO need.
:::

:::{grid-item-card} Profiling
:link: profiling/index

The profiling pipelines, what each sweep costs, and how the domain gates are recorded.
:::

:::{grid-item-card} Cost models
:link: cost-models/index

Rollout, training and kernel models: their functional form, fitted range, and how they
are validated.
:::

:::{grid-item-card} Simulation
:link: simulation/index

Policies, routing, prefix cache, backends, and the artifacts a run produces.
:::

:::{grid-item-card} Reference
:link: reference/index

CLI, Hydra configuration, JSONL/artifact schemas, validation results, and limitations.
:::

:::{grid-item-card} Development
:link: development/index

Package architecture, adding a kernel op or model family, and testing.
:::

::::

```{toctree}
:maxdepth: 2
:hidden:

getting-started/index
workload/index
profiling/index
cost-models/index
simulation/index
reference/index
development/index
```
