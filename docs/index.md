---
sd_hide_title: true
---

# SimRL Documentation

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

{doc}`getting-started/walkthrough/index` walks the whole path on one real model and one
real machine, from the first profiling command to a simulated run.

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
