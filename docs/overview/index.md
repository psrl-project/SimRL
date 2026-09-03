# Overview

## What DryRun-RL is

DryRun-RL answers questions of the form *"if I gave rollout 48 GPUs instead of 32, kept
staleness bounded at 2, and switched training from Megatron to FSDP2, what would
throughput be?"* — without running the job.

It does this by separating three concerns that are usually tangled together:

Measurement
: A **profiler adapter** drives a real system (vLLM, Megatron-Bridge, an FSDP2
  launcher, or individual CUDA kernels) and normalises what it observes into versioned
  JSONL. Adapters never model anything; they record.

Modelling
: A **cost model** is fitted from that JSONL and reduced to a small set of
  coefficients plus the input range they are valid over. Models never measure
  anything; they interpolate.

Simulation
: A **discrete-event simulator** drives a rollout/training loop under a staleness
  policy, asking the cost models for latencies and peak memory as it goes.

The boundary between the first two is a documented JSONL schema, which is what lets a
new backend be added without touching the simulator, and a new cost model be fitted
without re-profiling.

## Why the domain checks matter

A regression fitted over sequence lengths of 512 to 16384 tokens tells you nothing
about 65536 tokens — on the hardware this project was developed against, 65536 tokens
per rank does not run at all. Nine of the 108 attempts were out-of-memory failures.

So a `CostArtifact` stores `feature_ranges` alongside its coefficients, and a query
outside them raises rather than returning a number. Extrapolation is possible but must
be requested per feature, by name, and is recorded in the telemetry when it happens.
See {doc}`../simulation/domain-gates`.

This is also why {doc}`../reference/validation` exists as its own page: every accuracy
figure on this site carries its sample count and whether it was measured in-sample.

## What it does not do

- It does not simulate correctness or learning dynamics — only time and memory.
- It does not model network topology; collective costs come from measured tables at the
  world sizes that were actually swept.
- Mixture-of-experts and multi-head latent attention architectures are **refused at
  configuration time** rather than approximated. See {doc}`../development/model-families`.

```{toctree}
:maxdepth: 2

architecture
prior-art
```
