# The pipeline

SimRL is one pipeline with four stages. Each stage has a single job, and each hands the next
one a file format that is documented rather than implied. That is what lets a backend change
without touching the simulator, and a cost model change without re-profiling.

```{mermaid}
flowchart LR
    P["srl-profile<br/>measure a real system"] -->|JSONL records<br/>or kernel parquet| C["srl-regress<br/>fit coefficients"]
    C -->|CostArtifact JSON| S["srl<br/>discrete-event replay"]
    W["srl-workload<br/>prompt and response lengths"] --> S
    S -->|telemetry| O["srl-plot<br/>figures and SCG"]
```

| Stage | Page | Consumes | Produces |
|---|---|---|---|
| Profiling | {doc}`/pipeline/profiling/index` | a real GPU system and a sweep definition | versioned JSONL, or kernel parquet tables |
| Cost models | {doc}`/pipeline/cost-models/index` | those records | a `CostArtifact` JSON, or a kernel sum with no artifact |
| Workload | {doc}`/pipeline/workload/index` | a measured trace or hand-set parameters | groups of prompt and response lengths |
| Simulation | {doc}`/pipeline/simulation/index` | two cost models and a workload | telemetry, plots, staleness and SCG numbers |

## The three boundaries

**Records are not models.** A profiler adapter drives a real system and writes down what
happened. It never predicts anything, has no coefficients to tune, and could in principle be
replaced by a hand-written JSONL file. A record lists the parallelism it ran under, the
selectors that distinguish the configuration, the inputs, and the measured time.

**Models are not measurements.** A fitted cost model reduces many records to a small set of
coefficients plus the input range those coefficients are valid over. It never runs on a GPU.
The valid range travels with the artifact and becomes a prediction-time gate: asking outside it
raises instead of silently extrapolating.

**Simulation does not measure anything else.** Every duration and every memory figure in a run
comes from the two cost models selected in that run's config. This is why a simulated number can
be checked: turn on the record that produced the coefficients, and the model has to reproduce it.

## Where the stages connect

The two joints worth knowing:

- **The artifact file** between cost models and simulation. It is JSON, described on
  {doc}`/reference/schemas`, and the simulator loads it at startup. Nothing else is passed between
  the fitting half and the simulating half.
- **The kernel table** that bypasses fitting entirely. `ComposedRollout` and `ComposedTraining` read
  measured parquet tables and sum per-kernel times, so there is no regression step and no artifact
  ({doc}`/pipeline/cost-models/composed-rollout`).

Workload is orthogonal to both. It decides which request shapes flow through the simulated
system, and every other stage is indifferent to where those shapes came from.

## Reading order

Start with {doc}`/pipeline/profiling/index` to see what gets measured, then
{doc}`/pipeline/cost-models/index` to see what a fit looks like. If you only want to run the
simulator, skip both: {doc}`/getting-started/walkthrough/00-first-simulation` uses hand-set
constants and needs no GPU.

```{toctree}
:maxdepth: 2
:hidden:

profiling/index
cost-models/index
workload/index
simulation/index
```
