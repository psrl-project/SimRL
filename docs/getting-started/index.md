# Getting Started

Three paths lead through this documentation. Pick the one that matches what you have.

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} I just want to simulate
:class-card: dryrun-feature-card

You have no profiling data and possibly no GPU. Start with
{doc}`quickstart` — it runs an analytic cost model over a synthetic workload.
:::

:::{grid-item-card} I want to model my own system
:class-card: dryrun-feature-card

You have GPUs and a checkpoint. Read {doc}`concepts`, then
{doc}`../profiling/index` to measure, then {doc}`../cost-models/index` to fit.
:::

:::{grid-item-card} I want to extend it
:class-card: dryrun-feature-card

You want a new backend, kernel op, or staleness policy. Go to
{doc}`../development/index`.
:::

::::

Whichever path you take, {doc}`concepts` is worth ten minutes first: the rest of the
site uses *cell*, *slice*, *utility*, *feature range* and *version* as terms of art,
and guessing at them makes several pages read as more complicated than they are.

```{toctree}
:maxdepth: 2

installation
quickstart
concepts
```
