# Getting Started

SimRL has two halves. One measures a real system and reduces the measurements to
coefficients or tables. The other replays configurations you never ran, using those
cost models.

The fastest way to understand it is to walk both halves once on a real model.
{doc}`/getting-started/walkthrough/index` does that with Qwen3-8B on 8x H20, from the first profiling
command through to a simulated training job, with every input, output, and file format
named as it appears on disk.

If you only want to see the simulator run, {doc}`/getting-started/walkthrough/00-first-simulation` needs no
GPU and no profiling data, and finishes in seconds.

## Reading order

1. {doc}`/getting-started/installation` sets up the environment. The pinned third-party install in
   `third_party/install.sh` is only needed if you intend to profile.
2. {doc}`/getting-started/walkthrough/index` is the main path.
3. {doc}`/getting-started/concepts` defines the vocabulary the rest of the site assumes: *cell*, *slice*,
   *utility*, *feature range*, *version*. Several of these look like ordinary words but
   carry a precise meaning. Read it before or after the walkthrough, whichever you prefer.

If you want to add a backend, a kernel op, a cost model, or a staleness policy rather than
use the existing ones, go to {doc}`/development/index`.

## Pick your path

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} I have no GPU, show me the simulator
:link: /getting-started/walkthrough/00-first-simulation
:link-type: doc

Hand-set constants and a parametric workload. Runs in seconds, produces a full set of plots.
:::

:::{grid-item-card} I want to reproduce a real job
:link: /getting-started/walkthrough/06-simulate
:link-type: doc

Replay a measured trace with fitted cost models and compare the simulation against the run it
came from.
:::

:::{grid-item-card} I am measuring a new system
:link: /pipeline/profiling/index
:link-type: doc

The profiling commands, the sweeps they run, and the records they write. Start here instead of
at the walkthrough.
:::

::::

```{toctree}
:maxdepth: 2

installation
walkthrough/index
concepts
```
