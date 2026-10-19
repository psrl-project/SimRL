# Getting Started

SimRL has two halves. One measures a real system and reduces the measurements to
coefficients or tables. The other simulates configurations you never ran, using those
cost models.

The fastest way to understand it is to walk both halves once on a real model.
{doc}`walkthrough/index` does that with Qwen3-8B on 8x H20, from the first profiling
command through to a simulated training job, with every input, output, and file format
named as it appears on disk.

If you only want to see the simulator run, {doc}`walkthrough/00-first-simulation` needs no
GPU and no profiling data, and finishes in seconds.

## Reading order

1. {doc}`installation` sets up the environment. The pinned third-party install in
   `third_party/install.sh` is only needed if you intend to profile.
2. {doc}`walkthrough/index` is the main path.
3. {doc}`concepts` defines the vocabulary the rest of the site assumes: *cell*, *slice*,
   *utility*, *feature range*, *version*. Several of these look like ordinary words but
   carry a precise meaning. Read it before or after the walkthrough, whichever you prefer.

If you want to add a backend, a kernel op, a cost model, or a staleness policy rather than
use the existing ones, go to {doc}`../development/index`.

```{toctree}
:maxdepth: 2

installation
walkthrough/index
concepts
```
