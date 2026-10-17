# Development

SimRL has six extension points. Each is a small protocol with a registry, so adding one
does not require touching the simulator or the CLI.

## The extension points

Kernel op
: A kernel family that can be measured and priced: GEMM, attention, elementwise,
  collective. An op owns its table schema, its speed-of-light formula, its dispatch floor,
  and its sweep. See {doc}`kernel-ops`.

Model family
: The mapping from a model architecture to the kernels one of its layers runs. Ops say *how
  to price a shape*, families say *which shapes a layer produces*. See
  {doc}`model-families`.

Profiler adapter
: A backend that can be driven and measured, exposing `generate`, `run` and `collect`. See
  {doc}`adapters-and-cost-models`.

Rollout cost model
: Predicts one inference step's latency. It must also expose `decode_coeffs()`, which is not
  optional, because the simulator's closed-form pure-decode advance depends on it. See
  {doc}`adapters-and-cost-models`.

Training cost model
: Predicts one training step's latency and peak memory, as a separable latency model and
  memory model. See {doc}`adapters-and-cost-models`.

Staleness policy
: Decides which requests to admit and what to do with a completed request whose model
  version has moved on. The smallest extension point, roughly thirty lines. Documented at
  the end of {doc}`../simulation/policies`, where the five existing policies serve as worked
  examples.

{doc}`architecture` describes how the three layers fit together, the event loop, and the
package map.

## Before you write code

Read `.claude/coding-style.md` in the repository root. Two rules in it cause silent bugs
rather than style complaints, so they are worth repeating here:

- **Identifier strings never get a trailing period or a case change.** A log message ends
  with `.`, but a dictionary key, a status value, or a candidate ID does not. Adding one
  changes what the string *is*.
- **Comments explain why, not what.** A comment restating the code goes stale silently. A
  comment recording a rejected alternative saves the next person the experiment.

{doc}`testing` covers what the test suite gates versus what is a local instrument. That
distinction matters here because the measured perf tables are gitignored, so some tests skip
on a fresh clone rather than fail.

{doc}`docs` is the maintainer guide for this site, including the rule that keeps measured
numbers from drifting apart across pages.

```{toctree}
:maxdepth: 2

architecture
kernel-ops
model-families
adapters-and-cost-models
testing
docs
```
