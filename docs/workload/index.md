# Workload

A workload supplies the simulator with request shapes: how long each prompt is, how long each
response turns out to be, and which responses belong to the same prompt. The code is
`srl/workload/`, and the simulator consumes it through one class, `GroupedWorkload`: an
ordered, cycling sequence of **groups**, each a prompt length plus the lengths of its
responses.

## Two sources, and the choice between them

**Parametric distributions** (`source: distributions`) sample prompt and response lengths from
analytic families. No GPU, no data, instant. Prompt and response lengths are drawn
**independently**, and every group holds exactly one response, so `group_size` is 1. Use them
for sensitivity studies, for sweeping a shape you have no data for, and for any run where you
care about the simulator's behaviour rather than a specific model's workload.

**A joint group trace** (`source: trace`, config group `workload=dapo_trace`) replays measured
prompt and response lengths from a real model on a real dataset. It preserves each prompt's
length, all *n* of its response lengths, and the group boundary. Use it to reproduce a
specific training job, and for anything involving group-relative advantage (GRPO, DAPO), which
is meaningless without group boundaries. {doc}`trace-collection` covers how one is collected.

The distinction that matters is **correlation and grouping**. Parametric distributions give
you two independent marginals, so a long prompt is no more likely to produce a long response
than a short one, and eight responses never share a prompt. A joint trace preserves both. Group
boundaries also decide what the rest of the simulator can model: a group's responses are
admitted and consumed together, and responses from one group share prompt blocks in the KV
cache ({doc}`../simulation/prefix-cache`).

A parametric workload's `group_size` is 1, so `job.batch_size` is both groups and responses.
With a trace, `job.group_size` defaults to the trace's arity and an explicit value must match
it, otherwise the run stops with `group_size 4 does not match the workload's 8`. A step then
consumes `job.batch_size * group_size` responses.

There is deliberately **no distribution-fitting command**. Estimating a lognormal from a
measured trace and then sampling from it discards the correlation and grouping that made the
trace worth collecting. If you have a trace, replay it. If you want a parametric shape, the
next sections show how to choose parameters by hand from the statistics the trace header
already records.

## The parametric families

The samplers live in `srl/workload/distributions.py`. Each takes a count and a seed and returns
`list[int]`. A workload config has two blocks, `workload.output.*` and `workload.prompt.*`,
and each block names its family with `name`.

| `name` | Parameters | Shape |
|---|---|---|
| `fixed` | `lo` | every length equal to `lo`; the default prompt block |
| `uniform` | `lo`, `hi` | flat between the bounds, inclusive |
| `normal` | `mu`, `sigma`, `lo`, `hi` | Gaussian, rounded; draws below `lo` raised to `lo`, above `hi` clipped to `hi` |
| `lognormal` | `mu`, `sigma`, `lo`, `hi` | right-skewed; `mu` and `sigma` are in log space; `lo` defaults to 1 |
| `bimodal` | `short_len`, `long_len`, `long_frac` | two point masses, the shape for a truncated distribution |
| `powerlaw` | `alpha`, `lo`, `hi` | heavy tail, inverse-CDF sampled and clipped to `[lo, hi]` |

Every block also takes `n` (how many lengths to draw, default 200) and `seed` (default 42).
The parametric config groups are `uniform` (the default), `normal`, `lognormal`, `bimodal`,
and `powerlaw`, plus `dapo_trace` for a trace. Each parametric group sets the **output**
family and a prompt block, which is `fixed` at 512 tokens except in `normal`, where the prompt
is a second normal (seed 43, so the two dimensions are not paired). The defaults are in
`srl/config/defaults/workload/*.yaml`.

Select a group and override its keys:

```bash
srl workload=normal \
    workload.output.mu=8192 workload.output.sigma=512 \
    workload.prompt.mu=1024 workload.prompt.sigma=256
```

A key that is not already in the chosen group's YAML needs Hydra's `+` prefix, for example an
upper bound on a lognormal, which `lognormal.yaml` does not declare:

```bash
srl workload=lognormal +workload.output.hi=10240
```

Without the `+`, composition stops with `Could not override 'workload.output.hi'`. Print the
resolved block first with `srl --cfg job workload=lognormal`.

The two blocks may hold different counts. The workload then has `max(len(prompts),
len(outputs))` single-response groups, and each list cycles modulo its own length, so the
sequence repeats with that period. Per-request caps are not part of the family: set
`job.max_prompt_length` and `job.max_output_length` (default `null`, no cap), which truncate
the prompt and stop generation at that length.

## Choosing parameters by hand

Every collected trace stores summary statistics in its header, so you can derive a
parametric approximation without any fitting machinery. For Qwen3-8B on `dapo-math-17k` with
a 10240-token cap, the reference trace in {doc}`trace-collection` recorded this response
distribution:

| Statistic | Value |
|---|---|
| mean | 7804 |
| median | 8690 |
| p95 | 10240 |
| min | 989 |
| truncated at cap | 40.8% |

Two things stand out before you reach for a family:

**The mean is below the median.** That is left skew, which a lognormal cannot produce, because
lognormals are right-skewed by construction. Fitting one here would misrepresent the shape
whatever the parameters.

**41% of responses hit the cap.** The distribution is truncated, not merely skewed: a large
point mass sits at exactly 10240. No smooth two-parameter family reproduces that, and the
honest parametric approximation is `bimodal`:

```bash
srl workload=bimodal \
    workload.output.short_len=6000 \
    workload.output.long_len=10240 \
    workload.output.long_frac=0.41 \
    workload.prompt.name=normal +workload.prompt.mu=150 +workload.prompt.sigma=60 \
    workload.prompt.lo=75 workload.prompt.hi=1024
```

`short_len` is the body, near the median of the untruncated responses; `long_len` is the cap
where the point mass sits; `long_frac` is the truncated fraction. The prompt block is switched
from `fixed` to `normal` with the measured mean and a loose spread. `bimodal.yaml` ships the
prompt as `fixed` with `lo: 512, hi: 512`, so `mu` and `sigma` need the `+`, and `hi` must be
raised or the inherited 512 clips every prompt.

For a genuinely right-skewed length distribution with no cap effect, `lognormal` parameters
follow from the median and the mean:

$$\mu = \ln(\text{median}), \qquad \sigma = \sqrt{2\ln(\text{mean}/\text{median})}$$

which requires mean > median. When it does not hold, the data is telling you the family is
wrong.

## Replaying a trace

```bash
srl workload=dapo_trace \
    workload.trace.path=outputs/workload/qwen3_8b_dapo_tp1/joint.jsonl \
    job.n_versions=16 job.batch_size=64 rollout.n_instances=16 \
    job.max_prompt_length=1024 job.max_output_length=10240
```

`workload.trace.path` is required (`workload.trace.path is required when workload.source is
trace` otherwise). `workload.trace.n_groups` is `0` to read every group, or a positive cap for
a quick smoke run. The trace is read in file order; once the run has issued every request it
wraps around to the first group, and because the sequence is group-ordered, wraparound
restarts on a group boundary. `batch_size=64` with 8 responses per group consumes 64 prompts
(512 responses) per training step.

## A synthetic joint trace

When you need a trace whose dispersion is a controlled variable and have no GPU,
`examples/workload/make_dispersed_trace.py` writes one in the same format:

```bash
python examples/workload/make_dispersed_trace.py \
    --out outputs/workload/dispersed/cv1.0.jsonl \
    --n-groups 400 --mean 8000 --cv 1.0 --rho 0.6 --cap 32768
```

It draws log-lengths from a normal and solves the location so the mean survives truncation at
`--cap`. `--cv` is the coefficient of variation before truncation (a heavier right tail when
larger), and `--rho` is the share of log-length variance that belongs to the prompt rather than
the sample, so that a hard prompt makes all of its responses long. It also takes `--group-size`
(default 8), `--prompt-median`, `--prompt-sigma`, and `--seed`. The result loads with
`workload=dapo_trace`; its header marks `"provenance": "synthetic-dispersed"`.

```{toctree}
:maxdepth: 2

trace-collection
```
