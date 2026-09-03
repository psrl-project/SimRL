# Workload

A workload supplies the simulator with request shapes: how long each prompt is, how
long each response turns out to be, and which responses belong to the same prompt.

## Two sources, and the choice between them

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Parametric distributions
:class-card: dryrun-feature-card

Sample lengths from one of six analytic families. No GPU, no data, instant. Prompt and
response lengths are drawn **independently**, and every group has exactly one response.

Use for: sensitivity studies, sweeping a shape you do not have data for, and any run
where you care about the simulator's behaviour rather than a specific model's workload.
:::

:::{grid-item-card} Joint group trace
:class-card: dryrun-feature-card

Measure real prompt and response lengths by generating from an actual model on an actual
dataset. Preserves each prompt's measured length, all *n* of its response lengths, and
the group boundary.

Use for: reproducing a specific training job, and anything involving group-relative
advantage (DAPO, GRPO), which is meaningless without group boundaries.
:::

::::

The distinction that matters is **correlation and grouping**. Parametric distributions
give you two independent marginals, so a long prompt is no more likely to produce a long
response than a short one, and there is no notion of eight responses sharing a prompt. A
joint trace preserves both. If your algorithm computes advantage within a group, only
the trace path is faithful.

There is deliberately **no distribution-fitting command**. Estimating a lognormal from a
measured trace and then sampling from it discards the correlation and grouping that
made the trace worth collecting. If you have a trace, replay it. If you want a
parametric shape, {doc}`index` below shows how to choose parameters by hand from the
statistics the trace header already records.

## Choosing parameters by hand

Every collected trace stores summary statistics in its header, so you can derive
parametric approximations without any fitting machinery. For Qwen3-8B on
`dapo-math-17k` with a 10240-token cap, the measured response distribution was:

| Statistic | Value |
|---|---|
| mean | 7804 |
| median | 8690 |
| p95 | 10240 |
| min | 989 |
| truncated at cap | 40.8% |

Two things are immediately visible and worth internalising before reaching for a
distribution family:

**The mean is below the median.** That is left skew, which a lognormal cannot produce —
lognormals are right-skewed by construction. Fitting one here would misrepresent the
shape no matter how the parameters were chosen.

**41% of responses hit the cap.** The distribution is truncated, not merely skewed, and
a large point mass sits at exactly 10240. No smooth two-parameter family reproduces
that. The honest parametric approximation is `bimodal`:

```yaml
# workload/dapo_like.yaml
source: distributions
output:
  name: bimodal
  short_len: 6000      # the body, near the median of untruncated responses
  long_len: 10240      # the cap, where the point mass sits
  long_frac: 0.41      # the measured truncated fraction
prompt:
  name: normal
  mu: 150              # measured mean 149.7
  sigma: 60
  lo: 75               # measured min
```

For a genuinely right-skewed length distribution with no cap effect, `lognormal`
parameters follow from the median and mean:

$$\mu = \ln(\text{median}), \qquad \sigma = \sqrt{2\ln(\text{mean}/\text{median})}$$

which requires mean > median. When it is not, that is the data telling you the family
is wrong.

## The six families

All samplers live in `dryrun/workload/distributions.py`, take an integer count and a
seed, and return `list[int]`.

| Name | Parameters | Shape |
|---|---|---|
| `fixed` | `lo` | every length identical — the usual choice for prompts when only response length varies |
| `uniform` | `lo`, `hi` | flat between bounds |
| `normal` | `mu`, `sigma`, `lo`, `hi` | Gaussian, rounded and clipped |
| `lognormal` | `mu`, `sigma`, `lo`, `hi` | right-skewed; `mu`/`sigma` are in log space |
| `bimodal` | `short_len`, `long_len`, `long_frac` | two point masses — the right tool for a truncated distribution |
| `powerlaw` | `alpha`, `lo`, `hi` | heavy tail, inverse-CDF sampled |

Select one with `workload=<name>` and override its keys:

```bash
dryrun workload=normal \
    workload.output.mu=8192 workload.output.sigma=512 \
    workload.prompt.mu=1024 workload.prompt.sigma=256
```

Prompt and response distributions are configured independently under
`workload.prompt.*` and `workload.output.*`. When the two lists differ in length they
cycle independently, with a period of the longer one.

```{toctree}
:maxdepth: 2

trace-collection
```
