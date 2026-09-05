# Simulation

The simulator drives an asynchronous rollout/training loop under a staleness policy,
asking cost models for latency and peak memory as it goes. This section covers running
it, the policies it can run under, the guards that stop it producing numbers it cannot
justify, and what it writes.

## Running one

The quickest route is the example script, which wraps the Hydra CLI:

```bash
cd examples/simulate
TRAINING_BACKEND=fsdp \
TRAINING_ARTIFACT=../../outputs/cost/qwen3_8b/training_fsdp.json \
bash simulate.sh
```

Or drive `dryrun` directly, overriding whichever config groups you need:

```bash
dryrun policy=verl workload=normal rollout_cost=roofline training_cost=fixed \
    job.n_versions=20 job.batch_size=256 rollout.n_instances=4
```

Every configurable key is listed in {doc}`../reference/configuration`.

## The five things that most often surprise people

1. **`job.batch_size` counts responses, not prompts.** With a DAPO-style trace at eight
   responses per prompt, `batch_size=512` consumes 64 prompts.
2. **`job.max_staleness` means something different in each policy**: a version bound
   for some, a ratio fallback for others, and it is ignored outright by one. See
   {doc}`policies`.
3. **The default policy has an external dependency.** `policy=psrl` imports the
   separate `psrl` package from a hard-coded path. See {doc}`policies`.
4. **A cost model will refuse an out-of-domain query** rather than extrapolate. This is
   the single most common first-run failure. See {doc}`domain-gates`.
5. **All three batching modes can legitimately return identical results.** At
   `micro_batch_size=1` they must. See {doc}`domain-gates`.

## Sync and recompute costs

Beyond rollout and training, two smaller cost models participate, each with a fixed and
a derived variant:

`sync_cost`, weight synchronisation between training and rollout
: `fixed` takes `sync_time` in seconds. `bandwidth` derives it from parameter count and
  link bandwidth.

`recompute_cost`, the cost of re-prefilling a preempted request
: `fixed` takes `time_per_token`. `analytical` derives it from model FLOPs.

Neither is fitted from measurements. If weight sync is a material fraction of your step
time, measure it and set `fixed` from the measurement rather than trusting the derived
variant.

```{toctree}
:maxdepth: 2

policies
domain-gates
outputs
```
