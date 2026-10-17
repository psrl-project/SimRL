# Profiling

Profiling is the measurement half of SimRL. Its output is never a model. It is
versioned JSONL describing what a real system did, which
{doc}`../cost-models/index` then fits.

## Three verbs

Every backend exposes the same three subcommands, in this order:

```bash
srl-profile <backend> generate --config CFG   # enumerate candidates
srl-profile <backend> run      --config CFG   # execute them
srl-profile <backend> collect  --config CFG   # normalise into JSONL
```

`generate` writes a candidate list and does not touch a GPU, so it is the cheapest way
to see what a sweep will attempt. `run` supports `--resume`, which skips candidates
already marked `success`, which means it **re-runs every out-of-memory candidate**. That is
usually not what you want on a second pass. `collect` reads the raw artifacts each candidate
left behind and normalises them, and it never re-executes
anything, so the warmup window and other post-processing choices can change without
re-profiling.

## The four pipelines

| Pipeline | Measures | Produces | Regress? |
|---|---|---|---|
| {doc}`vllm` | end-to-end scheduler iterations | `rollout_*.json` | yes |
| {doc}`kernels` | individual GEMM / attention / elementwise / collective kernels | perf tables (parquet) | **no** |
| {doc}`tensor-parallel` | the same, sharded, plus the collectives a `tp>1` step pays | perf tables (parquet) | **no** |
| {doc}`megatron` | training steps under Megatron-Bridge | `training_hydraulis.json` | yes |
| {doc}`fsdp` | training steps under the SimRL FSDP2 launcher | `training_fsdp.json` | yes |

Read {doc}`methodology` before running a long sweep. It collects the measurement rules
this project learned the expensive way: how many steps to discard, which dispersion
statistic to report, and why an out-of-memory result is data you want rather than a
failure to retry.

## What a sweep costs

Measured for Qwen3-8B on 8x H20. Megatron 2026-09-03, the rest 2026-08-31:

| Pipeline | Candidates | Median each | Total | Of which is training |
|---|---|---|---|---|
| vLLM | 30 points | | ~6 min | |
| Kernels | 338 shapes | | ~1 h | |
| Megatron | 40 | 169.8 s | 2.23 h | 16 s (9%) |
| FSDP | 108 | 183 s | 6.9 h | 60 s (31%) |

**Roughly 70% of a training candidate's wall clock is process startup**, not
measurement: spawning eight processes, importing torch and transformers in every rank,
building a 36-layer model, forming the NCCL group, tearing it all down. That cost is
near-identical for both training backends, at 136 s for FSDP against 154 s for Megatron.

The practical consequence is that **cutting `max_steps` is not the lever it appears to
be.** Going from 12 steps to 8 saves about 2 s out of 196 s, roughly 1%, while halving
the retained sample count that the median and outlier rejection depend on. The real
lever would be amortising startup across candidates that share a topology: one
`torchrun` child, build the model once, iterate over batch shapes. That is not
implemented, and it is not free to implement, because candidates would then share allocator
state, which compromises the independence of the memory measurement.

FSDP's sweep is longer than Megatron's because it has 108 candidates against 40 and a
wider grid, not because FSDP is slower per candidate.

```{toctree}
:maxdepth: 2

methodology
vllm
kernels
tensor-parallel
megatron
fsdp
```
