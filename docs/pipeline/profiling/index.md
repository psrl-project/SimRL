# Profiling

Profiling is the measurement half of SimRL. Its output is never a model. It is either a
versioned JSONL file describing what a real system did, which {doc}`../cost-models/index`
then fits, or a tree of parquet kernel tables that a cost model reads directly.

All measurement goes through one console script, `srl-profile`.

## Commands

| Subcommand | Purpose | Page |
|---|---|---|
| `srl-profile kv-probe` | start a vLLM engine once and cache the KV capacity it allocates | {doc}`vllm` |
| `srl-profile vllm {generate,run,collect}` | end-to-end scheduler iterations from `vllm bench sweep serve` | {doc}`vllm` |
| `srl-profile kernels {run,fit-floor,holdout,verify-shapes,validate}` | individual kernels timed on synthetic tensors | {doc}`kernels`, {doc}`tensor-parallel` |
| `srl-profile megatron {generate,run,collect}` | training steps under Megatron-Bridge | {doc}`megatron` |
| `srl-profile fsdp {generate,run,collect}` | training steps under the SimRL FSDP2 launcher | {doc}`fsdp` |

## Where each pipeline ends

```{mermaid}
flowchart LR
    V["srl-profile vllm"] --> VJ["profiles/vllm.jsonl"]
    VJ --> RR["srl-regress rollout<br/>roofline / psrl / distserve"]
    K["srl-profile kernels"] --> KT["kernel/{gpu}/{family}/.../*.parquet"]
    KT --> CR["ComposedRollout<br/>ComposedTraining"]
    M["srl-profile megatron"] --> MJ["profiles/megatron_bridge.jsonl"]
    F["srl-profile fsdp"] --> FJ["profiles/fsdp.jsonl"]
    MJ --> RT["srl-regress training<br/>hydraulis"]
    FJ --> RF["srl-regress training<br/>fsdp"]
```

Only two `srl-regress` subcommands exist, `rollout` and `training`. The kernel pipeline has no
regression step: its parquet tables are the cost model's input. Tensor-parallel communication is
not a separate pipeline either. It is the `collective` kernel family, covered on
{doc}`tensor-parallel`.

## Three verbs

The vLLM, Megatron and FSDP pipelines expose the same three subcommands, in this order:

```bash
srl-profile <backend> generate --config CFG   # enumerate candidates or sweep jobs
srl-profile <backend> run      --config CFG   # execute them
srl-profile <backend> collect  --config CFG   # normalise into JSONL
```

`generate` never touches a GPU, so it is the cheapest way to see what a sweep will attempt.
For Megatron and FSDP it writes the candidate list to
`{output_dir}/control/{megatron_bridge,fsdp}/candidates.json`. For vLLM it writes the serve and
bench parameter files and the job manifests under `{output_dir}/control/vllm/`.

`run` accepts `--resume` and `--dry-run`. What `--resume` skips differs per backend and is
spelled out on each page, but in every case it **re-runs candidates that ended out of memory**,
because only a `success` status counts as done. Megatron and FSDP also accept `--index N` or
`--candidate-id ID` to run exactly one candidate.

`collect` reads the raw artifacts a run left behind and normalises them. It never re-executes
anything, and the warmup window is applied here rather than in the launcher, so the number of
discarded steps can change without re-profiling. See {doc}`methodology`.

The kernel pipeline has no `generate` or `collect`. It has `run` (measure and write parquet),
then the accuracy and calibration steps `fit-floor`, `holdout` and `validate`, plus the
GPU-free `verify-shapes`.

## Cost of a sweep

Every Megatron and FSDP candidate is its own `torchrun` child. Spawning the ranks, importing
torch and the training stack in each, building the model, forming the NCCL group and tearing
them down is paid once per candidate and is independent of how many steps are timed. Lowering
`max_steps` therefore removes only a small share of a candidate's wall clock while shrinking the
retained sample that the median and the dispersion diagnostics rest on.

The structural lever is sharing one process across candidates that have the same topology, by
building the model once and iterating over batch shapes. That is not implemented. It would also
let candidates share allocator state, which weakens the independence of the per-candidate memory
measurement.

FSDP's grid is wider than Megatron's by default, so its sweep is longer without any one
candidate being slower. Kernel and vLLM sweeps pay no per-candidate process cost of this kind:
the kernel runner times synthetic tensors in-process for one family at a time (collectives launch
one `torchrun` per topology and time all five ops inside it), and the vLLM adapter keeps one
server alive across many benchmark points.

Read {doc}`methodology` before running a long sweep. It collects the measurement rules the code
enforces and the traps the design guards against.

```{toctree}
:maxdepth: 2

methodology
vllm
kernels
tensor-parallel
megatron
fsdp
```
