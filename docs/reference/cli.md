# CLI reference

Five console scripts are installed into the active environment. Four of them are conventional
argparse programmes; `dryrun` is a Hydra app and is documented separately at the end of this
page. Cross-reference: {doc}`configuration` for every config key; {doc}`../getting-started/quickstart`
for the end-to-end walkthrough.

---

## `dryrun-profile`

Profiles a real system and normalises the raw artifacts into
{doc}`../reference/schemas`-compliant JSONL. Four backends, each with three verbs.

```
dryrun-profile {vllm,kernels,megatron,fsdp} ...
```

### Common pattern

Every backend except `kernels` exposes the same three verbs:

```bash
dryrun-profile <backend> generate --config CFG   # enumerate candidates; no GPU needed
dryrun-profile <backend> run      --config CFG   # execute them
dryrun-profile <backend> collect  --config CFG   # normalise into JSONL
```

`run --resume` skips candidates already marked `success`, but **re-runs every OOM candidate**
— see {doc}`../profiling/methodology` for why OOM results are data rather than failures to
retry.

### `dryrun-profile vllm`

| Subcommand | Key flags | Notes |
|---|---|---|
| `generate` | `--config CFG`, `--phase {all,prefill,decode}` | No GPU required |
| `run` | `--config CFG`, `--phase`, `--dry-run`, `--resume`, `--max-parallel auto\|N` | Defaults to all GPU slots |
| `collect` | `--config CFG`, `--phase`, `--output PATH` | Missing scheduler logs fail hard |

### `dryrun-profile kernels`

The kernel pipeline does not follow the three-verb model — it has five subcommands that must
run in order:

| Subcommand | Key flags | Notes |
|---|---|---|
| `run` | `--config CFG`, `--family {all,...,gemm}` | Every family by default |
| `fit-floor` | `--config CFG`, `--candidates-us US [US ...]` | Default: `[0,2,4,6,8,12,16,22,32,45,60,80]` µs |
| `holdout` | `--config CFG`, `--fraction F`, `--gate G`, `--output PATH` | Use `--fraction 1.0` |
| `validate` | `--config CFG`, `--baseline JSONL`, `--model-path PATH`, `--surface S` | See note below |
| `verify-shapes` | `--config CFG`, `--model-path PATH` | Sanity-checks GEMM shapes |

`validate` also accepts `--min-samples N`, `--gate G`, `--output PATH` (omitted above for width).
`--min-samples 1` is required to score prefill and mixed phases; the default (`20`) drops both from
a typical vLLM run.

`fit-floor` is **not optional**. Skipping it leaves the attention dispatch floor unfitted, which
measured 20.85% and 34.89% MRE for context and generation attention respectively before the fix.
See {doc}`../profiling/kernels` for the full accuracy table.

### `dryrun-profile megatron`

| Subcommand | Key flags | Notes |
|---|---|---|
| `generate` | `--config CFG`, `--candidates PATH` | Writes candidate manifest |
| `run` | `--config CFG`, `--candidates PATH`, `--candidate-id ID \| --index N`, `--dry-run`, `--resume` | |
| `collect` | `--config CFG`, `--candidates PATH`, `--candidate-id ID \| --index N`, `--output PATH` | |

### `dryrun-profile fsdp`

| Subcommand | Key flags | Notes |
|---|---|---|
| `generate` | `--config CFG`, `--candidates PATH` | Writes candidate manifest |
| `run` | `--config CFG`, `--candidates PATH`, `--candidate-id ID \| --index N`, `--dry-run`, `--resume` | |
| `collect` | `--config CFG`, `--candidates PATH`, `--candidate-id ID \| --index N`, `--output PATH` | |

---

## `dryrun-regress`

Fits cost model coefficients from a versioned profiling JSONL and writes a `CostArtifact` JSON.

```
dryrun-regress {rollout,training} ...
```

### `dryrun-regress rollout`

```
dryrun-regress rollout --model {roofline,psrl,distserve}
                       --input INPUT
                       --output OUTPUT
                       [--block-size BLOCK_SIZE]
                       [--validation-fraction VALIDATION_FRACTION]
                       [--hidden-size HIDDEN_SIZE]
                       [--intermediate-size INTERMEDIATE_SIZE]
```

| Flag | Default | Notes |
|---|---|---|
| `--model` | — | Required. Selects the regression formula |
| `--input` | — | Required. Path to the vLLM profiling JSONL |
| `--output` | — | Required. Path to write the `CostArtifact` JSON |
| `--block-size` | — | KV block size used to derive attention features |
| `--validation-fraction` | — | Fraction of records held out for validation metrics |
| `--hidden-size` | — | Architecture override (used by `psrl` and `distserve`) |
| `--intermediate-size` | — | Architecture override |

### `dryrun-regress training`

```
dryrun-regress training [--latency-model {hydraulis,fsdp}]
                        [--memory-model {hydraulis,fsdp}]
                        --input INPUT
                        --output OUTPUT
                        [--validation-fraction VALIDATION_FRACTION]
                        [--allow-rank-deficient]
```

| Flag | Default | Notes |
|---|---|---|
| `--latency-model` | — | Which latency formula to fit |
| `--memory-model` | — | Which memory formula to fit |
| `--input` | — | Required. Path to the training profiling JSONL |
| `--output` | — | Required. Path to write the `CostArtifact` JSON |
| `--validation-fraction` | — | Fraction held out; `0.2` produces a genuine held-out set for FSDP |
| `--allow-rank-deficient` | — | Suppress the rank-deficiency error; use only for debugging |

`--allow-rank-deficient` should never be used in production: a rank-deficient design means the
fit cannot separate two coefficients, and the resulting artifact is degenerate. See
{doc}`../getting-started/concepts` for the design-rank discussion.

---

## `dryrun-workload`

Collects and validates joint `(prompt, n responses)` group traces used by the `workload=dapo_trace`
simulation source.

```
dryrun-workload {collect-groups,merge-shards,list-workers,check-shard,verify-collection} ...
```

### `collect-groups`

Runs one worker's shard: spawns a vLLM server and generates `n` responses per assigned prompt.

```
dryrun-workload collect-groups
    --dataset DATASET          # Parquet dataset path
    --tokenizer TOKENIZER      # Tokenizer name or path
    --model MODEL              # Model passed to `vllm serve`
    --output OUTPUT            # Output shard JSONL path
    [--worker-index N]         # This worker's index (0-based)
    [--n-workers N]            # Total worker count
    [--sample-size N]          # Distinct prompts across all workers
    [--n N]                    # Responses sampled per prompt
    [--max-tokens N]           # Max output tokens per response
    [--temperature T]
    [--top-p P]
    [--tp N]                   # Tensor-parallel size for the vLLM server
    [--concurrency N]          # Max in-flight prompts per worker
    [--port N]
    [--seed N]
    [--gpu GPU]                # GPU to pin this worker to
    [--server-ready-timeout N]
    [--serve-arg KEY=VALUE]    # Extra `vllm serve` flag; repeatable
```

### `merge-shards`

Merges the completed worker shards into a single joint trace, sorted by `group_id`.

```
dryrun-workload merge-shards
    --shard-dir DIR            # Directory of worker-NN.jsonl shards
    --n-workers N              # Shards expected
    --output OUTPUT            # Merged joint trace path
    [--max-output-tokens N]    # If supplied, verified against every shard header
```

### `list-workers`

Expands a hosts file into `index host gpu port` rows, used to drive parallel collection.

```
dryrun-workload list-workers
    --hosts-file PATH          # Path to host/GPU inventory
    [--base-port N]
```

### `check-shard`

Exits successfully only when a worker shard is complete (header `n_groups_requested`
matched). Used by collection scripts to decide whether to skip re-running a worker.

```
dryrun-workload check-shard --path PATH
```

### `verify-collection`

Verifies the merged group count and diagnoses which worker shards are short.

```
dryrun-workload verify-collection
    --trace PATH               # Merged joint trace path
    --expected-groups N
    --shard-dir DIR
    --n-workers N
```

---

## `dryrun-plot`

Renders plots from an existing simulation run without re-simulating.

```
dryrun-plot run_dir [--logs-subdir NAME] [--plots-subdir NAME]
```

`run_dir` is a **positional** argument: the `--output_dir` from a prior `dryrun` run.
`--logs-subdir` selects the subdirectory under `run_dir` that holds the telemetry JSONL streams
(default `logs`). `--plots-subdir` names the output subdirectory to write under `run_dir`
(default `plots`).

---

## `dryrun` — the Hydra simulator

`dryrun` is not an argparse app. It is powered by
[Hydra](https://hydra.cc/) and accepts config-group overrides, not flags.

### Config-group overrides

The full set of config groups is:

| Group | Options |
|---|---|
| `rollout_cost` | `roofline` (default), `psrl`, `distserve`, `composed` |
| `training_cost` | `fixed` (default), `analytical`, `linear`, `hydraulis`, `fsdp` |
| `policy` | `psrl` (default), `areal`, `roll`, `slime`, `verl` |
| `workload` | `uniform` (default), `bimodal`, `lognormal`, `normal`, `powerlaw`, `dapo_trace` |

Override a group on the command line:

```bash
dryrun rollout_cost=roofline training_cost=hydraulis policy=verl workload=dapo_trace
```

Override individual keys inside a group:

```bash
dryrun job.batch_size=512 job.n_versions=16 rollout.n_instances=4
```

Point to a cost artifact:

```bash
dryrun rollout_cost=roofline rollout_cost.artifact_path=outputs/cost/qwen3_8b/rollout_roofline.json
```

### Useful Hydra flags

| Flag | Effect |
|---|---|
| `--multirun` | Run a sweep over multiple config combinations |
| `--cfg job` | Print the resolved config without running the simulation |
| `hydra.run.dir=PATH` | Override the run directory (default `outputs/`) |
| `+key=value` | Append a key not present in the schema |
| `~group` | Delete a config group from the composition |

```bash
# Print the resolved config without running anything
dryrun --cfg job rollout_cost=roofline

# Sweep two batch sizes
dryrun --multirun job.batch_size=256,512
```

See {doc}`configuration` for every config key and its default value.
