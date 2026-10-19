# CLI reference

`pyproject.toml` installs seven console scripts. `srl` is a Hydra app and takes `key=value`
overrides. The other six are argparse programs. Flags below were read from `--help` and from
`srl/cli/*.py`. See {doc}`configuration` for every `srl` key.

| Script | Entry point | Purpose |
|---|---|---|
| `srl` | `srl.cli.main:main` | Run one simulation |
| `srl-profile` | `srl.cli.profile:main` | Drive real systems and normalise their output |
| `srl-regress` | `srl.cli.regress:main` | Fit rollout and training cost artifacts from profile JSONL |
| `srl-plot` | `srl.cli.plot:main` | Re-render plots from a simulation's logs |
| `srl-psrl-plot` | `srl.cli.psrl_plot:main` | Convert a real PSRL run's logs and plot them |
| `srl-compare` | `srl.cli.compare:main` | Tabulate where a simulation differs from a converted real run |
| `srl-workload` | `srl.cli.workload:main` | Collect and validate joint grouped workload traces |

---

## `srl`

```bash
srl [--help] [--cfg job|hydra|all] [--multirun] [group=option ...] [key=value ...]
```

`srl` composes `srl/config/defaults/simulate.yaml` with the selected options of four groups
(`policy`, `rollout_cost`, `training_cost`, `workload`) and runs `run_simulate`.
`srl --help` prints the groups and the fully resolved default config.

```bash
# Select group options, then override individual keys.
srl policy=verl rollout_cost=roofline training_cost=fixed workload=bimodal \
    job.batch_size=32 job.n_versions=10 rollout.n_instances=2

# Print the resolved config without running.
srl --cfg job policy=roll training_cost=fsdp

# A key that is not in the composed config needs a leading plus sign.
srl +log_exclude='instance,rollout_step'
```

Hydra runs in struct mode, so a misspelt key such as `job.n_version=3` fails at composition
instead of being accepted. The groups and keys are listed in {doc}`configuration`.

### What a run produces

Hydra writes its own copy of the config under `outputs/<date>/<time>/.hydra/`. The simulator
writes to `output_dir` (default `outputs`, relative to the launch directory because Hydra does not
change directory):

| Path | Content |
|---|---|
| `run.log` | The `srl` logger's output at `log_level` |
| `logs/` | Telemetry JSONL streams, only when `log_telemetry=true`. Re-running into the same `output_dir` replaces the directory. See {doc}`schemas` |
| `plots/` | The plots rendered from `logs/` |

The process prints a summary to stdout: versions done, requests consumed and dropped, throughput,
the staleness distribution, the SCG(S) curve (tokens per second from trajectories whose
staleness is at most S), a staleness report, and wasted tokens by reason. It prints a warning when
the run livelocked. `log_exclude` is a comma-separated list of stream names or prefixes that the
store skips, which is how a sweep avoids writing the large per-step streams. It is not a schema
key, so it takes the `+` form shown above.

### Environment variables

| Variable | Read by | Meaning |
|---|---|---|
| `SIMRL_BACKEND` | `srl` | Rollout engine backend: `cpp` (default), `auto`, or `native` |
| `SIMRL_UNDOABLE_EVENTS` | `srl` | Comma-separated engine event kinds that are executed speculatively and can be undone. Default `request_complete,preempt,iteration_done`. `request_complete` must be present |
| `SIMRL_LOG_LEVEL` | `setup_logging` | Log level used when a caller passes none. `srl` itself passes `log_level` from the config, so this matters for the other scripts |
| `SIMRL_KV_CACHE` | `srl`, `srl-profile kv-probe` | Path of the on-disk KV capacity cache. Default `~/.cache/srl/kv_capacity.json` |

`SIMRL_BACKEND` selects between two implementations of the same instance engine:

`cpp` (default)
: Require the C++ engine. A missing module raises `ImportError`; a router strategy or eviction
  policy marked `python_engine_only` (`inventory_deadline`, `latest_deadline`) raises `ValueError`.

`auto`
: Use the C++ engine when `srl._engine_cpp` imports and no configured component is
  `python_engine_only`, otherwise the Python engine. Never raises on its own account.

`native`
: Use the Python engine (`NativeInstance`).

Any other value raises `ValueError`. The C++ module ships as
`srl/_engine_cpp.cpython-312-x86_64-linux-gnu.so`, so `auto` falls back to Python on any other
interpreter or platform until the module is rebuilt (see {doc}`../development/architecture`).

`srl` requires the `psrl` Python package at import time. See {doc}`limitations`.

---

## `srl-profile`

```bash
srl-profile {kv-probe,vllm,kernels,megatron,fsdp} ...
```

`vllm`, `megatron` and `fsdp` share three verbs. Each takes `--config` (required), a YAML
workflow file, and prints its work through the `srl` logger.

```bash
srl-profile <backend> generate --config CFG   # enumerate candidates, no GPU needed
srl-profile <backend> run      --config CFG   # execute them
srl-profile <backend> collect  --config CFG   # normalise into ProfileRecord JSONL
```

`--resume` skips a candidate whose `status.json` says `success` and re-runs every other one,
including candidates that ended in `oom`. `--dry-run` prepares and prints commands without
executing.

### `srl-profile kv-probe`

Start a vLLM engine once, read back the number of KV blocks it allocates, and cache it for
`srl` to use when `rollout.kv_cache_sizing.probe=true`.

| Flag | Default | Meaning |
|---|---|---|
| `--model` | required | Checkpoint path or hub id |
| `--tp` | `1` | Tensor-parallel degree |
| `--gpu-memory-utilization` | `0.9` | vLLM memory fraction |
| `--max-model-len` | `32768` | Longest sequence the engine must hold |
| `--dtype` | `bfloat16` | Weight dtype |
| `--max-num-batched-tokens` | `0` | vLLM `max_num_batched_tokens` |
| `--max-num-seqs` | `0` | vLLM `max_num_seqs` |
| `--gpu-name` | `unknown` | Cache key component. Two GPU models with equal memory do not share an entry |
| `--block-size` | `0` | Tokens per KV block |
| `--refresh` | off | Delete the cache file first, then probe again |

### `srl-profile vllm`

| Verb | Flags |
|---|---|
| `generate` | `--config`, `--phase {all,prefill,decode}` (default `all`) |
| `run` | `--config`, `--phase`, `--dry-run`, `--resume`, `--max-parallel auto\|N` |
| `collect` | `--config`, `--phase`, `--output PATH` |

`--max-parallel` bounds concurrent vLLM servers and defaults to `run.max_parallel` from the
config, or all available GPU slots. `collect` writes `<output_dir>/profiles/vllm.jsonl` unless
`--output` is given.

### `srl-profile kernels`

The kernel pipeline has five subcommands rather than the three verbs.

```bash
srl-profile kernels {run,fit-floor,holdout,verify-shapes,validate} ...
```

| Subcommand | Flags | Behaviour |
|---|---|---|
| `run` | `--config`, `--family {all,attention,attention_training,collective,elementwise,gdn,gemm,optimizer}` | Measure every registered family, or one, and write parquet tables. Needs a GPU. The family choices come from the op registry |
| `fit-floor` | `--config`, `--candidates-us US [US ...]` | Scan dispatch-floor candidates and write `floors.json` beside each floor-taking table. Defaults to 0, 2, 4, 6, 8, 12, 16, 22, 32, 45, 60, 80 microseconds |
| `holdout` | `--config`, `--fraction F` (1.0), `--seed N` (0), `--gate G` (5.0), `--output PATH` | Score leave-one-out interpolation accuracy per `(op, backend)` table |
| `verify-shapes` | `--config` or `--model-path`, `--model-path PATH` | Compare the declared GEMM shapes with a meta-device module graph. Exits non-zero on a mismatch |
| `validate` | `--config`, `--baseline JSONL`, `--model-path PATH`, `--surface {pchip,shepard}` (pchip), `--min-samples N` (20), `--gate G` (6.0), `--output PATH` | Score `ComposedRollout` against a vLLM profile JSONL, per phase |

`holdout` and `validate` write a report JSON (`holdout_report.json` and `validate_report.json`
under the table root by default) and exit non-zero when any row exceeds `--gate` or a row scored
no points. `holdout` warns when `--fraction` is below 1.0, because a sampled subset can flatter the
surface. `validate` warns about a phase that `--min-samples` dropped entirely. Pass
`--min-samples 1` to score a phase that has few iterations per bucket. The method behind both
commands is on {doc}`validation`.

### `srl-profile megatron` and `srl-profile fsdp`

| Verb | Flags |
|---|---|
| `generate` | `--config`, `--candidates PATH` |
| `run` | `--config`, `--candidates PATH`, `--candidate-id ID` or `--index N`, `--dry-run`, `--resume` |
| `collect` | `--config`, `--candidates PATH`, `--candidate-id ID` or `--index N`, `--output PATH` |

`--candidate-id` and `--index` are mutually exclusive, and omitting both selects every candidate.
`generate` writes a candidate manifest. When `--candidates` is omitted the Megatron manifest goes
to `<output_dir>/control/megatron_bridge/candidates.json`.

`srl-profile megatron run` adds multi-node launch flags:

| Flag | Default | Meaning |
|---|---|---|
| `--nnodes` | `1` | Nodes the candidate mesh spans. 1 means single-host `torchrun --standalone` |
| `--node-rank` | `0` | This host's index in `[0, nnodes)` |
| `--master-addr` | none | Rendezvous host, required when `--nnodes > 1` |
| `--master-port` | none | Rendezvous port, required when `--nnodes > 1` |

Every host runs the same command with its own `--node-rank`.

---

## `srl-regress`

```bash
srl-regress {rollout,training} ...
```

Fits a model from `ProfileRecord` JSONL and writes a `CostArtifact` JSON.

### `srl-regress rollout`

| Flag | Default | Meaning |
|---|---|---|
| `--model {roofline,psrl,distserve}` | required | Formula to fit |
| `--input PATH` | required | Rollout profile JSONL |
| `--output PATH` | required | Artifact path |
| `--block-size N` | `16` | KV block size used for attention features |
| `--validation-fraction F` | `0.0` | Fraction of records held out. 0 fits in sample |
| `--hidden-size N` | none | Architecture constant, used by `distserve` only |
| `--intermediate-size N` | none | Architecture constant, used by `distserve` only |

When every input record comes from the vLLM backend, the command also writes diagnostic plots to
`<output stem>_plots/` next to the artifact.

### `srl-regress training`

| Flag | Default | Meaning |
|---|---|---|
| `--latency-model {hydraulis,fsdp}` | `hydraulis` | Latency formula |
| `--memory-model {hydraulis,fsdp}` | `hydraulis` | Memory formula |
| `--input PATH` | required | Training profile JSONL |
| `--output PATH` | required | Artifact path |
| `--validation-fraction F` | `0.2` | Fraction of records held out |
| `--allow-rank-deficient` | off | Fit even when the latency design matrix lacks full rank |

The FSDP model must be selected for both flags together, and mixing `fsdp` with `hydraulis` raises
an error. `--allow-rank-deficient` produces an artifact whose coefficients are not identified, so
it exists for debugging a sweep that is too narrow.

---

## `srl-plot`

```bash
srl-plot RUN_DIR [--logs-subdir NAME] [--plots-subdir NAME] [--only PLOT] [--time-label TEXT]
```

Renders plots from an existing run without re-simulating.

| Argument | Default | Meaning |
|---|---|---|
| `RUN_DIR` | required | The `output_dir` of a previous run |
| `--logs-subdir` | `logs` | Directory under `RUN_DIR` that holds the JSONL streams |
| `--plots-subdir` | `plots` | Directory under `RUN_DIR` to write into |
| `--only` | all | One of `instance_load`, `rollout_step`, `staleness_distribution`, `request_timeline`, `throughput` |
| `--time-label` | `Simulation Time (s)` | X-axis label. Match the real run's label for a side-by-side figure |

---

## `srl-psrl-plot`

```bash
srl-psrl-plot LOG_DIR --output-dir DIR [options]
```

Converts a real PSRL run's log directory into the telemetry streams a simulation writes, then
renders the `instance_load` plot with the same plotter. `LOG_DIR` is read only. The converted
streams land in `<output-dir>/logs`, so `srl-plot DIR` can re-render them later and
`srl-compare` can read them.

| Flag | Default | Meaning |
|---|---|---|
| `--output-dir` | required | Where the converted logs and the plot go |
| `--role` | layout default | Replica role to plot. Validation replicas write the same stats files as rollout ones |
| `--instances` | none | Replica indices such as `0,2`. Overrides role selection |
| `--utc-offset-hours` | inferred | Offset of the text logs. Inferred from the logs when omitted |
| `--time-label` | `Wall-clock Time (s)` | X-axis label |
| `--logs-subdir`, `--plots-subdir` | `logs`, `plots` | Subdirectories under `--output-dir` |

A second group of flags retargets the converter at differently named log files. Each defaults to
the PSRL naming and is printed by `srl-psrl-plot --help`: `--stats-glob`, `--role-log-glob`,
`--gateway-stats`, `--phase-log`, `--train-worker-glob`, `--recompute-begin-marker`,
`--train-begin-marker`, `--train-end-marker`, `--sync-end-marker`, `--trajectory-glob` and
`--route-trace`.

---

## `srl-compare`

```bash
srl-compare SIMULATED REAL [--logs-subdir NAME]
```

Reads `<SIMULATED>/logs` and `<REAL>/logs` (a directory produced by `srl-psrl-plot`) and prints
one table with the simulated value, the real value and the error for: instance count, span,
training steps, train, recompute and sync seconds per step, peak and busy-mean running requests,
peak and busy-mean KV utilisation, and idle fraction. Occupancy averages are weighted by how long
each sample was in effect, so the two sides' different logging cadences do not bias them.

---

## `srl-workload`

```bash
srl-workload {collect-groups,merge-shards,list-workers,check-shard,verify-collection} ...
```

Builds the joint `(prompt, n responses)` traces that `workload=dapo_trace` reads. See
{doc}`../workload/trace-collection` for the procedure.

### `collect-groups`

Starts a vLLM server and generates `n` responses for each prompt assigned to this worker.

| Flag | Default | Meaning |
|---|---|---|
| `--dataset` | required | Parquet dataset path |
| `--tokenizer` | required | Tokenizer name or path |
| `--model` | required | Model passed to `vllm serve` |
| `--output` | required | Output shard JSONL path |
| `--worker-index` | `0` | This worker's index |
| `--n-workers` | `1` | Total worker count |
| `--sample-size` | `1024` | Distinct prompts across all workers |
| `--n` | `8` | Responses sampled per prompt |
| `--max-tokens` | `10240` | Maximum output tokens per response |
| `--temperature` | `1.0` | Sampling temperature |
| `--top-p` | `1.0` | Nucleus threshold |
| `--tp` | `1` | Tensor-parallel size of the server |
| `--concurrency` | `16` | In-flight prompts per worker |
| `--port` | `8700` | Server port |
| `--seed` | `42` | Sampling seed |
| `--gpu` | none | GPU to pin this worker to |
| `--server-ready-timeout` | `1800` | Seconds to wait for the server |
| `--serve-arg KEY=VALUE` | none | Extra `vllm serve` flag. Repeatable |

### `merge-shards`

| Flag | Meaning |
|---|---|
| `--shard-dir` | Directory of `worker-NN.jsonl` shards |
| `--n-workers` | Shards expected |
| `--output` | Merged joint trace path |
| `--max-output-tokens` | Optional. Verified against every shard header, and the shards are authoritative when omitted |

### `list-workers`

`--hosts-file PATH` (required) and `--base-port N` (default `8700`). Prints one
`index host gpu port` row per worker.

### `check-shard`

`--path PATH` (required). Exits 0 only when the shard holds every group it was asked for.

### `verify-collection`

`--trace`, `--expected-groups`, `--shard-dir` and `--n-workers`, all required. Prints the group
count on success and names the short shards on failure, with exit status 1.
