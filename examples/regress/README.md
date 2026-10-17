# Cost regression examples

Each command consumes normalized profile JSONL and writes a versioned artifact with explicit topology entries,
profile digest, coefficients, holdout metrics, feature ranges, and diagnostics.

```bash
bash examples/regress/regress_rollout_roofline.sh
bash examples/regress/regress_rollout_psrl.sh
bash examples/regress/regress_rollout_distserve.sh
bash examples/regress/regress_training_hydraulis.sh
```

Load exact entries programmatically:

```bash
python examples/regress/load_artifacts.py \
  --rollout-artifact outputs/cost/qwen3_8b/rollout_psrl.json \
  --training-artifact outputs/cost/qwen3_8b/training_hydraulis.json
```

Use the same artifacts in a simulation:

```bash
srl \
  rollout_cost=psrl \
  rollout_cost.artifact_path=outputs/cost/qwen3_8b/rollout_psrl.json \
  training_cost=hydraulis \
  training_cost.artifact_path=outputs/cost/qwen3_8b/training_hydraulis.json \
  job.batch_size=8 \
  training_cost.micro_batch_size=1
```

Training regression rejects rank-deficient designs by default. `--allow-rank-deficient` is intended only for
parser and smoke checks, not for an artifact used by search.
