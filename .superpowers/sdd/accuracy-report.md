## Fix A: GEMM per-(n,k) slicing and wgrad shapes

### Schema diff

**Before:**
```python
GEMM_SCHEMA = TableSchema(
    op_name="gemm", family="gemm",
    slice_columns=("gemm_dtype",),
    axis_columns=("m", "n", "k"),
)
```

**After:**
```python
GEMM_SCHEMA = TableSchema(
    op_name="gemm", family="gemm",
    slice_columns=("gemm_dtype", "n", "k"),
    axis_columns=("m",),
)
```

`GemmQuery.slice_key` changes from `(dtype,)` to `(dtype, n, k)`; `GemmQuery.coords` changes from
`(m, n, k)` to `(m,)`.  `KernelDatabase._sol` for GEMM now reads `n` and `k` from the query
rather than from `coords` (since coords no longer carries them).  `holdout._sol_for_gemm` unpacks
the three-element slice key for `n` and `k`.

### Test updates

| File | Test | What changed and why |
|---|---|---|
| `test_kernel_table.py` | `test_load_exposes_rows_and_slices` | Slice key `("bfloat16",)` → `("bfloat16", 4096, 4096)`; coords `(1024.0, 4096.0, 4096.0)` → `(1024.0,)`. |
| `test_kernel_table.py` | `test_distinct_dtypes_become_distinct_slices` | Both slice keys gain `4096, 4096` suffix. |
| `test_kernel_table.py` | `test_grid_derives_utility_from_sol` | Slice key updated; `utility` query coords drop `n`/`k`. |
| `test_kernel_table.py` | `test_grid_is_memoized_per_slice` | Slice key updated. |
| `test_kernel_table.py` | `test_unknown_slice_key_is_rejected` | Wrong-key probe uses new shape. |
| `test_kernel_table.py` | `test_duplicate_shapes_keep_the_fastest_measurement` | Slice key and coords updated. |
| `test_kernel_table.py` | `test_grid_does_not_alias_distinct_sol_functions` | Slice key updated; utility queries use 1-D coords. |
| `test_kernel_table.py` | `test_grid_cache_keeps_its_sol_function_alive` | Slice key updated. |
| `test_kernel_table.py` | `test_non_finite_sol_raises` | Slice key updated. |
| `test_kernel_holdout.py` | `test_a_single_dented_point_shows_up_as_the_worst_error` | `worst_coords` is now `(1024.0,)`: with 1-D interpolation the reported coordinate is just `m`. |
| `test_kernel_holdout.py` | `test_real_gemm_structure_scores_all_60_points` | Renamed and rewritten. Before: 60 rows in one slice, bracketing credited all 60 interior points. After: 60 rows split into 4 slices of 15 rows each; each 1-D slice loses its two m-endpoints (smallest and largest m are not bracketed in 1-D), giving 13 scorable points per slice × 4 slices = 52. The test now loops over slice keys, counts bracketed indices per slice, and asserts total == 52. |
| `test_kernel_holdout.py` | `test_holdout_agrees_with_the_database_on_a_full_table_point` | Slice key `("bfloat16",)` → `("bfloat16", 4096, 4096)`; coords `(2048.0, 4096.0, 4096.0)` → `(2048.0,)`. |
| `test_kernel_measurements.py` | `test_written_gemm_table_loads_through_kernel_table` | Same slice-key and coords update as table tests. |
| `test_kernel_gemm.py` | `test_profile_gemm_writes_a_loadable_table` | Slice key updated; `row_count` 8 → 16 (8 forward + 8 wgrad, no overlap at these token counts). |
| `test_kernel_gemm.py` | 3 new tests | `test_wgrad_shapes_are_derived_from_forward_projections`, `test_wgrad_shapes_cover_all_forward_projections`, `test_gemm_all_shapes_is_union_of_forward_and_wgrad`. |

### Wgrad sweep points added

For Qwen3-8B / TP=1 with the 16-token-count production sweep:

| Family | Count |
|---|---|
| Forward GEMM shapes | 64 (4 projections × 16 token counts) |
| Wgrad GEMM shapes | 64 (4 projections × 16 token counts, as `(n_fwd, m, k_fwd)`) |
| Overlap (coincide) | 2 (`(4096, 4096, 4096)` and `(4096, 4096, 12288)`) |
| **Net new measurement points** | **62** |

The two overlapping shapes are o_proj (n=4096, k=4096) and down_proj (n=4096, k=12288) at
token count m=4096, which equals the projection's n_fwd.

### Gate run 1: after per-(n,k) slicing (existing parquet, no re-sweep)

```
gemm [vllm]: 56 held-out points, MRE 4.08%, p90 9.62%, max 46.42% at (64.0,), utility 0.475-0.949
```

GEMM 5% gate: **PASS** (was 8.48% before slicing).

### Gate run 2: after wgrad shapes added (new parquet, post-sweep)

```
gemm [vllm]: 58 held-out points, MRE 4.16%, p90 10.15%, max 46.92% at (64.0,), utility 0.466-0.949, 30 slices skipped
```

GEMM 5% gate: **PASS**.

MRE is 4.16% with wgrad shapes (vs 4.08% without), because the new wgrad slices introduce new
interpolation surfaces with slightly different accuracy; the slices with fewer than 4 rows (most
wgrad slices, which have 3 or 1 rows per `(dtype, n_wgrad, k)` slice) are skipped.

Attention families still fail (separate dispatch-floor issue, not in scope for this task).

## Fix B: max-floor in the attention SOL

### Where the floor values live and why

The per-backend dispatch floors are stored in `ATTENTION_FLOOR_S` in
`dryrun/cost/kernel/sol.py` — a module-level dict keyed by backend name and phase.

**Alternatives considered:**

- `GpuSpec`: rejected. `GpuSpec` is per-GPU hardware; the floor is a property of the
  backend's Python/dispatch path cost, not the silicon. A vLLM floor applies on H20 and
  would apply equally on any other GPU.
- Parquet provenance field: most principled but requires a schema change, additional write-time
  tooling, and a read-time lookup. Adds complexity for a two-scalar per-backend value.
- Module-level mapping in `sol.py`: chosen. The fitted origin is visible at the definition
  site (backend, phase, date, hardware). The values are not arbitrary constants — they are
  labelled as fitted, with the measurement date and hardware in the docstring and the comment.
  Simpler than a schema change, cleaner than hiding them in a config YAML with no provenance.

### Floor values

Fitted from real H20 measurements (Qwen3-8B, TP=1, 2026-08-24):

| backend | context | generation |
|---|---|---|
| vLLM | 32 us | 22 us |
| FlashInfer | 4 us | 6 us |

vLLM's floors are larger because vLLM routes through a Python dispatch path with more overhead
than FlashInfer's lower-level C++ wrappers.

### Why max, not sum

Decisive evidence from the H20 parquet: decode latency is flat across a 64x increase in KV
work at small batch sizes (batch=1 measures ~70 us at step 128 and ~70 us at step 8192).
`floor + work` would predict 70 us + (small) = 70 us; that is numerically close, but when
work eventually grows large (batch=48, step=8192) the model would predict `floor + work` where
`floor + work > work`. Experiments with the additive model confirmed it barely helps:
generation_attention/vllm only goes 34.89% -> 33.23%. The max model works because the kernel
is entirely dispatch-limited at small work and entirely compute/memory-limited at large work;
the two regimes do not overlap.

### Round-trip consistency

Both `evaluate_holdout` and `KernelDatabase._grid`/`_sol` resolve the same floor from
`ATTENTION_FLOOR_S` using the resolved backend name. The floor is captured at grid-construction
time (inside the `sol_fn` closure in `_grid`), so the utility stored in the grid and the SOL
used at query time use identical floors. The existing
`test_holdout_agrees_with_the_database_on_a_full_table_point` test in `test_kernel_holdout.py`
guards this round trip and continues to pass.

### Anchor tests added

In `tests/test_kernel_sol.py`:

| Test | What it proves |
|---|---|
| `test_sol_components_has_three_fields` | `SolComponents` carries `floor_s` as a field |
| `test_floor_zero_reproduces_original_latency_s` | Default `floor_s=0.0` is provably unchanged |
| `test_floor_binds_when_work_is_negligible` | batch=1, step=1: floor dominates, returns floor |
| `test_floor_does_not_affect_large_shapes` | batch=64, step=8192: roofline dominates, floor ignored |
| `test_gemm_sol_has_no_floor_field_and_default_is_zero` | GEMM floor_s=0.0 and unchanged latency_s |
| `test_context_attention_floor_binds_at_tiny_shapes` | batch=1, seq_len=1: floor dominates |
| `test_attention_floor_s_mapping_covers_vllm_and_flashinfer` | Both backends and phases present |

### Full gate before and after

Before Fix B (as-shipped, GEMM after per-(n,k) slicing):

| family | backend | points | MRE | gate |
|---|---|---|---|---|
| `gemm` | vllm | 58 | 4.16% | PASS |
| `context_attention` | flashinfer | 59 | 11.39% | fail |
| `context_attention` | vllm | 59 | 20.85% | fail |
| `generation_attention` | flashinfer | 72 | 20.08% | fail |
| `generation_attention` | vllm | 72 | 34.89% | fail |

After Fix B (`ATTENTION_FLOOR_S` floors in max), existing parquet, `--fraction 1.0`:

| family | backend | points | MRE | p90 | max | gate |
|---|---|---|---|---|---|---|
| `gemm` | vllm | 58 | 4.16% | 10.15% | 46.92% | PASS |
| `context_attention` | flashinfer | 59 | 9.87% | 32.69% | 47.12% | fail |
| `context_attention` | vllm | 59 | 5.74% | 17.27% | 25.92% | fail |
| `generation_attention` | flashinfer | 72 | 6.01% | 14.61% | 39.67% | fail |
| `generation_attention` | vllm | 72 | 6.63% | 17.57% | 28.46% | fail |

GEMM is unchanged at 4.16% (no floor applied). All four attention families improved
substantially. `context_attention/flashinfer` at 9.87% remains above the 5% gate: the 4 us
floor resolves the dispatch mechanism but not the residual small-shape variance. Densifying
the grid in the overhead-limited regime is the next lever.

## Coverage: the logits projection

### vLLM sharding of the LM head

Verified in `third_party/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py`:
`ParallelLMHead` extends `VocabParallelEmbedding`, which sets
`num_embeddings_per_partition = divide(num_embeddings_padded, tp_size)`.
vLLM **does** shard the LM head over tensor-parallel ranks. The per-rank logits
projection weight is `(vocab_size / TP) x hidden_size`.

### Semantic difference and m-range choice

The four per-layer projections run once per layer (36 times per step) and their m
dimension is the total token count in the batch (up to 16384 in the production sweep).
The logits projection runs once per step and its m dimension is the request count:
one sampled token per request in decode, one last-token per sequence in prefill.

The baseline vLLM TP=1 profiles show decode request counts of 1..64. Using the full
`token_counts` sweep (up to 16384) would measure shapes the logits kernel can never
reach (no real vLLM run has 16384 active requests). A dedicated
`logits_request_counts` field was added to `GemmSweep`, defaulting to
`[1, 2, 4, 8, 16, 32, 48, 64]`, matching the observed decode batch-size range.

The logits slice `(bfloat16, 151936, 4096)` does not overlap with any per-layer
forward or wgrad slice, so it is cleanly distinguishable by its `(n, k)` key.
Its `GemmShape.name` is `lm_head`, which is absent from the four per-layer projection
names (`qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`), preventing accidental
multiplication by `num_layers` in future composition code.

### Peak allocation

At the largest measured point (m=64, n=151936, k=4096 in bfloat16):

- activation: 0.52 MB
- weight: 1.24 GB
- output: 19.5 MB
- **total peak: 1.265 GB**

No OOM; the H20 has ~95 GB free. Operands freed and `torch.cuda.empty_cache()` called
per point as the existing loop does.

### GEMM holdout gate after sweep

Re-run: `dryrun-profile kernels holdout --config examples/profile/kernel/qwen3_8b_h20_tp1.yaml --fraction 1.0`

| family | backend | points | MRE | p90 | max | gate |
|---|---|---|---|---|---|---|
| `gemm` | vllm | 64 | 3.91% | 7.70% | 47.98% | PASS |

Gate before this change: 4.16% (58 points, 30 slices skipped).
After this change: 3.91% (64 points, 30 slices skipped).

The 6 additional held-out points come from the new logits slice, which has 8 rows and
6 interior (bracketed) points after removing the two endpoints. The logits slice
interpolates well (all-memory-bandwidth-bound at these small m values), pulling the
MRE down slightly from 4.16% to 3.91%.

### End-to-end comparison (`.superpowers/sdd/e2e-check.py`)

Baseline: `outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl`.
Filtering: `phase == "decode"`, `sum_l2 == 0`, `n_tokens == request_count`.
Bucketing: `(request_count, round(mean_context / 256) * 256)`, min 20 iterations,
median latency per bucket.  TRUE mean context used (no power-of-two rounding).

Compose: `num_layers * (qkv + o + gate_up + down + generation_attn at m=rc) + lm_head at m=rc`.

| Variant | Buckets | MAE % | p90 % | ratio (pred/meas) |
|---|---|---|---|---|
| With logits | 210 | 5.00% | 16.42% | 1.0104 |
| Without logits | 210 | 4.51% | 11.55% | 0.9918 |

The logits projection shifts the median ratio from 0.99 to 1.01: adding the real work
closes the under-prediction but introduces a slight over-prediction, because the logits
kernel time measured in isolation (pure CUDA kernel with no vLLM scheduling overhead)
is a lower bound on what the full vLLM step charges. The MAE moves from 4.51% to 5.00%.

The previously reported 4.4% end-to-end figure was measured without the logits term;
4.51% (our without-logits variant) reproduces it closely (the small difference is
because the comparison now uses true mean context rather than power-of-two rounding).
