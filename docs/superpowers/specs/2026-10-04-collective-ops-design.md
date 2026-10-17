# Spec A: Collective Communication Op Family

**Date:** 2026-10-04
**Status:** implemented and gated. A0-A4 complete; see the phase table in section 7.
**Repo:** `/apdcephfs_zwfy/share_303760348/lhy/SimRL`
**Sequencing:** this is spec 1 of 2. Spec B (kernel-composed training latency) consumes
A's tables and is designed separately, after A's gates pass. §9 states the contract B
relies on, so A's interfaces can be judged against their only consumer.

## 1. Context

The rollout side already predicts end-to-end latency by summing measured kernels:
`srl/cost/kernel/` holds a parquet-backed `(shape -> latency)` table per op,
interpolates SOL-normalized utility at query time, and `ComposedRollout` sums a
declarative op list from `ModelOps`. Validated at `tp=2` on H20 / Qwen3-8B: prefill
3.60%, decode (`requests >= 5`) 2.96%, mixed 3.64% MRE.

The training side has no such path. `HydraulisTrainingCost` and `FsdpTrainingCost` fit a
4-term polynomial to whole-step wall times (`srl/cost/training/latency.py`). That
module's own docstring names the fix: "feeding training GEMMs through the same kernel
tables is the principled next step rather than another fitted polynomial term."

Two things block it, and both live in the collective layer:

**1.1 There is exactly one collective op, and it is the wrong one for training.**
`srl/cost/kernel/ops/collective.py` registers `all_reduce` and nothing else. But
`srl/profiler/adapters/megatron/launch.py:282` sets
`cfg.model.sequence_parallel = (tp > 1)`, and with sequence parallelism on, Megatron's
`RowParallelLinear` calls `reduce_scatter_to_sequence_parallel_region` instead of
`reduce_from_tensor_model_parallel_region`, paired with
`gather_from_sequence_parallel_region` on the column-parallel side
(`megatron/core/tensor_parallel/layers.py:297,302,422,1274,1278`). A Megatron `tp=2`
step therefore runs reduce-scatter / all-gather pairs over `tokens*hidden/tp`-sized
messages, not all-reduces over `tokens*hidden`. Pricing it from the `all_reduce` table
would be wrong in op identity, message size, and count simultaneously.

Also absent: pipeline `p2p` send/recv, and the DP gradient reduction.

**1.2 Topology is unrepresented, so a cross-node group is priced as an NVLink group.**
`srl/profiler/kernels/collective.py:126` pins `MASTER_ADDR=127.0.0.1` and spawns with
`mp.spawn`, so every measured world size is intra-node over NVLink.
`GpuSpec.link_bandwidth_bytes_per_s` is a single constant. `docs/profiling/tensor-parallel.md`
already records this as a known defect and names the trigger: "Acceptable for rollout,
where `tp` stays inside a node. It is the first thing to fix when DP or PP communication
is modelled." Spec B models DP gradient reduction, so the time has come.

This matters because `world_size` alone is not a schedule. 8 ranks on one node, 8 ranks
across 2 nodes, and 8 ranks across 8 nodes traverse different links in different counts,
and today all three would resolve to the same table row.

### Intended outcome

A collective family that covers every collective the two training backends and the
rollout backend actually issue, measured at the topology they actually run on, each op
gated by the same holdout mechanism as GEMM and attention. `ComposedRollout`'s validated
numbers must not move.

## 2. Verified environment

| Fact | Evidence |
|---|---|
| 3 distinct nodes, 8x H20 each (24 GPUs) | Distinct GPU UUIDs per host; `nvidia-smi -L` = 8 each |
| Nodes: `28.49.55.85`, `29.162.247.148` (this host), `28.49.196.175` | User-authorized; passwordless ssh verified (port 36000) |
| Shared CephFS at the repo path on all three | `ls` of the repo root succeeds on each |
| Intra-node: all-to-all `NV18` NVLink, `link_bandwidth` 4.5e11 B/s | `nvidia-smi topo -m`; `gpu_specs.yaml` |
| Inter-node: 42 active 200 Gb/s HDR ports, bonded into 8 x 400 Gb/s | `/sys/class/infiniband/*/ports/1/rate`; `/sys/class/net/bond*/speed` |
| Exactly 1 `PIX` NIC per GPU, all 8 | `nvidia-smi topo -m` |
| `NCCL_IB_HCA` pre-pinned to `mlx5_bond_1..8`; `NCCL_SOCKET_IFNAME=bond1` | environment |
| GPUs 0-3 on NUMA 0, 4-7 on NUMA 1 | `nvidia-smi topo -m` |
| `megatron-core` 0.17.1 at `third_party/Megatron-Bridge/3rdparty/Megatron-LM` | import path resolved |
| Megatron DDP bucket default `max(40M, 1M*dp)` params | `distributed_data_parallel.py:66` |

The NUMA split means a 4-rank group is not topology-free even inside one node, which §4
addresses by keying on `nodes` rather than asserting intra-node uniformity. The 1:1
GPU-to-bond affinity is what makes a single `inter_node_bandwidth` constant defensible at
these degrees; see §3's A3 detail.

## 3. Design decisions

| # | Decision | Rationale |
|---|---|---|
| A1 | One op class per collective, sharing a `_CollectiveOp` base that implements `sol()` once and takes `traffic_factor(world_size)` from the subclass | The five collectives differ *only* in their traffic factor. Five copies of `sol()` is exactly the drift risk `ops/base.py` was created to eliminate |
| A2 | `nodes` joins `world_size` as a **slice** column | A slice is matched exactly and refuses an unmeasured value. Interpolating between a 1-node and a 2-node 8-rank group would produce a number describing neither topology |
| A3 | `GpuSpec` gains `inter_node_bandwidth_bytes_per_s`; SOL takes `max(intra_term, inter_term)` | A cross-node ring is bounded by its slowest hop. `max` not sum, matching the existing `max(math, memory, floor)` physics |
| A4 | Collective measurement moves from `mp.spawn` to a torchrun child, driven by an ssh-based multi-node launcher | `mp.spawn` on `127.0.0.1` is structurally single-node. Nothing short of a real rendezvous measures a real NIC |
| A5 | Re-sweep intra-node rather than migrate the existing parquet | Adding `nodes` is a breaking format change, which the project contract permits. The existing doc measures an intra-node collective sweep in "seconds, not hours", so re-measuring is cheaper and safer than a migration path |
| A6 | `backend` stays the existing framework directory level (`vllm` / `nccl`) | Already load-bearing: rollout reads `vllm` because vLLM dispatches `custom_all_reduce`, while training uses `torch.distributed`. Training ops default to `nccl` |
| A7 | Every new op declares `fits_floor = True` | Measured precedent: a floor-less `all_reduce` SOL errs by 220x at 4096 bytes. Launch cost dominates at the small messages a decode step and an SP-sharded layer both live in |

### A1 in detail: the traffic factors

Per-rank bytes moved, standard ring lower bounds:

| op | `traffic_factor(N)` | issued by |
|---|---|---|
| `all_reduce` | `2(N-1)/N` | vLLM TP; Megatron TP without SP; DP grad reduce without distributed optimizer |
| `reduce_scatter` | `(N-1)/N` | Megatron TP **with SP**; DP grad reduce under distributed optimizer; FSDP2 per-layer grad |
| `all_gather` | `(N-1)/N` | Megatron TP **with SP**; FSDP2 per-layer param gather |
| `broadcast` | `1.0` | weight sync |
| `p2p` | `1.0` | PP stage boundary send/recv |

Two of these are **not** ring collectives, and their factors are deliberate declarations
rather than degenerate cases:

- `broadcast` is not `(N-1)/N`. A pipelined broadcast delivers the payload to every rank,
  so each non-root rank receives `bytes` once and the bound is `bytes / bandwidth`,
  independent of `N`. Giving it the reduce-scatter factor would understate it by up to 2x
  at large `N`.
- `p2p` is a single directed transfer, also `N`-independent. It keeps `world_size` as a
  slice anyway, because the *link* it traverses depends on which ranks are paired, which
  is a function of the mesh.

Both keep `fits_floor = True`, so the measured launch cost is still represented.

The base class owns `sol()`:

```python
@dataclass(frozen=True)
class _CollectiveOp:
    """Shared SOL binding for every ring collective.

    Subclasses declare name and traffic factor only. sol() exists once so the
    five ops cannot drift in how they bind (spec, slice_key, coords) onto a
    roofline, which is the failure ops/base.py was introduced to prevent.
    """

    family = "collective"
    slice_columns = ("dtype", "world_size", "nodes")
    axis_columns = ("message_bytes",)
    fits_floor = True

    def traffic_factor(self, world_size: int) -> float: ...

    def sol(self, spec, *, slice_key, coords, floor_s=0.0) -> SolComponents:
        dtype, world_size, nodes = slice_key
        ...
        return SolComponents(math_s=0.0, memory_s=memory_s, floor_s=floor_s)
```

Each subclass is then ~6 lines, and `register_op` is called per subclass at module scope
as the existing ops do.

### A2 in detail: `world_size` must leave `parallelism.tensor_parallel`

`examples/profile/kernel/qwen3_8b_h20_tp2.yaml` declares `parallelism.tensor_parallel: 2`,
and `profile_collective` spawns `config.tensor_parallel` ranks. World size is therefore a
function of the config's TP degree, which is why covering `tp=4` and `tp=8` today needs
three separate config files swept with `FAMILY=collective`
(`examples/profile/kernel/profile_qwen3_8b.sh` documents exactly this workaround).

That does not extend. A DP gradient reduction at `world_size=24` has nothing to do with a
tensor-parallel degree, and creating a `tensor_parallel: 24` config to measure it would be
a lie about what is being measured. So `CollectiveSweep` gains its own
`world_sizes`, `nodes`, and `hostnames`, and the collective family stops reading
`parallelism.tensor_parallel` entirely. One sweep then covers every degree and topology,
replacing the three-config workaround rather than extending it.

### Measurement granularity: one process group, five ops

`cli/profile.py:261` calls `ops[0].measure(...)` **once per family**, not once per op,
with a comment explaining why for attention: both phases come out of one backend sweep,
so calling per op would measure the grid twice and could pick a different backend per
phase.

The collective family wants the same treatment for a stronger reason: process-group setup
and the NCCL handshake dominate a small-message measurement, and a cross-node rendezvous
costs seconds. Timing all five collectives inside one group amortizes that once, and
guarantees all five rows describe the same topology and the same communicator. So
`_CollectiveOp.measure` is implemented once on the base, measures every registered
collective, and returns every path. The four subclasses inherit it.

This means the existing `ops[0].measure` dispatch needs **no CLI change** — which is the
second structural test of the abstraction, alongside A3's untouched `floor.py`.

### A3 in detail: the two-level SOL

A ring over `world_size` ranks laid out across `nodes` nodes has `world_size` hops, of
which exactly `nodes` cross a node boundary (the ring enters and leaves each node once).
So the **fraction** of the ring's hops that traverse a NIC is `nodes / world_size`, and
per-rank traffic `factor * message_bytes` splits accordingly:

```
inter_fraction = nodes / world_size          # 0 when nodes == 1, by the guard below
intra_s = factor * bytes * (1 - inter_fraction) / link_bw
inter_s = factor * bytes * inter_fraction    / inter_node_bw
memory_s = max(intra_s, inter_s)
```

`nodes == 1` is special-cased to `inter_fraction = 0.0` rather than falling out of the
formula, which matters: with one node the ring crosses no boundary, but `nodes/world_size`
would evaluate to `1/8`, not zero. Written without the guard this would silently charge an
intra-node sweep for NIC traffic and perturb the very numbers §4 gates on. With the guard,
`nodes == 1` reduces **exactly** to today's formula, which is what makes A5's re-sweep
verifiable against the current numbers rather than merely plausible.

`max` rather than sum: the two phases of a hierarchical collective overlap in the
best case, and the slower level bounds the result. This is the same choice the existing
`max(math_s, memory_s, floor_s)` makes, and it is an optimistic bound by construction.

#### The spec value, measured not assumed

`inter_node_bandwidth_bytes_per_s` is a property of the **cluster fabric**, not of the
GPU, which makes it the one field in `gpu_specs.yaml` that is not a vendor number. Read
off this cluster:

| fact | value | evidence |
|---|---|---|
| Active IB ports | 42, each 200 Gb/s 4X HDR | `/sys/class/infiniband/*/ports/1/{state,rate}` |
| NCCL's actual device list | 8 **bonds**, `mlx5_bond_1..8` | `NCCL_IB_HCA` already set in the environment |
| Per-bond netdev speed | 400000 Mb/s (2 slaves each) | `/sys/class/net/bond{1..8}/speed` |
| GPU-to-NIC affinity | exactly **1 PIX NIC per GPU**, all 8 | `nvidia-smi topo -m` |

So the per-GPU egress is one 400 Gb/s bond = **5.0e10 bytes/s**, and the node aggregate is
8 x 400 Gb/s = 3.2 Tb/s. The **per-GPU** figure is the right one for the spec, because SOL
is a per-rank bound and each rank owns one bond. Compare `link_bandwidth_bytes_per_s =
4.5e11` (NVLink): the NIC is ~9x slower, which is why `nodes` cannot be a free dimension.

Two things this settles that a vendor figure would not have. `NCCL_IB_HCA` is **already
pinned** in the environment, so the devices NCCL uses are known rather than guessed, and
the measurement will not silently pick a different fabric than the spec describes. And the
1:1 GPU-to-bond affinity means a cross-node ring does not contend for NICs at
`world_size <= 8` per node, which keeps the single-constant model defensible at the degrees
being swept.

The fitted utility absorbs the remaining gap, as it does for every other SOL
simplification. The value and this derivation go into `gpu_specs.yaml` as a comment, since
a future cluster with different bonding would need it re-derived rather than copied.

A deliberate simplification to record: a hierarchical collective (NCCL's usual choice
intra-then-inter) is not a flat ring, so `max` of the two terms understates a real
two-phase schedule. `nodes` being a slice column is what contains the damage: utility is
fitted per topology, so a systematically-wrong SOL at `nodes=2` is absorbed into
`nodes=2`'s own utility rather than leaking into `nodes=1`.

### A4 in detail: the multi-node launcher

`profile_collective` is rewritten as two pieces, mirroring how the training adapters
already split a torchrun child from its driver:

- `srl/profiler/kernels/collective_launch.py` — the torchrun child. Reads a JSON spec,
  joins the process group, times each `(op, message_size)` pair, and rank 0 writes rows.
  Imports torch at call time only.
- `srl/profiler/kernels/collective.py` — the driver. For `nodes == 1`, runs torchrun
  locally. For `nodes > 1`, ssh-launches torchrun on each host with a shared
  `--master_addr` / `--master_port` rendezvous and `--node_rank` per host, then waits.

Node list comes from the sweep config (`collective.hostnames:` as a list), so a topology is
declared data rather than discovered. The driver refuses to run if the requested `nodes`
exceeds the hostname count, rather than silently measuring fewer.

**The NCCL fabric variables are inherited, not set.** `NCCL_IB_HCA`,
`NCCL_SOCKET_IFNAME`, `NCCL_IB_GID_INDEX`, `NCCL_IB_TC`, `NCCL_IB_SL` and
`NCCL_IB_TIMEOUT` are already exported by the environment script (see §2). The launcher
must **not** override them: they describe the cluster's RoCE configuration, and a sweep
run under different values would measure a fabric that no real training job uses. The
driver logs the resolved values into the run directory so a table's rows can be traced to
the fabric configuration that produced them, and refuses to launch cross-node if
`NCCL_IB_DISABLE=1`, which would silently fall back to TCP sockets and produce rows that
look like a slow NIC rather than an absent one.

Reused rather than rebuilt: `_timing.py:time_cuda_call`, `PerfTableWriter.write`
(which already validates columns against `op.table_columns()`), and the
`_dispatched_backend` probe that reads vLLM's live communicator predicates.

**Timing a collective is not timing a kernel.** Rank 0's elapsed time includes however
long it waited for the slowest rank to arrive. The measured value must be the
*post-barrier* bracket: barrier, then time `repeat` iterations, so skew is paid once
outside the measurement rather than once per iteration. This is the opposite of the
training measure path, which deliberately avoids a barrier because there a barrier would
charge every rank for the slowest one. The difference is that a collective's cost
genuinely includes synchronization and a training step's does not.

## 4. Schema change and the neutrality gate

`slice_columns` becomes `("dtype", "world_size", "nodes")`. Under A5 the existing
`all_reduce_perf.parquet` files are re-measured, not migrated. A table lacking the
`nodes` column is **rejected** with a message naming the sweep command, never read with
an assumed `nodes=1`: a lenient reader would attribute cross-node measurements to NVLink
and the error would surface as an unexplained model defect much later.

The refactor touches code behind a passing accuracy gate, so **the numbers must not
move**. Baselines to capture into a committed file *before any source edit*, from
`docs/profiling/tensor-parallel.md` and the kernel-op-abstraction baseline table:

**Captured 2026-10-04 at commit `c1e557f`** into
`tests/test_collective_baseline_regression.py` (the gate) and
`outputs/baseline_a0/A0_BASELINE.json` (the raw capture). Every figure below was
*re-derived by running the code*, not transcribed from a document:

| quantity | measured baseline | required after |
|---|---|---|
| `all_reduce` holdout, `vllm` / `nccl`, ws=2 | 6.0122% / 4.9606%, 13 pts | within +-0.5pp |
| `all_reduce` holdout, `vllm` / `nccl`, ws=4 | 3.7643% / 7.5568%, 13 pts | within +-0.5pp |
| `all_reduce` holdout, `vllm` / `nccl`, ws=8 | **9.2969% / 10.6604%**, 13 pts | within +-0.5pp |
| `gemm` holdout, tp=2 | 3.9019%, 62 pts | **exact** |
| `context_attention` `vllm` / `flashinfer`, tp=2 | 6.6801% / 9.9274%, 59 pts | **exact** |
| `generation_attention` `vllm` / `flashinfer`, tp=2 | 4.5715% / 6.3715%, 72 pts | **exact** |
| `elementwise` holdout, tp=2 | 9.3544%, 52 pts | **exact** |
| rollout validate prefill, tp=2 | 3.597%, 29 buckets | within +-0.5pp |
| rollout validate decode, whole grid | 9.510%, 204 buckets | within +-0.5pp |
| rollout validate decode, `reqs>=5` | 2.957%, 137 buckets | within +-0.5pp |
| rollout validate mixed, tp=2 | 3.643%, 7 buckets | within +-0.5pp |
| fitted floors, all families | per `floors.json`, see test | within one candidate step |

Fifteen of the seventeen holdout figures reproduced **bit-exactly** against the published
documentation, which is what makes the two that did not diagnosable rather than ambiguous.

#### A0 finding: the `world_size=8` published figures describe data that no longer exists

`docs/profiling/tensor-parallel.md` publishes tp=8 at 7.56% (`vllm`) and 24.80% (`nccl`).
The tables on disk score **9.2969%** and **10.6604%** under current code, reproducibly.

The cause is file ordering, not a code change. In the `kernel_tp8` tree the parquet tables
have mtime `2026-09-02T16:46:50.5` while `holdout_report.json` has `16:46:49.5`, so the
report **predates by one second the tables it supposedly scores**. The `kernel_tp4` tree
has the correct order (tables `16:45:09`, report `16:46:41`). A re-sweep overwrote the tp=8
tables and `holdout` was never re-run against them. Since every other figure in both trees
reproduces exactly, a code regression is ruled out.

This corrects the direction of a published claim. The 24.80% `nccl` figure is cited as
evidence that the control backend degrades sharply with world size; on the surviving
tables it is 10.66%. The **ordering still holds** (`vllm` 9.30% beats `nccl` 10.66% at
ws=8, and the gap widens from ws=4 where `nccl` wins), so the qualitative conclusion that
the dispatched backend matters more as the world grows survives. Its magnitude was
overstated.

The A0 gate pins the **live** values, because a gate must compare against data that
exists. Re-running `fit-floor` then `holdout` on the tp=8 tree would settle the published
number; that is recorded as an A2 task rather than silently corrected, since A2 re-sweeps
that degree anyway under the new `nodes` schema.

Note the asymmetry, which is the honest version of the gate. The GEMM and attention
holdouts must be **bit-identical**: nothing in their path changes. The collective numbers
cannot be, because A5 re-measures them, and `docs/development/testing.md` records that
`latency_s` is not bit-reproducible (holdout MRE moved 3.896% -> 3.828% between identical
runs). So the collective gate is "within re-measurement noise", with the tolerance fixed
in advance at **+-0.5pp absolute** and a repeat sweep as the arbiter if a number lands
outside it. Claiming "identical" for a re-measured table would be a gate that cannot pass.

The end-to-end rollout numbers are the load-bearing check: they exercise the new slice
column, the new SOL with `nodes=1`, and the re-swept table together.

## 5. Sweep space

| dimension | values |
|---|---|
| `ops` | `all_reduce`, `reduce_scatter`, `all_gather`, `broadcast`, `p2p` |
| `world_sizes` | 2, 4, 8 at `nodes=1`; 16 at `nodes=2`; 24 at `nodes=3` |
| `nodes` | 1, 2, 3 |
| `dtype` | `bfloat16` (activations, grads), `float32` (fp32 grad reduce) |
| `message_bytes` | 4 KiB to 512 MiB, powers of two (see below) |
| `backend` | `vllm` and `nccl` from one sweep, as today |

`(world_size, nodes)` is enumerated as an explicit list of valid pairs, not a cross
product: `world_size=2, nodes=3` is not a configuration, and a cross product would spawn
runs that cannot form a mesh. The config declares pairs so an impossible topology is a
config error rather than a launcher failure.

### The grid must sweep message bytes directly, over a far wider span

The ops do not all send the same size at the same token count, so a `token_counts` grid
cannot describe them. Computed for Qwen3-8B (`hidden=4096`, 36 layers, 8.19B params):

| caller | message size | at |
|---|---|---|
| TP `all_reduce`, no SP | `tokens*hidden*2` | 8 KiB (1 tok) to 512 MiB (65536 tok) |
| TP `reduce_scatter`/`all_gather`, SP on, tp=2 | `tokens*hidden*2/tp` | 4 KiB to 256 MiB |
| DP gradient reduce, fp32 | **bucket-sized**, see below | ~153 MiB per bucket |

The DP figure needs care, and getting it wrong would have set the grid's ceiling three
orders of magnitude too high. Megatron does **not** reduce the whole 30.5 GiB fp32
gradient buffer in one collective: `distributed_data_parallel.py:66` sets
`bucket_size = max(40000000, 1000000 * dp_group.size())` **parameters**, so an 8.19B model
reduces ~205 buckets of 152.6 MiB each, and the per-rank ring message is
`bucket_bytes / dp` (6.4 MiB at dp=24, 76 MiB at dp=2). The `bucket_size` docstring states
the reason directly: larger DP needs larger buckets so collectives do not become
latency-bound.

So the grid sweeps `message_bytes` as a declared list spanning **4 KiB to 512 MiB** in
powers of two, which covers every row above. `message_bytes` is already the axis column,
so this is a grid-density decision rather than a schema one. The existing rollout sweep
spans only 8 KiB to 512 KiB; extending it is what makes a training query land inside the
measured frontier instead of clamping at its boundary.

Two consequences recorded for Spec B rather than solved here: the bucket count depends on
`bucket_size`, which depends on `dp`, so B must count buckets rather than assume one
collective per step; and `pad_buckets_for_high_nccl_busbw` (default `False`) changes the
message size if ever enabled, so B must read the resolved DDP config rather than
recompute the default.

`float32` is not optional: Megatron reduces gradients in fp32
(`srl/cost/training/inventory.py` documents the fp32 gradient buffer), so a
bf16-only table cannot price the DP reduction Spec B needs.

Cross-node runs at `world_size` 16 and 24 only, because fewer ranks spread over 2-3 nodes
is not a configuration either backend runs. `nodes=3` is included because 24 GPUs are
available and a 3-node ring has a different hop count from a 2-node one; without it,
`nodes` would be a two-valued column and any `nodes`-dependence would be unfalsifiable.

### Sweep size

5 ops x 18 message sizes x 2 dtypes x 5 `(world_size, nodes)` pairs = **900 rows per
backend, 1800 total**, from **5 process-group setups** (one per topology, all five ops
timed inside each). For comparison, the GEMM table is ~64 holdout points and attention
~72, so the collective family becomes the largest table in the tree. That is expected:
it is the only family with two slice dimensions beyond dtype.

`vllm` and `nccl` rows come from one sweep, as today. `p2p` is measured only for the pairs
a mesh can form; it has no meaning at `world_size=2, nodes=3`.

## 6. Files

### New
- `srl/cost/kernel/ops/collective.py` — rewritten: `_CollectiveOp` base plus five
  registered subclasses
- `srl/profiler/kernels/collective_launch.py` — torchrun child
- `examples/profile/kernel/collective_multinode.yaml`, `profile_collective.sh`
- `docs/superpowers/specs/2026-10-04-collective-ops-design.md` — this spec, committed

### Modified
- `srl/profiler/kernels/collective.py` — driver: local torchrun, ssh multi-node
- `srl/profiler/kernels/config.py` — `CollectiveSweep` gains `ops`, `world_sizes`,
  `nodes`, `hostnames`, `dtypes`
- `srl/cost/kernel/hardware.py` + `data/gpu_specs.yaml` — `inter_node_bandwidth_bytes_per_s`
- `srl/cost/kernel/database.py` — `CommQuery` gains `nodes` and an `op` field;
  `op_name` currently returns the literal `"all_reduce"` (`database.py:156`) and becomes
  the query's own op. `query_comm` and `_price` are otherwise untouched: `_price` is
  already op-agnostic
- `srl/cost/kernel/model_ops.py` — `comm_queries` passes `op="all_reduce"`, `nodes=1`
  explicitly, which keeps the rollout path's behaviour identical while making both
  assumptions visible rather than defaulted
- `docs/profiling/tensor-parallel.md`, `docs/reference/limitations.md`,
  `.claude/codebase-map.md`

### Untouched
`gemm.py`, `attention.py`, `elementwise.py` ops; `sol.py`'s other formulas; `surface.py`;
`interp.py`; `composed.py`'s GEMM and attention paths; `floor.py` and `holdout.py` must
need **no edit** to pick up four new ops. That they do not is the structural test of
whether the op abstraction is real, exactly as the elementwise/all_reduce addition was.

## 7. Phasing

| Phase | Scope | Gate |
|---|---|---|
| A0 | Capture §4 baselines into a committed gate | **Done 2026-10-04**: `tests/test_collective_baseline_regression.py`, 12 tests, deterministic across repeat runs; surfaced the ws=8 report-ordering defect above |
| A1 | `_CollectiveOp` base; `nodes` slice; `inter_node` SOL; re-sweep intra-node | **Done 2026-10-04**, gate passed in full: see below |
| A2 | ssh multi-node driver; cross-node sweep at `nodes` 2 and 3 | **Done 2026-10-04**: 24 GPUs over 3 nodes, 5 tables, MRE reported per topology; see below |
| A3 | `reduce_scatter`, `all_gather`, `broadcast`, `p2p` ops and measurement | **Done 2026-10-04** alongside A1: all five ops measured, `floor.py` and `holdout.py` unedited |
| A4 | Docs | **Done 2026-10-04**: every figure re-derived from the committed holdout reports |

A1 had to complete before A2: a cross-node measurement mixed into the schema change would
make the neutrality comparison meaningless. A3 turned out to be free to land with A1 rather
than after it, because `_CollectiveOp.measure` times every registered collective inside one
process group, so adding the four ops added no separate measurement pass.

### A1 + A3 results

The compute families were **bit-identical**, which is what makes the collective movement
attributable rather than ambiguous:

| family | baseline | after | required |
|---|---|---|---|
| `gemm/vllm` | 3.9019% | 3.9019% | exact |
| `context_attention/vllm` | 6.6801% | 6.6801% | exact |
| `generation_attention/vllm` | 4.5715% | 4.5715% | exact |
| `context_attention/flashinfer` | 9.9274% | 9.9274% | exact |
| `generation_attention/flashinfer` | 6.3715% | 6.3715% | exact |
| `elementwise/vllm` | 9.3544% | 9.3544% | exact |
| `all_reduce/vllm` | 6.0122% | 6.3064% (+0.294pp) | +-0.5pp |
| `all_reduce/nccl` | 4.9606% | 4.6572% (-0.303pp) | +-0.5pp |

End-to-end at tp=2 against the spliced serial baseline, the load-bearing check:

| phase | baseline | after | delta |
|---|---|---|---|
| prefill, 29 buckets | 3.597% | 3.673% | +0.076pp |
| decode, 204 buckets | 9.510% | 9.792% | +0.282pp |
| decode, `reqs>=5`, 137 buckets | 2.957% | 3.071% | +0.114pp |
| mixed, 7 buckets | 3.643% | 3.742% | +0.099pp |

The attention and elementwise **fitted floors did not move at all** (vllm 45/22 us,
flashinfer 16/6 us, elementwise 2 us), which is a second independent neutrality signal:
those tables were not re-measured, so their floors could not change.

Nine tables written, 108 rows each (18 message sizes x 2 dtypes x 3 topologies). `p2p` has
no `vllm` table, because vLLM's pipeline send/recv uses a separate group with its own
buffers and a `vllm`-labelled p2p row would misattribute the kernel:

| op | `vllm` MRE | `nccl` MRE | fitted floor (vllm / nccl) |
|---|---|---|---|
| `all_reduce` | 6.31% | 4.66% | 6 us / 2 us |
| `reduce_scatter` | 4.71% | 10.17% | 12 us / 2 us |
| `all_gather` | 3.16% | 7.11% | 6 us / 4 us |
| `broadcast` | 4.73% | 6.52% | 6 us / 6 us |
| `p2p` | n/a | 4.97% | n/a / 12 us |

Two structural gates also passed, and they are the real test of whether the op abstraction
holds: **`floor.py` and `holdout.py` were not edited**, yet both picked up four new ops
automatically (they iterate `ops_in(version_dir)`), and **`cli/profile.py` was not edited**,
because `--family` choices derive from the registry and `measure` is dispatched per family.

Three findings worth recording:

- **`reduce_scatter/nccl` at 10.17% is the worst table in the family**, with a utility floor
  of 0.006 and a 186% max error. The dispatched path scores 4.71% on the same shapes, so
  this is the control backend describing a schedule the server does not run, the same effect
  already documented for `all_reduce` at larger world sizes.
- **The dispatched and control backends disagree about which op is hard.** `nccl` beats
  `vllm` on `all_reduce` (4.66% against 6.31%) but loses on every other op, by 2-6pp. A
  single-op family could not have surfaced that.
- **`reduce_scatter`'s fitted floor differs 6x between backends** (12 us dispatched against
  2 us control). Fitting each op independently rather than sharing one collective constant
  was therefore load-bearing, not tidiness.

### A2 results: cross-node measured on 24 GPUs over 3 nodes

Five `nccl` tables at `(16, 2)` and `(24, 3)`, 64 held-out points each:

| op | MRE | fitted floor |
|---|---|---|
| `p2p` | 3.44% | 16 us |
| `all_gather` | 9.86% | 2 us |
| `reduce_scatter` | 11.77% | 6 us |
| `all_reduce` | 14.15% | 16 us |
| `broadcast` | 17.31% | 2 us |

**Cross-node rows carry no `vllm` column, and that is vLLM's behaviour rather than a
shortcut.** `custom_all_reduce.py:90` disables the custom kernel outright for a group
spanning nodes ("Custom allreduce is disabled because this process group spans across
nodes") and symmetric memory refuses the rendezvous for the same reason, so the dispatcher
has nothing left to select and falls through to pynccl, which the control table already
measures. The first attempt stood vLLM's parallel state up anyway and paid ~60 s per
topology to produce a duplicate of the control table. The launcher now skips it when
`nodes > 1` and leaves the column absent rather than filling it with a relabelled NCCL
measurement, which would misattribute the kernel in exactly the way the two-table
arrangement exists to prevent.

#### Verification 6 passes, and the spec's expected band was wrong

The falsification check required cross-node to measure materially slower than intra-node at
equal message size. It does, at every one of the 18 sizes:

| regime | measured ratio vs `(8, 1)` |
|---|---|
| 4-256 KiB (floor-dominated) | 2.1x - 3.8x |
| >= 64 MiB (bandwidth-dominated) | 1.34x - 1.84x |

The spec predicted "roughly 9x" at large messages. **That figure was an error in this
document, not a failure of the measurement.** 9x is the NVLink-to-NIC bandwidth ratio,
which would apply only if every hop crossed a boundary. A ring over 16 ranks on 2 nodes
crosses on `nodes / world_size` = 12.5% of its hops, and §3's own two-level SOL accordingly
predicts **1.21x**. Measured 1.34-1.84x is that roofline plus the usual 1.1-1.5x
inefficiency, i.e. exactly what the model says. The corrected expectation is "the ratio the
two-level SOL predicts for that topology", which is self-consistent rather than a constant.

The floor-dominated regime is the cleaner evidence anyway: a cross-node all-reduce sits on a
~50 us handshake against ~20 us intra-node, which no amount of loopback traffic could
produce.

#### The cross-node tables are 2-3x less accurate, and it is not contention

Intra-node `all_reduce/nccl` scores 4.66% where cross-node scores 14.15%. The obvious
suspect is a contended or degraded run, and the repo's own diagnostic for that is
monotonicity rather than outlier filtering (`docs/profiling/tensor-parallel.md`). Counting
non-monotonic steps in the message-size sweep:

| topology | non-monotonic steps |
|---|---|
| `(2, 1)` | 4 / 17 |
| `(4, 1)` | 3 / 17 |
| `(8, 1)` | 3 / 17 |
| `(16, 2)` | 3 / 17 |
| `(24, 3)` | 2 / 17 |

Cross-node is **no less monotonic than intra-node**, so contention is ruled out and the
higher MRE is a real property of the surface: utility ranges as low as 0.004, meaning the
flat-ring SOL is far from the measured schedule, which is the hierarchical-collective
simplification §3 recorded in advance. `nodes` being a slice column is what keeps that
confined to the cross-node rows instead of leaking into the validated intra-node ones.

Densifying the grid is the next lever if a consumer needs better than ~14%. Spec B should
treat cross-node DP reduction as priced to roughly +-15% rather than the +-5% the
intra-node tables support, and say so when it reports.

## 8. Verification

1. **Unit, no GPU.** `traffic_factor` per op against hand-computed ring bounds;
   `sol()` at `nodes=1` reproducing the current formula exactly; `nodes>1` bounded by the
   inter-node term; an old-format table (no `nodes` column) rejected with a message naming
   the sweep command; `PerfTableWriter.write` rejecting a missing or extra column per new op.
2. **Import hygiene.** `tests/test_kernel_package_imports.py` and
   `tests/test_kernel_shapes_import.py` must pass **unmodified** — the five new op classes
   must not drag `torch` into the cost path.
3. **Op coverage.** `tests/test_kernel_op_coverage.py` must see all five registered, and
   every parquet stem in the table tree must map to an op.
4. **Holdout, per op and per topology.** `srl-profile kernels holdout` reporting MRE
   per `(op, backend, world_size, nodes)` with an explicit `point_count > 0` check, since a
   zero-point result is a coverage failure that currently only warns.
5. **Neutrality, end to end.** Re-run `kernels validate` against
   `outputs/profile/qwen3_8b/vllm_tp2_serial/profiles/vllm.jsonl` and compare prefill /
   decode / mixed MRE against §4 **separately**, never pooled.
6. **Cross-node sanity, the number that matters for A2.** A `nodes=2, world_size=16`
   all-reduce must measure materially slower than a `nodes=1, world_size=8` one at the same
   message size. If it does not, the launcher is not actually crossing the NIC and every
   cross-node row is void. This is a falsification check, not a formality: it is the only
   evidence that the multi-node rendezvous did what it claims.

   It has a **quantitative** form, and getting that form right matters: an earlier draft of
   this section said "roughly 9x", which is the raw NVLink-to-NIC bandwidth ratio and would
   apply only if every hop crossed a boundary. A ring crosses on `nodes / world_size` of its
   hops, so the right expectation is simply **what §3's two-level SOL predicts for that
   topology** (1.21x for `(16, 2)` against `(8, 1)` at 64 MiB), times the usual 1.1-1.5x
   measurement inefficiency. Measured: 1.34-1.84x. A ratio near 1.0 means the traffic never
   left the node; a ratio in the hundreds means a TCP fallback (check `NCCL_IB_DISABLE`).

   The floor-dominated regime is the stronger signal and needs no model at all: a cross-node
   collective sits on a ~50 us handshake against ~20 us intra-node, a 2.1-3.8x ratio across
   4-256 KiB that loopback traffic cannot produce. Record both regimes. This is a sanity
   band, not a gate: the fitted utility is what carries accuracy.
7. **Dispatch probe.** Re-run `examples/profile/kernel/dispatch_probe.py` at each world
   size, including cross-node, and record which backend fires per message size. The existing
   `DISPATCH.md` table is intra-node only.

All GPU work requires `source /apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh`.
Multi-node runs use `28.49.55.85`, `29.162.247.148`, `28.49.196.175`.

### Test files

Following the repo's flat `tests/` layout and full-sentence test names:

| file | covers |
|---|---|
| `tests/test_kernel_ops_collective.py` | per-op `traffic_factor`; `sol()` reducing to the current formula at `nodes=1`; `p2p` being `N`-independent; invalid `nodes`/`world_size` refused |
| `tests/test_kernel_collective_topology.py` | `nodes` as a slice: an unmeasured topology raises `KernelCoverageError` rather than interpolating; an old-format table without `nodes` is rejected with a message naming the sweep command |
| `tests/test_kernel_collective_sweep.py` | `CollectiveSweep` parsing `(world_size, nodes)` pairs; refusing a pair whose `nodes` exceeds the hostname count; refusing a cross product |
| `tests/test_kernel_collective_launch.py` | spec JSON round-trip and command construction **without** a process group, mirroring `tests/test_fsdp_launch.py` |
| `tests/test_kernel_measurements.py` (extend) | `PerfTableWriter.write` rejecting a missing or extra column for each of the five ops |
| `tests/test_kernel_op_coverage.py` (existing, must pass) | all five ops registered; every parquet stem maps to an op |
| `tests/test_kernel_package_imports.py`, `tests/test_kernel_shapes_import.py` (existing, **unmodified**) | the five op classes do not drag `torch` into the cost path |
| `tests/test_kernel_floor.py` (extend) | floors fitted for a synthetic collective op registered only inside the test, with **no edit** to `floor.py` |

GPU-requiring tests use the established idiom
`pytestmark = pytest.mark.skipif(not _CUDA, reason="CUDA device is not available.")`.
Multi-node tests are skipped unless the three hosts answer, and that skip is reported
rather than passing vacuously, per `docs/development/testing.md`.

**No existing assertion may be weakened or deleted.** A test whose expected value changes
means A1 was not behaviour-neutral, and the fix belongs in the code.

## 9. The contract Spec B relies on

Stated here so A's interfaces are judged against their consumer, not designed in a
vacuum. **None of this is built in Spec A.**

Spec B will add `TrainingOps` (sibling to `ModelOps`) and `ComposedTraining`
(a `TrainingLatencyModel`, paired with the **existing** analytic memory model, since
memory is computed not composed). It will query:

| Training event | Op | Notes |
|---|---|---|
| Forward projections, dgrad, wgrad | `gemm` | Reuses the existing table; `gemm_all_shapes` already emits wgrad triples |
| Attention forward | `context_attention` | existing |
| Attention backward | `attention_backward` | **new op, Spec B** |
| Norm / rope / activation, fwd and bwd | `elementwise` | Existing table, and **summed** for training: there are no CUDA graphs, so the cost is genuinely exposed. The rollout model's deliberate omission does not transfer |
| TP collectives, SP on | `reduce_scatter` + `all_gather` | **Spec A.** Message size `tokens*hidden/tp` |
| TP collectives, SP off | `all_reduce` | Spec A, message size `tokens*hidden` |
| DP gradient reduction | `reduce_scatter` (distributed optimizer) or `all_reduce` | **Spec A, `float32`, and `nodes>1` when DP spans nodes** |
| FSDP2 per-layer param / grad | `all_gather` / `reduce_scatter` | Spec A |
| PP stage boundary | `p2p` | Spec A |
| Optimizer step | `optimizer_step` | **new op, Spec B** |
| Recompute | re-evaluated forward op list | composition, not a table |
| PP bubble | analytical, via the existing slot structure | not measured |

So Spec A must deliver: four new collective ops, `float32` coverage, and a `nodes` slice
reaching 2 and 3. Everything else B needs is either already measured or is B's own work.

Two B-side hazards worth recording now, because they bound what A can be blamed for:

- **Training's logits GEMM is not rollout's.** Rollout computes logits for one token per
  request (`m = request_count`); training computes them for every position
  (`m = tokens`), and `activation.py` documents that the logit path is **not** TP-divided.
  B's sweep must cover that much larger `m`, and A's sweep need not.
- **The psrl fixed-cost gap is not a kernel-sum error.** A recorded finding
  (`verl-charges-optimizer-offload-to-update-actor`) is that verl bills ~34s of optimizer
  offload to `update_actor`, which the profiler structurally cannot see. B must validate
  against the **profiler's own** step times, where that cost is absent, and treat the psrl
  comparison as a separate question. Conflating the two would make a composition error and
  an accounting difference indistinguishable.

## 10. Deliberately out of scope

- **Compute/communication overlap.** Still a plain sum. The existing tp=2 evidence is that
  overlap is *not* dominant there (every phase's signed error is negative, i.e. slight
  under-prediction, when unmodelled overlap would cause over-prediction). But Megatron's
  `overlap_grad_reduce` / `overlap_param_gather` make it a real question for B, and this
  spec only gives B the isolated numbers it needs to quantify it.
- **NUMA-aware intra-node topology.** GPUs 0-3 sit on NUMA 0 and 4-7 on NUMA 1, so a
  4-rank group's placement matters and `nodes` does not capture it. Mitigated rather than
  solved: all-to-all NV18 NVLink means the GPU-to-GPU path is uniform regardless of NUMA,
  so the exposure is host-side staging, not the collective itself. `world_size` 4 is
  measured at whatever placement torchrun gives, and that placement is logged so a later
  NUMA-sensitivity question is answerable from the existing rows.
- **NIC contention above 8 ranks per node.** The 1:1 GPU-to-bond affinity holds at
  `world_size <= 8` per node, which is every pair swept here. A configuration putting more
  ranks than bonds on one node would contend, and the single-constant
  `inter_node_bandwidth` would overstate throughput. Not swept, so not modelled.
- **Cross-hardware transfer.** Only H20 is available; D2 makes it possible but it stays
  unproven and must not be claimed.
- **MoE collectives** (all-to-all expert dispatch). Qwen3-8B is dense.
- **Any training op or `ComposedTraining`.** Spec B.

## 11. Risks

| risk | mitigation |
|---|---|
| Multi-node launcher silently falls back to loopback, voiding every cross-node row | Verification 6 is a falsification check: cross-node must measure materially slower than intra-node at equal message size |
| Schema change perturbs the validated rollout numbers | A0 captures baselines before any edit; A1's gate is the end-to-end tp=2 re-score, with tolerance fixed in advance |
| Old `all_reduce` table read with assumed `nodes=1`, attributing NIC measurements to NVLink | Explicit rejection naming the sweep command; no lenient path |
| Five `sol()` copies drift | One `sol()` on the shared base; subclasses declare only a traffic factor |
| `floor.py` / `holdout.py` need edits per new op, meaning the abstraction leaked | Made an explicit A3 gate: those files must stay unedited |
| Hierarchical NCCL schedule mismatches the flat-ring SOL | `nodes` is a slice, so a wrong SOL is absorbed into that topology's own utility rather than leaking across topologies |
| ssh/CephFS flakiness across 3 nodes misattributed to the model | Driver fails loudly on any non-zero node exit; the serial-baseline lesson from `tensor-parallel.md` applies (monotonicity, not outlier filtering, is the diagnostic) |
