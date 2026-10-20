# Staleness Policy 概览

本目录下的每个 `StalenessPolicy` 子类对应一个真实异步 RL 框架的 staleness 控制机制（详见 [`docs/design/blog.md`](../../docs/design/blog.md) 对应的源码调研）。所有策略共用同一组 hook，大致对应博客里的三层：

| hook | 对应层级 | 作用 |
|---|---|---|
| `admit_quota` | L1 准入控制 | 本时刻还能准入多少 response |
| `on_publish` | L2 Rollout 运行时控制 | 训练版本发布时，取消/替换哪些已准入的 group |
| `peek_batch` / `take_batch` | L3 Training 前过滤 | 组装训练 batch 时选取或丢弃哪些已完成 group |

`POLICY_REGISTRY`（见 [`registry.py`](registry.py)）把 preset 名字映射到这些类：`psrl`、`areal`、`roll`、`slime`、`verl`、`verl_v1`、`skyrl`、`openrlhf`、`dressage`、`nemo_rl`、`trl`、`prime_rl`。每个 preset 在 `srl/config/policies/` 下有对应的 Hydra/dataclass 配置，字段名与真实框架保持一致（没有统一的 `max_staleness`）；下面列的是各 **Policy 类自身构造函数**的参数。

部分策略（`psrl`、`areal`、`roll`）不在构造函数里接收阈值，而是读取 `SimState.max_staleness`——这个字段由 `SimConfig.staleness_horizon` 在每个训练步注入，对应各自 preset 配置里的 `staleness`、`max_head_offpolicyness`、`async_generation_ratio`。下表里这种情况会特别标出。

## 参数一览

| policy | 可调参数（默认值） | 与 staleness 大小最直接相关 |
|---|---|---|
| `psrl` | `proactive_filter=False`、`proactive_threshold=0` | **`max_staleness`**（来自 `SimConfig.staleness_horizon`，对应 preset 的 `staleness`，不是构造函数参数）。`proactive_*` 只是主动淘汰 RESERVED 条目的辅助开关 |
| `areal` | `max_inflight=None` | **`max_staleness`**（来自 `SimConfig.staleness_horizon`，对应 preset 的 `max_head_offpolicyness`）。`max_inflight` 只是额外的并发上限 |
| `roll` | `max_inflight=None` | **`max_staleness`**（来自 `SimConfig.staleness_horizon`，对应 preset 的 `async_generation_ratio`，取整后用作窗口宽度） |
| `slime` | `update_weights_interval=1`、`over_sampling_ratio=1.0`、`pause_mode`、`max_inflight=None` | **`update_weights_interval`**——没有真正的 staleness 阈值，最大 staleness 恒等于这个值 |
| `verl` | `staleness_threshold=0.1`、`require_batches=1`、`max_inflight=None` | **`staleness_threshold`**，但它是样本超发比例，不是版本数，不能换算成逐 token 的 staleness 上界 |
| `verl_v1` | `num_warmup_batches=1.0`、`max_off_policy_threshold=8`、`max_off_policy_strategy="drop"`、`pause_mode` | **`max_off_policy_threshold`**，且只在 `max_off_policy_strategy="drop"` 时才是硬上界；`"wait"` 不淘汰已完成的超限 group |
| `skyrl` | `max_staleness_steps=4`、`pause_mode`、`max_inflight=None` | `max_staleness_steps` 只驱动准入额度和违规计数，从不丢弃，因此不提供硬上界 |
| `openrlhf` | `async_queue_size=1`、`partial_rollout_enable=False`、`max_inflight=None` | `async_queue_size` 限制的是在途 batch-lump 数，不是逐样本年龄，没有真正的 staleness 阈值 |
| `dressage` | `keep_versions=0`、`max_active_groups=None`、`output_queue_size=1000`、`max_partial_rollout_preempts=None`、`pause_mode`、`max_inflight=None` | **`keep_versions`**（按每个 group 最后一个可训练 token 的版本过滤）；`max_partial_rollout_preempts` 管的是单条 trajectory 内部的版本切换次数，是另一个轴 |
| `nemo_rl` | `sampler="in_order"`、`max_staleness_versions=1`、`max_lookahead_versions=1`、`sample_freshest_first=False`、`max_inflight_prompts=32`、`max_buffered_rollouts=64`、`pause_mode` | 取决于 `sampler`：`windowed`/`weight_fifo` 看 **`max_staleness_versions`**，`in_order` 看 **`max_lookahead_versions`**，`ready_first` 完全不限制 staleness |
| `trl` | `max_staleness=4`、`max_inflight_tasks=-1`（`-1` 表示自动取 `max(max_staleness,1)*batch_size`）、`pause_mode` | **`max_staleness`**，直接比较生成版本与训练版本，是本文定义下最直白的硬上界 |
| `prime_rl` | `max_off_policy_steps=8`、`max_inflight=1024`、`target_lag=1`、`action="drop"` | **`max_off_policy_steps`**，L2、L3 共用同一个下界；`target_lag` 只影响准入提前量，不影响上界本身 |

## 各 Policy 的 L1 / L2 / L3

### `psrl`（[`psrl.py`](psrl.py)）
- **L1**：Reserve——按 `max_staleness` 窗口悲观占位到最高可用 buffer，从后往前填。
- **L2**：无显式取消；partial rollout 只重置引擎状态，保留原有占位。
- **L3**：Occupy/Consume——完成时贪心挪到最早 buffer，取 batch 前等待当前训练版本对应的 buffer 变为 READY。

### `areal`（[`areal.py`](areal.py)）
- **L1**：累积 token bucket（`ProgressQuota`），按 `accepted + running` 与 `max_staleness` 换算的额度门控。
- **L2**：无。
- **L3**：无——`peek_batch` 就是纯 FIFO，不做任何版本过滤。

### `roll`（[`roll.py`](roll.py)）
- **L1**：token bucket，首轮额度 `ratio*batch + batch`，之后每轮训练再加一个 `batch`。
- **L2**：`on_publish` 做 GC——把 `train_version - max_staleness` 之前的 cohort（在途和已完成的）整个取消并替换。
- **L3**：`peek_batch`/`take_batch` 只从未被 GC 的 cohort 里取，相当于一个消费下限。

### `slime`（[`slime.py`](slime.py)）
- **L1**：`WatermarkGate` 按过采样比例限制在途 group 数。
- **L2**：无版本相关取消；`pause_mode` 只决定权重更新时在途请求是否保留 KV。
- **L3**：无过滤，整批 FIFO 取出。最大 staleness 完全由 `update_weights_interval` 这个循环结构决定，不依赖任何版本比较。

### `verl`（legacy，[`verl.py`](verl.py)）
- **L1**：`LegacyOverage` 维护一个随权重同步重置的窗口（占用量租约）。
- **L2**：无。
- **L3**：有界交接队列溢出时丢最旧的（`drop_reason_for_complete` 标记为 `queue_overflow`）。

### `verl_v1`（[`verl_v1.py`](verl_v1.py)）
- **L1**：按训练进度补给（`num_warmup_batches` 预取一次，之后每轮一个 `batch`），不按 staleness 门控。
- **L2**：无。
- **L3**：`OffPolicySpan`——`drop` 在取样前淘汰超限的已完成 group（真正的硬上界）；`wait` 只阻塞在途 group，不淘汰已完成的。

### `skyrl`（[`skyrl.py`](skyrl.py)）
- **L1**：与 `areal` 相同的 `ProgressQuota` 公式。
- **L2**：无。
- **L3**：`WarnOnly`——取 batch 时统计超限 group 数，但从不丢弃。

### `openrlhf`（[`openrlhf.py`](openrlhf.py)）
- **L1**：许可池——每个空闲许可对应一整个 `batch_size` 的 lump，不能部分准入。
- **L2**：无；`partial_rollout_enable` 只改变许可何时释放。
- **L3**：无。

### `dressage`（[`dressage.py`](dressage.py)）
- **L1**：同时检查活跃 group 数（`max_active_groups`）和完成队列水位（`output_queue_size` 的 80%）。
- **L2**：`VersionSpanCap`——partial rollout 跨越的不同版本数超过 `max_partial_rollout_preempts` 就整条重试。
- **L3**：`DistinctVersionWindow`——按每个 group **最后一个可训练 token** 的版本判断是否落在 `keep_versions` 窗口内，丢弃时发一个补采样额度。

### `nemo_rl`（[`nemo_rl.py`](nemo_rl.py)）
四种 `sampler` 共用一个准入闸门（`max_inflight_prompts` + `max_buffered_rollouts` + dispatch-lag 窗口），差异都在 L2/L3：
- `windowed`：L2 在 `on_publish` 按 `max_staleness_versions` 取消运行中和已就绪但落后的 group；L3 再按同一窗口过滤。
- `weight_fifo`：L3 做同样的窗口过滤，但没有 L2 取消——落在窗口外的 group 会一直不被消费。
- `ready_first`：只看谁先完成，没有任何版本比较。
- `in_order`：L1 就给每个准入 cohort 绑定一个 `target_step`，L3 只取目标版本等于当前训练版本的 group，不需要比较生成版本本身。

### `trl`（[`trl.py`](trl.py)）
- **L1**：单一在途队列，上限 `max_inflight_tasks`。
- **L2**：无。
- **L3**：`VersionWindow`——每次发布都扫一遍就绪池，丢弃超过 `max_staleness` 的 group 并补采样。模拟器按 group 整体丢弃；真实 TRL 是逐样本丢弃，相关 TODO 见文件内注释。

### `prime_rl`（[`prime_rl.py`](prime_rl.py)）
- **L1**：固定并发上限 `max_inflight`，外加按 `target_lag` 限制的准入提前量，二者都与 staleness 窗口本身无关。
- **L2**：`CancelBelowVersion`——权重发布时取消生成版本早于 `max_off_policy_steps` 下界的在途 group。
- **L3**：`VersionWindow`——同一下界再扫一遍就绪池，两处结果按 group id 去重后合并。
