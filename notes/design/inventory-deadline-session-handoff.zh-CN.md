# inventory_deadline 当前状态与 E_k 建模说明

> 更新时间：2026-10-17
>
> 这是一份中文交接文档，供下一 session 继续使用。它总结当前实现、实验结论、加速比长期上不去的原因，以及 `E_k` 的具体建模过程。

## 1. 当前目标

`inventory_deadline` 的目标，是替代 psrl rollout 中固定的 per-instance concurrency cap。

静态 cap 的优点是简单：把并发限制在 decode throughput 的拐点附近。但它依赖 workload：请求长度分布、训练周期、staleness、GPU 数量变化后，最佳 cap 也会变化。

`inventory_deadline` 希望每次 admission 或 routing 决策都重新预测：

> 如果现在把这个 request/group 放到某个 instance，最近几个 training buffer 什么时候 ready、什么时候完成训练？

然后选择预测训练链最早的方案，让 cap 成为模型决策的副产物，而不是显式配置。

## 2. 当前代码与实验状态

### 2.1 主要代码位置

| 功能 | 文件 |
|---|---|
| 单 instance drain forecast | `dryrun/forecast/drain.py` |
| Fleet forecast 与 buffer 链 | `dryrun/forecast/fleet.py` |
| `E_k` pipeline 计算 | `dryrun/forecast/pipeline.py` |
| 一次 dispatch 内的增量 admission/routing 定价 | `dryrun/forecast/plan.py` |
| router 策略 | `dryrun/component/rollout/router/strategies/inventory_deadline.py` |
| psrl inventory 重放 | `psrl/workers/ps/inventory_forecast.py` |
| 主实验 | `examples/analysis/headroom_probe.py` |
| knob sweep | `examples/analysis/inventory_deadline_knobs.py` |
| decision-level instrumentation | `examples/analysis/inventory_deadline_decisions.py` |

### 2.2 已实现的实验功能

- `kv_watermark`：按预测 KV peak 限制新 group admission。
- `lookahead_defer`：用 `defer_until` 时刻预计已经减负的 fleet 重新估价 DEFER。
- `horizon_buffers`：预测多个 pending buffers。
- `latest_deadline` eviction：KV 压力下优先驱逐 deadline 更晚的 request。
- PPO/grid：`group_size=1, batch=256` 的验证网格。

完整测试基线：`pytest tests/` 为 1920 passed、3 skipped，新增 eviction 测试为 9 passed。

## 3. 当前实验结论

### 3.1 当前最好的 best-grid 配置

在 best 网格的 6 个点上，`wm06_h4`（`kv_watermark=0.6, horizon_buffers=4`）是当前最好配置：

| workload | shipped vs cap48 | wm06_h4 vs cap48 |
|---|---:|---:|
| `cv0.5_tau600` | 1.094x | **1.145x** |
| `cv1.0_tau600` | 1.029x | **1.066x** |
| `cv1.0_tau345` | 0.944x | **0.982x** |
| `cv0.5_tau345` | 0.947x | **1.015x** |
| `cv0.5_tau150` | 0.924x | **0.964x** |
| `cv0.5_tau600_s2` | 0.937x | **0.972x** |

`wm06_h4` 在 6/6 点相对 shipped 改善，但相对静态 cap48 仍有 3 个点没有超过 1.0x。

### 3.2 已否定的方向

| 方向 | 结论 |
|---|---|
| `horizon_buffers=1` 或 `4` 单独使用 | 与 shipped 基本完全相同，单纯扩大 E_k 向量没有帮助 |
| `lookahead_defer + watermark` | best 网格约 +3.8%，但 held-out 的 `mean=12000` 点下降约 7～9pp，泛化不稳定 |
| PPO：`group_size=1, batch=256` | 大多数点与静态 cap 持平或更差，不能解决根本问题 |
| `eps_frac` 调整 | 没有稳定收益 |
| `latest_deadline` eviction | 与对应 watermark 配置相同，eviction 不是当前主要瓶颈 |
| 将 `peak_blocks` 换成 `used_blocks + new_group_peak` | 某些 mean-shift 点改善，但整体退化，已回滚 |

## 4. E_k 到底如何建模

E_k 不是直接由一个神经网络预测出来的，也不是直接把某个 request 的 ETA 加上训练时间。当前实现是一个四层的确定性 replay/forecast：

```text
request-level finish time
        ↓
group ETA
        ↓
psrl inventory replay → buffer READY time F_k
        ↓
training-chain replay → training finish time E_k
```

下面按实际代码路径说明。

### 4.1 第一步：预测每个 request 什么时候生成完成

入口主要是：

- `dryrun/forecast/plan.py`
- `dryrun/forecast/fleet.py`
- `dryrun/forecast/drain.py`

对每个 instance，系统维护一个 `InstanceForecast`，里面有：

- 当前 running requests 的状态。
- 当前 waiting requests 的状态。
- 假设新 admission 会到达的 `Arrival`。
- 可能发生的训练 publish interrupt。

`InstanceForecast.finish_times()` 调用 `drain._replay()`，用 rollout engine 的 cost model 重放未来执行过程。它主要模拟两类阶段：

1. **decode phase**：根据当前 running batch 的 request 数量 `n_d`、上下文长度总和 `ctxsum`、剩余 token 数，调用 `decode_segment_time(...)` 估计一段 decode 的耗时。
2. **prefill phase**：在 token budget、KV block、chunk cap 等约束下，把 waiting request 分块 prefill，再用 `step_time(...)` 定价。

简化表示：

```text
T_next = cost_model.decode_segment_time(n_d, ctxsum, remaining)
```

如果有 waiting request，则会进一步考虑：

```text
step time = cost_model.step_time(
    decode_batch_size,
    prefill_cost,
    decode_context_sum,
    total_batch_size,
    prefill_context,
)
```

因此，新 request 并不是只增加自己的生成时间。它加入后会改变：

- 后续 decode 的 `n_d`。
- 每个 decode step 的 `ctxsum`。
- KV block 占用。
- waiting request 何时能进入 running batch。
- 所有已有 request 的完成时间。

重放最终得到：

```python
DrainResult.finish_times: dict[request_id, absolute_finish_time]
```

如果预测过程中 KV peak 超过池容量，`DrainResult.kv_overflow=True`，完成时间被视为不可用。这是 admission 的硬可行性检查。预测本身不把 preemption 当成可优化动作，而是把 KV peak 当作保守上界。

### 4.2 第二步：从 request finish time 聚合成 group ETA

训练以 group 为单位消费 response。一个 group 只有所有成员都生成完成，才算可以进入训练 buffer。

`dryrun/forecast/pipeline.py:group_etas()` 的逻辑是：

```python
ETA(group) = max(finish_time(member) for member in group)
```

如果某个 member 已经在真实 runtime 的 pending completion 中，则使用已知完成时间，如果既没有 forecast finish time，也没有已知完成时间，则该 group ETA 是 `inf`。

因此，group ETA 天然由该 group 最慢的成员决定：

```text
group_eta[g] = max(member_eta[g, 0], ..., member_eta[g, group_size-1])
```

当前 admission 评估时，假设新 group 被放置后，`_price_admission()` 会先预测新 group 的全部成员在 fleet 中的结束时间，再把新 group 作为 `new_groups=[(version, group_eta)]` 交给后续 inventory replay。

### 4.3 第三步：由 group ETA 重放 psrl inventory，得到 F_k

入口是 psrl 的：

```python
psrl.workers.ps.inventory_forecast.forecast_ready_times
```

输入包括：

- 当前 `InventorySnapshot`。
- 每个已 reserved group 的 ETA。
- 假设新 admission 的 `(version, eta)` 序列。

这不是简单排序。它重放真实 staleness inventory 的 reserve/occupy 规则：

1. 当前 buffer 有 occupied entries、reserved entries 和 empty slots。
2. group 在预测完成时刻到达。
3. 到达后按照 psrl 的版本/staleness 规则尝试填充最早可消费的 buffer。
4. 当某个 buffer 的所有 required entries 都被 occupy 时，该 buffer 的 ready time 就确定。

输出是：

```python
ready[buffer_id] = F_k
```

其中 `F_k` 表示 buffer k 预计什么时候 READY，可以交给 trainer。

重要的是：如果 group 的 ETA 晚，`F_k` 也会晚，如果 group 由于 staleness 约束只能进入更晚的 buffer，后续 buffer 的 ready 时间和版本关系也会随之改变。

### 4.4 第四步：由 F_k 链出 E_k

`PipelineForecast.evaluate()` 对 ready times 做训练链 replay。

设：

- `F_k`：buffer k 的 READY 时间。
- `tau_k`：buffer k 的训练周期，包含 train + sync + recompute。
- `L`：publish 后的 weight load 时间。
- `R`：额外 reconfig 时间。
- `E_k`：buffer k 的训练链结束时间。

代码的递推是：

```text
初始 e_prev = now                         （trainer idle）
或   e_prev = current_training_end + L    （当前训练还在进行）

publish_k = max(F_k, e_prev) + tau_k
E_k       = publish_k + L + R
下一步    e_prev = E_k
```

如果忽略 publish 后的 load/reconfig，只看核心结构，就是：

```text
E_k = max(F_k, E_{k-1}) + tau_k
```

但实际代码把 weight load 和 reconfig 也放进了 `E_k`，这样下一 buffer 的 start time 会等待上一个 publish 后的恢复时间。

当前实现的位置是：

```text
`dryrun/forecast/pipeline.py:87-127`
```

`Pipeline` 最终保存四个向量：

```python
Pipeline(
    buffers=(...),
    ready=(F_k0, F_k1, ...),
    train_end=(E_k0, E_k1, ...),
    publish_times=(publish_k0, publish_k1, ...),
)
```

### 4.5 第五步：用 E_k 向量比较 admission/routing 选项

每个候选项都会生成一个完整 `Pipeline`：

- 放到 instance 0。
- 放到 instance 1。
- DEFER。
- 对已 admitted member 换一个 instance。

默认 `horizon_buffers=2`，所以主要比较：

```text
(E_k0, E_k1)
```

比较是带容忍度的字典序：

```text
先比较 E_k0；
如果差值不超过 eps，再比较 E_k1；
继续向后比较，直到出现显著差异。
```

默认：

```text
eps = eps_frac * training_cycle
```

其中默认 `eps_frac=0.01`。

这意味着目标不是最小化总完成时间，也不是最小化所有 request ETA 的总和，而是：

> 优先让最近的 training buffer 尽早完成，只有最近 buffer 基本打平时，才看下一个 buffer。

## 5. 为什么这个 E_k 建模看起来合理，却仍然加速比上不去

### 5.1 request-level drain 本身不是完全错误

新 group 加入后，drain forecast 确实会重放整个 fleet，因此它并非完全不知道已有 request 会变慢。

问题在于：这些变慢最终如何被压缩成训练目标。

```text
所有 request 的 finish_times
        ↓ max per group
每个 group 一个 ETA
        ↓ inventory replay
少量 F_k
        ↓ training chain
少量 E_k
        ↓ lexicographic compare
一个 admission decision
```

这个压缩过程会丢掉大量非关键路径信息。

### 5.2 关键 blindspot：只保留 buffer critical path

实验中的代表性例子：

- 当前约 40 个 request。
- admit 一个 8-response group。
- 已有 request 的 ETA 每个约增加 17 秒。
- 所有已有 request 的 ETA 增量总和约 685 秒。
- 新 group 自身 ETA 约 154 秒。

drain forecast 能计算出这些 ETA 变化，但后面的目标只看 buffer ready/训练链上的关键 group。大量已有 request 虽然都被推迟，却没有改变当前最近 buffer 的完成时间，因此在 lexicographic 比较中几乎没有代价。

于是策略会认为：

```text
“新 group 让某个关键 buffer 更快或差不多，应该 admit。”
```

实际效果却是：

```text
“整个 fleet 被推入更高并发区，所有尾部 request 都变慢。”
```

### 5.3 high-CV 和 small-τ 为什么更容易暴露问题

- **high-CV**：少数长尾 request 的完成时间决定 group ETA。它们长时间占用 KV 和 decode 带宽，新 group 带来的额外并发会把这些尾部进一步推迟。
- **small-τ**：训练 buffer 很快就要消费，任何额外 drain 延迟都更容易在训练链上放大成等待，但等 E_k 明显变差时，过量 admission 已经发生。
- **静态 cap**：它虽然不理解 buffer deadline，但可以直接固定在经验上的 decode knee 附近，在这些 workload 上，这种粗糙规则反而避开了 over-admission。

## 6. 下一步最值得做什么

### 方向 A：把 fleet-wide ETA 外部性加入 admission objective

对同一个候选 admission，同时计算：

```text
base_finish[i] = 当前 fleet 下 request i 的预测完成时间
new_finish[i]  = admit 新 group 后 request i 的预测完成时间
externality    = Σ_i max(0, new_finish[i] - base_finish[i])
```

然后把 admission score 从单纯的 pipeline 比较扩展为：

```text
score = lexicographic(E_k vector) + λ * externality
```

第一版实验可以不改变 `Pipeline` 数据结构，而是在 `PlanSession` 中额外保存一个 fleet penalty，用于比较候选项。这样可以快速验证根因，验证有效后再决定是否把它正式建模进 Pipeline。

### 方向 B：先做 concurrency-knee gate 作为快速对照

从当前 cost model 估算当前 decode concurrency 是否已经超过 knee：

```text
if current_n_d > knee:
    strongly prefer DEFER
```

这相当于验证：单纯把 admission 压到正确并发区间，是否就能恢复 static cap 的收益。`kv_watermark=0.6` 已经说明“压低并发”有效，但它不是无量纲、可泛化的最终 objective。

### 方向 C：用极端 workload 做机制验证

优先尝试：

- `cv=3.0` 或 `4.0`。
- `mean=16000` 或 `20000`。
- 更短的 `tau`。
- 改变 `group_size` 和 batch，使总 response 数保持可比。

重点不是马上追求最好 speedup，而是收集：

```text
admit 前后每个 request 的 ETA 增量
critical buffer 的 E_k 增量
fleet-wide ETA externality
当前 n_d 与 decode throughput
```

如果大量点都出现“E_k 变化很小、externality 很大”，就能直接证明 objective blindspot，而不是继续盲目调 knob。

## 7. 结论

当前 `E_k` 的底层预测链条是有物理含义、且实现相当完整的：

```text
cost model drain replay
→ request finish time
→ group ETA
→ psrl inventory replay 得到 F_k
→ train/load/reconfig chain 得到 E_k
→ lexicographic admission decision
```

加速比长期上不去，不是因为完全没有预测 request 变慢，而是因为**预测结果在进入决策目标时被压缩成了少量 critical-buffer 的 E_k，丢失了 fleet-wide slowdown 的外部性**。

因此下一 session 最有价值的工作，不是继续调 `horizon`、`eps` 或 eviction 顺序，而是直接验证并实现 fleet-wide ETA externality objective。
