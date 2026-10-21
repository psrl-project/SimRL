# 异步 RL 中如何实现 True Staleness Control

> 一个参数叫 `max_staleness=4`，是否就意味着系统绝不会拿陈旧度大于 4
> 的 token 训练？
>
> 不一定。它可能限制的只是队列长度、并发任务数，或者“尚未被训练消费的工作量”。
> 这些机制可以改善平均情况，却不一定管得住最慢的那条 trajectory。

异步强化学习把 rollout 和 training 重叠起来：推理侧持续生成，训练侧持续更新，
两边不再等待彼此。这能显著提高硬件利用率，却也产生了一个无法回避的问题：
**用于训练的数据，可能来自若干个版本以前的策略。**

这个版本差通常被称为 **staleness（陈旧度）**。但在不同框架里，
“staleness control”可能指完全不同的东西：

- 有的系统限制 rollout 的准入数量。
- 有的系统在权重更新时暂停、恢复或取消正在生成的请求。
- 有的系统在训练前丢弃过旧样本。
- 还有的系统只记录告警，或者只在 loss 中修正“数据不是当前策略生成的”这个偏差。

本文想回答的核心问题是：

> **怎样的机制才能真正控制所有训练 token 的最大陈旧度？**

我们先解释 staleness 如何产生，再从 rollout 前、rollout 中和 rollout 后三个位置，
逐一分析 11 个异步 RL 框架，最后介绍如何用 DryRun-RL 对这些机制进行可重复实验。

---

## 第一部分：Staleness 如何产生？

### 1. 同一个系统里同时存在两个版本时钟

同步 RL 的循环很直观：

```text
rollout(v0) → train(v0→v1) → rollout(v1) → train(v1→v2) → ...
```

生成和训练之间有一道屏障，因此一批数据通常只服务紧邻的一次训练。

异步 RL 则把两者重叠起来：

```text
Rollout:  v0 ───────── v1 ───────── v2 ─────────►
Training:       train→v1    train→v2    train→v3 ►
```

当训练侧已经推进到 \(v_3\) 时，队列里仍可能留着 \(v_0\) 生成的数据。
一条长 trajectory 甚至可能在 \(v_0\) 开始、跨过两次权重更新，最后在 \(v_2\)
才结束。

![异步 RL 中 staleness 的产生](../../assets/async-rl-staleness/staleness-origin.svg)

异步不是 staleness 的偶然诱因，而是它产生的结构性原因：只要 rollout 和 training
能独立前进，两者的版本就可能分离。

### 2. 从 token 开始定义 staleness

记 \(e\) 为一次训练消费事件，\(\mathcal{G}_e\) 为该事件消费的 group 集合。
对其中 trajectory \(\tau\) 的第 \(i\) 个可训练 token，记：

- \(V_{\text{gen}}(\tau,i)\)：生成该 token 时 rollout engine 使用的策略版本。
- \(V_{\text{train}}(e)\)：本次消费事件真正执行梯度计算时使用的训练版本。

则 token 的 staleness 定义为：

\[
s(e,\tau,i)
=V_{\text{train}}(e)-V_{\text{gen}}(\tau,i)
\]

之所以从 token 定义，而不是只看 trajectory 的起始版本，是因为 partial rollout
可能跨越权重更新。同一条 trajectory 内，前半段和后半段 token 可能由不同版本生成。

trajectory 在该次消费中的 staleness，是其中最老 token 的年龄：

\[
s(e,\tau)=\max_i s(e,\tau,i)
\]

对于 GRPO 等按 group 训练的方法，还要在 group 内继续取最大值：

\[
s(e,g)=\max_{\tau\in g}s(e,\tau)
\]

而本文真正关心的 **max staleness**，是整个运行过程中所有实际进入训练的 group、
trajectory 和 token 的最大值：

\[
\boxed{
S_{\max}
=
\max_e\max_{g\in\mathcal{G}_e}\max_{\tau\in g}\max_i
\left(
V_{\text{train}}(e)-V_{\text{gen}}(\tau,i)
\right)
}
\]

这里的集合只包含实际进入梯度计算的数据。被取消或丢弃的任务不会破坏训练数据的
staleness 上界，但它们已经消耗的计算会计入浪费。对于不使用 group 的算法，
可以把每条 trajectory 看作一个单元素 group。若同一数据被 replay 多次，
每次消费都是独立事件，必须按当时的训练版本重新计算年龄。

许多系统只记录 trajectory 或 group 的起始版本。如果这个字段与训练端使用同一个
版本时钟、确实是该数据最早的行为版本，而且权重版本单调增加，那么用它代替所有
token 的最小生成版本是一种保守近似：

\[
V_{\text{start}}(\tau)\le \min_i V_{\text{gen}}(\tau,i)
\]

不过，这个近似成立仍有前提：系统必须在恢复、重试和跨版本 partial rollout 时
正确保留版本来源，不能把整条 trajectory 错标成最后一次恢复时的版本。

### 3. 什么是 True Staleness Control？

本文采用一个严格但可验证的定义。一套机制只有同时满足以下条件，才算
**true staleness control**：

1. **记录版本来源。** 对真正用于训练的单元，保存 token 生成版本，或保存一个
   可以保守覆盖所有 token 的最老行为版本。
2. **比较同一个版本时钟。** 明确计算
   \(V_{\text{train}}-V_{\text{behavior}}\)，而不是用队列长度、并发数或完成顺序代替。
3. **比较结果会改变训练集合。** 超限数据必须在梯度计算前被取消、丢弃、
   整组拒绝，或被证明对最终梯度完全无影响。简单把自身 loss mask 置零并不总是充分：
   在 GRPO 中，它仍可能通过 reward/advantage 的组内统计影响其他样本。
   仅记录 metric、warning、排序或暂停不算。
4. **没有旁路。** 启用的取样策略、存储后端、partial rollout 和恢复路径
   都必须经过同一个硬约束。

“让平均 staleness 更小”和“保证 \(S_{\max}\le S\)”是两件不同的事。
前者是性能优化，后者才是硬约束。

### 4. Staleness 会影响什么？

从算法角度看，staleness 增大了 behavior policy 与当前 policy 的差异。
这会改变 importance ratio、KL 和 clipping 的分布，也会让同一 GRPO group
内部的数据来自不同策略状态。AReaL 等系统会在 loss 中显式修正这种差异。
但“修正策略差异带来的偏差”并不等价于“限制样本年龄”。

从系统角度看，控制越严格，越可能付出以下成本：

- 取消正在生成的请求，浪费已经产生的 token。
- 在训练前丢弃完整 trajectory 或 group。
- 为补齐 batch 继续生成，增加等待时间。
- 不补齐 batch，则让实际 batch size 变小。
- 限制准入过严，降低 rollout engine 利用率。

反过来，如果只限制队列和并发，系统通常能获得更高吞吐，却无法阻止长尾请求无限变老。

> **TODO：收敛影响实验**
>
> 后续将固定模型、数据和总训练 token，扫描不同的真实 \(S_{\max}\)，比较 reward、
> validation 指标、KL、importance ratio 与训练稳定性。在实验完成前，本文不对
> “多大的 staleness 一定损害收敛”给出数值结论。

---

## 第二部分：现有框架如何控制 Staleness？

### 5. 三个可以采取行动的位置

从 prompt 进入系统到最终参与训练，staleness control 可以发生在三个位置：

![异步 RL 的三层 staleness 控制面](../../assets/async-rl-staleness/three-layer-control.svg)

#### L1：Rollout 前的 Admission Control

请求尚未开始生成。系统可以决定是否准入请求。例如一些系统会设置一个未被训练的全部请求的上限，当达到该上限时，不会再准入新的请求，以防止staleness过大。

这一层最直接，且不会引入额外成本。但只在这一层控制无法限制住某一个trajectory的staleness，例如xxx（举一个例子）。

#### L2：Rollout Runtime

trajectory已经开始生成。权重可以中途更新（partial rollout）。在这一过程中，系统可以判断请求是否已经staleness过大并暂停或者永久丢弃。

在这一层控制staleness可能存在成本，因为如果某trajectory已经生成了一段时候后再判断超过staleness阈值并丢弃，这些token的的生成就白白浪费掉了。

#### L3：Rollout 与 Training 之间

生成完成的数据正在 buffer/queue/batch assembler 中等待。系统可以按照一定顺序（基本上是FIFO）消费。但也可以在这一阶段丢弃超过staleness阈值的trajectory。丢弃后可补采样，或者直接用更小的 batch训练。

这是实现训练侧staleness硬保证最直接的位置：只要每个样本在进入梯度计算前都经过检查，
就能保证训练数据满足staleness阈值。但同时，也是成本最高的，因为这意味着一定会丢弃掉trajectory。

### 7. 十一个系统的三层机制

下图先给出全貌。后面每个框架都按L1-L3顺序讲

![十一个异步 RL 系统的三层机制](../../assets/async-rl-staleness/systems-layer-map.svg)

#### 7.1 verl

**L1**：verl支持设置`staleness_threshold` 以及`trigger_parameter_sync_step`来控制多采多少样本：

```python
self.max_required_samples = int(
    self.required_samples
    * (self.staleness_threshold + 1)
    * self.config.async_training.trigger_parameter_sync_step
)
```

其中`staleness_threshold`表示xxx、`trigger_parameter_sync_step`表示xxx。
然后这段
[`set_max_required_samples()`](https://github.com/verl-project/verl/blob/2bc6ac46043e31855cea194236ff7548c91817c8/verl/experimental/fully_async_policy/fully_async_rollouter.py#L488-L505)的结果`self.max_required_samples`就代表了
进入rollout前xxx，形成一个准入控制的限制。

**L2**：无staleness相关处理。

**L3**：无staleness相关处理。

**分析：** verl 只在L1进行了staleness控制，尽管通过限制`staleness_threshold` 可以减少系统里堆积的 rollout，但并不能控制每条样本最多旧多少。

#### 7.2 AReaL

**L1：`max_staleness` 用来控制系统里还有多少未训练的工作。**

AReaL 最容易误解的参数是
[`StalenessManager.max_staleness`](https://github.com/areal-project/AReaL/blob/bd741bd00f7c2af2acc611af9db31c9d367632d0/areal/infra/staleness_manager.py#L75-L115)。
名字像是在说“每条 trajectory 最多落后几个版本”，但代码实际做的是这个公式：

```python
ofp = self.max_staleness
sample_cnt = self.rollout_stat.accepted + self.rollout_stat.running
consumer_bs = max(1, self.consumer_batch_size)
staleness_capacity = (ofp + current_version + 1) * consumer_bs - sample_cnt

capacity = min(concurrency_capacity, staleness_capacity)
```

这里的 `accepted` 是已经完成并被接受的样本数，`running` 是正在生成的样本数。
训练版本 `current_version` 每增加 1，系统就多给一个 batch 的生成额度。和 verl 类似，
它管的是“总量”：系统里不要积压太多还没训练的样本。

**L2：权重更新时可以暂停、清 KV、续跑。** 这些操作处理的是请求怎么继续生成。
这一层没有 staleness 控制。

**L3：训练端会保存逐 token 的 `versions`。** AReaL 也会统计
[`current_version - versions`](https://github.com/areal-project/AReaL/blob/bd741bd00f7c2af2acc611af9db31c9d367632d0/areal/trainer/ppo/actor.py#L1882-L1934)。
但这些版本主要用于统计和 proximal log-probability 计算。rejection sampling
按 ratio/KL 等策略差异过滤。这里不会按版本差丢弃样本。

**小结：AReaL 不是硬 staleness 上限。** 举个例子：`max_staleness=1` 时，
一条 \(v_0\) 的长尾请求一直在生成。其他短请求不断完成并训练，模型已经更新到
\(v_5\)。这条长尾请求始终只占 `running` 里的一个位置，所以 L1 公式不会因为它变老
而触发取消。等它最后完成，仍可能进入训练。AReaL 这个参数更像“别让系统积压太多工作”，
不是“每条样本最多旧 1 个版本”。

#### 7.3 ROLL

**L1：`async_generation_ratio` 控制可以提前生成多少。** 这和 verl、AReaL 的思路接近：
先限制 rollout 不要跑得离 trainer 太远。直观地说，它允许 rollout 最多提前几步生成。

**L2：生成中会清理窗口外的任务。** ROLL 有 agentic 和 RLVR 两条路径。两条路径都把
scheduler step 当成版本时钟，用 `sampling_start_step` 或 `create_step` 记录样本出生时间。

以 RLVR 的
[`ReplayBuffer`](https://github.com/alibaba/ROLL/blob/d96d0cce56ec42c78c7b61e111b596a3a3a008a8/roll/distributed/scheduler/generate_scheduler.py#L232-L442)
为例，窗口边界是：

```python
min_step = self.current_step - math.ceil(self.async_generation_ratio)
max_gc_step = self.current_step - math.ceil(self.async_generation_ratio)
```

`gc()` 会清掉更早的 group，并取消对应 prompt task。

**L3：训练前也只取窗口内的数据。** `get_batch()` 只从窗口内收集数据，agentic 路径也会在
[`rollout_scheduler.py`](https://github.com/alibaba/ROLL/blob/d96d0cce56ec42c78c7b61e111b596a3a3a008a8/roll/distributed/scheduler/rollout_scheduler.py#L262-L346)
按 `step - create_step` 做训练前检查。

**小结：ROLL 比 verl / AReaL 多了后两层检查。** 如果 scheduler step 和权重版本是一一对应的，
这个窗口就可以控制样本版本差。不过两个路径有细节差异：

- agentic 路径要求运行时对象真的是 `int`。字段虽然声明为 `float`，却直接传给
  `range()`，任何 Python `float`（包括 `1.0`）都会抛出 `TypeError`。该路径成功运行
  时，上界是整数 \(S\)。
- RLVR 路径支持小数，并允许最老的一批样本只提供部分 batch，它的严格上界是
  `ceil(async_generation_ratio)`，而不是配置的小数本身。

#### 7.4 SkyRL

**L1：`max_staleness_steps` 也控制系统里积压多少工作。** 它和 AReaL 非常像，代码注释
也明确说它不能保证每个 group 都不超限。核心公式是：

```python
consumer_capacity = (
    self.max_staleness_steps + self._current_global_step
) * self.mini_batch_size
producer_staleness_capacity = (
    consumer_capacity - (self._stat.accepted + self._stat.running)
)
```

这段代码位于
[`_compute_capacity_unlocked()`](https://github.com/NovaSky-AI/SkyRL/blob/a1e34dbf21003f66bd7921487fd47db6c154aba2/skyrl/train/fully_async_trainer.py#L112-L184)。

**L2：权重更新时默认暂停后继续。** 默认 `PauseMode.KEEP`，也就是请求继续留在原来的生成流程里。
这里不按版本差判断要不要取消请求。

**L3：训练前会算一次超限并打日志。** SkyRL 会计算
`self.global_step - global_step_when_scheduled`，但超限后的分支是：

```python
if cur_staleness > self.max_staleness_steps:
    logger.warning("Staleness control violated ...")
    staleness_violation_count += 1
```

这段在
[`fully_async_trainer.py`](https://github.com/NovaSky-AI/SkyRL/blob/a1e34dbf21003f66bd7921487fd47db6c154aba2/skyrl/train/fully_async_trainer.py#L1031-L1066)。
源码 TODO 还写着“应该 drop、drop and resample，还是只 log？”当前答案是只 log。

**小结：SkyRL 是检测，不是硬控制。** 它会告诉你超了，但不会阻止超了的样本训练。

#### 7.5 OpenRLHF

**L1：`async_queue_size` 控制队列大小。** OpenRLHF 的 `async_queue_size` 同时限制
[`rollout_queue` 和 slot pool](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_trainer_async.py#L287-L301)，
意思是系统里最多放多少批 rollout。

**L2：partial rollout 会暂停后继续生成。** 代码路径是 `pause_generation(mode="keep")`，
请求不会因为版本差太大而被取消。

**L3：训练侧按 FIFO 取数据。** dynamic filtering 按 reward range 做过滤，不是按版本差过滤。
更根本的是当前
[`Experience`](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_utils/experience.py#L29-L70)
里不带 model version 字段，训练端连“这个样本是哪个版本生成的”都不知道。

**小结：OpenRLHF 当前不是 staleness control。** `async_queue_size` 可以减少排队，
但不能证明样本不会太旧。

#### 7.6 slime

**L1：semaphore、active concurrency、completed queue 水位控制并发和队列。**
这些参数决定同时跑多少请求、完成结果最多堆多少。

**L2：`flush-cache-interval` 控制权重更新方式。** 它决定权重更新时
[`pause(mode="abort")` 还是 `pause(mode="in_place")`](https://github.com/THUDM/slime/blob/05e3b017fed9e02de418c2fbac9d75602a534be5/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L102-L143)。
前者清 cache 并重提，后者保留 KV 原地更新。

**L3：训练前会记录和排序版本。** `weight_versions` 被
[`slime/utils/staleness.py`](https://github.com/THUDM/slime/blob/05e3b017fed9e02de418c2fbac9d75602a534be5/slime/utils/staleness.py#L10-L29)
用于统计，buffer 可以按版本排序，但排序不是丢弃。

**小结：slime 会影响 staleness 分布，但不提供最大版本差保证。**

#### 7.7 Dressage

**L1：`max_active_groups` 和 output queue 控制同时跑多少、队列放多少。**
它们限制并发和结果队列大小。

**L2：权重更新时可以 pause、abort，也可以 prefix 续跑。**
`max_partial_rollout_preempts` 限制的是一条 trajectory 最多被打断几次，不是样本年龄。

**L3：`dressage_staleness_keep_versions` 会丢一部分 group。** 代码在
[`StalenessTracker`](https://github.com/Accio-Lab/Dressage/blob/3e3142fe8ea07e4504c3b20a936a4c201a3de44c/dressage/rollout/staleness.py#L53-L132)：

```python
for info in trajectory_version_infos(group):
    if info.version not in self.versions:
        self.versions.append(info.version)

cutoff = max(0, len(self.versions) - self.config.keep_versions)
drop = self.version_index(info.version) < cutoff
```

问题是，`self.versions` 记录的是“结果返回时看到过哪些版本”，不是“当前训练版本减去生成版本”。
所以一个很早开始的 \(v_0\) 长尾，如果很晚才第一次完成，`v0` 反而会被当作刚观察到的新版本。

**小结：Dressage 会过滤一部分旧结果，但不能保证最大版本差。** 它比只限制队列更强，
但不是本文定义的 true staleness control。

#### 7.8 NeMo-RL

**L1：`max_inflight_prompts`、`max_buffered_rollouts` 控制 prompt 数和缓存结果数。**
它们限制同时有多少 prompt 在系统里。

**L2：部分取样策略会取消正在生成的旧请求。** 例如 `windowed` 这条路径，
会用请求开始时的 `start_weight` 判断它是否已经太旧，如果太旧，就可以取消。
但这取决于你选的取样策略和 rollout 后端，不是所有 NeMo-RL 配置都有。

**L3：`windowed` / `weight_fifo` 会在训练前按版本窗口取样。** 关键代码在
[`WindowedSampler`](https://github.com/NVIDIA-NeMo/RL/blob/370cc6f9fc828556dbbd85f0ca2682b438e7359e/nemo_rl/algorithms/async_utils/staleness_sampler.py#L357-L449)：

```python
min_valid_version = max(
    0, current_train_weight - self.max_staleness_versions
)
valid_idxs = [
    i
    for i, weight in enumerate(self._buffer.start_weight_list)
    if min_valid_version <= weight <= current_train_weight
]
```

这次是真的拿“当前训练版本”和“样本开始生成的版本”做比较。`weight_fifo` 也使用同类窗口。

**小结：NeMo-RL 要看取样策略。** `windowed` / `weight_fifo` 可以给训练数据版本差上限。
`ready_first` 会保留晚到样本。`in_order` 按目标 step 排序，不等价于按生成版本过滤。

#### 7.9 Trinity-RFT

**L1：`max_inflight_batches` 控制最多有多少 batch 在路上。**
它先限制系统里同时推进的 batch 数。

**L2：vLLM 会 pause、reset prefix cache、resume。** timeout 或 over-rollout
也可能中止任务，这些都不是按版本差做判断。

**L3：`sample_strategy=staleness_control` 会按版本差取样。** 关键配置是
`sample_strategy=staleness_control` 和有限的 `max_staleness`。

[`StalenessControlSampleStrategy`](https://github.com/agentscope-ai/Trinity-RFT/blob/24bf3405fec102bb2ef85d9b49179c6d62236086/trinity/algorithm/sample_strategy/sample_strategy.py#L80-L92)
计算：

```python
min_model_version = max(step - self.max_staleness - 1, 0)
exp_list = await self.exp_buffer.read(
    min_model_version=min_model_version
)
```

训练这批数据时使用的版本可以看作 `step - 1`，所以上面等价于：

\[
V_{\text{train}}-V_{\text{gen}}\le S
\]

[`QueueStorage`](https://github.com/agentscope-ai/Trinity-RFT/blob/24bf3405fec102bb2ef85d9b49179c6d62236086/trinity/buffer/storage/queue.py#L378-L409)
会跳过 `model_version < min_model_version` 的样本并继续取样，SQL 存储也会把阈值放进
`WHERE` 条件里。

**小结：Trinity-RFT 条件成立。** 必须同时满足：使用 `staleness_control`、
`max_staleness` 不是无限大，并且存储方式是 queue 或 SQL。默认取样策略或 file 存储
不能套用这个结论。

#### 7.10 TRL Async GRPO

**L1：`max_inflight_tasks` 控制同时在生成的任务数。** 如果用户没显式设置，TRL 会用
`max_staleness * samples_per_step` 推出最大在途任务数。这是为了别生成太多注定会被丢掉的样本。

**L2：权重同步时 pause、update、resume。** 正在生成的请求会继续走完。
这里不按版本差取消正在生成的请求。

**L3：训练前按版本差丢弃。** 每个 `RolloutSample` 带着创建时的 `model_version`，
trainer 的
[`RolloutQueueDataset`](https://github.com/huggingface/trl/blob/6b1e2d703d9b6f807bc2e874eb3eb6fbab33b688/trl/experimental/async_grpo/async_grpo_trainer.py#L303-L389)
直接执行：

```python
version = self.model_version_fn()
staleness = version - sample.model_version
if staleness > self.max_staleness:
    continue  # drop stale, pull next
```

**小结：TRL 成立。** 过期样本可能已经浪费了生成成本，但不会进入训练。

#### 7.11 prime-rl

**L1：`max_inflight`、burst budget、rate limit 控制系统里最多跑多少任务。**
这些限制先减少同时生成的任务量。

**L2：`max_off_policy_steps` 会取消正在生成、但已经太旧的 group。** 新版本生效前，
dispatcher 会算出最低可接受版本 `min_fresh_version`，然后取消太旧的 group：

```python
min_version = min_fresh_version(
    self.progress.step, self.max_off_policy_steps
)
stale_groups = [
    gid
    for gid, group in self.groups.items()
    if group.policy_version_at_start < min_version
]
for gid in stale_groups:
    await self.drop_group(gid, reason="stale")
```

这段逻辑在
[`on_version_pending()`](https://github.com/PrimeIntellect-ai/prime-rl/blob/9b1904e7d217bbfd125f3667582e87ea32b7563b/src/prime_rl/orchestrator/dispatcher.py#L397-L435)
里，会取消 asyncio task，也会处理还没 dispatch 的成员。

**L3：训练前还会再丢一次旧数据。** 因为 L2 可能有漏网情况：任务刚好完成，结果已经进了队列。
所以 `TrainSink` 在训练前又执行
[`_drop_stale()`](https://github.com/PrimeIntellect-ai/prime-rl/blob/9b1904e7d217bbfd125f3667582e87ea32b7563b/src/prime_rl/orchestrator/train_sink.py#L155-L198)：

```python
min_version = min_fresh_version(
    self.progress.step, self.config.max_off_policy_steps
)
if policy.start < min_version:
    del self.pending_batch[trace_id]
```

**小结：prime-rl 是完整控制。** L2 负责少浪费生成计算，L3 负责最终保证：
太旧的数据不会进入训练。

### 8. 哪些系统实现了 True Staleness Control？

按本文的严格定义，结论不是简单的“某框架支持/不支持”，而是分为四类。

#### 完整硬约束

- **prime-rl**：L2 取消正在生成的过期 group，L3 在训练前再扫一遍旧数据。
- **TRL Async GRPO**：L3 在取样给 trainer 前检查版本差，过期样本不会参与梯度。

#### 条件性硬约束

- **ROLL**：两条路径都执行 scheduler-age 窗口，但 agentic 要求实际 `int`，
  RLVR 的有效硬界为 `ceil(S)`，且都要求 scheduler step 与发布版本对齐。
- **NeMo-RL**：仅 `windowed` / `weight_fifo` 等按版本窗口取样的策略成立。
  正在生成的请求能否取消，还取决于 rollout 后端。
- **Trinity-RFT**：仅 `staleness_control` + 有限 `max_staleness` + queue/SQL 存储成立。

#### 检测、修正或近似，但不能给硬界

- **AReaL**：逐 token 记录和 loss correction，但不按年龄丢弃。
- **SkyRL**：检测到 violation 后只告警。
- **Dressage**：按 distinct observed-version 窗口过滤，不等价于训练版本差。
- **verl**：记录 version span，但仅用于 metric。

#### 容量或结构控制

- **OpenRLHF**：有界 queue/slot，`Experience` 不带 version。
- **slime**：通过 barrier、cache flush 和排序塑造 staleness，不执行年龄阈值。

这份分类关心的是**实际执行路径**，不是参数或 README 的命名。

### 9. 三层并不是互相替代，而是分工

一个生产系统理想的组合通常是：

1. **L1 控制浪费发生的概率。** 用队列大小、并发数、完成时间预测或准入门槛，
   尽量不启动大概率会过期的任务。
2. **L2 减少已经发生的浪费。** 每次版本推进时检查正在生成的请求，
   及时取消已经不可能保持新鲜的请求。
3. **L3 提供最终安全网。** 在梯度计算前再次按真实生成版本检查，
   保证任何晚到结果、重试或后端差异都不能绕过 \(S_{\max}\)。

只做 L1，长尾请求可能穿透。只做 L3，硬界成立但浪费可能很大。
只做 L2，则完成后排队过久的数据仍可能在 queue 中变老。

prime-rl 的路径之所以完整，不是因为它有一个更强的参数，而是因为三层职责被拆开了：
L1 控制别放太多任务，L2 及时取消已经太旧的任务，L3 在训练前做最终检查。

---

## 第三部分：用 DryRun-RL 做机制实验

### 10. 为什么需要模拟器？

源码阅读能告诉我们“代码试图保证什么”，却不能直接回答：

- 配置 `max_staleness=4` 时，真实 \(S_{\max}\) 到底是多少？
- 容量租约能把 P95 压低多少，又会留下多长的尾部？
- 生成中取消能节省多少无效 token？
- 训练前丢弃会损失多少吞吐？
- 相同 workload 下，不同机制的性能差异来自哪里？

线上系统很难公平比较这些问题。真实集群中的请求长度、实例速度、KV 容量和训练时间
同时变化，换一个框架还会连带改变 runtime 与并行策略。

因此，我们搭建了 **DryRun-RL**：一个面向异步 RL rollout/training loop 的
离散事件模拟器。它把 staleness policy 做成插件，让策略可以在三层控制面上分别接入：

- **L1：** `admit_quota` / `on_admit`
- **L2：** `on_interrupt` / `expire` / `on_discard`
- **L3：** `on_group_complete` / `peek_batch` / `take_batch`

同一个 workload、同一组 rollout 实例、同一个训练耗时模型，可以只替换 policy，
从而复刻 AReaL 式信用租约、ROLL 式 GC、verl 式 bounded queue、slime 式 barrier，
以及后续要补充的 prime-rl 式“容量 + runtime cancel + queue sweep”组合。

### 11. 第一组实验：配置值是否真的约束 Realized Max Staleness？

对每个框架机制，扫描它自己原生语义下的参数，而不是强行把所有参数都解释成版本数：

- AReaL / SkyRL：信用租约大小。
- verl：sample-overage ratio。
- ROLL / TRL：年龄窗口。
- slime：weight update / cache flush interval。
- prime-rl：年龄窗口与最大在途任务数。
- NeMo-RL / Trinity-RFT：取样策略、存储后端与窗口组合。

主要观测量为：

\[
S_{\max}^{\text{observed}}
=\max_e\max_{g\in\mathcal{G}_e}\max_{\tau\in g}\max_i
\left(V_{\text{train}}(e)-V_{\text{gen}}(\tau,i)\right)
\]

同时报告：

- mean / P50 / P95 / P99 staleness。
- throughput 与训练等待时间。
- 完成后丢弃、生成中取消的 group 数。
- 生成后未进入训练的 token。
- 因清 KV 或 abort 产生的重复 prefill / regeneration 成本。
- 不同长度分位的请求分别贡献了多少尾部 staleness。

> **TODO（图 4）：参数扫描**
>
> 横轴为各框架自己的 staleness 参数，纵轴同时展示 configured bound 与 observed
> \(S_{\max}\)。如果机制提供硬约束，曲线应始终不越界，若只是容量控制，
> 平均值可能下降，但最大值仍会被长尾请求拉高。

> **TODO（图 5）：成本分解**
>
> 对比 L1 block、L2 cancel 与 L3 drop 分别造成的 idle、sunk generation、
> queue wait 和 replacement generation。

### 12. 第二组实验：Max Staleness 如何影响收敛？

系统实验先回答“机制能否控制住真实最大值”。收敛实验再回答“控制到什么程度值得”。

计划固定：

- 初始 checkpoint 与数据顺序。
- 总训练 token 或总优化 step。
- rollout sampling 参数。
- batch/group 配置。
- 除 staleness policy 外的系统资源。

然后改变真实 \(S_{\max}\)，比较 reward、validation score、KL、clip fraction、
importance ratio 分布和 wall-clock efficiency。

> **TODO（图 6）：收敛曲线与质量/吞吐 Pareto frontier**
>
> 该实验尚未完成。当前博客只建立定义、机制分类和测量方法，不预填任何趋势。

---

## 当前阶段的 Takeaways

在实验数据加入前，源码调研已经给出几条相对稳固的结论。

### 1. 参数名不是语义

`max_staleness`、`staleness_threshold` 或 `max_head_offpolicyness`
可能控制版本年龄，也可能控制工作量、队列容量或样本比例。要看公式读取了什么状态，
以及超限分支真正执行了什么。

### 2. 控制未训练工作量，不能推出最大样本年龄

在吞吐近似固定的稳态中，队列和并发上限通常能限制平均驻留时间，但一个长尾请求可以只占
一个槽位却跨越任意多个训练版本。Little 定律描述平均量之间的关系，不能推出最大值。

### 3. True control 的最终判据在梯度入口

只要某条数据仍可能进入梯度计算，系统就没有提供训练侧硬保证。L2 取消很重要，
但它主要减少浪费。面对竞争时序、重试和晚到结果时，L3 的队列/训练前过滤
通常是最稳健、也最容易审计的最终安全网，它不是数学上唯一可能的实现位置。

### 4. Token 生成版本决定上界能否被证明

partial rollout 会让一条 trajectory 跨版本。最理想的是逐 token 记录生成版本。
至少也要可靠保存最老行为版本。只记录完成时版本、最后恢复版本或“观察顺序”，
无法证明全局 \(S_{\max}\)。

### 5. “支持”经常是有条件的

NeMo-RL 的结论取决于取样策略和 rollout 后端。
Trinity-RFT 的结论取决于取样策略与存储后端。
ROLL 的小数参数在不同 scheduler 中含义不同。框架级标签太粗，
真正需要审计的是启用配置下的完整数据路径。

### 6. 最实用的设计不是三选一，而是三层组合

L1 降低过期概率，L2 及时止损，L3 提供硬保证。三层分别优化准入成本、沉没成本和
训练正确性。把这三种职责塞进一个含义模糊的 knob，既难以解释，也难以验证。

后续实验将补充两类新的 takeaway：

- 哪些非硬约束机制在现实 workload 下已经足够接近目标。
- 为了把尾部从 P99 压到严格 max bound，需要付出多少吞吐和无效 token。

---

## 一句话结论

**真正的 staleness control，不是给系统加一个叫 `max_staleness` 的参数，
而是让每一条进入梯度的数据都带着可验证的生成版本，并在梯度入口前对最老 token
执行不可绕过的年龄上限。**
