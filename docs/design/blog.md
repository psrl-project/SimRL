# 分离式异步 RL 框架的 Staleness，究竟是怎么控制的？

分离式异步 RL 让 rollout 和 training 重叠运行，也带来了 staleness（陈旧度）问题：生成和排队期间，训练侧的策略可能已经更新了好几个版本。等到生成的内容被用于训练，生成它的策略就落后于当前策略了。

看起来，每个框架里都有个类似于 `max_staleness` 的参数。把它设成 1，或者更大一些的 4，似乎就能放心让 rollout 和 training 各跑各的。但这里的 `max_staleness=4`，究竟限制了什么？能保证所有参与训练的内容都最多落后 4 个策略版本吗？其实并不一定。

作为 PSRL 系列博客的第一篇，本文想从这个问题出发，看看 staleness 可以在哪里控制，以及目前不同的 RL 框架具体是怎么做的。

> **源码说明**：源码核对日期为 2026-10-16，选用当时各仓库默认分支的最新 commit，其中 ROLL、Dressage、TRL、prime-rl、NeMo-RL 于 2026-10-19 重新核对。关键逻辑在正文引用，完整版本与核对路径见[源码复核记录](publishing/源码复核记录.md)。文末讨论的实验尚未完成。

## 第一部分：Staleness 是怎么产生的？

### 1. 一条 Trajectory 的生命周期

一条 trajectory 通常从一个 prompt 开始。prompt 是生成的输入，trajectory 则是从这个 prompt 展开的生成过程及其内容，其中逐个生成的内容单位称为 **token**。系统先决定是否让 prompt 进入 rollout，这个动作下文统一称为**准入**。准入后，rollout 开始生成 trajectory，生成完成后再放进 queue 或 buffer 等待训练。

对于 GRPO，同一个 prompt 通常会生成多条 trajectory，组成一个 **group**。训练侧从 queue/buffer 中选出 trajectory 或 group，这个动作称为**训练取样**。选出的内容由 batch assembler 组装成一个 **batch**，也就是本轮一起参与训练的内容。

在分离式异步 RL 中，rollout 端和训练侧可以同时处理不同的 trajectory。rollout 端继续生成时，训练侧也在用之前收集的 batch 训练。因此，一条 trajectory 从开始生成到训练取样，可能经历多次策略版本更新。在这一过程中，一共涉及到三种模型版本：**准入版本**是 prompt 准入时记录的策略版本，**生成版本**是生成某个 token 时实际使用的策略版本，**训练版本**则是这条 trajectory 参与训练时实际使用的策略版本。

下图跟踪了其中一条 trajectory。它先用模型版本 v1 生成前半段。这期间，训练侧完成一轮训练，得到 v2，并把新权重同步到 rollout。这条 trajectory 采用了 **partial rollout**：保留已经生成的前缀，换用 v2 继续生成剩余部分。因此，同一条 trajectory 里既有 v1 生成的 token，也有 v2 生成的 token。

![一条 partial rollout trajectory 在 v1 和 v2 生成，等待完整的一次版本更新，再以 v3 参与训练](assets/async-rl-staleness-blog/staleness-origin.png)

生成结束也不意味着马上训练。此时训练侧正在处理前面的 batch，这条 trajectory 只能先在 queue/buffer 中排队。它在这里等待训练侧完成 v2→v3 的更新，直到训练版本为 v3 时才被取出，参与 v3→v4 的训练。v4 要等这轮训练完成后才会得到。

### 2. Staleness 的定义

顺着这个例子，staleness（陈旧度）就容易理解了。**对一个 token，训练版本减去生成版本，就是它的 staleness：**

$$
\text{token staleness}=V_{\text{train}}-V_{\text{gen}}
$$

图中这条 trajectory 在 v3 时参与训练，所以前半段 token 的 staleness 是 3 − 1 = 2，后半段则是 3 − 2 = 1。

> **口径说明**：本文的训练版本指训练开始时实际使用的版本。图中是 v3，而不是这轮训练完成后得到的 v4。

从中我们不难发现，**生成时间长，或者生成后等待得久，都可能让 staleness 增加**，因为这两段时间里，训练侧都可能继续更新，跨过数个策略版本。

如果要用一个数描述整条 trajectory 的 staleness，我们可以取 trajectory 中最大的 token staleness。因此，图中示例 trajectory 的 staleness 就是 2，因为最早用 v1 生成的那部分内容，在训练时落后了两个版本。对于 GRPO 的一个 group，我们类似地取组内全部 trajectory staleness 的最大值，作为这个 group 的 staleness。

因此，本文所说的“能否严格控制 staleness”，就是看：设定上限之后，**系统能否保证所有实际参与训练的 token，其 staleness 都不超过这个上限**。例如上限为 4，就要求每一个训练 token 的生成版本最多落后训练版本 4 个版本。只让平均 staleness 下降，或者在部分负载下没有观察到超限，都不等于具备这个保证。被取消或丢弃、最终没有参与训练的 trajectory 不计入这个最大值。

有了这个定义，接下来我们就可以进一步讨论：究竟该怎样控制 staleness？目前的异步 RL 框架分别是怎么做的？它们能否真正限制住整个训练过程中的最大 staleness？

## 第二部分：现有框架如何控制 Staleness？

### 3. 三个可以介入的位置

先回到一条 trajectory 从生成到训练的过程，我们不难发现，有三处位置系统可以直接介入：（1）系统可以在 rollout 前决定是否准入，（2）系统可以在生成过程中进行干预，（3）系统也可以在生成完成后、训练取样前进行干预。不同框架的做法虽然各有差异，但 staleness 控制位置基本都可以归到这三层，下面分别记作 L1、L2、L3。

![异步 RL 流水线上的 L1 准入控制、L2 Rollout 运行时控制和 L3 Training 前过滤](assets/async-rl-staleness-blog/three-layer-control.png)

#### L1：准入控制（Admission Control）

prompt 还没开始生成，系统先决定是否准入。常见做法是限制同时生成的 trajectory 数量，或者给“正在生成和已经完成、但还没被训练取样的 trajectory”设置总量上限。达到限制后，先暂停准入，等生成完成、已有 trajectory 被训练取样，或者训练版本更新后再恢复，具体取决于限制的对象。

这一层比较直接，因为被拦住的 prompt 还没有消耗生成计算。**但只限制数量，并不意味着能够限制每条 trajectory 的 staleness。** 假设系统允许同时生成 8 条 trajectory，其中一条在 v0 开始后一直没有结束。剩下 7 个槽位不断生成较短的 trajectory，让训练侧可以不断凑齐 batch，一路更新到 v5。那条长 trajectory 始终只占一个槽位，没有违反数量上限，但如果此时参与训练，最早那些 token 的 staleness 已经达到 5。

因此，这类 L1 控制可以减少积压，但仅靠数量限制，无法保证每条 trajectory 的 staleness 都不超过阈值。

#### L2：Rollout 运行时控制（Rollout Runtime Control）

trajectory 已经开始生成，也可能像图中那样通过 partial rollout 跨版本继续生成。系统可以在版本更新时比较当前训练版本与最早生成版本，取消已经超过 staleness 阈值的 trajectory。

例如，一条 trajectory 从 v1 开始生成，到了 v4 仍没结束。如果允许的最大 staleness 为 2，系统就可以在这次版本推进时取消它，而不必等它完整生成后再处理。这样省下了后续计算，但取消前已经生成的 token 仍然花费了计算资源。因此，**在 L2 层级上进行控制的成本要明显高于在 L1 层级上进行控制**。此外单纯的 L2 层级只能控制生成种产生的 staleness，而无法阻挡生成完排队耗时导致的 staleness，因此也无法仅仅靠 L2 层级干预来保证每条 trajectory 的 staleness 都不超过阈值。

#### L3：Training 前过滤（Training Input Filtering）

trajectory 已经生成完，会在 queue/buffer 中等待。系统可以按 FIFO（先进先出）等顺序训练取样，也可以在组装 batch 时比较生成版本与训练版本，丢弃超过 staleness 阈值的 trajectory 或 group。丢弃之后，可以继续准入新的 prompt 补齐 batch，也可以使用更小的 batch，具体取决于框架实现。

**L3 是实现严格 staleness 控制最直接的位置，但同时也是成本最大的。** 到了这里，系统已经知道这批 trajectory 将用哪个版本训练。只要保留了可靠的生成版本，并让所有 trajectory 在训练取样前接受检查，就可以判断它们是否超过 staleness 阈值。但这一层的问题是，丢弃时整条 trajectory 的生成成本已经发生。具体会浪费多少，还要看前两层是否减少了过量生成和超限 trajectory。如果大部分 trajectory 都能及时被取样训练，L3 也可能很少触发丢弃。

### 4. 不同框架的三层机制

下面先用一张图整理 10 个框架的做法，再按 L1 → L2 → L3 展开。L1-L3 的勾叉表示是否在对应层有进行 staleness 控制相关操作。最后一列是**严格 staleness 控制（按本文定义）**：能保证所有训练 token 都不超限，才记为勾，否则记为叉。

> **配置说明**：图中结论对应下文核对的配置。部分框架在不同配置下会采用不同机制，具体条件随实现一起说明。

![10 个系统的三层控制与严格 staleness 控制，具体配置见下文](assets/async-rl-staleness-blog/systems-layer-map.png)

点击框架名称，展开查看具体分析。

<details class="framework-detail" id="framework-verl">
<summary>4.1 verl</summary>

verl 采用 `trainer.v1.trainer_mode=separate_async` 时，通过 `max_off_policy_threshold` 设置阈值，再用 `max_off_policy_strategy` 选择 `drop` 或 `wait`。

**L1：** verl 按训练进度补充准入。`num_warmup_batches` 控制训练开始前额外预取多少批 prompt，支持小数。之后每进入一轮训练，还会正常准入 `train_batch_size` 个 prompt。例如，设 `train_batch_size=4`、`num_warmup_batches=1.5`，启动时先额外准入 `4 × 1.5 = 6` 个 prompt，首轮再正常准入 4 个，合计 10 个。之后每轮再准入 4 个。这里要把[启动预取](https://github.com/verl-project/verl/blob/5b550b0c3dad8c7bb7aab834a3fe195f8cdfc517/verl/trainer/ppo/v1/trainer_base.py#L624-L651)与[每轮正常准入](https://github.com/verl-project/verl/blob/5b550b0c3dad8c7bb7aab834a3fe195f8cdfc517/verl/trainer/ppo/v1/trainer_base.py#L1520-L1527)加在一起看。

**L2：** 不进行 staleness 控制。

**L3：** 准入 prompt 时，训练侧会记录当时的训练轮次 `global_steps`。`ReplayBufferAsync` 训练取样时，再比较当前训练轮次与准入时记录的训练轮次。下面分两种策略看。

**`drop`：超限就丢弃。** 如果下面的检查值超过阈值，就丢弃整个 group：

```python
if global_steps - prompt_global_steps.get(uid, global_steps) + 1 > self.max_off_policy_threshold
```

源码：[超限 group 的识别与淘汰入口](https://github.com/verl-project/verl/blob/5b550b0c3dad8c7bb7aab834a3fe195f8cdfc517/verl/trainer/ppo/v1/replay_buffer.py#L520-L539)。

> **口径说明**：`+1` 把准入当轮也算进去。若每个训练轮次只做一次策略更新，`drop` 的 staleness 上限是 `max_off_policy_threshold − 1`。verl 也支持 `parameter_sync_step` 大于 1，此时同一个 `global_steps` 中会连续取样和更新多次，门限需要按每轮的更新次数换算，不能直接把 `H−1` 当成 token staleness 上限。

过滤后，buffer 会重新准入同样数量的新 prompt，继续等待足够的合格 group，选出的 batch 还会再次接受检查。

**`wait`：先等慢 group 完成。** 当已准入、但尚未完成生成的 group 达到上述计数阈值时，即使已完成的 group 足够凑 batch，buffer 也不向训练侧返回。训练卡在取样处，后续版本发布也随之等待，给慢 group 留下完成的时间。等这些 group 全部完成，buffer 再优先选择准入更早的 group。

不过，**等到完成，不等于一定能在本轮被训练取样。** 每轮最多取 `train_batch_size` 个 group。如果达到阈值的 group 超过本轮取样量，剩下的仍会留到下一轮。而 `wait` 的[等待条件](https://github.com/verl-project/verl/blob/5b550b0c3dad8c7bb7aab834a3fe195f8cdfc517/verl/trainer/ppo/v1/replay_buffer.py#L541-L556)只检查尚未完成生成的 group，已经完成的 group 不会因 staleness 超限被丢弃。

我们可以考虑如下的例子：沿用 L1 的配置，并假设 `parameter_sync_step=1`，再设 `max_off_policy_threshold=1`（此时 `drop` 对应 staleness 上限 0）。L1 的预取量和这个阈值互不干扰，因此阈值设为 1，也不妨碍首轮准入 10 个 prompt。假设它们都用 v0 完成生成，`wait` 会等这 10 个 group 全部完成，但本轮只取 4 个参与 v0→v1 的训练，剩下 6 个留在 buffer。下一轮又会准入 4 个新 prompt，`wait` 也要等它们完成。不过，取样时会优先选择更早准入的 group，因此取到的仍是前一轮留下的 4 个 group，用 v1 训练时 staleness 已是 1。如果改用 `drop`，前一轮留下的 6 个 group 会在下一轮取样时被丢弃。

源码中的[对应测试](https://github.com/verl-project/verl/blob/5b550b0c3dad8c7bb7aab834a3fe195f8cdfc517/tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py#L676-L691)也单独验证了 `wait` 的这一行为：在没有未完成 group 的情况下，已经完成的超限 group 仍会被返回。

**分析：** `drop` 和 `wait` 在 L3 都有控制动作。`drop` 会在训练取样前淘汰超过准入轮次门限的已完成 group。固定每轮更新次数后，门限可以换算成所有训练 token 的 staleness 上界，因此最后一列记勾。verl 的[训练循环](https://github.com/verl-project/verl/blob/5b550b0c3dad8c7bb7aab834a3fe195f8cdfc517/verl/trainer/ppo/v1/trainer_base.py#L533-L610)会在同一个 `global_steps` 内按 `parameter_sync_step` 多次取样和更新。以每次 `_step_once()` 视为一个策略版本、每轮更新 P 次、阈值设为 H 为例，保守上界是 `P × H − 1`。当前配置的 [`parameter_sync_step=4`](https://github.com/verl-project/verl/blob/5b550b0c3dad8c7bb7aab834a3fe195f8cdfc517/verl/trainer/config/ppo_trainer.yaml#L262) 时，`H=1` 不等于 staleness 上限 0。`wait` 只阻塞仍在生成的 group，不会淘汰已经完成的超限 group，因此不能保证严格上界，最后一列记叉。

> **配置说明**：将 `max_off_policy_threshold` 设为 `null` 时，不执行这两种 L3 策略，此时主要由 L1 限制准入量。

</details>

<details class="framework-detail" id="framework-areal">
<summary>4.2 AReaL</summary>

**L1：** AReaL 有专门的 `StalenessManager` 模块来控制准入，允许通过 `max_head_offpolicyness` 配置生成提前量。类似于 verl，它让准入额度随训练进度推进，核心公式如下：

```python
sample_cnt = self.rollout_stat.accepted + self.rollout_stat.running
consumer_bs = max(1, self.consumer_batch_size)
staleness_capacity = (ofp + current_version + 1) * consumer_bs - sample_cnt
```

源码：[get_capacity()，其中 ofp = max_head_offpolicyness](https://github.com/areal-project/AReaL/blob/6cad8710bb5740a5480f8fba3a5970dd08638d01/areal/infra/staleness_manager.py#L100-L113)。

这里按每个 prompt 对应的生成任务计数。`accepted` 是累计生成完成并通过接收检查的数量，包括已经被训练取样的部分，`running` 是仍在生成的数量。生成完成时，只是从 `running` 移到 `accepted`，两者之和没有减少，因此不会释放准入额度。随着版本推进，`current_version` 每增加 1，公式才多给一个 batch 的准入额度。因此最终效果等同于 verl 的 L1 准入控制，即在初期会迅速用掉 `staleness_capacity`，之后每次版本更新才能再解锁一个 batch 的 prompt。

下面是一个具体的例子：每个 batch 有 32 个 prompt、`max_head_offpolicyness=1`，初始时，准入额度是 `(2 + 0 + 1) × 32 = 96`。因此系统会迅速填充 96 个 `running` 的 prompt，这些 prompt 逐渐生成并标注为 `accecpted`。当第一次训练完成，模型更新后，`current_version` 增加 1，从而使得 `staleness_capacity` 增加 32，可以准入新的 prompt。

**L2：** 无 staleness 控制。权重同步不会中断正在生成的请求；上面公式里的 `running` 只有在请求本身完成时才会转入 `accepted`，这个转移和权重版本的推进没有关系。

**L3：** AReaL 会测量 token staleness，记录平均值、最小值、最大值和旧 token 占比。不过，这段[staleness 统计](https://github.com/areal-project/AReaL/blob/6cad8710bb5740a5480f8fba3a5970dd08638d01/areal/trainer/ppo/actor.py#L2065-L2115)不会移除超限 trajectory。

**分析：** AReaL 的额度公式把版本推进和准入额度联系起来，可以减少 rollout 积压，最终实现类似于 verl 的效果。但其和 verl 存在同样的问题：一个 prompt 无论生成多久，都只占一个 `running` 额度。只要其他 prompt 完成生成、足以让训练继续，新的版本和额度就会不断产生，慢 trajectory 仍可以继续生成。与 verl 在 L3 的 `drop` 相比，AReaL 没有训练前淘汰超限 trajectory 的步骤，因此 `max_head_offpolicyness` 不能直接理解为每个训练 token 的 staleness 上限。

</details>

<details class="framework-detail" id="framework-roll">
<summary>4.3 ROLL</summary>

**L1：** 这里按 ROLL 的 RLVR 路径分析。首次准入时，ROLL 在正常 batch 之外增加 `ceil(batch_size × async_generation_ratio)` 个 prompt 额度，之后每轮再增加一个 batch。scheduler 将正在生成和已经完成的 prompt 数相加，与累计额度比较，额度用完就暂停准入。例如，每批 32 个 prompt、`async_generation_ratio=2`，首轮额度是 96，之后每轮再增加 32。trajectory 的起点训练轮次在生成开始时记录。

**L2：** 窗口推进时，scheduler 会回收窗口外的 group，并通过生成 scheduler 的 `gc()` [取消仍在运行的对应任务](https://github.com/alibaba/ROLL/blob/d96d0cce56ec42c78c7b61e111b596a3a3a008a8/roll/distributed/scheduler/generate_scheduler.py#L568-L577)。

**L3：** 已完成的 trajectory 还要经过训练取样窗口检查。RLVR 取 batch 时先计算：

```python
min_step = self.current_step - math.ceil(self.async_generation_ratio)
```

源码：[ReplayBuffer.get_batch()](https://github.com/alibaba/ROLL/blob/d96d0cce56ec42c78c7b61e111b596a3a3a008a8/roll/distributed/scheduler/generate_scheduler.py#L351-L386)。

这里的 `current_step` 是 scheduler 的训练轮次计数，group 也记录了开始生成时的轮次。训练取样会跳过早于下界的 group。例如，当前轮次为 5、`async_generation_ratio=2`，就只取开始生成的轮次不早于 3 的 group。

**分析：** ROLL 的 RLVR 路径把 `async_generation_ratio` 同时用于 L1 准入额度和 L3 训练取样窗口，不能只把它理解为多采样比例。L3 排除窗口外的已完成 trajectory，L2 则取消窗口外的在途任务。只要 scheduler 轮次与策略版本对齐，这个窗口就能限制最大 staleness。

这个参数也不总能直接读成 staleness。RLVR 中设成 1.5，取整后的训练轮次窗口为 2，小数部分还影响窗口边缘的取样量。

RLVR pipeline 还在配置加载时直接断言这个比例不能小于 1（`async_generation_ratio >= 1.0`），比 scheduler 自身更宽松的 `>= 0` 校验更严。这是因为 agentic 的另一条路径复用同一个字段，且不需要这个下限。换句话说，RLVR 下看到的 `async_generation_ratio` 永远至少是 1，首轮额度也永远不会小于两个 batch。

</details>

<details class="framework-detail" id="framework-skyrl">
<summary>4.4 SkyRL</summary>

**L1：** SkyRL 沿用了 AReaL 的额度思路，源码也注明了这一点。它用 `max_staleness_steps` 和当前训练进度计算累计准入额度，再减去 `accepted + running`，同时受最大并发数限制。额度用完就暂停准入，训练每完成一轮，再增加一个 batch 的额度。

**L2：** 无 staleness 控制。

**L3：** SkyRL 在训练取样 group 时也会计算 `global_step - global_step_when_scheduled`。超过 `max_staleness_steps` 后，它会打印 warning 并累计违规次数。源码还留下了这样一个问题：

```python
# TODO(Charlie): should we drop, drop and resample, or just log?
```

源码：[训练取样时的 staleness 检查](https://github.com/NovaSky-AI/SkyRL/blob/38146ee9248eb61d884e7df7b44747d97a6fc381/skyrl/train/fully_async_trainer.py#L1037-L1070)。

当前这个版本的选择是“只记日志”。group 在检查前已经加入训练 batch，warning 分支没有把它移除，也没有跳过训练。

**分析：** 与 AReaL 相比，SkyRL 多了训练取样时的超限告警，但没有进一步丢弃。它能发现某个 group 的 staleness 超限，却仍会训练这个 group，所以也不能保证最大 staleness 不超过配置值。

</details>

<details class="framework-detail" id="framework-openrlhf">
<summary>4.5 OpenRLHF</summary>

**L1：** OpenRLHF 用有限数量的生成许可连接 generator 和 trainer。初始化时，它创建一个容量为 `async_queue_size` 的 queue，再准备同样数量的生成许可。generator 每准入一个 batch 前先取走一个许可，用完就等待。trainer 取到一个 batch 后，[先归还许可，再调用 `train_step()`](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_trainer_async.py#L227-L232)，generator 才能继续准入。

例如，`async_queue_size=2` 时，最多两份许可可以分配给正在生成或等待训练取样的 batch。batch 一旦被取走，许可就已归还，即使它还在训练也不再占用。这里限制的是流水线中这些阶段的 batch 数。

**L2：** 开启 partial rollout 后，系统在权重同步时暂停和恢复生成，但没有按 staleness 阈值取消超限 trajectory。这个开关不改变严格控制的结论。

**L3：** trainer 取到 batch 后就归还许可并进入训练，中间没有比较生成版本与训练版本，也没有过滤超限 trajectory。

**分析：** OpenRLHF 用许可池限制 batch 积压，与 AReaL 的累计额度控制有所不同。不过，这里仍然没有逐条检查生成版本，`async_queue_size=2` 限制的是 batch 数，不能直接解释成 staleness 上限为 2。

</details>

<details class="framework-detail" id="framework-slime">
<summary>4.6 slime</summary>

**L1：** 这里看 slime 默认 object-store 路径的 fully async worker。它先按回答并发预算除以每个 prompt 的回答数，换算成 group 上限，还可以用预取系数扩大这个上限。例如，允许 64 条回答并发、每组 8 条、预取系数为 1，就最多同时准入 8 个 group。

准入时，worker [同时检查活跃 group 数和完成 queue 水位](https://github.com/THUDM/slime/blob/2f4280e3b0e6934b954db93153bd79d9c9312830/slime/rollout/fully_async_rollout.py#L199-L223)。沿用上面的例子，正在生成的 group 达到 8 个，就先不再准入 prompt。即使有空位，只要完成 queue 已有 8 个 group，也停止准入，等训练侧取走 group 后再继续。这里的 8 是停发水位，queue 本身无界，已经在运行的 trajectory 仍能完成，让 queue 暂时超过水位。

**L2：** 中断的 group 可以保留已生成的内容，重新准入后继续生成，但没有按 staleness 阈值永久取消。`mask_offpolicy_in_partial_rollout` 可以屏蔽旧前缀的 loss，不过它没有限制剩余 token 在排队后的 staleness。

**L3：** 训练侧从完成 queue 取 group，直到凑够 `rollout_batch_size`。slime 会[统计 rollout 当前版本与 trajectory 最早生成版本的差](https://github.com/THUDM/slime/blob/2f4280e3b0e6934b954db93153bd79d9c9312830/slime/utils/staleness.py#L10-L29)，但内置训练取样逻辑没有据此删除超限 trajectory。这里统计用的版本来自 rollout，仍需与第一部分的训练版本区分。

**分析：** 与 OpenRLHF 共用一池许可不同，slime 分别检查“有多少 group 正在生成”和“有多少 group 已经完成”。但与前面的 L1 方案一样，一个慢 group 可以一直占着运行槽位，其他 group 则不断完成并参与训练。并发和 queue 水位都没有超限，也不能保证这个慢 group 最终参与训练时的 staleness。

</details>

<details class="framework-detail" id="framework-dressage">
<summary>4.7 Dressage</summary>

**L1：** 类似于 slime，Dressage 也同时检查活跃 group 数和完成 queue 水位。不同的是，它用 `max_active_groups` 单独配置活跃上限，并使用有界 output queue，将容量的 80% 作为停发水位。两项都没达到限制，主循环才继续准入下一组 prompt，否则等待已有 trajectory 完成或训练侧取样。

**L2：** Dressage 的 partial rollout 使用 `max_partial_rollout_preempts` 限制一次 session（保留生成历史的会话）可跨过的版本数。proxy 收集历史和当前响应使用的版本，去重后计算：

```python
version_span = len(versions)
version_switches = max(0, version_span - 1)
if version_switches <= max_partial_rollout_preempts:
    return
```

源码：[_raise_if_partial_version_span_exceeded()](https://github.com/Accio-Lab/Dressage/blob/3e3142fe8ea07e4504c3b20a936a4c201a3de44c/dressage/proxy/server.py#L297-L340)。

例如设成 2，最多允许涉及 3 个不同版本，出现第 4 个时就拒绝继续生成，重试耗尽后放弃这个 group。源码把“不同版本数减 1”记为 `version_switches`，但这不等于 staleness。

**L3：** Dressage 使用 `dressage_staleness_keep_versions` 保留最近观察到的若干版本，并淘汰窗口之外的 trajectory。这里的[版本提取与窗口维护](https://github.com/Accio-Lab/Dressage/blob/3e3142fe8ea07e4504c3b20a936a4c201a3de44c/dressage/rollout/staleness.py#L32-L132)使用末尾 token 的版本，并按新版本被观察到的顺序维护窗口，没有直接比较最早 token 的生成版本与训练版本。

开启过滤后，负责收集 group 的 collector 还会重新检查已经收集、尚未凑齐 batch 的 group，删除窗口外的内容后继续收集，直到补足目标 group 数。

**分析：** 相比 slime，Dressage 在 L2、L3 都增加了实际拒绝旧 trajectory 的动作。不过，一条 trajectory 从 v1 直接切到 v5，只有一次版本切换，版本编号却已经相差 4。若又用末尾 token 的 v5 标记整条 trajectory，前缀中 v1 token 的 staleness 就体现不出来。生成后继续排队，还可能让训练版本继续向前推进。

所以，**Dressage 仍不能按本文定义严格控制 staleness**。它的 L1 仍然限制未训练工作量，但这不能替代对每个训练 token 的版本检查。

</details>

<details class="framework-detail" id="framework-nemo-rl">
<summary>4.8 NeMo-RL</summary>

NeMo-RL 的 single-controller 可以选择四种 sampler，也就是决定如何准入和训练取样的策略。它们采用不同机制，需要分开来看。真实系统的默认值是 `in_order`，不是下面篇幅最长的 `windowed`。`windowed` 在 L1、L2、L3 三层都有动作，容易让人读完误以为它才是默认选择。

**L1：** 所有 sampler 都受生成并发 `max_inflight_prompts` 和 buffer 容量 `max_buffered_rollouts` 限制。buffer 许可从准入前就开始占用，到 group 被淘汰或训练取样后才释放，所以正在生成的 group 也占额度。除了 `windowed`，另外三种 sampler 还会限制生成批次相对训练版本的提前量，额度用完就等训练推进。

**L2：** 只有 `windowed` 会判断起点版本是否早于 `当前训练版本 - max_staleness_versions`，超限就取消运行中的 group。它目前也只在 [native 权重同步路径](https://github.com/NVIDIA-NeMo/RL/blob/726d5eb225a8ae21e14b6778e56abfab2d794513/nemo_rl/algorithms/single_controller.py#L5448-L5456)执行，NeMo-Gym 路径有 TODO，暂时跳过。其他三种 sampler 没有这项取消。

**L3：** `windowed` 按生成版本的数值窗口取已完成 group，核心条件如下：

```python
min_valid_version = max(0, current_train_weight - self.max_staleness_versions)
valid_idxs = [
    i
    for i, weight in enumerate(self._buffer.start_weight_list)
    if min_valid_version <= weight <= current_train_weight
    and self._buffer.ready_list[i]
]
```

源码：[WindowedSampler.select()](https://github.com/NVIDIA-NeMo/RL/blob/726d5eb225a8ae21e14b6778e56abfab2d794513/nemo_rl/algorithms/async_utils/staleness_sampler.py#L425-L445)。

例如，当前训练版本为 v5、阈值为 2，就只取生成起点在 v3 到 v5 之间的 group。`weight_fifo` 也执行这个窗口限制，但优先取窗口内最早生成版本的 group，并等它达到本轮取样要求。两者都会保留最早生成版本，因此能覆盖 partial rollout 前缀和生成后的排队时间。

`ready_first` 则只看谁先完成，不检查最早版本下界，也不淘汰超限 group。它虽然在 L1 限制提前量，慢 group 仍可能被后来的快 group 绕过，等到很久以后才参与训练。

`in_order` 采用另一种办法：每批生成任务在准入时就绑定一个 `target_step`，指定它要用于哪个训练版本，训练取样只选目标与当前训练版本一致的 group。如果 `max_lookahead_versions=2`，目标为 v5 的 batch 最早在训练版本 v3 时准入，并且只能用于 v5 的训练。慢 batch 会让这轮训练等待，不能任意留到 v6、v7 再训练。因此，在正常策略训练、生成权重与训练版本同步的条件下，最大 staleness 不超过 2。这个保证来自“限制准入提前量 + 指定训练版本”，没有依赖训练前比较生成版本。四种 sampler 的关键差异可对照[准入与取样实现](https://github.com/NVIDIA-NeMo/RL/blob/726d5eb225a8ae21e14b6778e56abfab2d794513/nemo_rl/algorithms/async_utils/staleness_sampler.py#L498-L712)。

| sampler | L1 | L2（native） | L3 | 严格 staleness 控制（按本文定义） |
|---|---|---|---|---|
| `windowed` | ✓ | ✓ | ✓ | ✓ |
| `weight_fifo` | ✓ | ✗ | ✓ | ✓ |
| `ready_first` | ✓ | ✗ | ✗ | ✗ |
| `in_order` | ✓ | ✗ | ✓ | ✓ |

> **配置说明**：这里将 `in_order` 按指定训练版本取样也计入 L3 控制。NeMo-Gym 路径的 L2 均为叉，总览图中的三层勾选对应 `windowed + native`。

**分析：** `windowed` 和 `weight_fifo` 都在训练取样前按版本窗口过滤，`ready_first` 主要控制 L1 的未训练工作量。`in_order` 则说明，等待也可以形成严格上限，关键是它还限定了这一批 trajectory 只能在哪一轮训练。相比之下，verl 的 `wait` 没有为已完成 group 绑定训练轮次，旧 group 仍可能继续积压。L2 是否提前取消，影响的是无效生成成本，不能单凭这一项判断有没有严格上限。

</details>

<details class="framework-detail" id="framework-trl">
<summary>4.9 TRL</summary>

这里关注 TRL 的 **Async GRPO recipe**，不将结论扩展到它的其他 trainer。

**L1：** TRL 用 `max_inflight_tasks` 限制同时生成的 trajectory 数。worker 拿到空闲槽位后才能准入下一条 trajectory 的生成任务，生成完成便归还槽位，不需要等训练侧取样。同一个 prompt 对应的多条 trajectory 分别占用槽位。未显式配置时，默认槽位数随 `max_staleness` 和每轮训练使用的 trajectory 数计算，并至少保留一轮的容量。

**L2：** 没有根据 `max_staleness` 取消运行中的 trajectory。

**L3：** `max_staleness` 的直接检查放在 `RolloutQueueDataset` 中。每个训练 trajectory 都带有 `model_version`，训练取样时执行：

```python
version = self.model_version_fn()
staleness = version - sample.model_version
if staleness > self.max_staleness:
    logger.info(f"dropping stale sample (staleness={staleness}, max={self.max_staleness})")
    self.metrics["sample/dropped_stale_total"].append(1.0)
    continue  # drop stale, pull next
```

源码：[RolloutQueueDataset 的训练取样分支](https://github.com/huggingface/trl/blob/9d76f2f8c0421fcf8a4a72113a9e226ce796e592/trl/experimental/async_grpo/async_grpo_trainer.py#L367-L372)。

一旦超限，就丢掉这一条，继续取下一条。worker 使用的是 group 首次建立时的版本，后续会把它写入训练 trajectory，不会在生成完成时用新版本覆盖起点。

**分析：** 和 verl 的 `drop` 一样，TRL 会在 L3 排除超限 trajectory。区别是 TRL 直接比较训练版本和 trajectory 的生成版本，所以 `max_staleness=4` 就是本文定义下的 staleness 上限 4。verl 按准入训练轮次计数，其门限还要结合 `parameter_sync_step` 换算，两边的参数值不能直接类比。与 SkyRL 的告警相比，真正让 TRL 阈值生效的是最后那个 `continue`，超限 trajectory 不会进入训练 batch。

</details>

<details class="framework-detail" id="framework-prime-rl">
<summary>4.10 prime-rl</summary>

**L1：** prime-rl 的 dispatcher 用 `max_inflight - current_inflight` 检查还有多少并发空位，空位用完就暂停准入。生成完成、空位释放后，再继续准入新的 prompt。

**L2：** 对使用当前训练策略生成的 group，新权重即将发布时，dispatcher 会算出可接受的最早生成版本。group 的最早生成版本若早于这个下界，就被[逐组取消](https://github.com/PrimeIntellect-ai/prime-rl/blob/d61df5690af1ccc2019c844403f8d2f2c5ed3273/src/prime_rl/orchestrator/dispatcher.py#L404-L438)。

**L3：** 已完成的 group 由 `TrainSink` 收集成 batch。L2、L3 共用同一个函数计算可接受的最早生成版本：

```python
def min_fresh_version(step: int, max_off_policy_steps: int) -> int:
    return (step - 1) - max_off_policy_steps
```

源码：[min_fresh_version()](https://github.com/PrimeIntellect-ai/prime-rl/blob/d61df5690af1ccc2019c844403f8d2f2c5ed3273/src/prime_rl/orchestrator/utils.py#L41-L44)。

> **口径说明**：这里的 `step` 是训练轮次，对应的训练版本是 v(step−1)。减去 `max_off_policy_steps`，得到的就是允许参与这轮训练的最早生成版本。

`TrainSink` 逐条检查 trajectory 记录的最早生成版本 `policy.start`，低于下界就[从 pending_batch 删除](https://github.com/PrimeIntellect-ai/prime-rl/blob/d61df5690af1ccc2019c844403f8d2f2c5ed3273/src/prime_rl/orchestrator/train_sink.py#L151-L192)。版本推进时，已经排队的 trajectory 会重新检查，新到的 trajectory 同样接受检查。过滤后仍需凑够 batch，才会开始训练。

例如，训练版本为 v5、`max_off_policy_steps=2` 时，只接受最早生成版本不低于 v3 的 trajectory。设为 0 则只接受当前版本，和 TRL 直接计算 staleness 的口径一致。

**分析：** 相较于 TRL，prime-rl 多了 L2 的提前取消，能少做一些后续注定无法参与训练的计算。L3 继续检查生成后因排队而超限、或者取消时恰好完成的 trajectory。两层共用版本下界，L2 减少无效生成，L3 保证实际参与训练的 token 不超过 staleness 上限。

</details>

### 5. 哪些系统具有严格 Staleness 控制？

现在可以回答开头的问题了：同样是 staleness 参数，有的用于计算准入额度，有的用于超限告警，有的则真正决定哪些 trajectory 可以进入训练。按第一部分的定义，严格控制只有两种结论：能保证所有训练 token 都不超限，或者不能保证。

| 严格 staleness 控制（按本文定义） | 框架或配置 | 实现与适用条件 |
|---|---|---|
| ✓ | prime-rl、TRL | L3 根据生成版本与训练版本过滤，prime-rl 还在 L2 取消超限生成 |
| ✓ | ROLL、NeMo-RL（`windowed`、`weight_fifo`、`in_order`） | ROLL 要求 scheduler 轮次与策略版本对齐。NeMo-RL 按所选 sampler 的版本规则控制 |
| ✗ | AReaL、SkyRL、OpenRLHF、slime、NeMo-RL（`ready_first`） | 主要在 L1 限制并发或未训练工作量，统计或告警不会阻止超限 trajectory 进入训练 |
| `drop` ✓ / `wait` ✗ | verl | `drop` 在训练取样前淘汰超限 group，固定更新次数后可换算出策略版本上界。`wait` 只等待在途 group，已完成的超限 group 仍可能参与训练 |
| ✗ | Dressage | 检查不同版本数量和观察版本窗口，没有覆盖全部 token 相对训练版本的 staleness |

这些结论有一个共同前提：记录的生成版本和检查时的训练版本，需要对应第一部分的定义。对于 partial rollout，逐 token 记录版本最清楚，保留最早生成版本也可以用来约束最大值。如果只记录末尾 token 的版本，就可能漏掉前缀的 staleness。同样，trajectory 如果从未经检查的另一条路径进入训练，已有的过滤器也无法为它提供保证。

因此，比较框架时，需要同时确认启用了什么配置、trajectory 具体从哪里进入训练。严格控制回答了“最大 staleness 能否被限制住”，但还没有回答这样做是否划算。只限制未训练工作量，也可能在常见负载下表现很好。执行严格过滤，也可能因为大量丢弃而降低有效吞吐。

### 6. 三层应该怎样配合？

前面有些配置主要依靠 L1，有些采用 L1 加 L3，还有些把三层都用上了。NeMo-RL 不同 sampler 就落在不同组合中。它们的差别既体现在能否限制最大 staleness，也体现在何时停止为超限 trajectory 花费计算。

L1 限制准入量，可以减少已完成 trajectory 的积压，但无法仅凭数量发现那条一直没生成完的长 trajectory。L2 会在生成过程中取消超限 trajectory，省下剩余 token 的计算，但已经完成的 trajectory 还可能在 queue/buffer 里继续等待。L3 检查真正要参与训练的 trajectory，能够覆盖这段等待，也能处理取消时恰好完成的旧 trajectory，不过此时生成成本已经发生。

所以，在需要严格控制最大 staleness 的系统中，一个自然的组合是：**L1 控制准入量，L2 取消运行中的超限 trajectory，L3 过滤训练前的超限 trajectory。** ROLL 的 RLVR 路径、NeMo-RL 的 `windowed + native` 和 prime-rl 的相应实现已经体现了这种配合。NeMo-RL 的 `in_order` 则展示了另一种组合：L1 限制准入提前量，L3 指定每个 batch 的训练版本，不必在 L2 取消。

接下来的问题是，各层配合后能减少多少浪费？为了把最大 staleness 限制在指定范围内，又要损失多少吞吐？这些需要放到相同负载下比较。

## 第三部分：如何验证这些机制？

### 7. 把机制放进同一个实验环境

源码可以说明一个机制控制了什么，以及超限后会发生什么。但最大 staleness、丢弃比例和有效吞吐之间的关系，还取决于实际负载。例如，同样的窗口，在回答长度相近时可能很少触发过滤，遇到长回答和多轮工具任务时却可能频繁丢弃。

直接比较两个框架的端到端性能，又会同时引入推理后端、并行策略和资源配置等差异。我们更想先单独知道：固定其他条件，只替换 staleness 控制机制，会发生什么？

这里可以用 **SimRL** 做机制对照。它通过离散事件模拟 rollout 和 training 的并行过程，允许在相同的 trajectory 长度分布、rollout 资源和训练耗时下，替换不同的 L1/L2/L3 策略。

例如，先只限制未训练工作量，再加入 L3 过滤，观察最大 staleness 和丢弃成本如何变化。然后在相同条件下加入 L2 取消，比较提前结束超限 trajectory 能省下多少计算。负载中还需要加入生成耗时特别长的 trajectory，以及在取消时恰好完成的 trajectory，检查前文分析的情况是否确实发生。

### 8. 先验证实际 staleness，再计算成本

第一组实验先看配置与实际 staleness 的关系：调小参数之后，训练 token 的平均 staleness 和最大 staleness 分别下降了多少？如果配置声称提供严格上限，实际是否出现过超限 token？

这里不能简单地给所有框架都设一个“4”再比较。AReaL 和 SkyRL 把参数换算成准入额度，verl 的计数包含准入当轮，而 TRL 的 `max_staleness` 直接对应生成版本与训练版本的差。我们需要按照各自的含义扫描参数，再测量实际参与训练的 token staleness。

对每种机制，至少同时记录：

- 实际训练 token 的平均、P95、P99 和最大 staleness。
- 有效训练吞吐和训练侧等待时间。
- 生成中取消、完成后丢弃的数量及对应 token 成本。
- trajectory 中断后恢复生成时，重新处理已有前缀产生的额外计算成本。

这些指标需要一起看。一条超限 trajectory 如果被取消、没有参与训练，就不计入训练 token 的最大 staleness，但已经生成的 token 要计入浪费。反过来，训练 token 的 staleness 很低，也可能只是因为系统丢掉了大量长 trajectory。因此，staleness 下降了多少和为此浪费了多少计算，应当放在同一组实验结果中展示。

还要区分“实验中没有超限”和“机制保证不会超限”。有限次运行可能恰好没有遇到生成耗时足够长的 trajectory，严格保证仍需要结合完整的准入、生成和训练取样路径判断。实验则帮助我们发现实现中的遗漏，并测量控制成本。

### 9. 再验证控制到什么程度值得

确认机制能够控制最大 staleness 后，还需要回答一个更实际的问题：把上限从 4 收紧到 2，甚至 1，是否值得？更低的 staleness 可能改善训练，但如果大量时间花在等待和重新生成上，达到同等质量的总耗时未必更短。这需要通过收敛实验判断。

可以固定初始 checkpoint、采样参数和资源配置，在相同训练 token 预算或优化步数下，比较 reward、验证集表现、KL、clip fraction 和 importance ratio 分布。同时记录达到同等质量所需的实际时间，观察训练质量与有效吞吐之间的取舍。

还需要检查过滤后的 trajectory 构成。长回答和多轮工具任务更容易超过 staleness 阈值，留下来的 trajectory 可能偏向短、快的生成。如果训练质量发生变化，就要进一步区分：变化来自更低的 staleness，还是来自训练 trajectory 组成的改变？

这些实验尚未完成，本文暂时只给出机制分析和验证思路。阈值应该设成多少，以及不同机制能带来多大的收益，留待后续实验回答。

## 最后，回到那个参数

回到最初的 `max_staleness=4`，理解它需要沿着一条 trajectory 看清三件事：这个 4 限制的是准入量，还是 staleness？生成版本在哪里记录，又在哪里与训练版本比较？超过阈值后，这条 trajectory 还会不会参与训练？

如果参数只用于计算准入额度，它能够减少积压，但无法仅凭这项限制保证最大 staleness 不超过 4。对于采用过滤的机制，生成版本与训练版本的检查需要覆盖所有参与训练的 trajectory，并确实排除超限内容，这个阈值才落实到了本文定义的最大 staleness 上。

沿着 **L1 准入控制、L2 Rollout 运行时控制、L3 Training 前过滤** 这三层看实现，我们就能把参数名还原成具体行为，也能进一步讨论：系统控制住了多少 staleness，又为此付出了多少计算成本。
