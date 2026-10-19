# 实践中的陈旧度控制：十二个系统的调研

> 本文是 `staleness-in-practice.md` 的中文翻译与分类整理版。为方便核查，代码、
> 参数名、文件路径和行号均保持原样；内容则按控制机制重新编排。

## 1. 文档目的

SimRL 在某种*陈旧度策略*（staleness policy）下模拟 rollout/训练循环
（参见 `docs/simulation/policies.md`）。每个内置策略都声称自己刻画了一个真实系统，
所以这些策略是否可信，取决于我们对相应系统实现的理解是否准确。本文记录受调研框架
实际实现的机制，给出精确到源码行的引用，并提炼它们的共同结构。

本文位于 `docs/design/`，该目录不会进入 Sphinx 构建（`docs/conf.py:54`）；
它是一份工作笔记，而不是已发布文档。

简而言之：**引用最广泛的几个系统——verl 的同步模式、AReaL 和 SkyRL——都把陈旧度参数
实现为“尚未消费的工作量上限”，而不是“任一样本的年龄上限”。** 但这并非所有系统的共同
做法。在所调研的十二个系统中，有七个确实实施了真正的逐样本年龄限制，其中一个
（prime-rl）甚至会在生成尚未完成时，针对仍在进行中的工作实施该限制。这个设计空间
更适合用两个维度描述，而不能简单排成一条“约束有多严格”的直线；详见“二维分类框架”
一节。

下述结论均来自源码阅读，而非仅依据文档。若某系统的文档或字段帮助文本与代码冲突，
本文会同时引用二者并明确指出差异。外部仓库均按引用的版本以 tarball 形式从
`codeload.github.com` 拉取；具体方法见下文“如何复现本次调研”一节。

## 2. 调研范围与证据深度

本文对七个系统进行了端到端阅读，另外五个系统只核查了陈旧度参数及其执行路径。
verl 的两种模式位于同一仓库，但彼此是互不相关的机制，因此分别阅读并单独引用。
除另有说明外，所有版本截至 2026-10-10。verl v1、AReaL、ROLL、SkyRL、slime、
Dressage、NeMo-RL、TRL、prime-rl 和 OpenRLHF 已于 2026-10-16 针对下表所列
commit 完成核查校正（完整记录见 `publishing/源码复核记录.md`）；其中 ROLL、
Dressage、TRL、prime-rl 和 NeMo-RL 在本文中的具体修正内容，已于 2026-10-19
独立重新抓取并再次核查。

| 系统 | 版本 | 调研深度 |
|---|---|---|
| verl（legacy 同步模式） | `0.9.0.dev`，commit `e5ca4acb` | 完整（本地 checkout：`psrl/third_party/verl`） |
| verl v1（`trainer.v1.trainer_mode=separate_async`） | `verl-project/verl@5b550b0c` | 完整（源码 tarball） |
| AReaL | `areal-project/AReaL@6cad8710` | 完整（源码 tarball） |
| ROLL | `alibaba/ROLL@d96d0cce` | 完整（源码 tarball） |
| slime | `THUDM/slime@2f4280e3` | 完整（源码 tarball） |
| SkyRL | `NovaSky-AI/SkyRL@38146ee9` | 完整（源码 tarball） |
| Dressage | `Accio-Lab/Dressage@3e3142fe` | 完整（源码 tarball） |
| prime-rl | `PrimeIntellect-ai/prime-rl@d61df569` | 完整（源码 tarball） |
| TRL | `huggingface/trl@9d76f2f8` | 完整（源码 tarball） |
| NeMo-RL | `NVIDIA-NeMo/RL@726d5eb2` | 完整（源码 tarball） |
| OpenRLHF | `OpenRLHF/OpenRLHF@dc2a7ad3` | 已核查参数及执行路径 |
| Trinity-RFT | `agentscope-ai/Trinity-RFT@main` | 仅参数，由子调研完成 |

Trinity-RFT 这一行来自委派给其他 agent 的调查，本文如实保留其来源；
这一结论尚未逐行复核，在论文中引用前应再次验证。

## 3. 分类总览

### 3.1 按“约束了什么”分类

1. **约束未消费工作量**
   - 信用累积式租约：AReaL、SkyRL
   - 占用量重测式租约：verl
   - 退化的计数租约：OpenRLHF
   - 容量与流水线深度：Trinity-RFT
2. **约束逐样本年龄**
   - 生成后、消费时丢弃：ROLL、Dressage、TRL
   - 生成过程中响应式取消：prime-rl、NeMo-RL 的 `windowed` sampler
3. **不直接设置陈旧度参数**
   - 由流水线结构决定：slime

### 3.2 按“何时执行”分类

- **事前（ex-ante）**：通过限制准入或取消已不可能及时完成的进行中任务，尽量减少
  无效计算。
- **事后（ex-post）**：样本已经生成，再在组 batch 或消费时丢弃；能够硬性限制年龄，
  但会浪费已经付出的计算。
- **结构性（structural）**：不比较样本年龄，由流水线深度、同步点或队列容量间接决定
  陈旧程度。

## 4. 第一类：未消费工作量租约

### 4.1 共同结构

引用最广泛的三个系统——AReaL、SkyRL 和 verl——都用同一类不等式控制准入；
OpenRLHF 则实现了它的退化形式。记：

- $A(t)$：累计已准入且已完成的单元数；
- $R(t)$：当前正在生成的单元数；
- $v(t)$：训练版本；
- $B$：训练 batch size；
- $S$：配置参数。

则准入条件为：

```text
A(t) + R(t)  <=  (S + v(t) + 1) * B
```

由于 $v(t)B$ 恰好是训练到当前为止已消费的工作量，可改写为：

```text
outstanding(t)  =  A(t) + R(t) - v(t)*B  <=  (S + 1) * B
```

这里的版本项并不是样本年龄，而是一个恰好每个训练 step 增长一次的*消费计数器*。
因此，该机制实际上是：

> 预先装入 $(S{+}1)$ 个 batch 的生成任务；此后训练每消费一个 batch，就再释放一个
> batch 的准入额度。

本文将这一不变量称为**未消费工作量租约**（unconsumed-work lease）：调度器最多放出
$(S{+}1)B$ 个尚未被训练消费的任务单元，每个训练 step 再续租一个 batch。这里不使用
“capacity”一词，是因为 SimRL 已经用它表示 KV-cache 容量；二者是不同且会同时
生效的约束。

### 4.2 为什么租约不能限制陈旧度

租约限制的是未完成单元的数量。根据 Little 定律，在到达率为 $\lambda$ 时，它会限制
一个单元处于未完成状态的*平均*时间：$W=L/\lambda$。然而，陈旧度关心的是分布的
*尾部*。

一个长时间运行的请求只占租约中的一个单元，却可以无限期占据该槽位；其余
$(S{+}1)B-1$ 个单元仍可不断循环新任务。上述不等式不会随该请求变老而增大，因此也
不会触发驱逐。该请求最终完成时，无论落到哪个训练版本，都会被消费并用于训练。

这不是隐蔽的边界故障，而是公式本身规定的行为；SkyRL 的源码也明确承认了这一点。

### 4.3 两种租约子类型

**信用累积（AReaL、SkyRL）**

状态是一个只增不减的累计计数器。若训练停顿，信用桶会耗尽，生产者将彻底停止。

**占用量重测（verl）**

上限是没有版本项的常数 $M$。系统不累积信用，而是在每次权重同步时，把计数器重置为
当前实际占用量。

两者在稳态下一致，在训练停顿时则会分化：AReaL 的生产者会干净地阻塞；verl 会持续
重测，并依赖有界交接队列。该队列溢出时会驱逐**最旧**条目，因此当系统最为落后时，
verl 可能恰好丢掉最接近被消费的任务。

### 4.4 AReaL：信用租约

`areal/infra/staleness_manager.py:109`：

```python
staleness_capacity = (ofp + current_version + 1) * consumer_bs - sample_cnt
```

其中：

```python
sample_cnt = self.rollout_stat.accepted + self.rollout_stat.running
```

`accepted` 在 `on_rollout_accepted` 中递增，并且从不递减。系统在
`areal/infra/workflow_executor.py:803,811` 中以整数个数消费额度：

```python
capacity = min(cap_staleness, cap_queue)
...
for _ in range(min(batch_size, capacity)):
```

对应参数是 `max_head_offpolicyness: int = 0`
（`areal/api/cli_args.py:2859`）。帮助文本写道：
“如果当前版本比请求领先超过这么多个版本，则不接受该请求。”
**代码中不存在这种检查。** 在整个仓库中搜索样本生成版本与当前版本的比较，只能找到
`cli_args.py:3036` 的一条 `deterministic_sampling` 告警。

AReaL 没有 L2 陈旧度控制。权重同步从不会因为陈旧度而中断正在进行的生成；上面的
`running` 计数只有在请求本身完成时，才会转入 `accepted`。

AReaL 确实使用逐 token 版本，但它用于**目标函数**，而不是准入控制。
`areal/trainer/ppo/actor.py:1414-1420` 计算：

```python
alpha = (v_proximal - v_behave) / (v_theta - v_behave)
```

该值用于解耦 PPO 中近端 log-probability 的插值。其 `RejectionSamplingConfig` 按
*策略分歧*（ratio/KL）而非年龄过滤。因此，AReaL 的整体立场是自洽的：它不按年龄
限制准入，而是在 loss 中修正年龄带来的偏差。

### 4.5 SkyRL：信用租约，并明确承认缺口

`skyrl/train/fully_async_trainer.py:184`：

```python
consumer_capacity = (self.max_staleness_steps + self._current_global_step) * self.mini_batch_size
producer_staleness_capacity = consumer_capacity - (self._stat.accepted + self._stat.running)
```

除 `+1` 外，这与 AReaL 完全相同。差异仅仅来自语义：SkyRL 的
`_current_global_step` 表示正在处理的版本，而不是已完成版本的数量。源码还注明了
机制来源：“Modeled after AReal's StalenessManager … The idea of this controller is
from section 5.1 of AReal's paper.”

对应参数为 `max_staleness_steps: int = 4`
（`skyrl/train/config/config.py:1035`）。

SkyRL 是本次调研中最有价值的引用，因为其类 docstring
（`fully_async_trainer.py:119-127`）直接说明了限制：

> 该约束通过容量规则实施，并不对每个 group 的陈旧度提供硬保证。系统限制领先于训练的
> group 总数，使陈旧度在稳态下保持在 `max_staleness_steps` 预算内。
>
> 在病态情况下（例如运行时间极长的 trajectory），单个 group 的生成过程可能跨越超过
> `max_staleness_steps` 个训练 step。对于这些少见情况，系统仍然接受该 trajectory，
> 只记录陈旧度指标并发出告警。

SkyRL 甚至会计算并比较每个 group 的年龄，却明确选择不采取行动。
`fully_async_trainer.py:1038-1065`：

```python
cur_staleness = self.global_step - cur_generated_output_group.global_step_when_scheduled
...
generator_outputs.append(cur_generated_output_group.generator_output)   # line 1041
...
if cur_staleness > self.max_staleness_steps:                            # line 1061
    # TODO(Charlie): should we drop, drop and resample, or just log?
    logger.warning("Staleness control violated despite using AsyncStalenessManager: ...")
    staleness_violation_count += 1
```

该 group 在第 1041 行先被加入训练 batch，第 1061 行才检查是否违规。违规分支既没有
`continue`，也没有移除操作，只有一个用于
`async/staleness_violation_count` 指标的计数器。SkyRL 文档列出了“丢弃”和
“丢弃后重采样”两个替代方案，但都标为“尚不支持”。

在六个端到端调研的系统中，SkyRL 对设计权衡的表述最清楚：

> 另一种办法是过量生成，并在训练时丢弃过于陈旧的 group。我们更倾向主动准入控制，
> 以避免浪费计算，并简化 buffer 管理。

### 4.6 verl（legacy 同步模式）：占用量租约

`verl/experimental/fully_async_policy/fully_async_rollouter.py:431`：

```python
self.max_required_samples = int(
    self.required_samples * (self.staleness_threshold + 1)
    * self.config.async_training.trigger_parameter_sync_step
)
```

`_should_pause_generation` 中有两个计数检查（`:988`、`:996`）：一个比较队列深度与
`max_queue_size`，另一个比较 `staleness_samples` 与 `max_required_samples`。
每次权重同步时，计数器不是获得新增信用，而是按实际占用量重新测量（`:475`）：

```python
self.staleness_samples = len(self.active_tasks) + await self.message_queue_client.get_queue_size()
```

交接队列是一个 `deque(maxlen=...)`，溢出时执行 `popleft()`，即 FIFO 驱逐；
如果队头恰好是新鲜样本，被丢弃的就会是该新鲜样本。

在租约类系统中，verl 对参数语义的描述最诚实。`staleness_threshold: float = 0.1`，
README 将其定义为“允许使用的陈旧样本的最大比例”——它表示比例，而不是年龄。

verl 在训练时同样不按年龄过滤。逐样本标签 `min_global_steps` /
`max_global_steps` 在 `verl/trainer/ppo/v1/agent_loop_tq.py:191` 引入，注释写道：
“这些标签用于 off-policy 陈旧度控制；若一条 trajectory 跨越过多 global step，
就需要将它过滤掉。”然而，除测试外，它们唯一的使用方是
`detach_utils.py:151-159`，这里只计算 `partial_ratio`、`max_partial_span` 和
`param_version_diversity` 用于日志记录。**不存在过滤器。**

### 4.7 verl v1（`separate_async`）：按步补给的准入，一个真正的年龄过滤器

通过 `trainer.v1.trainer_mode=separate_async` 启用，这是与上面的 legacy 同步模式
完全不同的机制，位于同一个仓库中。

**L1。** 准入跟随训练进度，而不是未消费工作量计数器。`num_warmup_batches`
（支持小数）设置训练开始前额外预取多少批 prompt；之后每一轮训练都会在此基础上再
准入一个 `train_batch_size`（`verl/trainer/ppo/v1/trainer_base.py:624-651,1520-1527`，
commit `5b550b0c`）。以 `train_batch_size=4, num_warmup_batches=1.5` 为例：启动时
额外准入 6 个 prompt，首轮再准入 4 个（合计 10 个），之后每轮再准入 4 个。

**L2。** 无。

**L3，真正生效的路径。** 每个 prompt 准入时记录当时的 `global_steps`
（`prompt_global_steps`，按 uid 索引）。训练取样时，`ReplayBufferAsync` 将当前
`global_steps` 与其比较，分两种策略：

`drop` 在已完成的 group 被取样前，淘汰超过阈值的 group（`replay_buffer.py:520-539`）：

```python
if global_steps - prompt_global_steps.get(uid, global_steps) + 1 > self.max_off_policy_threshold
```

这里的 `+1` 把准入当轮也算进去。若每个训练轮次只做一次更新，`drop` 的 staleness
上限是 `max_off_policy_threshold - 1`。`parameter_sync_step > 1` 会在同一个
`global_steps` 内连续做多次取样和更新，把 batch 拆成共用同一个版本的多个
minibatch——它不会放大这个上限；把它折算进去后，保守上界是
`parameter_sync_step * max_off_policy_threshold - 1`。

`wait` 只在某个仍在生成的 group 达到阈值时阻塞取样（`replay_buffer.py:541-556`），
给它留出完成的时间；已经完成、超过阈值的 group 永远不会被淘汰。本轮没取走的已完成
group 并未丢失——下一轮取样会优先选择更早准入的 group，所以它会是下一轮第一个被
取走的，此时它又多陈旧了一个版本。`wait` 自己的测试
（`tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py:676-691`）印证了这一点：在没有
仍在生成的 group 时，一个已完成、超过阈值的 group 仍会被返回。

**分析。** `drop` 在取样前淘汰，其阈值能直接折算成每个训练 token 的 staleness
上界，因此按本文定义记为严格控制。`wait` 只会阻塞在途工作，永远不会淘汰一个已完成
但超限的 group，因此不能阻止它最终在任意晚的版本上参与训练。将
`max_off_policy_threshold` 设为 `None` 会同时关闭这两种 L3 策略，只剩 L1 作为
唯一的界限。

### 4.8 OpenRLHF：退化的计数租约

OpenRLHF 提供 `--train.async_queue_size: int = 1`，实现方式是字面意义上的计数信号量。
系统完全不记录逐样本版本，因此陈旧度只能由队列深度间接限制。它可视为一个
$S$ 固定为队列大小的退化租约。

### 4.9 Trinity-RFT：深度与容量约束

Trinity-RFT（`agentscope-ai/Trinity-RFT`）提供：

- `sync_offset: int = 0`：结构性约束；
- `max_inflight_batches: int = 2`：容量约束。

它在 synchroniser 层仔细跟踪 `model_version`，却不把版本附加到单条 experience 上，
因此无法执行年龄比较。这是“有跟踪、无比较”模式最纯粹的实例。

## 5. 第二类：真正的逐样本年龄限制

### 5.1 生成后执行：ROLL

ROLL 是端到端调研系统中第一个真正限制年龄的系统。它的参数为
`async_generation_ratio: float = 0`（`roll/configs/base_config.py:552`），并在
两个不同调度器中承担两种用途。

在 agentic 路径（`roll/distributed/scheduler/rollout_scheduler.py`）中，它既表示
流水线深度，也表示年龄上限：

```python
for _ in range(self.async_generation_ratio):          # line 277  - depth
    ...
if step - group.create_step > self.async_generation_ratio:   # line 283  - age
    expired_episodes.append(episode_id)
```

过期 group 会被永久弹出。`GroupQueueManager.get_batch` 在获取 batch 时也执行相同
比较，而 `put()` 会丢弃已经被驱逐的 group 所返回的结果。`GroupData.create_step`
表示 group 的出生 step，所以 `step - create_step` 是以训练 step 计量的真实年龄。

在 RLVR 路径（`roll/distributed/scheduler/generate_scheduler.py`）中，首轮额度是
`ceil(B*r) + B`（B 为 batch size，r 为比率），在 `advance_step` 中一次性设定
（`:244-249`）：`self.batch_size = math.ceil(batch_size * self.async_generation_ratio)`，
随后 `self.batch_size += batch_size`。之后每一轮只再加一个 `B`。以 `B=32, r=2`
为例：首轮额度 96，之后每轮再加 32。这个窗口在取样时（`min_step`，`:358`）和主动
GC（`max_gc_step`，`:425`）两处都会生效。

**RLVR 路径要求 `r >= 1.0`，这是在配置时强制校验的，不只是建议。**
`roll/pipeline/rlvr/rlvr_config.py:222`：`assert self.async_generation_ratio >= 1.0,
"async_generation_ratio must be >= 1.0"`。`generate_scheduler.py` 自己只断言
`>= 0`，因为它与下面的 agentic 路径共用，而 agentic 路径没有这个下限。真正收紧这个
下限的是 RLVR 自己的 pipeline 配置，在 scheduler 看到小于 1.0 的值之前就已经拦住了。

**ROLL 在 RLVR 路径中有意支持小数值。** 小数部分决定 batch 中可以有多大比例来自
最老的合格 cohort（`:376-379`）：

```python
if self.async_generation_ratio % 1 == 0:
    expected = expected_samples - collected_samples
else:
    expected = int(expected_samples * (self.async_generation_ratio % 1))
```

窗口边界则由 `math.ceil(self.async_generation_ratio)` 设置。这是一种真正的小数陈旧度
语义：$S=1.5$ 表示“允许回退一个完整 step，再从前两个 step 的 cohort 中取半个
batch”。

但两个路径存在不对称：agentic 的 `rollout_scheduler.py` 在第 277 行把同一字段传给
`range()`，它要求整数；若传入小数，会抛出 `TypeError`。该字段声明为 `float`，
且除 `base_config.py:727` 中
`assert async_generation_ratio == 0 or generate_opt_level == 1` 外没有
`__post_init__` 校验。因此，实际行为取决于运行的是哪个调度器。

### 5.2 生成后执行：Dressage

Dressage 基于 slime 构建，并加入其 README 所称的
**“True Staleness Control（真正的陈旧度控制）”**。一个基于 slime 的框架认为有必要
增加该功能，甚至特意称之为“真正的”控制，本身就说明业界已经认识到本文所讨论的缺口。

`dressage/rollout/staleness.py`：

```python
@dataclass(frozen=True)
class StalenessConfig:
    keep_versions: int | None = None          # line 13

@property
def cutoff_version_index(self) -> int | None:
    return max(0, len(self.versions) - int(self.config.keep_versions))   # line 100

def should_drop_group(self, group: list[Any]) -> bool:                   # line 121
    cutoff = self.cutoff_version_index
    if cutoff is None:
        return False
    return any(self.version_index(info.version) < cutoff
               for info in trajectory_version_infos(group))
```

`trajectory_version_infos` 做的事情比上面这段 `any(...)` 看起来要多：对 group 里每个
不同的 trajectory key，它先把所有分段折叠成 `segment_index` 最大的那一段，再读取
那一段自己的版本（`dressage_end_token_version`）。所以 `should_drop_group` 实际
比较的是每条 trajectory 的**最后一个可训练 token 的版本**与截止线，对 group 里的
每条 trajectory 各比较一次，而不是对观察到的所有分段做一次原始扫描。一条
trajectory 即使第一段很旧，只要最后一段够新就能通过；反过来，哪怕它早期的分段都很
新，只要它自己的最后一段落后了，也会被淘汰。

其中有三项设计值得注意：

1. **版本按观察顺序跟踪。** `observe_group`（`:111`）会按遇到顺序追加版本，既不排序，
   也不做减法。`keep_versions` 统计的是*已观察到的不同版本*，而不是 step 数。这是
   本次调研中唯一一个整数类型在语义上确有必要、而不只是 `range()` 所致的系统。
2. **丢弃粒度是 GRPO group。** 文档明确解释，删除单条 trajectory 会扰动组内相对
   advantage 的归一化。
3. **被丢弃的 group 会被补齐，而不是让 batch 缩水。** collector 的外层循环
   （`generate_rollout_async`，`fully_async_rollout.py`）是一个朴素的
   `while len(data) < target_data_size`：被 `keep_group` 丢弃、或者在版本前进后被
   `filter_pending` 重新检查丢弃的 group，根本不会被加入 `data`，循环会持续拉取
   新完成的 group，直到凑够目标数量，无论路上丢了多少个。这与 ROLL 的重采样是
   同一种形状，不是本文之前认为的“batch 简单缩水”。

Dressage 还限制*单条 trajectory 内部*的版本混合，这是其他受调研系统都没有的。
在 partial rollout 中，同一条 trajectory 可能跨权重更新被中断和恢复；代理通过
`--max-partial-rollout-preempts` 实施限制，其中
`version_switches = max(0, version_span - 1)`。超限时返回 HTTP 502：
`partial_rollout_staleness_exceeded`。

过滤发生在 batch 组装阶段，因此陈旧 group 被丢弃前已经消耗了 GPU 时间。这些计算
会被浪费而非事先阻止；这正是 SkyRL 的准入控制设计试图避免的成本。

### 5.3 生成后执行：TRL

TRL（`huggingface/trl`，`trl/experimental/async_grpo/`）使用
`max_staleness: int = 4`。帮助文本为：“一个 rollout 样本最多可以落后当前模型版本
多少个权重更新 step，超过后将被丢弃。”

该参数在消费时执行（`async_grpo_trainer.py:362-367`）：

```python
staleness = version - sample.model_version
```

随后通过 `continue` 丢弃。

`sample.model_version` 只在 `RolloutGroup` 第一次创建时被标记一次
（`async_rollout_worker.py`，`pending_groups[group_id] = RolloutGroup(...,
model_version=self.model_version, ...)`，由 `if group_id not in pending_groups`
把关），此后只是把这个值从 group 原样传给每条单独的 sample
（`model_version=group.model_version`），生成完成时从不重新读取当时的实际模型版本。
一个要跨越好几次权重同步才能生成完的 group，staleness 是按它*开始*时的版本算的，
不是它最终落地时的版本。

值得注意的是，它的容量参数由年龄参数*推导*而来：`max_inflight_tasks` 默认为
`max_staleness * samples_per_step`。其明确理由是：如果生成更多样本，多出来的部分
最终会因陈旧而被丢弃，从而浪费计算。这个推理方向与租约类系统相反。

### 5.4 生成过程中执行：prime-rl

prime-rl（`PrimeIntellect-ai/prime-rl`）实施了本次调研中最严格的限制：

```python
max_off_policy_steps: int = Field(8, ge=0)
min_fresh_version(step, S) = (step - 1) - S
```

定义位置为 `orchestrator/utils.py:42-45`。这里的 `step` 参数指的是即将运行的训练步，
不是它所使用的版本：第 `n` 步训练使用的是第 `n-1` 步结束后发布的策略，即版本
`v(n-1)`，所以这个下界的含义是“任何在 `v(n-1) - S` 之前派发的 episode 都不能参与
这一批训练”。关键在于，
`dispatcher.py:427-439` 会取消**仍在生成**的 group：

```python
stale_groups = [... if group.policy_version_at_start < min_version]
for gid in stale_groups:
    cancelled += await self.drop_group(gid, reason="stale")
```

`drop_group` 会取消进行中的 asyncio task，并释放其 permit。

prime-rl 同时还用容量上限 `max_inflight` 控制准入，因此它会同时实施工作量上限和
年龄上限。此外，它把陈旧度拆分为生成时间与队列时间：

- `off_policy/in_flight/*`：生成中的陈旧度；
- `off_policy/in_queue/*`：排队中的陈旧度。

SimRL 可以直接复现这一划分。

### 5.5 可插拔执行：NeMo-RL

NeMo-RL（`NVIDIA-NeMo/RL`）把陈旧度暴露为可插拔策略。它使用一个带判别字段的
联合类型，包括：

- `windowed`
- `ready_first`
- `weight_fifo`
- `in_order`

每种 sampler 都有 `max_staleness_versions: NonNegativeInt = 1`。

**真实系统的默认值是 `in_order`**（`single_controller_utils/config.py`：
`sampler: SamplerConfig = Field(default_factory=InOrderSamplerConfig)`），不是
`windowed`——这里需要明确说出来，因为 `windowed` 在 L1、L2、L3 三层都有动作，篇幅上
得到的讨论最多，容易让人误以为它是默认值。

其中，`in_order` 只表示流水线深度；`windowed` 则是纯年龄窗口，并提供显式的
`should_abort_inflight()`。背压参数 `max_inflight_prompts` 和
`max_buffered_rollouts` 被有意与陈旧度参数分离。这是与 SimRL 自身可插拔策略
设计最接近的外部系统。

## 6. 第三类：由结构决定陈旧度

### 6.1 slime：没有陈旧度参数

slime 确实没有 staleness *参数*，但它在准入侧有一个水位控制。全异步 object-store
worker 把并发预算（并发回答数除以每个 prompt 的回答数）换算成 group 上限，还可以用
预取系数扩大这个上限（`slime/rollout/fully_async_rollout.py:199-223`）。准入时同时
检查活跃 group 数是否达到这个上限，以及完成 queue 的深度是否达到同一个水位：只要
任一个满了，准入就暂停，直到训练侧把 queue 排空。以 64 条并发回答、每组 8 条、
预取系数为 1 为例，这将活跃生成上限设为 8 组，并在完成 queue 也达到 8 组时暂停准入。
queue 本身无界，所以一个已经在运行的 trajectory 仍能完成并把 queue 推过水位，这里
暂停的只是新的准入。这限制的是*有多少未消费的工作量*，与 AReaL/SkyRL/verl 的 lease
是同一种形状，不是任何一个样本的年龄。

`Sample.weight_versions` 在整个仓库中有五处读取：

- 一处指标统计；
- 两处*排序键*；
- 一处遥测字段；
- 一处写入对应的读取。

搜索任何对版本或陈旧度数值应用比较运算符的代码，都没有结果。文档也两次明确说明
这一后果，例如 `docs/en/advanced/rollout-scheduling.md`：
“排序不会自动丢弃陈旧样本。”

除水位控制之外，陈旧度仍由循环屏障自然产生：生成在权重推送前分派、在权重推送后
消费，因此上限由流水线形状决定。真正影响 off-policyness 的参数是
`--flush-cache-interval`（`slime/utils/arguments.py:350`，`int`，默认值 1）。该参数
决定在权重更新时，进行中的请求是被中止，还是保留 KV 并原地暂停。

另外有两项与 SimRL 成本模型有关的发现：

1. slime 的恢复逻辑在预算上是正确的：`max_new_tokens -= response_length`，
   因此不存在 PSRL 中“恢复时重新获得完整新预算”的行为。
2. 当 `flush-cache-interval <= 0` 时，KV 会跨权重更新原地保留，所以 partial
   rollout **不需要**重新支付 prefill 成本。

## 7. 横向对照

| 系统 | 参数 | 类型 | 支持小数 | 实际约束 | 执行方式 |
|---|---|---|---|---|---|
| slime | *无* | — | — | 流水线形状 | 结构性；版本只用作排序键 |
| OpenRLHF | `async_queue_size` | `int` = 1 | 否 | 未完成工作量 | 信号量；不记录版本 |
| Trinity-RFT | `sync_offset`、`max_inflight_batches` | `int` = 0 / 2 | 否 | 深度 + 工作量 | 结构性；记录版本但不比较 |
| AReaL | `max_head_offpolicyness` | `int` = 0 | 否 | 未消费工作量 | 信用租约；在 loss 中修正年龄；无 L2 |
| verl（legacy） | `staleness_threshold` | `float` = 0.1 | **是** | 未消费工作量 | 占用量租约；溢出时 FIFO 丢弃 |
| verl v1 | `max_off_policy_threshold` | `int` = 8 | 否 | 年龄（`drop`）或无（`wait`） | `drop` 在取样前淘汰；`wait` 只阻塞在途工作 |
| SkyRL | `max_staleness_steps` | `int` = 4 | 未校验 | 未消费工作量 | 信用租约；仅记录违规，不执行丢弃 |
| ROLL | `async_generation_ratio` | `float` = 0 | **是**（RLVR 路径） | 深度 **和** 年龄 | GC + 消费下限；重采样 |
| Dressage | `..._keep_versions` | `int` = 0 | 否 | 年龄（组间 + 轨迹内） | 组装时丢弃；不重采样 |
| TRL | `max_staleness` | `int` = 4 | 否 | 年龄 | 消费时丢弃；容量由年龄推导 |
| NeMo-RL | `max_staleness_versions` | `int` = 1 | 否 | 年龄或深度（可插拔） | 窗口驱逐；可选中止进行中任务 |
| prime-rl | `max_off_policy_steps` | `int` = 8 | 否 | 工作量 **和** 年龄 | **取消进行中任务** + 容量门控 |

## 8. 二维分类框架

用单一维度衡量“约束有多严格”无法解释上述证据。更合适的是两个正交维度：

1. **约束在哪里执行**
   - 事前（ex-ante）：限制准入，或在样本完成前取消；
   - 事后（ex-post）：丢弃已经生成的工作。
2. **约束的对象是什么**
   - 未完成工作的*数量*；
   - 单个样本的*年龄*，即版本差。

```text
                     约束工作量                         约束样本年龄
事前（ex-ante）      AReaL、SkyRL、verl（legacy）、     prime-rl（取消进行中任务）、
                     OpenRLHF、Trinity-RFT              NeMo-RL（windowed sampler）

事后（ex-post）      --                                 ROLL、Dressage、TRL、verl v1（drop）
```

右上角并非空集，这是对直觉化叙事最重要的修正：完全可以在计算完成前限制年龄。
prime-rl 和 NeMo-RL 的 `windowed` sampler 都会取消已经开始、但不可能及时完成的任务。

这种做法要求系统在新权重版本到达时，响应式地检查仍在运行的请求是否已经过期。
两个系统都不会在分派时预测请求能否保持新鲜，而是等待它已经明确失败后再处理。

SimRL 的价值恰好体现在这里。响应式取消仍要支付取消点之前全部生成工作的成本；
它只能减少沉没成本，不能消除沉没成本。若要在*准入时*判断哪些请求在完成时仍会新鲜，
就需要预测完成时间分布——模拟器能够提供这种信息，而在线系统很难低成本获得。

各系统明确给出了两种彼此冲突的设计理由：

- SkyRL：“我们更倾向主动准入控制，以避免浪费计算。”
- TRL 也以避免浪费为理由，但从年龄上限反推容量上限，使很少有样本能够老化到需要
  丢弃。

二者都希望减少同一种浪费，分歧在于应该固定哪一个量。

## 9. 对 SimRL 的启示

上文调研的十二个系统，现在在 `srl/policy/` 下都有对应的 `StalenessPolicy`；所有
会在运行时取消任务的策略共用的分层 `Cancel`/`on_publish` 机制，已经完全取代了早期
更狭窄的 `expire()` hook，因此不再存在“某个系统还缺一个尚未实现的 hook”这类逐系统
空缺。

- `ArealPolicy`（`srl/policy/areal.py`）精确复现了 AReaL 的公式，包括
  `accepted + running` 项，并声明不存在 L2。策略文档称它“不限制单个样本的年龄”是
  正确的；更准确的说法是，AReaL 中根本不存在任何逐样本年龄检查。
- `RollPolicy` 的消费下限和由 `on_publish` 触发的 GC 对应真实的上游机制，并非模拟
  中的自由发挥——包括 RLVR pipeline 配置强制要求的 `r >= 1.0` 下限，而这个模拟器
  目前并未在自己构造 `RollPolicy` 时对此做断言。这也解释了为什么把 SCG 实验绑定到
  `policy=roll` 后，会暴露其他策略隐藏的浪费 token：ROLL 是本次调研中会丢弃过龄
  工作的几个系统之一，也是本项目最先实现的一个。
- `DressagePolicy`（`srl/policy/dressage.py`）已经把窗口的 key 设为每个 group 最后
  一个可训练 token 的版本（`last_version = max(req.v_max for req in group)`），并且
  在每次 L3 丢弃时已经会发放一个补采样额度（`replacement_groups_on_drop` 返回
  `len(groups)`）——这两点都与上文修正后的真实系统行为一致。本文此前的版本曾错误
  地认为 Dressage 丢弃后不会重采样；实际上它会，而这个策略在这次修正之前就已经正确
  建模了这一点。
- 忠实的 **SkyRL** 策略应当是去掉 `+1` 的 `areal`，再加一个不改变行为的违规计数器；
  `SkyRLPolicy` 实现的正是这个。
- `VerlV1Policy` 实现了上文修正后的 `drop`/`wait` 区分：`drop` 的阈值能折算成严格的
  逐 token 上界，`wait` 只阻塞仍在生成的工作，因此被标记为非严格。
- slime 根本不应使用基于年龄的策略建模；`SlimePolicy` 的上界来自上文所述的循环屏障
  形状加准入水位，而不是任何版本比较。
- `PrimeRLPolicy` 把一个固定的 inflight 上限与一个 dispatch-lag 准入闸门结合起来，
  再加上共享的 `CancelBelowVersion`/`VersionWindow` 原语，分别用于取消进行中任务和
  清扫 ready pool——这正是本文过去认为“无法实现”的“租约加响应式取消”组合。
- 值得采用 prime-rl 把陈旧度拆分为生成时间与排队时间的做法，作为一个上报指标：它
  能区分“请求本身很慢”和“请求等待过久”，而当前单一的陈旧度数字把二者混在一起。
  `StalenessReport` 还没有做这个拆分。

## 10. 如何复现本次调研

集群禁止使用 GitHub code-search API 和 `git clone`，但允许通过
`codeload.github.com` 使用 `curl`：

```bash
mkdir -p /tmp/src && curl -sL \
  https://codeload.github.com/<owner>/<repo>/tar.gz/refs/heads/main | tar xz -C /tmp/src
```

下载 tarball 后在本地搜索，比逐文件网页抓取更快也更可靠；后者会截断长文件，并用摘要
替代原文引用。
