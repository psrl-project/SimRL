# 异构 TP + 长度分层路由：经验、优势来源与边界

Qwen3-8B / H20 / composed cost model / psrl 策略，2026-10-11 至 10-12 的一组仿真研究。
目标：在长尾 workload 上，让异构 TP 的 span 优于**同构配置在最优 cap 下**的 span。

> ⚠️ 状态：代码在 `/tmp/capstudy/snap` 快照中，未进仓库；TP4 / TP8 kernel 表在 `/tmp/hetero/profile`。

---

## 1. 结论速览

- **能赢，但有条件**：在"大量中等长度请求 + 少数特别长 straggler"的 workload 上，
  `[TP4, TP1×4]` + 长度分层路由比同构最优 cap 快 **1.20–1.29x**（S≤1、训练够快）。
  同构基线扫遍 TP1×8 / TP2×4 / TP4×2 的 cap，最优点在网格内部。
- **必须有 TP4**：只用 TP1/TP2 时，异构相对"同构 + 同样聪明的调度"只多约 1%。
- **S=2 输**（0.93–0.98x），**宽尾分布（dapo_32k）约 1.0x**，**训练慢（DP=1）时被稀释**。
- **长度预测的召回率是生死线**：漏判 10% 的长请求就从 1.29x 掉到 0.93x；误报很便宜。
- **16 卡**：对同构最优 cap 1.32–1.36x；扣掉"同构 + makespan 调度"的收益后，异构本身每步 8 条 straggler 时 1.18–1.26x、4 条时 1.08x。
- **TP8 不值得**：比 TP4 只快约 6%（固定开销占大头），`[8,1×8]` 与 `[4,4,1×8]` 结果相当。
- **平凡分布下收益主要来自调度**：lognormal CV 1.0 对 no-cap 1.15–1.18x，但异构本身只占 1.01–1.06x；dapo_32k 约 1.0x（见第 10 节）。
- **往 TP4 混入短请求**是平衡杠杆：主体更重时把纯分层的 0.76–0.87x 救回到 1.12–1.19x，但不抬高峰值；`auto_mix` 免去配额搜索。

---

## 2. 优势从哪来

### 2.1 TP 是"延迟 vs 吞吐"的权衡，且方向随负载翻转

decode 是访存受限的：每步每卡读一次自己那份权重 + 自己那份 KV，再加一个不随 TP 缩小的固定开销
（kernel launch、elementwise、all-reduce，约 1.6 ms@TP1 → 2.9 ms@TP4）。

**低负载、长 context（straggler 所处的区间）**：同 4 张卡、4 条 30k context 的请求

| 布局 | 每 token 延迟 | 每 GPU 吞吐 | 每卡每步访存（估算） |
|---|---|---|---|
| 4×TP1，各 1 条 | 8.19 ms | 122 tok/s | 权重 16.4 GB + KV 4.4 GB |
| 2×TP2，各 2 条 | 7.33 ms | 136 tok/s | 8.2 + 4.4 GB |
| 1×TP4，4 条 | **6.38 ms** | **157 tok/s** | 4.1 + 4.4 GB |

TP 越高，每卡要读的权重越少，而这份读取由 batch 内的请求分摊——TP4 在这里**延迟和吞吐同时更好**。
TP2 只把权重读砍半，优势只有 TP4 的一半左右，撬不动后面的权衡。

**高负载、短 context（主体请求所处的区间）**：2k context、每 GPU 64 条

| | TP1 | TP2 | TP4 | TP8 |
|---|---|---|---|---|
| tok/s/GPU | **3079** | 2626 | 2514 | 2134 |

权重读取已被 64 条请求摊薄，TP 的固定开销成为主导，TP1 吞吐最高。

### 2.2 关键路径 = max(主体吞吐时间, straggler 延迟时间)

一个版本要等 batch 内所有轨迹生成完（S=0 时严格如此，S=1 时有一步缓冲）：

```
rollout 窗口 ≈ max( 主体 token 数 / 主体吞吐 ,  L_max × 每 token 延迟(straggler 所处负载) )
```

同构配置只能在两种坏结果之间取折中：

- cap 大 / TP 小：吞吐高，但 straggler 和大 batch 挤在一起，每 token 30–40 ms，单条就要 1000+ s；
- cap 小 / TP 大：straggler 快了，但主体吞吐不够。

异构把两侧拆开：**straggler 去 TP4 低负载，主体去 TP1 高负载**，各自工作在自己擅长的区间。
以 `4k 主体 + 32k straggler` 为例，每个版本：

| 层 | GPU | 产出 token | 每 GPU 吞吐 | 完成时间 |
|---|---|---|---|---|
| TP4（4 条 straggler） | 4 | 131k（6%） | 约 157 tok/s | 32768 × 约 6.4 ms ≈ 210 s |
| TP1（508 条短请求） | 4 | 2.08M（94%） | 约 3000 tok/s | 约 170 s |

一半 GPU 只产出 6% 的 token，看似浪费，但它们卡在关键路径上：用一部分 GPU 的吞吐换关键路径的延迟。

### 2.3 收益上限约 1.5x

异构对"两种同构里较好的那种"的优势，大致不超过
min(TP1/TP4 高负载吞吐比 ≈ 1.54, TP1/TP4 straggler 延迟比)，且只有主体与 straggler 两侧耗时相当时才逼近。
离线 oracle 见到的最大值 1.40x，仿真实测 1.27–1.29x。

---

## 3. 实验结果

8 卡 rollout，batch 64 组 × 8，8 个版本，所有对比组消费的轨迹数和 token 数相同（例：17.695M token、32 条长轨迹）。
比值 = 同构最优 cap 配置 span ÷ 异构 span（>1 表示异构更快）；括号内为相对"同构 + 智能调度（LPT / makespan 放置）"。

| workload | 训练 | S=0 | S=1 | S=2 |
|---|---|---|---|---|
| 4k 主体 + 32k straggler | DP=4 | **1.290x**（1.247） | **1.204x**（1.180） | 0.943x |
| 2k 主体 + 16k straggler | DP=4 | **1.263x**（1.233） | 1.049x（1.000） | — |
| 4k（CV 0.5）+ 32k straggler | DP=4 | **1.206x**（1.090） | **1.126x**（1.062） | 0.935x |
| 4k 主体 + 32k straggler | DP=1 | 1.142x | 0.983x | 0.980x |
| 2k 主体 + 16k straggler | DP=1 | 1.134x | 0.987x | — |
| 4k（CV 0.5）+ 32k straggler | DP=1 | 1.118x | 1.002x | 0.952x |
| dapo_32k（宽尾），batch 16 | DP=4 | 1.057x（0.995） | 1.006x（1.020） | — |

S=1 下的 SCG（4k+32k，DP=4）：

| 方案 | span | SCG(0) | SCG(1) | SCG(1) 对比 |
|---|---|---|---|---|
| 同构最优 cap（2×TP4, cap 64） | 1947 s | 1828 | 9074 | 1.000x |
| 同构 + makespan（4×TP2） | 1908 s | 1099 | 9275 | 1.022x |
| 异构分层 | 1617 s | 1449 | 10944 | **1.206x** |
| 异构分层 + rescue@3000 | 1536 s | 1440 | 11521 | **1.270x** |
| 异构，无预测器，仅 rescue | 1765 s | 1235 | 10025 | 1.105x |

⚠️ SCG(0) 下降（staleness-0 占比 20% → 13%）：rollout 变快后跑得更超前，更多轨迹以 staleness 1 被消费（推测，未验证）。

### 只用 TP1/TP2 时（早期结果，4 卡与 8 卡）

| 场景 | 异构最优 | 同构 + 智能调度 |
|---|---|---|
| lognormal CV1.0，4 卡，S=0 | 1.106x | 1.073x |
| dapo_32k，4 卡，S=0 | 1.058x | 1.061x |
| dapo_32k，8 卡 batch 16，S=0 | 1.081x | 1.087x |
| 各种 S=1 / DP=2,4 | ≤1.014x | 同构 LPT 通常最好 |

大部分"异构收益"其实是**长度感知调度**带来的，同构 fleet 上同样能拿到。

---

## 4. 什么时候不成立

| 条件 | 原因 |
|---|---|
| 只有 TP1/TP2 | TP2 的延迟优势仅 1.12–1.35x，又损失吞吐 |
| S=2 | psrl 把晚到的组放进后面的 buffer，straggler 不再卡训练，问题退化为纯吞吐，同构更好 |
| 宽尾分布（dapo_32k：p50 11k，p90 25k） | 长请求太多，一个 TP4 层装不下，TP4 在高负载下吞吐又差 |
| 训练慢（DP=1） | rollout 不在关键路径上 |
| 每 GPU 负载低（如 8 卡 batch 16，约 16 条/GPU） | 这个负载下同构 TP4 吞吐不输 TP1、延迟更好，直接全上 TP4 即可 |

**甜点 workload 的两个条件必须同时满足**：
1. 足够多的中等请求，让 TP1 工作在高负载（约 100+ 条/GPU）；
2. 每个版本的 straggler 数 ≈ TP4 实例仍然快的 batch（约 4 条）。

---

## 5. 算法

代码均在 `/tmp/capstudy/snap/srl/component/rollout/router/strategies/`。

### 5.1 `length_tiered_tp`：按预测长度分两层

1. **分层**：最高 TP 的实例为长请求层，其余为短请求层。
2. **标记**：`remaining ≥ long_threshold` 为长请求（实验用 oracle 长度；`miss_rate` / `false_alarm_rate` 模拟预测误差）。
3. **各层 cap 分开**：长请求层 `long_cap`（约 4，保持低负载）；短请求层 `short_cap`（64–128，保持高吞吐）。
4. **出队**：`pick_queued` 按最长优先扫描整个版本队列，取第一个"某层现在能收"的请求——长请求层满了也不会堵住短请求。
5. **层内**：驻留最少者优先，平手看总 context。
6. **reveal-and-rescue**：每次 dispatch 前刷新长请求层空位；短请求层上生成量超过 `rescue_threshold` 的请求被暂停（保留 token）、回到队列、改放到 TP4，付一次 re-prefill。

### 5.2 `makespan_tp`：无参数的贪心放置

对每个候选实例，用**该实例自己的 cost model** 模拟其驻留请求按剩余长度依次排空（每段固定 batch 的耗时用中点估算），
把请求放到使 fleet 整体预测完成时间增加最少的实例上。配合最长优先出队，就是带 batch 相关速度的 LPT list scheduling。
无需阈值和 cap，长请求自然流向快实例；比调好的分层版本约差 5%，但它在同构 fleet 上也是很强的对照组。

### 5.3 离线 fleet 规划器（`/tmp/hetero/oracle_grid.py`）

给定每个版本的长度集合，用 5.2 的排空模型对多种 fleet 形状做 LPT 贪心，预测窗口与最优形状。
4k+32k 上预测 1.336x，仿真 1.29x，可以先用它判断某个 workload 值不值得上异构。

---

## 6. 鲁棒性：预测误差

| 预测误差 | 4k+32k S=0 | 4k+32k S=1 | CV0.5 S=1 |
|---|---|---|---|
| 完美 | 1.290x | 1.204x | 1.126x |
| 漏判 10% 长请求 | **0.927x** | **0.882x** | **0.837x** |
| 误报 10% 短请求 | 1.171x | 1.231x | 1.090x |

一条漏判的 straggler 落到高负载 TP1 上（30–40 ms/token），单条就决定了窗口。**预测器必须优先保召回**。

rescue 的效果：

- **S=1 有效**：完美预测 + rescue 从 1.204x 提到 1.27x；**完全不用预测器**、只靠 rescue 也有 1.10x。
- **S=0 结构上救不回来**：dispatch 只在 KV 释放时触发，短请求一起在第 4096 步左右结束后再没有触发点；
  更关键的是损害发生在前 4096 步（漏判的 straggler 和约 127 条短请求挤在 TP1 上），
  任何高于短请求长度的 reveal 阈值都来不及。

---

## 7. 经验与陷阱

1. **一定要有"同构 + 智能调度"对照组**。只和"同构 + 默认路由"比，会把调度收益误记成异构收益（TP1/TP2 阶段就是这样）。
2. **同构 cap 网格要越过最优点**。第一版网格的最优 cap 落在边界，扩展后基线变强，比值从 1.38x 降到 1.29x。
3. **比 span/SCG 前先确认消费的轨迹数和 token 数相同**。psrl 取先完成的组，理论上可能因为少训长轨迹而"变快"。
4. **LPT/makespan 调度已经把尾巴压没了**：剩 ≤8 条时的尾部时间从 14% 降到 1%。版本中途弹性改 TP 没有剩余空间可赚。
5. **跨实例的决策需要 fleet 级的新鲜信息**。rescue 一度用了过期的空位计数，S=0 下静默地从不触发——结果和不开 rescue 逐位相同就是信号。
6. **大量并行仿真不要 `conda activate`**：几百个并发激活会在锁上排队（一度 238 个 conda 进程对 36 个仿真）；用只 export 变量的轻量 env 脚本。
7. **共享节点测 kernel 必须全程监控 GPU**，并在测量结束后复查日志，不能只看开始时刻。
8. **借用已有 kernel 表要 `cp -a`，不要 symlink**：`fit-floor` 会顺着链接写回原目录。

---

## 8. 往 TP4 层混入短请求（`mixed_short_cap`）

TP4 实例上长短请求分开计数：长请求受 `long_cap` 限制，短请求另有配额 `mixed_short_cap`，短请求永远占不到 straggler 的名额。

**代价比直觉大。** cost model 实测（TP4，4 条长请求 @16k context，加 k 条 2k context 的短请求）：

| k | 每步耗时 | straggler 变慢 | 短请求吞吐 |
|---|---|---|---|
| 0 | 5.26 ms | — | — |
| 8 | 6.46 ms | +23% | 1238 tok/s |
| 32 | 7.57 ms | +44% | 4226 tok/s（约 1060/GPU） |
| 64 | 9.82 ms | +87% | 6519 tok/s |

每多一条请求不只是多读一点 KV：GEMM 行数变化、attention 与 elementwise 随 batch 增长，前 8 条就有一次跳变（之前按带宽手算的"32 条只慢 13%"偏乐观）。
短请求在 TP4 上只有 TP1 每 GPU 吞吐的约 1/3。所以混入只在**主体侧是瓶颈**时才划算——它是平衡杠杆，不是额外加速来源。

**仿真结果**（8 卡，batch 64，DP=4，干净 TP4 表；比值 = 同构最优 cap ÷ 异构）：

| workload | S | 纯分层（m=0） | 分层 + 混入（最优 m） |
|---|---|---|---|
| 4k 主体 + 4 条 32k | 0 | **1.290x** | 同左（最优即 m=0） |
| 4k 主体 + 4 条 32k | 1 | **1.204x** | 同左 |
| 6k 主体 + 4 条 32k | 0 | 1.005x | **1.186x**（m=128） |
| 6k 主体 + 4 条 32k | 1 | 0.874x | **1.147x**（m=32） |
| 8k 主体 + 4 条 32k | 0 | 0.878x | **1.120x**（m=192） |
| 8k 主体 + 4 条 32k | 1 | 0.759x | **1.143x**（m=64） |
| 4k 主体 + 8 条 32k | 0 | 1.037x | 同左 |
| 4k 主体 + 8 条 32k | 1 | 0.890x | 1.062x（m=16） |

结论：混入把异构的获胜区间从"恰好平衡的甜点"扩展到主体更重的 workload（纯分层在这些 workload 上会输到 0.76x），
但峰值仍是 4k+32k 上的 1.29x / 1.20x。straggler 翻倍到 8 条时 TP4 层本身负载升高，收益明显变小。

### 8.1 自动配额（`auto_mix`）

不再网格搜索 `mixed_short_cap`，而是逐个请求判断：用排空模型预测"放到 TP4 上该实例何时排空"与"放到最好的 TP1 实例上何时排空"，
只有前者不晚于后者的 `(1 + mix_slack)` 倍才放 TP4。短请求层满时，与最慢的 TP1 实例当前排空时间比较。
这是把 `makespan_tp` 的判据只用在"TP4 收不收短请求"这一个决策上，同时保留分层结构（长请求名额、LPT、rescue）。

| workload | S | 网格搜索最优配额 | auto，slack=0.1 | auto，最优 slack |
|---|---|---|---|---|
| 4k + 4 条 32k | 0 | 1.290x | 1.269x | 1.290x（slack 0） |
| 4k + 4 条 32k | 1 | 1.204x | 1.184x | **1.246x**（slack 0） |
| 6k + 4 条 32k | 0 | 1.186x | 1.186x | 1.186x |
| 6k + 4 条 32k | 1 | 1.147x | **1.177x** | 1.177x |
| 8k + 4 条 32k | 0 | 1.120x | 1.081x | 1.120x（slack 0.5） |
| 8k + 4 条 32k | 1 | 1.143x | 1.080x | 1.095x（slack 0.25） |
| 4k + 8 条 32k | 0 | 1.037x | 1.037x | 1.037x |
| 4k + 8 条 32k | 1 | 1.062x | 1.045x | 1.045x |

一个固定的 slack（0.1）在所有格子里都在网格最优的约 6% 以内，有 2 格反超；`short_cap` 仍需搜索。
slack 为 0 时偏保守：判据一次只看一个请求，看不到它后面排队的短请求，所以会少往 TP4 放。

## 9. TP8 与 16 卡

**TP4 → TP8 几乎不再变快**（cost model，干净的 TP8 表）：

| 8 条 straggler 放 8 张卡，30k context | 每 token 延迟 | 每 GPU 吞吐 |
|---|---|---|
| 8×TP1 | 8.19 ms | 122 tok/s |
| 4×TP2 | 7.33 ms | 136 tok/s |
| 2×TP4 | 6.38 ms | 157 tok/s |
| 1×TP8 | **6.01 ms**（比 TP4 快 6%） | 166 tok/s |

单条请求：TP4 4.63 ms，TP8 4.33 ms。ctx≈0 时固定开销 TP4 3.87 ms、TP8 3.52 ms：固定开销占大头，权重再切一半每卡只少读约 0.5 GB。
高负载（2k context、每 GPU 64 条）吞吐 TP8 只有 2134 tok/s/GPU（TP4 2514，TP1 3079）。**延迟杠杆在 TP4 基本用尽**。

**16 卡仿真**（batch 128，训练 DP=8；同构基线扫了 16×TP1 / 8×TP2 / 4×TP4 / 2×TP8 的 cap，最优点在网格内部或是真实平台）：

| workload | S | 对同构最优 cap | 对同构 + makespan | TP8 层 `[8,1×8]` | TP4 层 `[4,4,1×8]` |
|---|---|---|---|---|---|
| 4k + 每步 8 条 32k | 0 | **1.363x** | **1.261x** | 1.363x | 1.349x |
| 4k + 每步 8 条 32k | 1 | **1.213x** | **1.179x** | 1.204x | 1.213x |
| 4k + 每步 4 条 32k | 0 | 1.317x | 1.083x | 1.317x | 1.286x |
| 4k + 每步 4 条 32k | 1 | 1.339x | 1.083x | 1.231x | 1.339x |

- 16 卡上对"同构最优 cap"的优势到了 1.32–1.36x，比 8 卡（1.20–1.29x）略高。
- 但**"同构 + makespan 放置"在每步 4 条 straggler 时很强**（1.22–1.24x），扣掉调度收益后异构本身只剩 1.08x；
  每步 8 条时异构本身仍有 1.18–1.26x。
- TP8 层与 TP4 层（2 个 TP4 实例）结果相当，S=1 时 TP4 层更好。TP8 不值得。

## 10. 加上 no-cap 基线，以及更平凡的分布

no-cap = 同构 fleet、不设 router cap（`ACTIVE_CAP=0`，只受 KV 容量与 `max_concurrency=1024` 约束），在 TP1×8 / TP2×4 / TP4×2 中取最好。
"同构 + 调度" = 同构 fleet 上的 LPT 分层或 `makespan_tp`。异构列取所有异构方案（纯分层、auto_mix、`makespan_tp`、`[4,1×4]` / `[4,2,2]`）中最好的。

| workload | 卡数 | S | 对 no-cap | 对最优 cap | 对同构 + 调度 |
|---|---|---|---|---|---|
| 4k + 每步 4 条 32k（构造） | 8 | 0 / 1 | **1.33x / 1.38x** | 1.29x / 1.20x | 1.25x / 1.18x |
| 6k + 每步 4 条 32k（构造） | 8 | 0 / 1 | 1.23x / 1.29x | 1.19x / 1.18x | 1.17x / 1.11x |
| 8k + 每步 4 条 32k（构造） | 8 | 0 / 1 | 1.12x / 1.18x | 1.12x / 1.14x | 1.09x / 1.12x |
| 4k + 每步 8 条 32k（构造） | 8 | 0 / 1 | 1.26x / 1.17x | 1.07x / 1.06x | 0.96x / 1.06x |
| 4k + 每步 8 条 32k（构造） | 16 | 0 / 1 | 1.36x / 1.36x | 1.36x / 1.21x | 1.26x / 1.18x |
| 4k + 每步 4 条 32k（构造） | 16 | 0 / 1 | 1.32x / **1.47x** | 1.32x / 1.34x | 1.08x / 1.08x |
| **lognormal 均值 4k，CV 1.0** | 8 | 0 / 1 | 1.15x / 1.18x | 1.13x / 1.06x | 1.06x / 1.01x |
| **lognormal 均值 4k，CV 0.6** | 8 | 0 / 1 | 1.10x / 1.12x | 1.09x / 0.95x | 1.02x / 0.93x |
| **dapo_32k（真实分布 + 补回尾部）** | 8 | 0 / 1 | 1.04x / 0.99x | 1.04x / 0.93x | 0.95x / 0.93x |

所有格子消费的轨迹数相同。读法：

- **no-cap 是最弱的基线**。S=0 时最优 cap 往往就是"很大的 cap"，与 no-cap 几乎相同；S=1 时 cap 才明显有用（no-cap 比最优 cap 慢 7–20%）。
- **平凡分布下，大部分收益来自长度感知调度，不是异构**。lognormal CV 1.0：对 no-cap 1.15–1.18x，但其中只有 1.01–1.06x 归功于异构。
  lognormal CV 0.6 在 S=1 时异构反而输；dapo_32k（每 512 条里约 130 条 ≥16k）异构基本无用，最好的是同构 TP1 + LPT。
- 异构真正有用的前提仍是**每步少量（约 4–8 条）、长度远超主体的 straggler**。平凡分布里长请求要么太少（CV 0.6：每步约 1 条 ≥16k），
  要么太多（dapo_32k），都不在这个区间。

## 11. 长度感知规划器（`TierPlanner`）：TailSieve 与异构并行的结合

代码已进仓库：`srl/simulator/dynamic_parallel/planner.py`、`srl/component/rollout/router/{estimator.py,strategies/planned_tier_tp.py}`、`tests/test_tier_planner.py`。
配置：`dynamic_parallel.strategy=planner`、`dynamic_parallel.algorithm=dp|pp|twopool`、`dynamic_parallel.switch_margin`、`rollout.router.strategy=planned_tier_tp`。
旧的 `ParallelismSolver` 没有删除。

**每个权重版本重新规划一次**：输入是上一版本遗留的 partial rollout（带已生成长度）加上即将放行的新 prompt 组，输出是 TP 形状、每个 tier 的长度区间和 cap。
所有求解器共用同一个排空模型：一个实例按剩余长度依次完成，`T = sum_j tau(b_j) * (L_j - L_{j-1})`，`tau` 取该 TP 自己的 cost model，cap 和 KV 容量用事件驱动模拟处理。

| 求解器 | 做法 | 单次规划耗时 |
|---|---|---|
| `twopool` | TailSieve 形式：一个尾池（最高 TP）加一个主体池，二分隔离规模使两池完成时间相等 | 约 20 ms |
| `pp` | 枚举 TP 形状，对排好序的请求做连续划分，对 makespan 二分，段代价随段增长所以贪心填充就是可行性检验 | 约 90 ms |
| `dp` | Libra 形式：状态 `(已用 GPU, 请求类前缀)`，每次转移加一个实例。类按长度突降处切开，否则类均值会把 straggler 抹平 | 约 800 ms |

**oracle 长度下的结果**（8 卡，batch 64，DP=4，对同构最优 cap 与"同构 + 调度"中更强者，10 个格子几何均值）：`twopool` 1.075x、`pp` 1.066x、`dp` 1.056x。
峰值是构造的 4k + 4 条 32k 在 S=0 的 1.25x，lognormal 为 1.01–1.09x，dapo_32k 约 1.0x，S=2 时同构更好（与前文一致）。
DP 并没有比阈值或二分更好：最优结构本来就是"尾部隔离加均衡"，两层就够了，DP 的额外自由度没有用上。

**部署相关的结论比 oracle 数字重要**：

| 长度信息 | 相对"同构 + oracle 调度"的加速比范围 |
|---|---|
| oracle（trace 里的真长度） | 1.00–1.25x |
| 噪声预测器 σ=0.25 | 0.93–1.20x |
| σ=0.5 | 0.80–1.09x |
| σ=1.0 | 0.67–0.93x |
| 在线估计（prompt 历史、生存、总体分位数） | 0.28–0.93x |

基线这里用的是 oracle 调度，所以这一列同时包含了"没有 oracle 的代价"，只比较异构本身时要和 σ 对应的同构调度一起看。
rescue（请求长出所在 tier 后暂停并重新放置）在 S=1 有帮助，在 S=0 不起作用：伤害发生在最前面的步骤里。

## 12. 下一步

1. 把 `length_tiered_tp`、`makespan_tp`、router 的 `pick_queued` / `observe_fleet` 钩子和 TP4 / TP8 表移植进仓库。
2. 对"同构 + makespan"做与异构同等程度的调参（目前它是零参数的），确认 1.08x / 1.18–1.26x 这组"纯异构收益"。
3. 在真实 RL 长度分布（非两类构造）上验证；用一个真实的长度预测器替换 oracle。

---

## 13. 复现

依赖（均不在仓库中）：

| 内容 | 路径 |
|---|---|
| 代码快照（含新策略、router 改动、允许 TP4） | `/tmp/capstudy/snap` |
| TP4 kernel 表（GEMM、attention、elementwise；collective 从 kernel_tp2 复制，含 world size 8） | `/tmp/hetero/profile/kernel_tp4/kernel` |
| 轻量 env（不含 conda activate） | `/tmp/hetero/env_light.sh` |
| 两类 trace | `/tmp/hetero/traces/mix4k_32k.jsonl` 等 |
| 批量运行与汇总脚本 | `/tmp/hetero/run.sh`、`summ.py`、`report.py` |

TP4 KV 容量（probe 实测）：2,330,112 tokens（已缓存在 `~/.cache/srl/kv_capacity.json`）。

**trace 构造**：每组 8 条、prompt 32 token；全部 4096 token，每 16 组把第一条换成 32768 token。
每个 batch（64 组 = 512 条）恰好 508×4096 + 4×32768：均值 4320，CV 0.58，straggler 占请求 0.78%、占 token 5.9%。

**运行**（S=1 示例）：

```bash
SNAP=/tmp/capstudy/snap
K=/apdcephfs_zwfy/share_303760348/lhy/SimRL/outputs/profile/qwen3_8b
K4=/tmp/hetero/profile/kernel_tp4/kernel
source /tmp/hetero/env_light.sh
export PYTHONPATH=$SNAP SIMRL_BACKEND=native CUDA_VISIBLE_DEVICES="" POLICY=psrl \
       MAX_STALENESS=1 TRAINING_DP=4 BATCH_SIZE=64 N_VERSIONS=8 MAX_OUTPUT_LENGTH=32768
BASE="workload.trace.path=/tmp/hetero/traces/mix4k_32k.jsonl rollout.max_concurrency=1024 \
rollout.router.max_waiting_after_preempt=1024 rollout.router.reject_preempted_on_waiting=true \
+log_exclude='instance,rollout_step'"

# 同构基线（需对 TP1×8 / TP2×4 / TP4×2 各扫 5–6 个 cap，确认最优点在网格内部）
N_INSTANCES=2 ROLLOUT_TP=4 KERNEL_ROOT=$K4 ACTIVE_CAP=64 OUTPUT_DIR=/tmp/r/hom_tp4x2_cap64 \
  EXTRA_OVERRIDES="$BASE" bash $SNAP/examples/simulate/psrl_aligned.sh

# 异构 [4,1,1,1,1] + 长度分层 + rescue（ACTIVE_CAP 必须为 0，否则 psrl_aligned.sh 默认的 50 会覆盖分层 cap）
SA=+rollout.router.strategy_args
N_INSTANCES=5 ACTIVE_CAP=0 OUTPUT_DIR=/tmp/r/het EXTRA_OVERRIDES="$BASE \
  dynamic_parallel.enabled=true dynamic_parallel.n_gpus=8 dynamic_parallel.tp_candidates=[1,2,4] \
  dynamic_parallel.strategy=fixed dynamic_parallel.tp_per_instance=[4,1,1,1,1] \
  +dynamic_parallel.kernel_table_roots.1=$K/kernel_tp1/kernel \
  +dynamic_parallel.kernel_table_roots.2=$K/kernel_tp2/kernel \
  +dynamic_parallel.kernel_table_roots.4=$K4 \
  rollout.router.strategy=length_tiered_tp $SA.long_threshold=12000 $SA.long_cap=4 \
  $SA.short_cap=64 $SA.order=longest_first $SA.rescue_threshold=3000" \
  bash $SNAP/examples/simulate/psrl_aligned.sh
```

每个 `run.log` 末尾打印 span、Consumed 与 SCG(S)。

---

## 14. 注意事项

- **TP4 elementwise 表已重测**：初测（04:24–04:42）期间有其他租户占满 8 张卡，旧表比重测表慢（中位 3.5%，最大 34%）。
  已换成干净的重测表（全程监控无外来进程），mix4k 的结果逐秒不变（2715 s / 2105 s），所以 1.29x / 1.20x 不受影响。
  旧表留在 `/tmp/hetero/profile/elementwise_contaminated_0424`。
- **TP8 表**：GEMM 在外来进程出现之前测完；attention 与 elementwise 初测时与外来任务同在 GPU 0，已移到 `tp8_contaminated_1209`（个别点偏差达 2.3x），
  重测时用 `CUDA_VISIBLE_DEVICES=4,5,6,7,2,3,0,1` 让 microbenchmark 跑在空闲的 GPU 4 上，逐 GPU 监控确认无外来进程。
  TP8 KV 容量（probe 实测）：4,773,952 tokens。表在 `/tmp/hetero/profile/kernel_tp8/kernel`。
- **长度是 oracle**（trace 中的真实长度）；每个配置只有一个 seed。
- **cost model 局限**：所有数字来自 composed cost model；TP2 的 attention 表在 b=2 / b=4 处有非单调现象。
- **仓库文件被覆盖**：`outputs/profile/qwen3_8b/kernel_tp4/kernel/h20/collective/{vllm,nccl}/.../floors.json`
  被我误写为空拟合（symlink 导致）。原件（9 月 2 日）无法再生成，但当前代码也读不了那张 collective 表（缺 `nodes` 列），
  目前没有计算依赖它。如有备份请恢复。

相关研究：[README_14b_fleet_shapes.md](README_14b_fleet_shapes.md)（此前 TP1/TP2 异构在 S=2 上不划算的结论与本文一致）。
