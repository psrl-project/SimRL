# inventory_deadline 调度：完整交接文档（2026-10-15 收尾）

> 这是给新 session 的一站式交接：motivation、算法、与基线的全部对比、已实现/未实现、已知问题、
> 下一步优先级、复现命令和环境坑，全部在这一份里。
> 完整的发现时序和历史在英文权威设计文档
> `docs/superpowers/specs/2026-10-13-inventory-deadline-scheduling-design.md`
> （同步副本：`/apdcephfs_zwfy10/share_303541817/lhy/vibe_coding/lhy-claude-data/plans/paper-trajectory-routing-apdcephfs-zwfy-fuzzy-squid.md`）。
> 本文取代之前的时序快照（那些内容的历史版本在 git 里）。

---

## 一、Motivation：这条研究线为什么存在

**要替代的东西**：在 psrl policy 下，给每个 rollout instance 写死一个"最多同时跑 N 条请求"的
静态 cap（调到 decode 拐点附近）能有收益，但 cap 是魔法数字，换 workload（长度分布、训练/
rollout 耗时比、GPU 数）就要重调，算法上立不住，发不了 paper。

**目标**：一套会算账的模型驱动调度器。对每个"要不要把这条请求放到某台 instance"的决策，预测
它会让训练流水线里最近的未 ready buffer 晚多久，选延迟最小的方案。cap 的效果应当是模型的
副产品。**"零旋钮"的强形式已被证伪**（见第四节）。目前成立的弱形式是：唯一旋钮是无量纲的
KV 池占比分数，但它在 mean shift 下也出了问题（见第六节问题 1）。

**原始设计的两块**：
1. **按轨迹的准入 + 路由**（本 session 的主战场，已基本成型，见第二、四节）。
2. **动态异构并行**：每次权重更新后重选 rollout 集群的 TP 形状（**完全没写**，见第五节）。

**为什么 partial rollout 下重新配置是免费的**：每次权重发布本来就 abort 所有在跑请求、重新
prefill，所以在发布边界重排所有请求/换 TP 形状不付 KV 迁移代价。真实系统侧 psrl 已有
dynamic-parallel 机制（slot + awake set，切换实测 ~5s），接口是 `CohortPlanner.plan(table,
state) -> ParallelismPlan`。

**收益的理论来源（已验证的边界）**：饱和 fleet 序无关。headroom 只在 τ（训练时长）与 rollout
接近平衡的地带、且 over-knee 并发 / S=1 drain tail / straggler isolation 三处。长度离散度是
收益杀手（CV 1.5→收益趋 0）。

---

## 二、算法是什么（从下往上四层）

### 第 1 层：单 instance drain 预测
"假设把这条请求放到 instance A"→ 把 A 当前所有在跑/排队请求用解析流转模拟（`dryrun/forecast/
drain.py`）精确重演一遍：闭式 decode segment + 分块 prefill，得到每个人的完成时间和**未来 KV
峰值**（`peak_blocks`，保守上界：共享 prompt 按兄弟各计一次、不建模抢占、含未放置 tail）。

### 第 2 层：buffer READY 时间
group 完成时间变晚 → 它占的训练 buffer 变晚 READY。用照抄 psrl 真实 reserve/occupy 规则的
重放函数（psrl 仓库 `psrl/workers/ps/inventory_forecast.py` 的 `forecast_ready_times`）算出
未来各 buffer 的 `F_k`。

### 第 3 层：训练完成时间 E_k
```
E_k = max(F_k, E_{k-1}) + τ_k   （训练 + 同步 + recompute + 权重加载）
```
partial rollout 每次发布打断所有 rollout 重头生成，预测要计入，无需不动点迭代，严格按序往后
推（`dryrun/forecast/fleet.py` `chain_buffers`）。

### 第 4 层：决策
每个候选方案（某台可用 instance，或 DEFER 留在队列）算出 `(E_k0, E_k0+1, ...)` 向量（默认
看 2 个 buffer），**字典序更小者胜**，比较带容忍度 `eps`（默认 = 1% × 训练周期）。字典序 +
DEFER 就是"cap 被替代"的机制：会拖慢眼看要凑齐的 buffer 的方案自动输给 DEFER。

### 准入 vs 路由（两个不同的决策，同一套定价）
- **路由**：已 admit 的成员放哪台 instance（必须在某台落位）。
- **准入**：staged 的新 group 接不接（DEFER_GROUP 是合法选项）。**准入多一道 KV 闸门**：
  `PlanSession(kv_watermark=)`：假设准入后该 instance 的预测 KV 峰值超过 `watermark × 池子`
  就拒绝（等效 DEFER）。只作用于准入，不作用于路由。

### 处理顺序
已 admit 成员 > staged group，内部按 deadline buffer 近者先、长度长者先，**staged 的长度排序
被 `window_groups=32`（到达序分块，块内才重排）约束**。这是修掉的真实 bug（无界重排会饿死
长度分布的一端，并污染消费集合）。deadline 是真实紧迫性，永不加窗。某版本队列被 DEFER/放不下
则整个 dispatch 停止，不允许新版本插队（`allow_version_bypass` 恒 False）。

---

## 三、代码与实验资产在哪

| 组件 | 位置 |
|---|---|
| drain 预测（第 1 层） | `dryrun/forecast/drain.py` |
| buffer/训练链（第 2、3 层） | `dryrun/forecast/pipeline.py`、`fleet.py` |
| 一次 dispatch 增量定价 + KV 闸门 | `dryrun/forecast/plan.py`（`PlanSession`） |
| 策略本体 | `dryrun/component/rollout/router/strategies/inventory_deadline.py` |
| router 钩子 | `router/router.py`（`begin_dispatch`/`pick_queued` 能力探测，`observe_enqueue`） |
| 只读运行时绑定 | `dryrun/simulator/runtime.py` |
| psrl 只读 inventory API | psrl 仓库 `psrl/workers/ps/inventory_forecast.py`（editable 装入） |
| LPT 基线（保留在用的旧策略） | `strategies/length_aware.py`（`window_groups`/`kv_watermark` 约定的出处） |

| 实验脚本（`examples/analysis/`） | 回答什么问题 |
|---|---|
| `headroom_probe.py run|report best` | 主网格四臂对比（default / 逐点扫 cap / cap+LPT / ours） |
| `headroom_probe.py run|report heldout` | **held-out 验证**（6 个未调参点，含闸门臂） |
| `routing_ablation.py` | 路由单独贡献（cap=48 钳死，只比放置） |
| `kv_watermark_sweep.py` | 闸门分数 {0,0.5..0.9} × 6 点 |
| `inventory_deadline_knobs.py` | 单旋钮扫描（含 `wm06_*` 八个"闸门下"变体，**已就绪未跑**） |
| `inventory_deadline_decisions.py` | 决策级插桩（模型定价 vs 实际选择 vs 集群并发） |
| `inventory_sched_sweep.py` | toy 规模四臂（G1 pass 1） |
| `forecast_accuracy.py` | 预测精度 G0（含 `--strategy` 自洽性检查） |

结果 JSON 都在 `outputs/simulate/<脚本名>/<grid>/results.json`。关键 commit：`b19f38f`（策略）
→ `b3885eb`（自洽性）→ `b7de5de`（toy G1 + tie-break 修复）→ `1135f72`（网格/旋钮/插桩）
→ `bde954b`（window_groups 修复 + 路由消融）→ `279d7b3`（KV 闸门）→ `aa46f6c`（heldout
网格 + wm06 旋钮族）。测试基线：**`pytest tests/` 1920 passed, 3 skipped**（2026-10-15）。

---

## 四、与基线的全部对比（截至 2026-10-15，全部可复现）

**口径纪律**（每张表都遵守）：(1) 三个基线：`psrl_default`（不限制）、**固定 cap=48**（可部署
口径，一个数打天下）、逐点扫最优 cap（oracle，仅供参照）。(2) 每个对比都查 `tokens_per_group`
漂移，>1% 的点标记不可信（"重排不能改变被消费的集合"）。(3) 静态基线的 cap 每点重新扫
{24,32,40,48,56,64}，不沿用。(4) LPT 的 `window_groups=32`（一个训练 batch）。

### 4.1 预测精度（G0，前置门槛）
oracle 长度下 F 中位 0.0-2.0%、p90 ≤8.4%（带 cap 的 regime 过门）。toy 规模自洽性极好
（9/10 regime F 中位 0.000%）。无 cap regime 的 8.8-15% 误差是诊断采样器用了简化代理所致，
不是 `PlanSession` 本身的问题（它有 227 个测试对拍暴力重算和真实引擎）。

### 4.2 路由单独（cap=48 钳死准入，只比放置，`routing_ablation.py`）
6 点全部 `tok/grp` 漂移 0.00%：

| 点 | ours@48 / cap_lpt48 |
|---|---|
| cv0.5_τ600 | **1.174x** |
| cv1.0_τ600 | **1.118x** |
| cv0.5_τ345 | **1.019x** |
| cv1.0_τ345 | 1.020x |
| cv0.5_τ600,S=2 | **1.060x** |
| cv0.5_τ150 | 0.971x |

**5/6 赢，几何平均 1.058x。路由（放置）本身干净且有真实正贡献。**（toy 规模上与 default
逐位一致，那里本来就没油水。）

### 4.3 全策略、无闸门（调参网格 `best`，刷新于 window_groups 修复后）

| 点 | default | ours | ours 并发 | 最优静态并发 |
|---|---|---|---|---|
| cv0.5_τ600 | 11610 | 9353 | 60.4 | 37.8 |
| cv1.0_τ600 | 12795 | 11964 | 44.2 | 34.8 |
| cv1.0_τ345 | 11946 | 11673 | 44.8 | 36.9 |
| cv0.5_τ345 | 10469 | 9193 | 63.8 | 43.7 |
| cv0.5_τ150 | 9412 | 8808 | 67.8 | 47.6 |
| cv0.5_τ600,S=2 | 10254 | 9115 | 77.8 | 48.5 |

几何平均：vs default **1.109x**（6/6 赢），vs 固定 cap48 **0.980x**（2/6 赢，净输）。
**并发仍是最优静态的 1.2-1.6 倍，准入过松是独立于路由的问题。**

### 4.4 加 KV 闸门后（调参网格，36 arm 漂移全 0.00%）

| watermark | vs default | vs 固定 cap48 | vs oracle cap | 赢 cap48 |
|---|---|---|---|---|
| 0 | 1.109x | 0.980x | 0.977x | 2/6 |
| 0.5 | 1.145x | 1.013x | 1.009x | 3/6 |
| **0.6** | **1.160x** | **1.025x** | **1.022x** | **4/6** |
| 0.7 | 1.145x | 1.012x | 1.009x | 3/6 |
| 0.8 | 1.117x | 0.988x | 0.985x | 2/6 |
| 0.9 | 1.125x | 0.994x | 0.991x | 2/6 |

调参网格上 [0.5, 0.7] 整个区间赢 cap48，wm=0.6 最优，并发压到 36.5-55.0（与静态基线同量级）。
**但这不是最终结论，见 4.5。**

### 4.5 held-out 验证（真相，`heldout` 网格，6 个未调参点，0.6 原封不动）

| 点 | wm0 vs 最优静态 | wm=0.6 vs 最优静态 | 闸门自身作用 |
|---|---|---|---|
| cv0.75_τ475 | 1.030x | **1.065x 赢** | +3.5% |
| cv0.5_τ225 | 0.952x | 0.988x | +3.8% |
| cv1.5_τ345 | 0.946x | 0.935x | -1.2% |
| mean=12000_τ345 | **1.001x 平** | **0.920x 输** | **-8.8%** |
| ρ=0.5_τ600 | 1.117x | **1.156x 赢** | +3.5% |
| S=3_τ600 | 0.920x* | 0.960x* | +4.1% |

（*S=3 点静态基线自身漂移 0.9-1.2%，static-vs-ours 不可信，闸门前后对比不受影响。
ours 系全部 12 arm 漂移 0.00%。）

**几何平均：wm=0.6 vs oracle 1.001x、vs cap48 1.001x，恰好打平**（wm0 是 0.992x）。
调参网格上的 +2.5% 里约 2 个点是同网格挑最优值的泡沫。**幸存下来的是闸门在 4/6 未见点上
真实的 +3.5~4.1%，被 mean=12000 的 -8.8% 抵消。**

### 4.6 对比结论一句话
路由一半**已解决**（5/6 赢、1.058x、消费集合干净）。准入一半**方向对了但没完成**：闸门
把"按 token 而非人头限流"做对了大半（4/6 点稳定增益），但在长均值分布上过紧（问题见下），
目前整体与最优静态**打平**，还没有净赢。

---

## 五、已实现 / 未实现（对照原始设计）

| 原设想 | 状态 |
|---|---|
| psrl 只读 inventory API + 重放函数 | ✅ |
| drain / buffer / 训练链预测（四层） | ✅ 精度过 G0 |
| `PlanSession` 增量定价 | ✅（227 测试对拍） |
| 字典序比较 + eps + DEFER | ✅ |
| 策略本体 + router 钩子 + 运行时绑定 | ✅ |
| **"准入 = 预测 KV 峰值放得下"** | ✅ 已落地（`kv_watermark`），**但对照量有缺陷**（问题 1） |
| **"cap 是模型副产品"** | ⚠️ 弱形式（唯一旋钮 = 无量纲分数）部分成立，强形式（零旋钮）证伪 |
| **动态异构 TP replanner** | ❌ **一行没写**：`dynamic_parallel/` 只有 `fleet.py`/`plan.py`，`replanner.py` 不存在，config 无 `tp_candidates`。已知基础设施债：`Fleet._new_instance` 永远建 `NativeInstance` 且不传 eviction |
| G1 门禁 | ⚠️ 调参网格过（+2.5%），heldout 打平 |
| G2 门禁（动态 TP） | ❌ 未开始 |
| 长度噪声鲁棒性（oracle → noisy） | ❌ 未评估（`NoisyOracleLength` 已存在于 `forecast/length.py`） |
| 真实 trace（dapo_32k / psrl_aligned）上的评估 | ❌ 只在合成 MediumRegime 上 |

---

## 六、存在的问题（按定位程度排序）

1. **闸门对照"保守上界"在 mean shift 下过紧（已定位机制，修法方向明确）**。drain 峰值是
   保守上界（共享 prompt 按兄弟各计、不建模抢占、含整个未放置 tail），其松弛随 mean 和 tail
   深度放大：mean 8k→12k，同一 0.6 分数等效并发 52→36（实测 51.7→36.3），mean=12000 点上
   闸门 -8.8%。而静态 cap48 同点 47 并发仅 42 次抢占：**现实容忍偶尔抢占，闸门却要求上界
   不越线**。佐证：该点无闸门的策略本身打平最优静态（1.001x），损失全是闸门的。
   修法候选：(a) 对照量换成 committed-now + 本组自身预测增长（`length_aware` 式 + 组内预测）。
   (b) 对上界保守性做校准（比如按实测 peak/预测 peak 比值缩放）。**教训已存 memory：
   对照量不能是"保守性随 workload 轴变化"的量，否则分数不是真无量纲。**
2. **cv=1.5 高离散度下策略本身输 5-7%（已定位现象，未诊断）**。有无闸门都输，与准入无关。
   调参网格上最弱的 cv1.0_τ345（0.98x）外推恶化。指向字典序 `E` 目标或 DEFER 时序。下一步：
   `inventory_deadline_decisions.py` 在 cv1.5 regime 上跑一遍看定价 vs 选择 vs 并发。
3. **`eps` 的角色在新闸门下未知**。旧结论（"eps 既当容忍度又当节流阀、正确方向随 regime
   反转"）是在 wm=0 下测的，闸门接管节流后 eps 能否归零/纯当容忍度，决定最终一 or 两个旋钮。
   `wm06_*` 八个变体已写进 knobs 脚本，未跑。
4. **S=3 上静态基线自身消费集合漂移**（0.9-1.2%），高压力下"哪个 group 错过 deadline"依赖
   调度本身。ours 系不漂移（window_groups 修复的功劳），但 static-vs-ours 在这些点不可信。
   需要更长的 trace 或受控消费顺序。
5. **测试脚手架未交叉验证**：进程内 `MediumRegime` vs `psrl_headroom_sweep.py` 的 CLI 路径，
   已知差异 `rollout.max_concurrency`（那边 1024，这边未设）。
6. 次要：长度预测噪声鲁棒性未测，规模放大（>2 instance）未测，诊断脚本量尺不够细。

---

## 七、下一步（优先级 + 判据）

1. **修闸门保守上界问题**（问题 1）→ 改完直接跑 `headroom_probe.py run heldout` 验证。
   判据：mean12k 从 -8.8% 回到 ≥0，其余 5 点不退（尤其 ρ0.5/cv0.75 那两个赢点）。
2. **诊断 cv1.5**（问题 2）：`inventory_deadline_decisions.py cv1.5` 看模型定价是否系统性
   偏、DEFER 选择是否合理、并发形态。
3. **`eps` 重扫**（问题 3）：`python examples/analysis/inventory_deadline_knobs.py run best
   --knobs wm06 wm06_eps0 wm06_eps_small wm06_eps_big ...`（变体已就绪）。
4. CLI harness 对拍（问题 5）。
5. **动态 TP replanner**（第五节）：设想里唯一完全没动的部分。基础设施清单在英文 plan 文档
   的 "Dynamic heterogeneous TP replanner" 一节（含 psrl 侧 `CohortPlanner` 接口、KV 随
   物化度数缩小、reconfig ~5s 实测值、静态臂要在两种 KV 利用率下各跑一次的实验设计）。
6. 长噪声鲁棒性 → 真实 trace → 规模放大。

---

## 八、新 session 实用信息

**环境**：所有命令先 `source /apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh`。psrl 是
editable strict 安装进来的（`psrl.workers.ps.*` 直接 import）。**psrl 仓库的改动不提交**。
strict 模式下 psrl 新增文件要重装（`pip install --no-deps -e <psrl> --config-settings
editable_mode=strict`）。

**跑长任务**：sim 一臂 30s~6min，一个网格 30-90min，一律后台跑（`run_in_background`），前台
2min 超时会杀掉。注意外层 `| grep` 会块缓冲，要看中途输出用 `grep --line-buffered` 或直接
看进程 CPU 时间判断死活。

**复现命令**（都在仓库根目录）：
```bash
python examples/analysis/headroom_probe.py run best --workers 6     # 主网格（~40min）
python examples/analysis/headroom_probe.py report best
python examples/analysis/headroom_probe.py run heldout --workers 6  # held-out 验证（~1h）
python examples/analysis/headroom_probe.py report heldout
python examples/analysis/routing_ablation.py                        # 路由消融（~30min）
python examples/analysis/kv_watermark_sweep.py run|report           # 闸门扫描（~40min）
python examples/analysis/inventory_deadline_knobs.py run|report best
python -m pytest tests/ -q                                           # ~16min, 1920 passed
```

**其他坑**（都吃过亏）：共享盘 2T 常年满，遥测写 `/tmp`。`fit-floor` 会顺着 symlink 写表目录，
借表用 `cp -a` 别 `ln -s`。别的 session 原地重编 `.so` 会打断正在跑的 sim，长跑前后记 md5。
`assets/` 目录是别的 session 的，别动。

**当前干净状态**：工作树与 `aa46f6c` 一致，两份设计文档（中文本文档 + 英文 plan 副本）已同步。
