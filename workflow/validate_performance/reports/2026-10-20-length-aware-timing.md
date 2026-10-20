# 为什么这次 length_aware 验证跑了 2.5+ 小时

验证本身在跑（`a0e19e895d713328e`，测 `length_aware(lpt,load)` vs `request_num_balance`），用一个
fork 的 `validate-performance` agent 单独检查了这次慢在哪——方法是把每个 `tool_use` 的派发时间和对应
`tool_result` 的返回时间配对算真实耗时，而不是猜。

## 结论

**不是 sweep 慢，是三次 Write/Edit 卡在共享盘上，占了总时长的 80%。** 24 路并行的真实模拟从启动到全部
跑完只用了约 12 分钟——这个量级对 256k token 的极端长尾 workload 是合理的，不是瓶颈。

## 时间线

| 阶段 | 耗时 | 说明 |
|---|---|---|
| 读 SOP/代码 | ~11 min | 正常范围，单次工具调用 0–12s |
| 环境探测（`source simrl.sh` 等） | ~7 min | env.sh 本身要 20–130s，正常 |
| **写新脚本** `long_tail_lpt_load.py`（10.4KB） | **~15.4 min** | 单次 `Write` |
| **改脚本**（KV/batch 修正，3.2→3.4KB） | **~14.7 min** | 单次 `Edit` |
| **改脚本**（40→21 字符的小修正） | **~97.7 min** | 单次 `Edit`，内容极小但卡了近一个半小时 |
| 语法校验 + 启动 sweep | ~2 min | `ast.parse`、后台启动 |
| 真正的模拟 sweep | **~12 min** | 24 个并行 worker，每点 10 个 version，单点 5–12 分钟 |

三次 Write/Edit 合计约 128 分钟，而三次改动的内容都很小（10KB 新文件、3KB 编辑、40 字符的单行修正）——
不存在"内容大所以慢"的解释。

## 根因

不是这块共享盘的固有属性（事后用 `time cat` 探测同一文件，读取只要 4ms，盘是通的），而是那个时间窗口内
的瞬时拥塞/锁争用。`ListAgents` 显示当时同一台机器上还有 **18 个 peer session**（多个 `psrl-*`/`simrl-*`/
`lhy-research-*` 交互式会话 + 几个后台 `Request_chunk`/`design_discuss` 任务）在跑，很可能都挂在同一个
`/apdcephfs_zwfy` 挂载点上。

这和已有 memory 里记录的两个问题是同一类根因（共享盘 + 并发 session）的第三种表现形式：

- `shared-fs-quota-kills-sweeps`：配额满，表现为 `Errno 122`/FAILED。
- `cpp-so-rebuilt-in-place-kills-running-sims`：并发编译覆盖 `.so`，表现为 segfault。
- **这次：`Write`/`Edit` 本身挂起不报错**，表现为模型"卡住不产出"，97 分钟量级比前两者记录的案例更极端。

## Token 开销

输出 token 总计 64,670，其中 38,030（59%）是不可见的 extended thinking token。峰值上下文（cache_read）
约 198K token，是前 20 分钟密集读文件积累的，但这本身和 2.5 小时的墙钟时间不成比例——真正烧时间的是那
3 次 I/O 卡顿，token 开销不是主因。

## 没有必要重跑

日志已经证明 sweep 本身很快（~12 分钟跑完 24 路并行），拖慢整个任务的是前段三次文件写操作的偶发卡顿，
不是某个 workload 点太慢。重跑无法验证"是否 sweep 慢"这个问题，因为这个问题已经被证伪了。
