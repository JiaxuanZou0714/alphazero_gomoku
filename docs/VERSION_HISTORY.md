# 版本记录

最后更新：2026-06-17

按时间顺序记录每一次迭代——不断改进**训练逻辑、模型架构、基础设施、网页前端**。编号递增、最新在下；🎯 标记带模型产出的里程碑。

---

### 1. 项目初始化 · 2026-06-11
AlphaZero Gomoku 基础框架：10×10 棋盘、self-play 强化学习、MCTS + policy-value 网络。

### 2. KataGo 式改进 · 2026-06-12
MCTS 与网络引入 global pooling、辅助 soft policy head、dynamic cPUCT、MCTS value target、FPU、forced playouts、playout cap randomization；修复根节点估值符号、FPU 冻结防守点等 bug 并补单元测试；评估加随机开局与胜率 early stopping。
🎯 **模型产出 · v1 / old best**：`192x12` 网络，A100 训 `100` 轮（晋升第 `95` 轮),成为后续所有实验的正式基线。

### 3. 全面中文化与文档 · 2026-06-13
UI 全面中文化，README 补充算法原理与自我对弈训练目标。

### 4. 静态网页对弈 app · 2026-06-14
GitHub Pages 上线：浏览器内 onnxruntime-web（WebGPU/wasm）推理 + 浏览器端 MCTS；含搜索树可视化、分析/推荐面板、实时胜率追踪；模型用 Git LFS 跟踪。

### 5. v2 长训尝试 · 2026-06-14
加 v2 preset 与远端启动脚本，从 old best 继续长训并调 replay / 步数 / 搜索预算。
⚠️ **模型产出 · v2（失败）**：`96 -> 112` 轮曲线恶化（`loss` 升、`top1`/`value_acc` 降），未稳超 old best，判废，仅留分析产物。

### 6. 轻量 student 蒸馏 · 2026-06-15
用 old best 蒸馏出 `128x8` student（policy `12`/value `6`/value_hidden `384`），`24` 步 raw distill + `16` 步 MCTS 微调，过最低准入 benchmark。
🎯 **模型产出 · distill seed**：轻量 student，作为后续 RL 的起点，降低训练成本。

### 7. 并行评估 infra 与训练稳定化 · 2026-06-15
新增 `--eval-workers`/`--eval-devices`/`--train-data-workers` 等；eval 改多 worker 多 GPU 并行（`20` 局 `9m37s -> 2m42s`），checkpoint/best/replay 改原子替换；修首轮空 loader、Windows 临时目录竞争。

### 8. v3 student 大规模 RL · 2026-06-15
从 distill seed 起 KataGo-style RL：每轮 `96` 盘 self-play、replay `25k` 起训、每 5 轮 champion gate，晋升第 `90` 轮。
🎯 **模型产出 · v3-student**：`128 sims` 对 v1 复核 **`54-10-0`（`0.844`）**，明显超 old best，成为**网页默认模型**。

### 9. 网页多模型选择器 · 2026-06-15
`catalog.json` + 选择器，可在 v1/v3 间切换；新增网页版本记录页。

### 10. 大型 infra 重构 · 2026-06-16
共享 `inference.py`、`PRESETS` 字典化、self-play/eval 改 `ProcessPoolExecutor`（修 GIL 无并行 + 全局 RNG 污染）、`eval_cache` FIFO、`value_acc` 加权 bug 修复、`train_epoch` 拆分；弃用 Windows 专用代码转为 Linux-only。

### 11. GPU 训练循环提速 · 2026-06-16
训练时 bf16 AMP + cuDNN autotune + 减少 per-batch 同步 + 向量化对称增广；3080 实测 **`1.61x`** 训练步加速，CPU 结果不变。

### 12. KataGo 扩展：EMA 与 ownership 头 · 2026-06-16
opt-in EMA-of-weights（评估/晋升/存 best 用 EMA 快照，gate 失败不回滚训练模型）与 ownership 辅助头（对称增广同步变换 + MSE 损失）；默认关闭以保持 CPU 结果一致。

### 13. v4 训练：warm-start + 开局多样化 · 2026-06-16
从 v3 additive-head warm-start（`resume_allow_partial` 只随机初始化 ownership 头、走全新 schedule），叠加 EMA、ownership、self-play 开局多样化，本地 3080 RL 至晋升第 `35` 轮 EMA best。
🎯 **模型产出 · v4-student-3080**：对 v3 `60` 局 @`256 sims` 为 **`33-27-0`（`0.550`，噪声区内）**，但黑白更均衡、对非常规开局更鲁棒；已上线为**可选模型**，v3 仍默认。

### 14. v4 上线网页 + 每模型架构图 · 2026-06-16
v4 加入 `catalog.json`、版本面板展示；新增每模型 SVG 架构图与结构选择器；前端审查修复（sw 改 manifest/catalog network-first、`*.onnx.part*` 交 IndexedDB 按 `sha256` 失效）。

### 15. 网页推理 infra 提速 · 2026-06-16
WebGPU 固定形状 batch + 加载预热，消除首手 shader 编译延迟（仅 WebGPU，wasm 路径不变）；v4 导出改 **fp16**（权重半精度、I/O 仍 float32，worker 不变），下载 `11.95MB -> 6.0MB`，policy argmax 与 fp32 `100%` 一致。

### 16. 训练 infra 大提速：批量自对弈 + Gumbel MCTS · 2026-06-17
面向「更轻、更强」的 v5：① **批量跨对局自对弈引擎**（`selfplay_batched.py`）——单进程内 `N` 局并行、每 tick 把所有局的叶子评估融成一个大 GPU batch，单卡实测 **`~6x`** 于串行/多 worker 路径，产出样本与串行等价（胜率/value 分布一致），opt-in 接入训练（`--selfplay-batched`）。② **Gumbel AlphaZero**（`gumbel.py` + `9` 单测）——Gumbel-top-m + Sequential Halving 根选择 + `softmax(logits+σ(completedQ))` 改进策略目标，低 sims 下标签噪声更小（破 `policy_top1=0.68` 停滞的杠杆），实测吞吐 **∝ 1/sims**（半 sims → `2x`），自对弈胜率更均衡（`0.50/0.50`）。③ 修复 `distill_old_best.py` 的 `MCTS` 漏导入 bug + 加 `--dataset-cache`（多尺寸共享同一份教师目标）。④ MCTS `Node __slots__` + eval 守卫。`64` 单测全过。
🔬 **关键发现**：自对弈吞吐**与网络大小无关**（串行被 GPU 启动开销、批量被 Python 树操作主导），「更小=更快训练」在本栈不成立——小模型只为部署体积/延迟服务，提速靠批量 + Gumbel 减 sims。故 v5 尺寸按「能保住棋力的最小」选，而非按速度。

### 17. v5：更小且 SOTA + 网页棋力榜 · 2026-06-17
蒸馏 v4（教师）进 `64×5`（`policy 8`/`value 4`/`value_hidden 192`，仅 `~65万`参数）做 init，再用批量+Gumbel(`96 sims`)+损失重平衡（`soft_policy 6`/`value 0.8`）+ fresh cosine(`1e-4`)+开局多样化 `0.7` 做 RL，3080 上约 `3h` 收敛（`policy_top1` 破 `0.68` 至 `~0.76`、晋升至 iter `95` EMA best）。
🎯 **模型产出 · v5-tiny-3080（新 SOTA，网页默认）**：循环赛各 `60` 局 @`128 sims`、颜色互换——对 **v4 `45-15`（`0.750`）**、对 **v3 `47-13`（`0.783`）**、对 **v1 `60-0`（`1.000`）**，全部明确反超。Elo：**v5 `1607` > v4 `1426` > v3 `1358` > v1 `1000`**。体积 fp16 ONNX 仅 **`1.17MB`**（v4 `6MB`、v1 `38MB`）。用 `5×` 更小的网 + 半 sims 全面超越 v4，验证「批量+Gumbel 提速 → 更多 RL 数据 + 更优标签」的路线。
### 18. 网页棋力榜（Elo 柱状图）· 2026-06-17
新增「棋力榜」面板：循环赛 Bradley-Terry Elo 横向柱状图（`docs/assets/leaderboard.json` 驱动），直观对比 v1–v5 棋力；网页默认模型切换为 **v5**。

---

## checkpoint 晋升规则

新 checkpoint 只有同时满足以下条件，才可替代 `v1 / old best`：

1. 通过训练内 champion gate；
2. 单独对 old best 做 head-to-head 评估；
3. 随机开局和黑白互换下结果稳定；
4. 曲线没有明显退化（重点看 `policy_kl`、`policy_top1`、`value_acc` 和 value 稳定性）。

在这些条件满足之前，`v1 / old best` 仍是正式 baseline。
</content>
