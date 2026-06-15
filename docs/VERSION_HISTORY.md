# 版本历史

最后更新：2026-06-15

这份文档记录五子棋 AlphaZero 项目的主要实验版本。这里的版本号表示训练路线和基础设施迭代，不等同于 Python 包或网页应用的发布版本。

## 状态定义

| 状态 | 含义 |
| --- | --- |
| `baseline` | 已验证的强基线，用于后续对比。 |
| `active` | 当前主线或仍在推进的路线。 |
| `archived` | 保留用于复现，不推荐作为新实验起点。 |
| `failed` | 没有超过基线，或训练曲线显示明显问题。 |
| `infra` | 训练基础设施改动，不直接代表某个棋力 checkpoint。 |

## 总览

| 版本 | 状态 | 目标 | 当前结论 |
| --- | --- | --- | --- |
| `v1 / old best` | `baseline` | A100 上完整训练出的强模型。 | 仍是当前已验证的最强基线。 |
| `v2` | `failed` | 从 old best 继续长训，尝试超过旧基线。 | 失败：曲线恶化，对战结果不稳定。 |
| `v3-local` | `archived` | 在本地直接从 old best 继续训练。 | 仅保留复现实验，不作为推荐路线。 |
| `distill-oldbest-128x8` | `active` | 用 old best 蒸馏轻量 student。 | 已通过最低准入，作为 v3 student 的起点。 |
| `v3-student-local` | `active` | 让轻量 student 进入 KataGo-style RL。 | 当前主线，仍需最终对 old best 做大样本验证。 |
| `v3-infra-20260615` | `infra` | 优化 eval、保存、worker 生命周期。 | 已采用，显著降低 eval 阶段耗时。 |

## v1 / old best

目标：建立强基线和可部署的人机对弈模型。

主要 checkpoint：

```text
outputs/checkpoints/a100-4-prod-v3/gomoku10_best.pt
```

主要特征：

- 成功训练线约为 `192` channels、`12` residual blocks，比当前 student 更大。
- 使用全局上下文、软策略头、replay 加权、MCTS value target、root policy temperature、shaped Dirichlet、dynamic cPUCT、FPU、forced playouts、playout cap randomization 等 KataGo-style 改进。
- 来自完整 A100 训练线，是后续版本的默认比较对象。

当前结论：保持为 `baseline`。除非后续模型通过足够大的 head-to-head 评估，否则不替换这个基线判断。

## v2

目标：从 old best 继续长训，尝试用更长训练和调整后的搜索预算超过旧基线。

主要产物：

```text
outputs/checkpoints/v2/
outputs/metrics/v2.jsonl
outputs/plots/v2-failed/
```

观察结果：

- 大约在 `96 -> 112` 轮，训练曲线持续走坏：`loss` 上升，`policy_top1` 和 `value_acc` 下降，`policy_kl` 恶化。
- 多个 checkpoint 的对战复核没有稳定超过 old best。
- v2 内部选出的 checkpoint 只能代表 v2 低样本筛选结果，不代表全局最强模型。

当前结论：判定为 `failed`。保留产物用于分析，不继续作为主线。

## v3-local

目标：保留从 old best 直接继续训练的本地复现路线。

代表命令：

```bash
python -m alphazero_gomoku.train --preset v3-local
```

当前结论：`archived`。它可以用于复现和对比，但不再作为推荐主线。推荐路线是先蒸馏轻量 student，再进入 student RL。

## distill-oldbest-128x8

目标：先用 old best 蒸馏一个更轻的 student，再让 student 进入大规模强化学习。

主要产物：

```text
outputs/checkpoints/distill-oldbest-128x8/gomoku10_student_best.pt
outputs/checkpoints/distill-oldbest-128x8/gomoku10_student_final.pt
outputs/metrics/distill-oldbest-128x8.jsonl
```

student 架构：

- `128` channels
- `8` residual blocks
- `12` policy channels
- `6` value channels
- `384` value hidden size
- 开启 global pooling 和 soft policy head

训练思路：

- 第一阶段用 raw policy/value 蒸馏 old best 的网络行为。
- 如果 raw distill 不够，再用 old best 的 MCTS targets 微调。
- 只有 benchmark 不明显落后时，才进入 RL 阶段。

当前结论：作为 `v3-student-local` 的 active seed 使用。

## v3-student-local

目标：以轻量 student 为起点，继续进行 KataGo-style 强化学习，同时用 champion gate 控制退化风险。

代表命令：

```bash
python -m alphazero_gomoku.train --preset v3-student-local
```

关键设置：

- 从 `distill-oldbest-128x8/gomoku10_student_best.pt` 启动。
- 每轮 `96` 盘 self-play。
- replay 未达到 `25k` 原始局面前跳过训练。
- 每轮训练最多扫 replay `2` 遍，避免小 replay 过拟合。
- 每 5 轮做一次 champion gate eval。
- 实际目标仍是超过 `v1 / old best`。

当前 A100 产物：

```text
checkpoint: outputs/checkpoints/v3-student-local/
metrics:    outputs/metrics/v3-student-local.jsonl
replay:     outputs/replay/v3-student-local_replay.pt
```

当前结论：`active`。基础设施层面已经健康；棋力层面仍需要最终 head-to-head 验证。

## v3-infra-20260615

目标：降低 A100 空转和 eval 阶段等待时间，增强长训练的可靠性。

触发原因：

- 旧 eval 是串行流程，20 局评估会长期占住训练进程。
- self-play 和 train 本身已经较健康，主要瓶颈集中在 eval 阶段。

主要改动：

- 新增 `--eval-workers` 和 `--eval-devices`。
- eval 局面并行分发到多个 worker 和多张 GPU。
- 不减少 `eval_games`；并行 eval 会完整跑满评估局数，因此关闭 early cutoff。
- 新增 `--train-data-workers` 和 `--train-prefetch-factor`。
- self-play/eval 临时模型快照在 `finally` 中清理。
- replay、checkpoint、`gomoku10_best.pt` 改为先写临时文件，再原子替换。
- 修复 CPU 模式下 `auto` 设备误显示 CUDA 的问题。

直接效果：

- 旧串行 eval：`8/20` 局约 `9m37s`。
- 新并行 eval：`20/20` 局约 `2m42s`，使用两张 A100。
- 重启后的前几轮训练健康：`skipped=0`，`value_clamps=0`。

当前结论：`adopted`。后续 A100 训练默认应使用这套 infra。

## 晋升规则

一个新 checkpoint 只有同时满足以下条件，才应该替代 `v1 / old best`：

1. 通过训练内 champion gate。
2. 单独对 `v1 / old best` 做 head-to-head 评估。
3. 在随机开局和黑白互换下结果稳定。
4. 曲线没有明显退化，重点看 `policy_kl`、`policy_top1`、`value_acc` 和 value 稳定性。

在这些条件满足之前，`v1 / old best` 仍是正式基线。
