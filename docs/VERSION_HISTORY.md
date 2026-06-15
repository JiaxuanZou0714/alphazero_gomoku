# 版本记录

最后更新：2026-06-15

本文只记录训练版本和基础设施版本，不记录普通代码提交。每个版本统一使用以下字段：

- 状态
- 目标
- 起点/产物
- 关键改动
- 结果
- 结论

## v1 / old best

- 状态：`baseline`
- 目标：建立当前最强的已验证基线模型。
- 起点/产物：`outputs/checkpoints/a100-4-prod-v3/gomoku10_best.pt`
- 关键改动：A100 训练 `100` 轮，中间 resume 两次；使用较大的 `192x12` 网络；引入 global context、soft policy head、MCTS value target、dynamic cPUCT、FPU、forced playouts、playout cap randomization 等 KataGo-style 改进。
- 结果：成为后续所有实验的默认比较对象。
- 结论：继续作为正式 baseline，除非新模型通过足够大的 head-to-head 评估。目录名 `a100-4-prod-v3` 是历史命名，不代表它属于 v3。

## v2

- 状态：`failed`
- 目标：从 old best 继续长训，尝试超过旧基线。
- 起点/产物：`outputs/checkpoints/v2/`、`outputs/metrics/v2.jsonl`、`outputs/plots/v2-failed/`
- 关键改动：延长训练，并调整 replay、训练步数和搜索预算。
- 结果：`96 -> 112` 轮曲线恶化，表现为 `loss` 上升、`policy_top1/value_acc` 下降、`policy_kl` 变差；对战复核也没有稳定超过 old best。
- 结论：判定失败。保留 v2-failed 分析产物，不继续作为主线。

## v3-local

- 状态：`archived`
- 目标：保留本地从 old best 直接继续训练的复现路径。
- 起点/产物：`--preset v3-local`
- 关键改动：使用 v3 的训练配置直接从 old best 继续 RL。
- 结果：不是当前推荐路线。
- 结论：仅用于复现或对比；新实验优先走 distill -> student RL。

## distill-oldbest-128x8

- 状态：`active seed`
- 目标：用 old best 蒸馏一个更轻的 student，降低后续 RL 成本。
- 起点/产物：`outputs/checkpoints/distill-oldbest-128x8/gomoku10_student_best.pt`
- 关键改动：student 使用 `128` channels、`8` residual blocks、`12` policy channels、`6` value channels、`384` value hidden，并保留 global pooling 和 soft policy head。
- 结果：通过最低准入 benchmark，作为 v3 student RL 的起点。
- 结论：保留为当前 student 主线的 seed。

## v3-student-local

- 状态：`active`
- 目标：让轻量 student 进入大规模 KataGo-style RL，尝试超过 old best。
- 起点/产物：`--preset v3-student-local`、`outputs/checkpoints/v3-student-local/`、`outputs/metrics/v3-student-local.jsonl`
- 关键改动：从 `distill-oldbest-128x8` 启动；每轮 `96` 盘 self-play；replay 达到 `25k` 原始局面后开始训练；每轮最多扫 replay `2` 遍；每 5 轮做 champion gate eval。
- 结果：基础设施优化后训练流程健康；棋力仍需最终对 old best 做大样本验证。
- 结论：当前主线，尚未替代 old best。

## v3-infra-20260615

- 状态：`infra adopted`
- 目标：解决 eval 阶段串行导致的 A100 时间浪费，并提高长训可靠性。
- 起点/产物：`train.py` infra revision；远端新日志 `v3_student_a100_gpu23_newinfra_*.out.log`
- 关键改动：新增 `--eval-workers`、`--eval-devices`、`--train-data-workers`、`--train-prefetch-factor`；eval 改为多 worker 多 GPU 并行；self-play/eval 临时快照自动清理；checkpoint、best checkpoint、replay 改为临时文件写入后原子替换。
- 结果：旧串行 eval 约 `8/20` 局耗时 `9m37s`；新并行 eval `20/20` 局耗时约 `2m42s`。重启后前几轮 `skipped=0`、`value_clamps=0`。
- 结论：后续 A100 训练默认使用这套 infra。

## checkpoint 晋升规则

新 checkpoint 只有同时满足以下条件，才可以替代 `v1 / old best`：

1. 通过训练内 champion gate。
2. 单独对 old best 做 head-to-head 评估。
3. 随机开局和黑白互换下结果稳定。
4. 曲线没有明显退化，重点看 `policy_kl`、`policy_top1`、`value_acc` 和 value 稳定性。

在这些条件满足之前，`v1 / old best` 仍是正式 baseline。
