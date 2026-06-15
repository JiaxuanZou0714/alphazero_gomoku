# 10x10 AlphaZero 五子棋

这是一个面向 `10x10` 棋盘的 AlphaZero 风格五子棋项目，包含自我对弈、MCTS 搜索、神经网络训练、checkpoint 评估和 GitHub Pages 静态对弈页面。

旧的后端网页入口已经移除：仓库不再保留 `web/` 和 `web_play.py`，浏览器页面只维护 `docs/` 静态版。

## 当前状态

正式 baseline：

```text
outputs/checkpoints/a100-4-prod-v3/gomoku10_best.pt
```

该模型约 `115 MB`，不随 GitHub 仓库分发，需要作为本地或远端训练产物保存。`a100-4-prod-v3` 是历史目录名；按实验语义它是 `v1 / old best`，不是 v3。

当前主线实验：

```text
v3-student-local
```

真正的 v3 从蒸馏得到的 `128x8` student 出发，再做大规模 self-play RL。最新 A100 训练后，通过内部 gate 的 best 是：

```text
outputs/checkpoints/v3-student-local/gomoku10_best.pt
```

注意：这个 best 对应第 `90` 轮，不是最后保存的 `iter_0096.pt`。

最新对 old best 复核：

```text
v3_best_iter90 vs old_best
128 sims, 64 games
54 胜 / 10 负 / 0 和
score = 0.84375
执黑：31 胜 / 1 负
执白：23 胜 / 9 负
```

这个结果说明 v3 已经明显强于 v1 / old best 的 `128 sims` 设置；正式晋升仍建议继续补更高 sims 和更大样本评估。

## 项目结构

```text
game.py      五子棋规则、状态转移、胜负判断
mcts.py      神经网络引导的 MCTS
model.py     policy-value 网络
train.py     自我对弈、训练、评估、checkpoint 保存
play.py      命令行人机对弈
utils.py     模型和设备工具
scripts/     画图、benchmark、蒸馏、Pages 导出脚本
docs/        GitHub Pages 静态页面和版本记录
tests/       单元测试
outputs/     本地训练产物、metrics、plots
```

## 训练曲线

### v1 / old best

```text
outputs/metrics/a100-4-prod-v3.jsonl
```

`a100-4-prod-v3` 是历史目录名，实际语义是 v1 / old best。它训练 `100` 轮，中间 resume 两次。

![v1 training overview](outputs/plots/a100-4-prod-v3/metrics_overview.png)

更多 v1 图：

```text
outputs/plots/a100-4-prod-v3/losses.png
outputs/plots/a100-4-prod-v3/accuracy_value.png
outputs/plots/a100-4-prod-v3/entropy.png
outputs/plots/a100-4-prod-v3/selfplay_outcomes.png
outputs/plots/a100-4-prod-v3/data_timing_eval.png
```

### distill / 128x8 student

```text
outputs/metrics/distill-oldbest-128x8.jsonl
```

蒸馏分两段：前 `24` 步是 raw distill，后 `16` 步是 MCTS fine-tune。

![distill overview](outputs/plots/distill-oldbest-128x8/metrics_overview.png)

更多蒸馏图：

```text
outputs/plots/distill-oldbest-128x8/losses.png
outputs/plots/distill-oldbest-128x8/policy_value.png
outputs/plots/distill-oldbest-128x8/policy_kl.png
outputs/plots/distill-oldbest-128x8/metrics.csv
```

蒸馏末尾：`policy_top1 ~= 0.712`，`policy_kl ~= 0.509`，`value_mae ~= 0.120`。

### v3 / student RL

```text
outputs/metrics/v3-student-a100-final.jsonl
```

![v3 training overview](outputs/plots/v3-student-a100-final/metrics_overview.png)

更多 v3 图：

```text
outputs/plots/v3-student-a100-final/losses.png
outputs/plots/v3-student-a100-final/policy_value.png
outputs/plots/v3-student-a100-final/entropy_kl.png
outputs/plots/v3-student-a100-final/selfplay_outcomes.png
outputs/plots/v3-student-a100-final/timing_replay.png
outputs/plots/v3-student-a100-final/metrics.csv
```

简要诊断：

- replay 达到最低门槛后，训练才真正开始。
- 真实训练阶段 loss 和 policy KL 整体下降。
- policy top-1 最高约 `0.60`。
- value accuracy 稳定在约 `0.88`。
- eval 波动较大，不能只靠曲线晋升 checkpoint。
- 当前 v3 best 是第 `90` 轮，第 `96` 轮不是 accepted champion。

## 测试

Windows 本地建议从父目录运行，保证包导入路径正确：

```powershell
cd "C:\Users\Jiaxuan Zou\Documents\GitHub"
$env:PYTHONPATH="C:\Users\Jiaxuan Zou\Documents\GitHub"
& "C:\Users\Jiaxuan Zou\.conda\envs\alphazero-gomoku\python.exe" -m unittest discover -s alphazero_gomoku\tests -t .
```

## 命令行对弈

```bash
python -m alphazero_gomoku.play \
  alphazero_gomoku/outputs/checkpoints/a100-4-prod-v3/gomoku10_best.pt \
  --simulations 256 \
  --human white
```

人类执白后手，AI 执黑先行。

## 画训练曲线

```bash
python alphazero_gomoku/scripts/plot_training_metrics.py \
  --metrics alphazero_gomoku/outputs/metrics/v3-student-a100-final.jsonl \
  --out-dir alphazero_gomoku/outputs/plots/v3-student-a100-final
```

## checkpoint 对战评估

不要只看 loss 曲线晋升模型。正式比较必须做 head-to-head：

```bash
python alphazero_gomoku/scripts/benchmark_checkpoints.py \
  --candidate alphazero_gomoku/outputs/checkpoints/v3-student-local/gomoku10_best.pt \
  --baseline alphazero_gomoku/outputs/checkpoints/a100-4-prod-v3/gomoku10_best.pt \
  --candidate-sims 128,256,512 \
  --baseline-sims 128 \
  --games 64 \
  --opening-moves 4 \
  --device cuda
```

评估时应使用随机开局，并交替黑白。

## GitHub Pages 静态页面

`docs/` 是无需 Python 后端的静态对弈页面。

导出模型：

```powershell
conda run -n alphazero-gomoku python scripts\export_pages_model.py `
  --checkpoint outputs\checkpoints\a100-4-prod-v3\gomoku10_best.pt `
  --out-dir docs\assets\model `
  --chunk-mib 24
```

本地预览：

```powershell
conda run -n alphazero-gomoku python -m http.server 8780 --bind 127.0.0.1 --directory docs
```

打开：

```text
http://127.0.0.1:8780/
```

## 版本记录

简版版本记录：

```text
docs/VERSION_HISTORY.md
```
