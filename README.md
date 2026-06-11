# 10x10 AlphaZero 五子棋

这是一个面向 `10x10` 棋盘的 AlphaZero 风格五子棋训练项目。当前仓库包含训练代码、网页对弈界面、一次 4 卡 A100 训练日志、训练曲线图，以及最新保留的 checkpoint：

```text
outputs/checkpoints/a100-4-prod-v3/gomoku10_iter_0030.pt
```

项目没有写入开局库、活三活四、威胁搜索或人工估值函数。代码里只编码了五子棋环境规则：

- 棋盘大小为 `10x10`；
- 黑白双方轮流在空位落子；
- 任意方向连成五子即获胜；
- 棋盘下满且无人连五则为平局。

策略和价值都由神经网络从自我对弈中学习。MCTS 使用网络给出的 policy/value 作为先验与叶子估值，训练目标来自 MCTS 访问分布和最终胜负。

## 项目结构

```text
game.py        五子棋规则、状态转移和胜负判断
mcts.py        MCTS 搜索
model.py       Policy-Value 网络
train.py       自我对弈、训练、评估和 checkpoint 保存
play.py        命令行人机对弈
web_play.py    本地网页对弈服务
web/           前端棋盘界面
logs/          训练日志
outputs/       checkpoint、metrics、plots
```

## 当前模型

仓库只保留了最新的第 30 轮 checkpoint：

```text
outputs/checkpoints/a100-4-prod-v3/gomoku10_iter_0030.pt
```

这个文件约 `99 MB`。GitHub 会提示它超过推荐的 `50 MB` 单文件大小，但它低于 GitHub 的硬限制，已经随仓库提交。

## 快速验证训练循环

从项目父目录运行：

```bash
cd /Users/jiaxuanzou/Documents

python -m alphazero_gomoku.train \
  --iterations 1 \
  --games-per-iteration 1 \
  --simulations 4 \
  --epochs 1 \
  --channels 8 \
  --residual-blocks 1
```

这个命令只用于验证训练流程，不会得到强棋力模型。

## 使用 A100 预设训练

在远端 A100 机器上，从 `~/jiaxuanzou` 运行：

```bash
cd ~/jiaxuanzou
conda activate modded-nanogpt

python -m alphazero_gomoku.train \
  --preset a100-4 \
  --checkpoint-dir alphazero_gomoku/outputs/checkpoints/a100-4-prod-v3 \
  --replay-path alphazero_gomoku/outputs/replay/a100-4-prod-v3_replay.pt \
  --metrics-path alphazero_gomoku/outputs/metrics/a100-4-prod-v3.jsonl
```

`a100-4` 预设使用更大的 SE-ResNet、固定每轮训练步数、cosine learning-rate schedule、16 个并行自我对弈 worker，以及多 GPU 训练更新。

## 命令行对弈

```bash
cd /Users/jiaxuanzou/Documents

python -m alphazero_gomoku.play \
  alphazero_gomoku/outputs/checkpoints/a100-4-prod-v3/gomoku10_iter_0030.pt \
  --simulations 128 \
  --human black
```

行列坐标均从 `1` 开始。

## 本地网页对弈

```bash
cd /Users/jiaxuanzou/Documents

python -m alphazero_gomoku.web_play \
  alphazero_gomoku/outputs/checkpoints/a100-4-prod-v3/gomoku10_iter_0030.pt \
  --simulations 64
```

然后打开：

```text
http://127.0.0.1:8765
```

## 训练曲线

下面的图来自：

```text
logs/a100_4_prod_v3_stable_20260610_194644.log
```

日志中第 `1-30` 轮有完整的 self-play、train 和 eval summary；第 `31` 轮只有部分 self-play 进度，没有完整训练 summary，因此下图只统计完整的 `1-30` 轮。

### 总览

![metrics overview](outputs/plots/a100_4_prod_v3_stable_20260610_194644/metrics_overview.png)

### 损失曲线

![losses](outputs/plots/a100_4_prod_v3_stable_20260610_194644/losses.png)

前 3 轮是最明显的学习阶段：总损失从 `4.5253` 降到 `1.9173`，policy loss 从 `4.4537` 降到 `1.8296`，`policy_kl` 也从 `3.0433` 降到 `0.6739`。这说明模型很快从接近均匀策略转向能拟合 MCTS 访问分布的策略。

第 4 轮之后，总损失大多在 `1.9-2.0` 附近波动，而不是继续单调下降。这在自我对弈训练里是正常现象：数据分布会随着模型变强而移动，后续样本并不是固定监督集。

### 策略和值网络指标

![accuracy value](outputs/plots/a100_4_prod_v3_stable_20260610_194644/accuracy_value.png)

`policy_top1` 从第 1 轮的 `0.1480` 快速升到第 3 轮的 `0.6920`，之后基本维持在 `0.68-0.70`。这说明网络对 MCTS 首选落点的拟合已经较稳定。

`value_acc` 从 `0.9715` 缓慢下降到 `0.9323`，`value_mae` 从约 `0.10` 上升到约 `0.19`。这不一定代表训练崩坏，更可能说明后期自我对弈局面更短、更尖锐，胜负标签更集中，价值头面对的分布发生了变化。后续如果要继续增强，应重点观察独立评估胜率，而不是只看 value loss。

### 熵和策略确定性

![entropy](outputs/plots/a100_4_prod_v3_stable_20260610_194644/entropy.png)

`pred_entropy` 从 `4.5976` 快速降到约 `1.8`，说明网络输出从接近全棋盘均匀分布变得更集中。与此同时，后期 `target_entropy` 和 `selfplay_entropy` 有所回升，表示 MCTS 目标并不是完全塌缩到单一落点，仍然保留一定搜索分歧。

这组曲线整体是健康的：网络策略更确定，但 MCTS 目标没有完全退化成硬标签。

### 自我对弈结果

![selfplay outcomes](outputs/plots/a100_4_prod_v3_stable_20260610_194644/selfplay_outcomes.png)

平均步数从第 1 轮的 `37.4` 降到第 30 轮的 `14.3`，说明模型越来越快地进入决定性局面。训练集中没有平局，黑棋胜率从约 `0.56` 升到约 `0.82`，白棋胜率对应下降。

这反映了两个信息：

- 模型学到了更直接的胜负线路，棋局明显变短；
- 当前 `10x10` 设置和自我对弈采样下存在很强的先手优势或先手偏置。

如果后续要做更公平的棋力评估，建议固定一组开局、交换先后手，并单独统计黑白胜率。

### 数据量、耗时和评估

![data timing eval](outputs/plots/a100_4_prod_v3_stable_20260610_194644/data_timing_eval.png)

replay buffer 从 `28.7k` 增长到 `441.6k` 样本。由于棋局变短，每轮产生的 `raw_examples` 和增强后的 `examples` 后期明显减少。

每轮耗时主要在 `7-10` 分钟之间，自我对弈部分占主要时间。learning rate 在前 5 轮 warmup 到 `2e-4`，之后按 cosine schedule 缓慢下降，到第 30 轮约为 `1.71e-4`。`grad_norm` 在第 5 轮附近达到高点，之后整体下降，说明训练后期更新幅度更稳定。

评估分数大部分为 `1.0`。第 5 轮候选模型评估失败，`eval_score=0.0`；第 20 轮只有 `0.5`，低于晋升阈值；第 30 轮重新达到 `1.0`。这说明训练过程中有少数候选 checkpoint 没有超过 champion，但整体没有出现持续性退化。

### 全部数值指标

![all numeric metrics](outputs/plots/a100_4_prod_v3_stable_20260610_194644/all_numeric_metrics.png)

解析后的表格保存在：

```text
outputs/plots/a100_4_prod_v3_stable_20260610_194644/metrics_from_log.csv
```

## 测试

```bash
python -m compileall -q alphazero_gomoku
```

如果后续补充 `tests/` 目录，也可以使用：

```bash
python -m unittest discover tests
```
