# 10x10 AlphaZero Gomoku

This is a small AlphaZero-style training project for 10x10 Gomoku.

The implementation does **not** use hand-written Gomoku knowledge such as openings,
threat patterns, live-threes, live-fours, or handcrafted evaluations. The only
rules encoded are the environment rules required to generate data:

- the board is 10x10;
- players alternate legal moves on empty intersections;
- five stones in a row wins;
- a full board without a winner is a draw.

The neural network learns from self-play. MCTS uses the network's policy and value
outputs, and the network is trained from MCTS visit distributions plus final game
outcomes.

## Run a Tiny Smoke Training

```powershell
python -m alphazero_gomoku.train --iterations 1 --games-per-iteration 1 --simulations 4 --epochs 1 --channels 8 --residual-blocks 1
```

This only verifies the loop. It will not produce a strong player.

## Train Longer

```powershell
python -m alphazero_gomoku.train --iterations 50 --games-per-iteration 20 --simulations 128 --epochs 3
```

Checkpoints are written to `outputs/checkpoints`.

The default device is `auto`: CUDA is used when the active Python environment has
a CUDA-enabled PyTorch build, otherwise CPU is used. On this machine, the GPU
Torch environment is:

```powershell
C:\Users\123\miniconda3\python.exe -m alphazero_gomoku.train --iterations 50 --games-per-iteration 20 --simulations 128 --epochs 3
```

## 4x A100 Server Preset

On `labserver`, use the `modded-nanogpt` environment and launch from the parent
directory of the package:

```bash
cd ~/jiaxuanzou
conda activate modded-nanogpt

python -m alphazero_gomoku.train --preset a100-4
```

The `a100-4` preset uses a larger SE-ResNet, larger training batches, fixed
optimizer steps per iteration, cosine LR decay, 16 parallel self-play workers
spread over the visible CUDA devices, and multi-GPU training for the supervised
update step.

For the fastest first model, use `a100-turbo`:

```bash
python -m alphazero_gomoku.train --preset a100-turbo
```

This uses fewer MCTS simulations, a smaller network, 16 self-play workers, TF32
math on CUDA, and batched MCTS leaf evaluation. It is designed to get a playable
checkpoint quickly. After that, use `a100-fast` or `a100-4` for refinement:

```bash
python -m alphazero_gomoku.train --preset a100-fast
```

For the production run on the lab server, launch from `~/jiaxuanzou` and keep
all artifacts inside the package directory:

```bash
python -m alphazero_gomoku.train \
  --preset a100-prod \
  --checkpoint-dir alphazero_gomoku/outputs/checkpoints/a100-prod-v2 \
  --replay-path alphazero_gomoku/outputs/replay/a100-prod-v2_replay.pt \
  --metrics-path alphazero_gomoku/outputs/metrics/a100-prod-v2.jsonl
```

Do not judge training by total loss alone. Early policy targets can be high
entropy, so policy loss often sits near `ln(100) = 4.605`. Watch `policy_kl`,
`target_entropy`, `pred_entropy`, `policy_top1`, `value_mae`, `value_acc`, and
candidate-vs-champion `eval_score`.

For a shorter test run:

```bash
python -m alphazero_gomoku.train \
  --preset a100-4 \
  --iterations 1 \
  --games-per-iteration 8 \
  --simulations 16 \
  --epochs 1 \
  --checkpoint-dir alphazero_gomoku/outputs/checkpoints/a100-smoke
```

## Play Against a Checkpoint

```powershell
python -m alphazero_gomoku.play outputs/checkpoints/gomoku10_iter_0001.pt --simulations 128 --human black
```

Rows and columns are 1-indexed.

## Local Web Board

```powershell
C:\Users\123\miniconda3\python.exe -m alphazero_gomoku.web_play `
  outputs\checkpoints\a100-turbo\gomoku10_iter_0012.pt `
  --simulations 64
```

Then open:

```text
http://127.0.0.1:8765
```

## Tests

```powershell
python -m unittest discover tests
```
