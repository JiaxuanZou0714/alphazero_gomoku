# v2 failed experiment

The v2 remote run is kept as a failed continuation experiment.

Summary:

- Baseline: `outputs/checkpoints/v1-old-best/gomoku10_best.pt` (= v1 / old best)
- Remote baseline alias: `/home/featurize/alphazero_gomoku/outputs/checkpoints/gomoku10_best_old.pt`
- v2 checkpoints tested: `gomoku10_iter_0096.pt` through `gomoku10_iter_0112.pt`
- Result: v2 did not reliably beat the old best.

Key matches:

```text
v2 0101 vs old best: 9-7, score 0.5625
v2 0112 vs old best: 8-8, score 0.5000
v2 0096 vs old best: 6-10, score 0.3750
v2 0096 vs v2 0101: 7-9, score 0.4375
```

Files:

- `v2_selection_summary.json`: full screen/final ranking from the remote selection run.
- `v2_selection_20260615.jsonl`: one JSON row per matchup in the selection run.
- `v2_096_recheck_20260615.jsonl`: targeted recheck for the curve-favored `0096` checkpoint.
- `../../plots/v2-failed/`: plots generated from `outputs/metrics/v2.jsonl`.
