from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, object]]:
    by_iteration: dict[int, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        iteration = int(row.get("iteration", -1))
        if iteration >= 0:
            by_iteration[iteration] = row
    return [by_iteration[key] for key in sorted(by_iteration)]


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "iteration",
        "loss",
        "policy_loss",
        "soft_policy_loss",
        "value_loss",
        "policy_kl",
        "policy_top1",
        "value_acc",
        "value_mae",
        "target_entropy",
        "pred_entropy",
        "selfplay_entropy",
        "black_win_rate",
        "white_win_rate",
        "draw_rate",
        "avg_moves",
        "replay",
        "lr",
        "grad_norm",
        "iter_seconds",
        "total_seconds",
        "evaluated",
        "eval_score",
        "promoted",
        "checkpoint",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(rows: list[dict[str, object]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    x = [int(row["iteration"]) for row in rows]

    def values(name: str) -> list[float | None]:
        result: list[float | None] = []
        for row in rows:
            value = row.get(name)
            result.append(float(value) if isinstance(value, (int, float)) else None)
        return result

    def plot_lines(filename: str, title: str, names: list[str]) -> None:
        plt.figure(figsize=(11, 6), dpi=150)
        for name in names:
            y = values(name)
            if any(value is not None for value in y):
                plt.plot(x, y, marker="o", markersize=2.5, linewidth=1.4, label=name)
        plt.title(title)
        plt.xlabel("iteration")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename)
        plt.close()

    plot_lines("losses.png", "training losses", ["loss", "policy_loss", "soft_policy_loss", "value_loss"])
    plot_lines("policy_value.png", "policy and value quality", ["policy_top1", "value_acc", "value_mae"])
    plot_lines("entropy_kl.png", "entropy and policy KL", ["policy_kl", "target_entropy", "pred_entropy", "selfplay_entropy"])
    plot_lines("selfplay_outcomes.png", "self-play outcomes", ["black_win_rate", "white_win_rate", "draw_rate"])
    plot_lines("timing_replay.png", "timing and replay", ["iter_seconds", "replay"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
    panels = [
        (axes[0, 0], ["loss", "policy_loss", "value_loss"], "Loss"),
        (axes[0, 1], ["policy_top1", "value_acc"], "Policy/value accuracy"),
        (axes[1, 0], ["policy_kl", "selfplay_entropy"], "KL/entropy"),
        (axes[1, 1], ["black_win_rate", "white_win_rate"], "Self-play win rates"),
    ]
    for ax, names, title in panels:
        for name in names:
            y = values(name)
            if any(value is not None for value in y):
                ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.4, label=name)
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("training metrics")
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_overview.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AlphaZero Gomoku training metrics.")
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = load_rows(args.metrics)
    if not rows:
        raise SystemExit(f"no metric rows found in {args.metrics}")
    plot_metrics(rows, args.out_dir)
    write_csv(rows, args.out_dir / "metrics.csv")
    print(f"wrote {len(rows)} rows to {args.out_dir}")


if __name__ == "__main__":
    main()
