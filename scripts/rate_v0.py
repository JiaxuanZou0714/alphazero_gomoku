"""Rate the v0 hand-written heuristic against the learned models and slot it
into the Elo leaderboard.

Methodology mirrors the existing leaderboard round-robin: 128-sim noiseless MCTS
for the neural models, colour-swapped games, shared random openings. v0 itself
uses no search. Because the v1..v5 ratings are already published, we keep them
fixed and fit *only* v0's rating by maximum likelihood on the same Elo scale, so
adding a new player does not silently move the existing numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from alphazero_gomoku.game import GomokuState
from alphazero_gomoku.heuristic_v0 import select_move
from alphazero_gomoku.inference import greedy_action, mcts_config_from_cfg, random_opening
from alphazero_gomoku.utils import format_duration, load_model, resolve_device

CHECKPOINTS = {
    "v1": "outputs/checkpoints/v1-old-best/gomoku10_best.pt",
    "v3": "outputs/checkpoints/v3-student-local/gomoku10_best.pt",
    "v4": "outputs/checkpoints/v4-student-3080/gomoku10_best.pt",
    "v5": "outputs/checkpoints/v5-tiny-3080/gomoku10_best.pt",
}


def play_vs_model(model, cfg, *, sims, games, opening_moves, seed, device):
    """Play v0 (heuristic) against one neural model. Returns v0's W/L/D."""
    size = int(cfg.get("board_size", 10))
    win_length = int(cfg.get("win_length", 5))
    rng = random.Random(seed)
    openings = [random_opening(size, win_length, opening_moves, rng) for _ in range((games + 1) // 2)]
    mcfg = mcts_config_from_cfg(cfg, sims, for_eval=True)
    model.eval()
    wins = losses = draws = 0
    started = time.monotonic()
    for game_index in range(games):
        state = GomokuState.new(size=size, win_length=win_length)
        for action in openings[game_index // 2]:
            state = state.apply(action)
        v0_player = 1 if game_index % 2 == 0 else -1
        while not state.is_terminal:
            if state.current_player == v0_player:
                # Deterministic (rng=None) so the rated engine plays EXACTLY the
                # browser's v0Evaluate; game variety comes from random openings.
                action = select_move(state, None)
            else:
                action = greedy_action(model, mcfg, state, device)
            state = state.apply(action)
        if state.winner == 0:
            draws += 1
        elif state.winner == v0_player:
            wins += 1
        else:
            losses += 1
        print(
            f"v0_game game={game_index + 1}/{games} W={wins} L={losses} D={draws} "
            f"elapsed={format_duration(time.monotonic() - started)}",
            flush=True,
        )
    return {"wins": wins, "losses": losses, "draws": draws, "games": games}


def expected_score(r_self: float, r_opp: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_opp - r_self) / 400.0))


def fit_rating(results: dict[str, dict], opp_elo: dict[str, float]) -> float:
    """1-D MLE of v0's Elo given fixed opponent ratings.

    The negative log-likelihood is convex in the rating, so a coarse grid scan
    followed by a ternary-search refinement finds the global optimum reliably.
    """

    def nll(r0: float) -> float:
        total = 0.0
        for opp, res in results.items():
            n = res["games"]
            score = (res["wins"] + 0.5 * res["draws"]) / n
            e = min(max(expected_score(r0, opp_elo[opp]), 1e-12), 1 - 1e-12)
            total -= n * (score * math.log(e) + (1 - score) * math.log(1 - e))
        return total

    seed = min(range(0, 4001, 5), key=nll)
    lo, hi = seed - 5.0, seed + 5.0
    for _ in range(60):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if nll(m1) < nll(m2):
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Rate v0 heuristic and update leaderboard.json")
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--games", type=int, default=30, help="games per opponent (colour-balanced)")
    parser.add_argument("--opening-moves", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--leaderboard", type=Path, default=REPO_DIR / "docs/assets/leaderboard.json")
    parser.add_argument("--dry-run", action="store_true", help="do not write leaderboard.json")
    args = parser.parse_args()

    device = resolve_device(args.device)
    board = json.loads(args.leaderboard.read_text(encoding="utf-8"))
    opp_elo = {row["id"]: float(row["elo"]) for row in board["ratings"]}

    results: dict[str, dict] = {}
    for index, (name, rel) in enumerate(CHECKPOINTS.items()):
        model, cfg = load_model(REPO_DIR / rel, device)
        print(f"=== v0 vs {name} ({rel}) sims={args.sims} games={args.games} ===", flush=True)
        res = play_vs_model(
            model, cfg,
            sims=args.sims, games=args.games, opening_moves=args.opening_moves,
            seed=args.seed + index * 1009, device=device,
        )
        res["v0_score"] = (res["wins"] + 0.5 * res["draws"]) / res["games"]
        results[name] = res
        print(f"RESULT v0 vs {name}: {json.dumps(res)}", flush=True)
        del model

    r0 = fit_rating(results, opp_elo)
    print(f"\nFitted v0 Elo = {r0:.1f} (opponents fixed: {opp_elo})", flush=True)
    for name, res in results.items():
        print(f"  vs {name}: v0 score {res['v0_score']:.3f}  (expected {expected_score(r0, opp_elo[name]):.3f})")

    # ---- splice v0 into leaderboard.json -------------------------------------
    arch = {"v0": "heuristic"}
    board["ratings"].append({"id": "v0", "elo": int(round(r0)), "arch": "手写启发式"})
    board["ratings"].sort(key=lambda row: row["elo"], reverse=True)
    matrix = board.setdefault("matrix", {})
    matrix["v0"] = {"v0": None}
    for name, res in results.items():
        s = round(res["v0_score"], 3)
        matrix["v0"][name] = s
        matrix.setdefault(name, {})["v0"] = round(1 - s, 3)
    board["v0_games_per_opponent"] = args.games
    board["note"] = "循环赛 Elo · 128 sims · 颜色互换 · v1–v5 各 60 局，v0 各 30 局"
    board["updated"] = "2026-06-17"

    if args.dry_run:
        print("\n[dry-run] leaderboard NOT written. Preview:")
        print(json.dumps(board, ensure_ascii=False, indent=2))
        return
    args.leaderboard.write_text(json.dumps(board, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.leaderboard}")


if __name__ == "__main__":
    main()
