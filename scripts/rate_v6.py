"""Rate the v6 model against the existing ladder and splice it into the Elo
leaderboard.

Same methodology as the v0 rater and the original round-robin: 128-sim noiseless
MCTS, colour-swapped games, shared random openings. v6 plays the four neural
rivals (v5/v4/v3/v1) net-vs-net, and the v0 hand-written heuristic as the fifth
opponent. The existing v0..v5 ratings are kept fixed and only v6's rating is fit
by maximum likelihood on the same Elo scale, so adding v6 does not move the
published numbers.
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

V6_CKPT = "outputs/checkpoints/v6cont-3080/gomoku10_best.pt"
NEURAL_RIVALS = {
    "v5": "outputs/checkpoints/v5-tiny-3080/gomoku10_best.pt",
    "v4": "outputs/checkpoints/v4-student-3080/gomoku10_best.pt",
    "v3": "outputs/checkpoints/v3-student-local/gomoku10_best.pt",
    "v1": "outputs/checkpoints/v1-old-best/gomoku10_best.pt",
}


def play_vs_neural(v6, v6_cfg, opp, opp_cfg, *, sims, games, opening_moves, seed, device):
    """v6 vs one neural rival. Returns v6's W/L/D."""
    size = int(v6_cfg.get("board_size", 10))
    win_length = int(v6_cfg.get("win_length", 5))
    rng = random.Random(seed)
    openings = [random_opening(size, win_length, opening_moves, rng) for _ in range((games + 1) // 2)]
    v6_mcfg = mcts_config_from_cfg(v6_cfg, sims, for_eval=True)
    opp_mcfg = mcts_config_from_cfg(opp_cfg, sims, for_eval=True)
    v6.eval()
    opp.eval()
    wins = losses = draws = 0
    started = time.monotonic()
    for game_index in range(games):
        state = GomokuState.new(size=size, win_length=win_length)
        for action in openings[game_index // 2]:
            state = state.apply(action)
        v6_player = 1 if game_index % 2 == 0 else -1
        while not state.is_terminal:
            if state.current_player == v6_player:
                action = greedy_action(v6, v6_mcfg, state, device)
            else:
                action = greedy_action(opp, opp_mcfg, state, device)
            state = state.apply(action)
        if state.winner == 0:
            draws += 1
        elif state.winner == v6_player:
            wins += 1
        else:
            losses += 1
        print(f"v6_game vs_neural game={game_index + 1}/{games} W={wins} L={losses} D={draws} "
              f"elapsed={format_duration(time.monotonic() - started)}", flush=True)
    return {"wins": wins, "losses": losses, "draws": draws, "games": games}


def play_vs_v0(v6, v6_cfg, *, sims, games, opening_moves, seed, device):
    """v6 vs the v0 heuristic. Returns v6's W/L/D."""
    size = int(v6_cfg.get("board_size", 10))
    win_length = int(v6_cfg.get("win_length", 5))
    rng = random.Random(seed)
    openings = [random_opening(size, win_length, opening_moves, rng) for _ in range((games + 1) // 2)]
    v6_mcfg = mcts_config_from_cfg(v6_cfg, sims, for_eval=True)
    v6.eval()
    wins = losses = draws = 0
    started = time.monotonic()
    for game_index in range(games):
        state = GomokuState.new(size=size, win_length=win_length)
        for action in openings[game_index // 2]:
            state = state.apply(action)
        v6_player = 1 if game_index % 2 == 0 else -1
        while not state.is_terminal:
            if state.current_player == v6_player:
                action = greedy_action(v6, v6_mcfg, state, device)
            else:
                # Deterministic (rng=None) so v0 plays EXACTLY the browser engine;
                # variety comes from random openings.
                action = select_move(state, None)
            state = state.apply(action)
        if state.winner == 0:
            draws += 1
        elif state.winner == v6_player:
            wins += 1
        else:
            losses += 1
        print(f"v6_game vs_v0 game={game_index + 1}/{games} W={wins} L={losses} D={draws} "
              f"elapsed={format_duration(time.monotonic() - started)}", flush=True)
    return {"wins": wins, "losses": losses, "draws": draws, "games": games}


def expected_score(r_self: float, r_opp: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_opp - r_self) / 400.0))


def fit_rating(results: dict[str, dict], opp_elo: dict[str, float]) -> float:
    """1-D MLE of v6's Elo given fixed opponent ratings (convex -> grid + ternary)."""

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
    parser = argparse.ArgumentParser(description="Rate v6 and update leaderboard.json")
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--games", type=int, default=40, help="games per opponent (colour-balanced)")
    parser.add_argument("--opening-moves", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--leaderboard", type=Path, default=REPO_DIR / "docs/assets/leaderboard.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    board = json.loads(args.leaderboard.read_text(encoding="utf-8"))
    opp_elo = {row["id"]: float(row["elo"]) for row in board["ratings"]}

    v6, v6_cfg = load_model(REPO_DIR / V6_CKPT, device)

    results: dict[str, dict] = {}
    for index, (name, rel) in enumerate(NEURAL_RIVALS.items()):
        opp, opp_cfg = load_model(REPO_DIR / rel, device)
        print(f"=== v6 vs {name} ({rel}) sims={args.sims} games={args.games} ===", flush=True)
        res = play_vs_neural(v6, v6_cfg, opp, opp_cfg, sims=args.sims, games=args.games,
                             opening_moves=args.opening_moves, seed=args.seed + index * 1009, device=device)
        res["v6_score"] = (res["wins"] + 0.5 * res["draws"]) / res["games"]
        results[name] = res
        print(f"RESULT v6 vs {name}: {json.dumps(res)}", flush=True)
        del opp

    print(f"=== v6 vs v0 (heuristic) sims={args.sims} games={args.games} ===", flush=True)
    res0 = play_vs_v0(v6, v6_cfg, sims=args.sims, games=args.games,
                      opening_moves=args.opening_moves, seed=args.seed + 9090, device=device)
    res0["v6_score"] = (res0["wins"] + 0.5 * res0["draws"]) / res0["games"]
    results["v0"] = res0
    print(f"RESULT v6 vs v0: {json.dumps(res0)}", flush=True)

    r6 = fit_rating(results, opp_elo)
    print(f"\nFitted v6 Elo = {r6:.1f} (opponents fixed: {opp_elo})", flush=True)
    for name, res in results.items():
        print(f"  vs {name}: v6 score {res['v6_score']:.3f}  (expected {expected_score(r6, opp_elo[name]):.3f})")

    # ---- splice v6 into leaderboard.json -------------------------------------
    board["ratings"] = [row for row in board["ratings"] if row["id"] != "v6"]
    board["ratings"].append({"id": "v6", "elo": int(round(r6)), "arch": "64×5"})
    board["ratings"].sort(key=lambda row: row["elo"], reverse=True)
    matrix = board.setdefault("matrix", {})
    matrix["v6"] = {"v6": None}
    for name, res in results.items():
        s = round(res["v6_score"], 3)
        matrix["v6"][name] = s
        matrix.setdefault(name, {})["v6"] = round(1 - s, 3)
    board["v6_games_per_opponent"] = args.games
    board["note"] = "循环赛 Elo · 128 sims · 颜色互换 · v1–v5 各 60 局，v0/v6 各 " + str(args.games) + " 局"
    board["updated"] = "2026-06-18"

    if args.dry_run:
        print("\n[dry-run] leaderboard NOT written. Preview ratings:")
        print(json.dumps(board["ratings"], ensure_ascii=False, indent=2))
        return
    args.leaderboard.write_text(json.dumps(board, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.leaderboard}")


if __name__ == "__main__":
    main()
