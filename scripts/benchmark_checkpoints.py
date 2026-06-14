from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from alphazero_gomoku.game import GomokuState
from alphazero_gomoku.mcts import MCTS, MCTSConfig, visit_count_policy
from alphazero_gomoku.train import format_duration
from alphazero_gomoku.utils import load_model, resolve_device


def parse_simulations(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("simulations must be positive integers")
    return values


def config_value(cfg: dict, name: str, default: object) -> object:
    return cfg.get(name, default)


def select_action(model, cfg: dict, state: GomokuState, simulations: int, device: str) -> int:
    root = MCTS(
        model,
        MCTSConfig(
            simulations=simulations,
            c_puct=float(config_value(cfg, "mcts_c_puct", 1.5)),
            dirichlet_alpha=float(config_value(cfg, "mcts_dirichlet_alpha", 0.3)),
            dirichlet_fraction=float(config_value(cfg, "mcts_dirichlet_fraction", 0.25)),
            eval_batch_size=min(int(config_value(cfg, "mcts_batch_size", 32)), simulations),
            amp_dtype=str(config_value(cfg, "mcts_amp_dtype", "bf16")),
            root_policy_temp=float(config_value(cfg, "mcts_root_policy_temp", 1.0)),
            shaped_dirichlet=bool(config_value(cfg, "mcts_shaped_dirichlet", False)),
            dynamic_cpuct=bool(config_value(cfg, "mcts_dynamic_cpuct", False)),
            fpu_reduction=float(config_value(cfg, "mcts_fpu_reduction", 0.0) or 0.0),
        ),
        device=device,
    ).search(state, add_exploration_noise=False)
    policy = visit_count_policy(root, state.action_size, temperature=0.0)
    if policy.sum() <= 0:
        return int(state.legal_actions()[0])
    return int(policy.argmax())


def random_opening(size: int, win_length: int, opening_moves: int, rng: random.Random) -> list[int]:
    state = GomokuState.new(size=size, win_length=win_length)
    opening: list[int] = []
    for _ in range(opening_moves):
        if state.is_terminal:
            break
        action = int(rng.choice(list(state.legal_actions())))
        opening.append(action)
        state = state.apply(action)
    return opening


def play_matchup(
    candidate,
    candidate_cfg: dict,
    baseline,
    baseline_cfg: dict,
    *,
    candidate_simulations: int,
    baseline_simulations: int,
    games: int,
    opening_moves: int,
    seed: int,
    device: str,
) -> dict[str, object]:
    size = int(config_value(candidate_cfg, "board_size", config_value(baseline_cfg, "board_size", 10)))
    win_length = int(
        config_value(candidate_cfg, "win_length", config_value(baseline_cfg, "win_length", 5))
    )
    rng = random.Random(seed)
    openings = [random_opening(size, win_length, opening_moves, rng) for _ in range((games + 1) // 2)]
    wins = losses = draws = 0
    started = time.monotonic()
    candidate.eval()
    baseline.eval()
    for game_index in range(games):
        state = GomokuState.new(size=size, win_length=win_length)
        for action in openings[game_index // 2]:
            state = state.apply(action)
        candidate_player = 1 if game_index % 2 == 0 else -1
        while not state.is_terminal:
            if state.current_player == candidate_player:
                action = select_action(
                    candidate, candidate_cfg, state, candidate_simulations, device
                )
            else:
                action = select_action(baseline, baseline_cfg, state, baseline_simulations, device)
            state = state.apply(action)
        if state.winner == 0:
            draws += 1
            outcome = "draw"
        elif state.winner == candidate_player:
            wins += 1
            outcome = "candidate"
        else:
            losses += 1
            outcome = "baseline"
        score = (wins + 0.5 * draws) / max(1, game_index + 1)
        print(
            "benchmark_game "
            f"candidate_sims={candidate_simulations} baseline_sims={baseline_simulations} "
            f"game={game_index + 1}/{games} outcome={outcome} moves={state.moves_played} "
            f"wins={wins} losses={losses} draws={draws} score_so_far={score:.3f} "
            f"elapsed={format_duration(time.monotonic() - started)}",
            flush=True,
        )
    score = (wins + 0.5 * draws) / max(1, games)
    return {
        "candidate_simulations": candidate_simulations,
        "baseline_simulations": baseline_simulations,
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": score,
        "seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two Gomoku checkpoints at different MCTS simulation budgets."
    )
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate-sims", type=parse_simulations, default=[128, 256, 512])
    parser.add_argument("--baseline-sims", type=int, default=512)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--opening-moves", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.games <= 0:
        raise SystemExit("--games must be positive")
    device = resolve_device(args.device)
    candidate, candidate_cfg = load_model(args.candidate, device)
    baseline, baseline_cfg = load_model(args.baseline, device)

    summaries = []
    for index, candidate_sims in enumerate(args.candidate_sims):
        print(
            "benchmark_start "
            f"candidate={args.candidate} baseline={args.baseline} "
            f"candidate_sims={candidate_sims} baseline_sims={args.baseline_sims} "
            f"games={args.games} opening_moves={args.opening_moves} device={device}",
            flush=True,
        )
        summary = play_matchup(
            candidate,
            candidate_cfg,
            baseline,
            baseline_cfg,
            candidate_simulations=candidate_sims,
            baseline_simulations=args.baseline_sims,
            games=args.games,
            opening_moves=args.opening_moves,
            seed=args.seed + index * 1009,
            device=device,
        )
        summaries.append(summary)
        print("benchmark_result " + json.dumps(summary, sort_keys=True), flush=True)

    payload = {
        "candidate": str(args.candidate),
        "baseline": str(args.baseline),
        "baseline_simulations": args.baseline_sims,
        "results": summaries,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
