from __future__ import annotations

import argparse
from pathlib import Path

from .game import GomokuState
from .mcts import MCTS, MCTSConfig, visit_count_policy
from .utils import load_model, resolve_device


def parse_move(raw: str, state: GomokuState) -> int:
    parts = raw.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError("enter row and column, for example: 5 6")
    row, col = int(parts[0]) - 1, int(parts[1]) - 1
    action = state.coord_to_action(row, col)
    if not state.legal_mask()[action]:
        raise ValueError("that point is already occupied")
    return action


def main() -> None:
    parser = argparse.ArgumentParser(description="Play against a trained Gomoku model.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--human", choices=["black", "white"], default="black")
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    model, cfg = load_model(args.checkpoint, args.device)
    state = GomokuState.new(
        size=int(cfg.get("board_size", 10)), win_length=int(cfg.get("win_length", 5))
    )
    human_player = 1 if args.human == "black" else -1
    mcts = MCTS(
        model,
        MCTSConfig(
            simulations=args.simulations,
            c_puct=float(cfg.get("mcts_c_puct", 1.5)),
            dirichlet_alpha=float(cfg.get("mcts_dirichlet_alpha", 0.3)),
            dirichlet_fraction=float(cfg.get("mcts_dirichlet_fraction", 0.25)),
            eval_batch_size=min(int(cfg.get("mcts_batch_size", 1)), max(1, args.simulations)),
            amp_dtype=str(
                cfg.get(
                    "mcts_amp_dtype",
                    str(cfg.get("amp_dtype", "bf16")) if bool(cfg.get("amp", True)) else "none",
                )
            ),
        ),
        device=args.device,
    )

    while not state.is_terminal:
        print(state.render())
        if state.current_player == human_player:
            while True:
                try:
                    action = parse_move(input("Your move (row col): "), state)
                    break
                except (ValueError, IndexError) as exc:
                    print(f"Invalid move: {exc}")
        else:
            root = mcts.search(state)
            policy = visit_count_policy(root, state.action_size, temperature=0.0)
            action = int(policy.argmax())
            row, col = state.action_to_coord(action)
            print(f"AI move: {row + 1} {col + 1}")
        state = state.apply(action)

    print(state.render())
    if state.winner == 0:
        print("Draw.")
    elif state.winner == human_player:
        print("You win.")
    else:
        print("AI wins.")


if __name__ == "__main__":
    main()
