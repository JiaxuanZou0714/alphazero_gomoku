from __future__ import annotations

import random
import unittest

import numpy as np
import torch

from alphazero_gomoku.game import GomokuState
from alphazero_gomoku.mcts import MCTS, MCTSConfig


class UniformNet(torch.nn.Module):
    """Uniform policy, zero value: search behaviour comes from the game alone."""

    def __init__(self, action_size: int = 100) -> None:
        super().__init__()
        self.action_size = action_size
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        n = x.shape[0]
        return torch.zeros(n, self.action_size), None, torch.zeros(n)


def winning_position() -> GomokuState:
    """Black has an open four at (0,0)-(0,3); black to move wins at (0,4)."""
    board = np.zeros((10, 10), dtype=np.int8)
    board[0, 0:4] = 1
    board[5, 0:4] = -1
    return GomokuState(board=board, current_player=1, moves_played=8)


class MCTSTest(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(0)

    def test_finds_winning_move_and_root_value_sign(self) -> None:
        state = winning_position()
        mcts = MCTS(UniformNet(), MCTSConfig(simulations=300), device="cpu")
        root = mcts.search(state, add_exploration_noise=True)
        best_action = max(root.children.items(), key=lambda kv: kv[1].visit_count)[0]
        self.assertEqual(best_action, state.coord_to_action(0, 4))
        # root.value is from the current player's perspective: black is winning
        self.assertGreater(root.value, 0.5)

    def test_policy_target_normalised(self) -> None:
        state = winning_position()
        mcts = MCTS(UniformNet(), MCTSConfig(simulations=64), device="cpu")
        root = mcts.search(state, add_exploration_noise=True)
        target = mcts.policy_target(root, state.action_size)
        self.assertAlmostEqual(float(target.sum()), 1.0, places=5)
        self.assertGreaterEqual(target.min(), 0.0)

    def test_policy_target_pruning_keeps_best_move(self) -> None:
        state = winning_position()
        cfg = MCTSConfig(simulations=300, forced_playouts=True, fpu_reduction=0.2)
        mcts = MCTS(UniformNet(), cfg, device="cpu")
        root = mcts.search(state, add_exploration_noise=True)
        pruned = mcts.policy_target(root, state.action_size, pruned=True)
        raw = mcts.policy_target(root, state.action_size, pruned=False)
        win = state.coord_to_action(0, 4)
        self.assertEqual(int(pruned.argmax()), win)
        self.assertAlmostEqual(float(pruned.sum()), 1.0, places=5)
        # pruning removes forced visits, so the best move's share cannot shrink
        self.assertGreaterEqual(pruned[win], raw[win] - 1e-6)

    def test_raw_prior_unaffected_by_noise(self) -> None:
        state = GomokuState.new(size=10)
        mcts = MCTS(UniformNet(), MCTSConfig(simulations=8, dirichlet_fraction=0.5), device="cpu")
        root = mcts.search(state, add_exploration_noise=True)
        raw = np.array([child.raw_prior for child in root.children.values()])
        # uniform net: raw priors stay exactly uniform even after root noise
        self.assertTrue(np.allclose(raw, raw[0]))
        noised = np.array([child.prior for child in root.children.values()])
        self.assertFalse(np.allclose(noised, noised[0]))

    def test_tree_reuse_accumulates_visits(self) -> None:
        state = GomokuState.new(size=10)
        mcts = MCTS(UniformNet(), MCTSConfig(simulations=16), device="cpu")
        root = mcts.search(state)
        action = next(iter(root.children))
        child = root.children[action]
        prior_visits = child.visit_count
        next_state = state.apply(action)
        reused = mcts.search(next_state, reuse_root=child, simulations=16)
        if prior_visits > 0 and child.expanded:
            self.assertIs(reused, child)
        self.assertGreaterEqual(reused.visit_count, 16)

    def test_terminal_root_raises(self) -> None:
        state = GomokuState.new(size=2, win_length=3)
        for action in range(4):
            state = state.apply(action)
        mcts = MCTS(UniformNet(action_size=4), MCTSConfig(simulations=4), device="cpu")
        with self.assertRaises(ValueError):
            mcts.search(state)

    def test_seeded_search_is_reproducible(self) -> None:
        state = GomokuState.new(size=10)
        visits = []
        for _ in range(2):
            random.seed(123)
            mcts = MCTS(UniformNet(), MCTSConfig(simulations=32), device="cpu")
            root = mcts.search(state, add_exploration_noise=True)
            visits.append({a: c.visit_count for a, c in root.children.items()})
        self.assertEqual(visits[0], visits[1])


if __name__ == "__main__":
    unittest.main()
