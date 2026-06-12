from __future__ import annotations

import random
import tempfile
import unittest
from collections import deque
from pathlib import Path

import numpy as np
import torch

from alphazero_gomoku.mcts import MCTSConfig
from alphazero_gomoku.model import PolicyValueNet
from alphazero_gomoku.train import (
    TrainConfig,
    apply_random_symmetries,
    load_replay,
    play_self_game,
    replay_to_dataset,
    save_replay,
    train_epoch,
)


def tiny_model() -> PolicyValueNet:
    return PolicyValueNet(channels=8, residual_blocks=1, value_hidden=16)


def make_examples(n: int) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    examples = []
    for i in range(n):
        state = np.zeros((2, 10, 10), dtype=np.float32)
        state[0, i % 10, (i * 3) % 10] = 1.0
        policy = np.full(100, 1.0 / 100, dtype=np.float32)
        examples.append((state, policy, 1.0 if i % 2 == 0 else -1.0, 1.0))
    return examples


class SymmetryTest(unittest.TestCase):
    def test_policy_follows_state(self) -> None:
        torch.manual_seed(0)
        n, size = 32, 10
        states = torch.zeros(n, 2, size, size)
        policies = torch.zeros(n, size * size)
        for i in range(n):
            r, c = i % size, (i * 7) % size
            states[i, 0, r, c] = 1.0
            policies[i, r * size + c] = 1.0
        out_states, out_policies = apply_random_symmetries(states, policies, size)
        # the policy mass must land exactly where the marked stone moved to
        self.assertTrue(torch.equal(out_states[:, 0].reshape(n, -1), out_policies))
        # all 8 transforms are bijections: mass is conserved
        self.assertTrue(torch.equal(out_states.sum(dim=(1, 2, 3)), states.sum(dim=(1, 2, 3))))


class SelfPlayTest(unittest.TestCase):
    def test_play_self_game_smoke(self) -> None:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        cfg = TrainConfig(
            simulations=8,
            mcts_batch_size=4,
            temperature_moves=4,
            mcts_value_weight=0.5,
            playout_cap_randomization=True,
            full_search_prob=0.5,
            fast_simulations=4,
            mcts_forced_playouts=True,
            mcts_fpu_reduction=0.2,
            selfplay_tree_reuse=True,
        )
        cfg.device = "cpu"
        mcts_cfg = MCTSConfig(
            simulations=cfg.simulations,
            eval_batch_size=cfg.mcts_batch_size,
            amp_dtype="none",
            forced_playouts=True,
            fpu_reduction=0.2,
        )
        examples, kls, stats = play_self_game(tiny_model(), cfg, mcts_cfg)
        self.assertGreater(len(examples), 0)
        self.assertEqual(len(kls), len(examples))
        for state, policy, value, policy_weight in examples:
            self.assertEqual(state.shape, (2, 10, 10))
            self.assertAlmostEqual(float(policy.sum()), 1.0, places=4)
            self.assertLessEqual(abs(value), 1.0 + 1e-6)
            self.assertIn(policy_weight, (0.0, 1.0))
        self.assertIn(stats["winner"], (-1.0, 0.0, 1.0))
        self.assertGreater(stats["full_search_rate"], 0.0)


class TrainEpochTest(unittest.TestCase):
    def test_empty_loader_does_not_crash(self) -> None:
        cfg = TrainConfig(batch_size=64, train_steps_per_iteration=2)
        cfg.device = "cpu"
        cfg.amp = False
        model = tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        replay = deque(make_examples(8))
        stats = train_epoch(model, optimizer, replay, cfg, scaler=None)
        self.assertEqual(stats.optimizer_steps, 0)

    def test_one_step_runs(self) -> None:
        cfg = TrainConfig(batch_size=16, train_steps_per_iteration=2)
        cfg.device = "cpu"
        cfg.amp = False
        model = tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        replay = deque(make_examples(64))
        stats = train_epoch(model, optimizer, replay, cfg, scaler=None)
        self.assertEqual(stats.optimizer_steps, 2)
        self.assertTrue(np.isfinite(stats.loss))


class ReplayRoundTripTest(unittest.TestCase):
    def test_v2_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "replay.pt")
            examples = make_examples(6)
            kls = [0.1 * i for i in range(6)]
            save_replay(deque(examples), deque(kls), path)
            cfg = TrainConfig(replay_path=path, replay_size=100)
            replay, kl_buffer = load_replay(cfg)
            self.assertEqual(len(replay), 6)
            self.assertEqual(list(kl_buffer), kls)
            dataset, weights = replay_to_dataset(replay, kl_buffer)
            self.assertEqual(len(dataset), 6)
            self.assertIsNotNone(weights)

    def test_legacy_v1_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "replay.pt")
            legacy = [(e[0], e[1], e[2]) for e in make_examples(4)]
            torch.save(legacy, path)
            cfg = TrainConfig(replay_path=path, replay_size=100)
            replay, kl_buffer = load_replay(cfg)
            self.assertEqual(len(replay), 4)
            self.assertEqual(len(kl_buffer), 4)
            # legacy examples get policy_weight=1.0 appended
            self.assertEqual(replay[0][3], 1.0)


if __name__ == "__main__":
    unittest.main()
