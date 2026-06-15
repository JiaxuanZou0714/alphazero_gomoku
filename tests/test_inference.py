from __future__ import annotations

import random
import unittest

from alphazero_gomoku.game import GomokuState
from alphazero_gomoku.inference import (
    greedy_action,
    mcts_config_from_cfg,
    random_opening,
)
from alphazero_gomoku.model import PolicyValueNet


def tiny_model() -> PolicyValueNet:
    model = PolicyValueNet(channels=8, residual_blocks=1, value_hidden=16)
    model.eval()
    return model


class MctsConfigFromCfgTest(unittest.TestCase):
    def test_reads_keys_and_caps_batch(self) -> None:
        cfg = {"mcts_c_puct": 1.25, "mcts_batch_size": 64, "mcts_fpu_reduction": 0.2}
        mcfg = mcts_config_from_cfg(cfg, simulations=8)
        self.assertEqual(mcfg.c_puct, 1.25)
        self.assertEqual(mcfg.simulations, 8)
        self.assertEqual(mcfg.eval_batch_size, 8)  # min(64, 8)
        self.assertEqual(mcfg.fpu_reduction, 0.2)

    def test_for_eval_omits_forced_playouts(self) -> None:
        cfg = {"mcts_forced_playouts": True, "mcts_forced_playout_k": 2.0}
        train_mcfg = mcts_config_from_cfg(cfg, 16, for_eval=False)
        eval_mcfg = mcts_config_from_cfg(cfg, 16, for_eval=True)
        self.assertTrue(train_mcfg.forced_playouts)
        self.assertFalse(eval_mcfg.forced_playouts)

    def test_amp_default_respects_amp_flag(self) -> None:
        self.assertEqual(mcts_config_from_cfg({"amp": False}, 4).amp_dtype, "none")
        self.assertEqual(
            mcts_config_from_cfg({"amp": True, "amp_dtype": "bf16"}, 4).amp_dtype, "bf16"
        )


class GreedyActionTest(unittest.TestCase):
    def test_returns_legal_action(self) -> None:
        model = tiny_model()
        state = GomokuState.new()
        mcfg = mcts_config_from_cfg({}, simulations=8, for_eval=True)
        action = greedy_action(model, mcfg, state, device="cpu")
        self.assertTrue(state.legal_mask()[action])


class RandomOpeningTest(unittest.TestCase):
    def test_seeded_and_legal(self) -> None:
        a = random_opening(10, 5, 4, random.Random(1))
        b = random_opening(10, 5, 4, random.Random(1))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 4)
        self.assertEqual(len(set(a)), 4)


if __name__ == "__main__":
    unittest.main()
