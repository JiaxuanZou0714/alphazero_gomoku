from __future__ import annotations

import unittest

import torch

from alphazero_gomoku.model import (
    PolicyValueNet,
    build_model_from_config,
    model_kwargs_from_config,
)


class ModelTest(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        model = PolicyValueNet(channels=8, residual_blocks=1, value_hidden=16)
        model.eval()
        x = torch.zeros(3, 2, 10, 10)
        policy_logits, soft_logits, value = model(x)
        self.assertEqual(policy_logits.shape, (3, 100))
        self.assertIsNone(soft_logits)  # off by default
        self.assertEqual(value.shape, (3,))
        self.assertTrue(torch.all(value >= -1.0) and torch.all(value <= 1.0))

    def test_soft_policy_head_present_when_enabled(self) -> None:
        model = PolicyValueNet(
            channels=8, residual_blocks=1, value_hidden=16, use_soft_policy=True
        )
        _, soft_logits, _ = model(torch.zeros(1, 2, 10, 10))
        self.assertIsNotNone(soft_logits)
        self.assertEqual(soft_logits.shape, (1, 100))

    def test_build_from_config_round_trip(self) -> None:
        cfg = {
            "board_size": 10,
            "channels": 16,
            "residual_blocks": 2,
            "policy_channels": 4,
            "value_channels": 2,
            "value_hidden": 32,
            "use_global_pool": True,
            "use_soft_policy": True,
        }
        model = build_model_from_config(cfg)
        # a checkpoint built from a config must reload into a model built from the
        # config's own model_kwargs (the round-trip train.py relies on).
        kwargs = model_kwargs_from_config(cfg)
        twin = PolicyValueNet(**kwargs)
        twin.load_state_dict(model.state_dict())
        self.assertEqual(kwargs["channels"], 16)
        self.assertTrue(kwargs["use_global_pool"])


if __name__ == "__main__":
    unittest.main()
