from __future__ import annotations

import json
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
    PRESETS,
    TrainConfig,
    apply_random_symmetries,
    build_train_loader,
    compute_step_losses,
    evaluate_candidate,
    effective_train_steps,
    load_replay,
    mcts_config_from_train,
    play_self_game,
    preset_config,
    random_opening,
    replay_to_dataset,
    run_training,
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
    def test_small_replay_falls_back_to_small_batches(self) -> None:
        # replay smaller than one batch must still train (drop_last disabled)
        # instead of silently skipping the whole iteration
        cfg = TrainConfig(batch_size=64, train_steps_per_iteration=2)
        cfg.device = "cpu"
        cfg.amp = False
        model = tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        replay = deque(make_examples(8))
        stats = train_epoch(model, optimizer, replay, cfg, scaler=None)
        self.assertEqual(stats.optimizer_steps, 2)
        self.assertTrue(np.isfinite(stats.loss))

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

    def test_effective_train_steps_caps_replay_passes(self) -> None:
        cfg = TrainConfig(
            batch_size=16,
            train_steps_per_iteration=10,
            max_train_replay_passes=1.0,
        )
        self.assertEqual(effective_train_steps(cfg, examples=56), 4)

    def test_train_epoch_accepts_step_override(self) -> None:
        cfg = TrainConfig(batch_size=16, train_steps_per_iteration=10)
        cfg.device = "cpu"
        cfg.amp = False
        model = tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        replay = deque(make_examples(64))
        stats = train_epoch(model, optimizer, replay, cfg, scaler=None, train_steps_to_run=3)
        self.assertEqual(stats.optimizer_steps, 3)
        self.assertTrue(np.isfinite(stats.loss))


class EvaluationTest(unittest.TestCase):
    def test_random_opening_is_legal_and_seeded(self) -> None:
        cfg = TrainConfig(eval_opening_moves=4)
        opening_a = random_opening(cfg, random.Random(42))
        opening_b = random_opening(cfg, random.Random(42))
        self.assertEqual(opening_a, opening_b)
        self.assertEqual(len(opening_a), 4)
        self.assertEqual(len(set(opening_a)), 4)  # no repeated squares

    def test_paired_openings_smoke(self) -> None:
        torch.manual_seed(0)
        cfg = TrainConfig(eval_games=2, eval_simulations=4, eval_opening_moves=2)
        cfg.device = "cpu"
        result = evaluate_candidate(tiny_model(), tiny_model(), cfg, rng=random.Random(7))
        self.assertEqual(
            result["wins"] + result["losses"] + result["draws"], float(cfg.eval_games)
        )
        self.assertGreaterEqual(result["win_rate"], 0.0)
        self.assertLessEqual(result["win_rate"], 1.0)

    def test_eval_early_cutoff_stops_impossible_match(self) -> None:
        torch.manual_seed(0)
        cfg = TrainConfig(
            eval_games=4,
            eval_simulations=2,
            eval_opening_moves=0,
            eval_early_cutoff=True,
            promotion_threshold=2.0,
        )
        cfg.device = "cpu"
        result = evaluate_candidate(tiny_model(), tiny_model(), cfg, rng=random.Random(7))
        self.assertLess(result["games"], float(cfg.eval_games))
        self.assertEqual(result["early_cutoff"], 1.0)


class EarlyStopTest(unittest.TestCase):
    def test_stops_after_consecutive_failed_evals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = str(Path(tmp) / "metrics.jsonl")
            cfg = TrainConfig(
                iterations=5,
                games_per_iteration=1,
                simulations=4,
                mcts_batch_size=2,
                epochs=1,
                train_steps_per_iteration=1,
                batch_size=16,
                channels=8,
                residual_blocks=1,
                value_hidden=16,
                temperature_moves=4,
                eval_interval=1,
                eval_games=2,
                eval_simulations=4,
                promotion_threshold=2.0,  # impossible: every eval fails
                early_stop_evals=1,
                checkpoint_dir=str(Path(tmp) / "ckpt"),
                metrics_path=metrics_path,
                device="cpu",
            )
            run_training(cfg)
            rows = [json.loads(line) for line in Path(metrics_path).read_text().splitlines()]
            self.assertEqual(len(rows), 1)  # stopped after the first iteration
            self.assertTrue(rows[0]["early_stopped"])
            self.assertEqual(rows[0]["failed_evals"], 1)
            self.assertFalse(rows[0]["promoted"])
            # never promoted: the "best" checkpoint must not point at a stagnant model
            self.assertFalse((Path(tmp) / "ckpt" / "gomoku10_best.pt").exists())


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


class StepLossTest(unittest.TestCase):
    def test_compute_step_losses_matches_manual_cross_entropy(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(4, 100)
        policies = torch.softmax(torch.randn(4, 100), dim=1)
        values = torch.tensor([1.0, -1.0, 0.0, 1.0])
        predicted = torch.zeros(4)
        pw = torch.ones(4)
        cfg = TrainConfig()
        parts = compute_step_losses(logits, None, predicted, policies, values, pw, cfg)
        # policy_loss is the (pw-weighted) cross-entropy of policies vs softmax(logits)
        expected = -(policies * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
        self.assertAlmostEqual(float(parts.policy_loss), float(expected), places=5)
        self.assertEqual(float(parts.soft_policy_loss), 0.0)  # no soft head
        self.assertTrue(torch.isfinite(parts.loss))

    def test_policy_weight_zero_excludes_value_only_samples(self) -> None:
        torch.manual_seed(1)
        logits = torch.randn(2, 100)
        policies = torch.softmax(torch.randn(2, 100), dim=1)
        values = torch.zeros(2)
        predicted = torch.zeros(2)
        cfg = TrainConfig()
        # Second sample is value-only (pw=0): policy_loss must equal the first row alone.
        pw = torch.tensor([1.0, 0.0])
        parts = compute_step_losses(logits, None, predicted, policies, values, pw, cfg)
        row0 = -(policies[0] * torch.log_softmax(logits, dim=1)[0]).sum()
        self.assertAlmostEqual(float(parts.policy_loss), float(row0), places=5)


class LoaderTest(unittest.TestCase):
    def test_build_train_loader_yields_expected_batch(self) -> None:
        dataset, _ = replay_to_dataset(deque(make_examples(8)))
        cfg = TrainConfig(batch_size=4)
        loader = build_train_loader(dataset, None, cfg)
        states, policies, values, weights = next(iter(loader))
        self.assertEqual(states.shape, (4, 2, 10, 10))
        self.assertEqual(policies.shape, (4, 100))
        self.assertEqual(values.shape, (4,))
        self.assertEqual(weights.shape, (4,))


class PresetTest(unittest.TestCase):
    def test_all_presets_build(self) -> None:
        for name in PRESETS:
            cfg = preset_config(name)
            self.assertEqual(cfg.preset, name)
            self.assertGreater(cfg.iterations, 0)

    def test_unknown_preset_raises(self) -> None:
        with self.assertRaises(ValueError):
            preset_config("does-not-exist")

    def test_mcts_config_for_eval_omits_forced_playouts(self) -> None:
        cfg = preset_config("v3-student-local")  # has forced playouts on
        train_mcfg = mcts_config_from_train(cfg, 64, for_eval=False)
        eval_mcfg = mcts_config_from_train(cfg, 64, for_eval=True)
        self.assertTrue(train_mcfg.forced_playouts)
        self.assertFalse(eval_mcfg.forced_playouts)


class SymmetryTest(unittest.TestCase):
    def test_permutation_matches_rot90_flip(self) -> None:
        from alphazero_gomoku.train import symmetry_permutations

        b = 10
        perm = symmetry_permutations(b, torch.device("cpu"))
        self.assertEqual(perm.shape, (8, b * b))
        torch.manual_seed(0)
        x = torch.randn(3, 2, b, b)
        p = torch.softmax(torch.randn(3, b * b), dim=1)
        for k in range(8):
            if k < 4:
                ref_x = torch.rot90(x, k, dims=(2, 3))
                ref_p = torch.rot90(p.view(3, b, b), k, dims=(1, 2)).reshape(3, -1)
            else:
                ref_x = torch.flip(torch.rot90(x, k - 4, dims=(2, 3)), dims=(3,))
                ref_p = torch.flip(torch.rot90(p.view(3, b, b), k - 4, dims=(1, 2)), dims=(2,)).reshape(3, -1)
            gi = perm[k]
            got_x = torch.gather(
                x.reshape(3, 2, b * b), 2, gi.view(1, 1, -1).expand(3, 2, b * b)
            ).reshape(3, 2, b, b)
            got_p = torch.gather(p, 1, gi.view(1, -1).expand(3, -1))
            self.assertTrue(torch.equal(got_x, ref_x), f"state mismatch k={k}")
            self.assertTrue(torch.equal(got_p, ref_p), f"policy mismatch k={k}")

    def test_apply_random_symmetries_shapes_and_identity(self) -> None:
        b = 10
        x = torch.randn(5, 2, b, b)
        p = torch.softmax(torch.randn(5, b * b), dim=1)
        out_x, out_p = apply_random_symmetries(x.clone(), p.clone(), b)
        self.assertEqual(out_x.shape, x.shape)
        self.assertEqual(out_p.shape, p.shape)
        # each output row is a permutation of the corresponding input policy row
        self.assertTrue(torch.allclose(out_p.sum(1), p.sum(1), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
