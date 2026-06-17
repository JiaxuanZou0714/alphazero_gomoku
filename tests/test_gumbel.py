import math
import unittest

import numpy as np

from alphazero_gomoku.gumbel import (
    completed_q,
    gumbel_top_m,
    improved_policy,
    sequential_halving_plan,
    sigma,
)


class TestGumbelMath(unittest.TestCase):
    def test_sigma_monotonic_and_visit_scaled(self):
        q = np.array([-1.0, 0.0, 0.5, 1.0])
        s_low = sigma(q, max_visit=0)
        s_high = sigma(q, max_visit=100)
        # monotonic increasing in q
        self.assertTrue(np.all(np.diff(s_low) > 0))
        # more visits => larger magnitude (Q trusted more)
        self.assertGreater(abs(s_high[-1]), abs(s_low[-1]))

    def test_improved_policy_reduces_to_prior_with_no_visits(self):
        logits = np.array([0.0, 1.0, 2.0, -1.0])
        prior = np.exp(logits) / np.exp(logits).sum()
        visits = np.zeros(4)
        q = np.zeros(4)
        pi = improved_policy(logits, prior, q, visits, value=0.0)
        np.testing.assert_allclose(pi, prior, atol=1e-6)
        self.assertAlmostEqual(float(pi.sum()), 1.0, places=6)

    def test_improved_policy_sharpens_toward_high_q(self):
        logits = np.zeros(3)  # uniform prior
        prior = np.ones(3) / 3
        visits = np.array([10, 10, 10])
        q = np.array([-0.5, 0.0, 0.8])  # third action clearly best
        pi = improved_policy(logits, prior, q, visits, value=0.0)
        self.assertEqual(int(np.argmax(pi)), 2)
        # with uniform prior the improved policy must favor the best-Q action
        self.assertGreater(pi[2], pi[0])
        self.assertGreater(pi[2], pi[1])

    def test_completed_q_uses_vmix_for_unvisited(self):
        prior = np.array([0.25, 0.25, 0.25, 0.25])
        q = np.array([0.6, 0.4, 0.0, 0.0])
        visits = np.array([5, 5, 0, 0])
        value = -0.2
        cq = completed_q(prior, q, visits, value)
        # visited entries unchanged
        self.assertAlmostEqual(cq[0], 0.6, places=6)
        self.assertAlmostEqual(cq[1], 0.4, places=6)
        # unvisited entries share one v_mix value strictly between value and the
        # visited weighted-Q (here 0.5), since total_n>0
        self.assertAlmostEqual(cq[2], cq[3], places=9)
        self.assertGreater(cq[2], value)
        self.assertLess(cq[2], 0.5)

    def test_completed_q_no_visits_is_value(self):
        prior = np.ones(4) / 4
        q = np.zeros(4)
        visits = np.zeros(4)
        cq = completed_q(prior, q, visits, value=0.3)
        np.testing.assert_allclose(cq, np.full(4, 0.3), atol=1e-9)

    def test_sequential_halving_plan_spends_budget(self):
        for budget in (16, 32, 64, 128, 200, 224):
            for m in (2, 4, 8, 16):
                plan = sequential_halving_plan(budget, m)
                spent = sum(alive * per for alive, per in plan)
                # the plan spends exactly the budget (leftover folded into a final phase)
                self.assertEqual(spent, budget, msg=f"budget={budget} m={m} plan={plan}")
                # alive counts are non-increasing and start at the (possibly
                # budget-clamped) candidate count
                alives = [a for a, _ in plan]
                self.assertLessEqual(alives[0], m)
                self.assertTrue(all(x >= y for x, y in zip(alives, alives[1:])))
                # for the realistic regime no clamping happens
                if budget >= 2 * m:
                    self.assertEqual(alives[0], m)

    def test_sequential_halving_single_candidate(self):
        plan = sequential_halving_plan(50, 1)
        self.assertEqual(plan, [(1, 50)])

    def test_gumbel_top_m_picks_highest_keys_deterministically(self):
        rng = np.random.default_rng(0)
        logits = np.array([0.0, 5.0, 1.0, 4.0, 2.0])
        considered, g = gumbel_top_m(logits, m=2, rng=rng)
        self.assertEqual(len(considered), 2)
        self.assertEqual(g.shape, (5,))
        # the two considered are exactly the top-2 by logit+g
        keys = logits + g
        expected = set(np.argsort(-keys)[:2].tolist())
        self.assertEqual(set(considered.tolist()), expected)
        # returned in descending key order
        self.assertGreaterEqual(keys[considered[0]], keys[considered[1]])

    def test_gumbel_top_m_full_when_m_ge_n(self):
        rng = np.random.default_rng(1)
        logits = np.array([1.0, 2.0, 3.0])
        considered, _ = gumbel_top_m(logits, m=10, rng=rng)
        self.assertEqual(sorted(considered.tolist()), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
