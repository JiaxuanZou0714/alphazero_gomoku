"""Gumbel AlphaZero root planning primitives (Danihelka et al., 2022).

This module implements the *math* of Gumbel policy improvement as pure,
testable functions over plain numpy arrays. The search orchestration (which
action to simulate next, when to halve) lives in the self-play engine; here we
only provide:

* ``sigma`` — the monotonic Q transform used to rank actions.
* ``v_mix`` / ``completed_q`` — value completion for unvisited actions.
* ``improved_policy`` — softmax(logits + sigma(completedQ)), the *training
  target*. At zero simulations this collapses to the network prior, and with
  more visits it sharpens toward the searched Q values. This target is far less
  noisy than the visit-count distribution at the low simulation counts we run,
  which is the whole point of adopting Gumbel: it breaks the policy-target
  quality ceiling that pinned ``policy_top1`` at 0.68.
* ``sequential_halving_plan`` — how many visits each considered action gets per
  halving phase for a given simulation budget.
* ``gumbel_top_m`` — Gumbel-top-k sampling of the considered root actions.

Conventions: ``logits`` are per-action network logits (we pass ``log(prior)``,
which differs from the true logit only by a constant that cancels in every
softmax/argmax used here). ``q`` values are from the *root player's*
perspective and live in ``[-1, 1]``. Illegal actions are handled by the caller
via masking before these functions are used (arrays here are over the set of
legal/considered actions).
"""

from __future__ import annotations

import math

import numpy as np

C_VISIT_DEFAULT = 50.0
C_SCALE_DEFAULT = 1.0


def sigma(q: np.ndarray, max_visit: float, c_visit: float = C_VISIT_DEFAULT, c_scale: float = C_SCALE_DEFAULT) -> np.ndarray:
    """Monotonic transform of Q used to rank/weight actions.

    ``(c_visit + max_visit) * c_scale * q``: the more the most-visited child has
    been searched, the more the Q values are trusted relative to the prior.
    """
    return (c_visit + float(max_visit)) * c_scale * q


def completed_q(
    prior: np.ndarray,
    q: np.ndarray,
    visits: np.ndarray,
    value: float,
) -> np.ndarray:
    """Return completed Q per action: searched Q where visited, ``v_mix`` else.

    ``visits`` is the per-action visit count (0 for unvisited). ``prior`` is the
    network prior (softmax of logits) over the same action set; ``value`` is the
    network value estimate at the node, from the root player's perspective.
    """
    visited = visits > 0
    total_n = float(visits.sum())
    if visited.any() and total_n > 0:
        visited_prior = float(prior[visited].sum())
        if visited_prior > 0:
            weighted_q = float((prior[visited] * q[visited]).sum()) / visited_prior
        else:
            weighted_q = float(value)
        mix = (value + total_n * weighted_q) / (1.0 + total_n)
    else:
        mix = float(value)
    out = np.where(visited, q, mix).astype(np.float64)
    return out


def improved_policy(
    logits: np.ndarray,
    prior: np.ndarray,
    q: np.ndarray,
    visits: np.ndarray,
    value: float,
    c_visit: float = C_VISIT_DEFAULT,
    c_scale: float = C_SCALE_DEFAULT,
) -> np.ndarray:
    """Gumbel improved policy target: softmax(logits + sigma(completedQ)).

    Returns a probability distribution over the given action set. With all
    ``visits == 0`` this reduces to ``softmax(logits)`` (the network prior).
    """
    cq = completed_q(prior, q, visits, value)
    max_visit = float(visits.max()) if visits.size else 0.0
    scores = logits.astype(np.float64) + sigma(cq, max_visit, c_visit, c_scale)
    scores -= scores.max()
    exp = np.exp(scores)
    total = exp.sum()
    if total <= 0:
        out = np.ones_like(exp) / len(exp)
        return out
    return exp / total


def sequential_halving_plan(sim_budget: int, m_considered: int) -> list[tuple[int, int]]:
    """Per-phase ``(num_alive, visits_per_action)`` schedule for Sequential Halving.

    Distributes ``sim_budget`` simulations across ``ceil(log2(m))`` phases; each
    phase visits every still-alive considered action ``visits_per_action`` times,
    then the bottom half is dropped. Any visit budget left after the planned
    phases (from flooring) is appended as extra visits on the final survivor so
    the full budget is spent.
    """
    m = max(1, int(m_considered))
    budget = max(1, int(sim_budget))

    # Clamp the candidate count so the unavoidable minimum of one visit per alive
    # action per halving phase fits the budget (sum of m + m/2 + ... + 2). Real
    # runs (sims >= 64, m <= 16) never bind; this only guards tiny budgets.
    def halving_minimum(k: int) -> int:
        total, alive = 0, k
        while alive > 1:
            total += alive
            alive //= 2
        return total

    while m > 1 and halving_minimum(m) > budget:
        m //= 2
    if m == 1:
        return [(1, budget)]

    phases_left = max(1, math.ceil(math.log2(m)))
    plan: list[tuple[int, int]] = []
    alive = m
    remaining = budget
    while alive > 1:
        per = max(1, remaining // (phases_left * alive))
        if alive * per > remaining:  # only when the budget is nearly exhausted
            per = max(1, remaining // alive)
        spend = alive * per
        plan.append((alive, per))
        remaining -= spend
        phases_left = max(1, phases_left - 1)
        alive = alive // 2
    if remaining > 0:
        plan.append((1, remaining))
    return plan


def gumbel_top_m(
    logits: np.ndarray,
    m: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample Gumbel noise and return (considered_indices, gumbel_values).

    ``considered_indices`` are the ``m`` actions with the largest
    ``logit + g``; ``gumbel_values`` is the full per-action Gumbel sample (so the
    caller can keep scoring survivors with the same noise).
    """
    n = len(logits)
    m = max(1, min(int(m), n))
    g = rng.gumbel(size=n)
    keys = logits + g
    if m >= n:
        considered = np.argsort(-keys)
    else:
        # top-m by key (unordered partition then sort the top slice)
        top = np.argpartition(-keys, m - 1)[:m]
        considered = top[np.argsort(-keys[top])]
    return considered.astype(np.int64), g
