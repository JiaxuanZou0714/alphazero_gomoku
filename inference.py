from __future__ import annotations

import random

from .game import GomokuState
from .mcts import MCTS, MCTSConfig, visit_count_policy
from .model import PolicyValueNet


def mcts_config_from_cfg(
    cfg: dict,
    simulations: int,
    *,
    for_eval: bool = False,
) -> MCTSConfig:
    """Build an :class:`MCTSConfig` from a checkpoint/training config dict.

    This is the single source of truth for turning the ``mcts_*`` keys stored in
    a checkpoint's ``config`` into a runtime MCTS configuration. It replaces the
    near-identical hand-rolled constructions that used to live in ``play.py``,
    ``scripts/benchmark_checkpoints.py`` and ``scripts/distill_old_best.py``.

    ``for_eval=True`` omits forced playouts: evaluation and head-to-head play
    must not inject extra exploration, so the forced-playout knobs are dropped
    rather than silently inherited.
    """
    amp_default = str(cfg.get("amp_dtype", "bf16")) if bool(cfg.get("amp", True)) else "none"
    kwargs: dict = dict(
        simulations=simulations,
        c_puct=float(cfg.get("mcts_c_puct", 1.5)),
        dirichlet_alpha=float(cfg.get("mcts_dirichlet_alpha", 0.3)),
        dirichlet_fraction=float(cfg.get("mcts_dirichlet_fraction", 0.25)),
        eval_batch_size=min(int(cfg.get("mcts_batch_size", 32)), max(1, simulations)),
        amp_dtype=str(cfg.get("mcts_amp_dtype", amp_default)),
        root_policy_temp=float(cfg.get("mcts_root_policy_temp", 1.0)),
        shaped_dirichlet=bool(cfg.get("mcts_shaped_dirichlet", False)),
        dynamic_cpuct=bool(cfg.get("mcts_dynamic_cpuct", False)),
        fpu_reduction=float(cfg.get("mcts_fpu_reduction", 0.0) or 0.0),
    )
    if not for_eval:
        kwargs["forced_playouts"] = bool(cfg.get("mcts_forced_playouts", False))
        kwargs["forced_playout_k"] = float(cfg.get("mcts_forced_playout_k", 2.0))
    return MCTSConfig(**kwargs)


def greedy_action(
    model: PolicyValueNet,
    mcts_cfg: MCTSConfig,
    state: GomokuState,
    device: str,
) -> int:
    """Run a noiseless MCTS search and return the most-visited legal action."""
    root = MCTS(model, mcts_cfg, device=device).search(state, add_exploration_noise=False)
    policy = visit_count_policy(root, state.action_size, temperature=0.0)
    if policy.sum() <= 0:
        return int(state.legal_actions()[0])
    return int(policy.argmax())


def random_opening(
    size: int,
    win_length: int,
    opening_moves: int,
    rng: random.Random,
) -> list[int]:
    """Sample a random sequence of opening plies (used by eval/benchmark)."""
    state = GomokuState.new(size=size, win_length=win_length)
    opening: list[int] = []
    for _ in range(opening_moves):
        if state.is_terminal:
            break
        action = int(rng.choice(list(state.legal_actions())))
        opening.append(action)
        state = state.apply(action)
    return opening
