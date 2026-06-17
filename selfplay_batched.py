"""Batched cross-game self-play engine.

The serial ``play_self_game`` runs one game at a time and batches only the
leaves *within a single* MCTS search (``eval_batch_size``). On a single GPU the
per-search Python overhead dominates, so the network's batch stays tiny (~32)
and the GPU is starved — measured throughput is nearly identical for a 3.1M and
a 0.36M parameter net, because FLOPs are not the bottleneck.

This engine advances ``max_parallel`` games concurrently and, on every tick,
collects exactly one leaf (or root-expansion) request per active game into a
*single* network forward pass that spans all games. Because each tree emits at
most one outstanding leaf per tick, there is no intra-tree collision, so virtual
loss is unnecessary and the search is identical to a serial search run with
``eval_batch_size == 1`` (slightly higher quality than the batched-leaf serial
path, never lower). Finished games are refilled with fresh ones so the network
batch stays close to ``max_parallel`` until the game budget is exhausted.

All tuned MCTS logic (FPU, dynamic cPUCT, forced playouts + target pruning,
shaped Dirichlet, the eval cache, root policy temperature) is reused verbatim
from :class:`alphazero_gomoku.mcts.MCTS`; this module only changes the
orchestration around those primitives. The per-move bookkeeping, opening
diversity, playout-cap randomization, value blending and ownership targets are a
faithful port of ``play_self_game`` so the produced training examples are drawn
from the same distribution.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from . import gumbel as gumbel_lib
from .game import GomokuState
from .inference import random_opening
from .mcts import MCTS, MCTSConfig, Node, visit_count_policy

Example = tuple  # (encoded, policy, value, policy_weight[, ownership])


def _policy_entropy(policy: np.ndarray) -> float:
    probs = policy[policy > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log(probs + 1.0e-12)).sum())


def _sample_action(policy: np.ndarray) -> int:
    total = float(policy.sum())
    if total <= 0:
        raise ValueError("cannot sample from an empty policy")
    return int(np.random.choice(len(policy), p=policy / total))


class _GumbelRoot:
    """Per-move Gumbel + Sequential Halving state machine for one root.

    Drives which considered root action to simulate next under the tick-based
    batched engine and, at move end, produces the improved-policy training target
    and the winning action. Q/visit statistics are read live from the shared tree
    (``root.children``); this object only holds the halving schedule and the
    Gumbel noise.
    """

    def __init__(self, actions, logits, g, considered_idx, plan, c_visit, c_scale):
        self.actions = actions          # list[int]: legal root actions (array-aligned)
        self.logits = logits            # np.float64[len(actions)] = log(prior)
        self.g = g                      # np.float64[len(actions)]: gumbel noise
        self.plan = plan                # list[(num_alive, visits_per_action)]
        self.c_visit = c_visit
        self.c_scale = c_scale
        self.phase = 0
        self.alive = list(considered_idx[: plan[0][0]])   # indices into actions
        self.per_remaining = {i: plan[0][1] for i in self.alive}
        self._rr = 0
        self.done = len(self.actions) <= 1

    def next_candidate(self):
        """Return (action, idx) needing a visit this phase, or None if exhausted."""
        live = [i for i in self.alive if self.per_remaining[i] > 0]
        if not live:
            return None
        i = live[self._rr % len(live)]
        self._rr += 1
        return self.actions[i], i

    def record_visit(self, idx):
        self.per_remaining[idx] -= 1

    def _scores(self, root):
        max_visit = max(
            (root.children[self.actions[i]].visit_count for i in self.alive), default=0
        )
        scale = (self.c_visit + max_visit) * self.c_scale
        out = []
        for i in self.alive:
            ch = root.children[self.actions[i]]
            q = -ch.value if ch.visit_count > 0 else 0.0
            out.append((i, self.g[i] + self.logits[i] + scale * q))
        return out

    def advance_phase(self, root):
        """Halve the alive set by score; return False when planning is finished."""
        self.phase += 1
        if self.phase >= len(self.plan):
            self.done = True
            return False
        keep = self.plan[self.phase][0]
        ranked = sorted(self._scores(root), key=lambda kv: kv[1], reverse=True)
        self.alive = [i for i, _ in ranked[:keep]]
        self.per_remaining = {i: self.plan[self.phase][1] for i in self.alive}
        self._rr = 0
        return True

    def winner(self, root):
        return self.actions[max(self._scores(root), key=lambda kv: kv[1])[0]]

    def target(self, root, action_size, root_value):
        prior = np.array([root.children[a].raw_prior for a in self.actions], dtype=np.float64)
        prior = np.clip(prior, 1e-12, 1.0)
        visits = np.array([root.children[a].visit_count for a in self.actions], dtype=np.float64)
        q = np.array(
            [(-root.children[a].value if root.children[a].visit_count > 0 else 0.0) for a in self.actions],
            dtype=np.float64,
        )
        pi = gumbel_lib.improved_policy(
            self.logits, prior, q, visits, value=root_value,
            c_visit=self.c_visit, c_scale=self.c_scale,
        )
        full = np.zeros(action_size, dtype=np.float32)
        for a, p in zip(self.actions, pi):
            full[a] = float(p)
        return full


@dataclass
class _Slot:
    """State machine for one in-flight game within the batched engine."""

    state: GomokuState
    root: Node | None = None
    needs_root_eval: bool = False
    is_full: bool = False
    sims_target: int = 0
    sims_done: int = 0
    next_root: Node | None = None
    history: list = field(default_factory=list)  # (encode, target, player, mcts_value, weight)
    entropies: list = field(default_factory=list)
    kl_surprises: list = field(default_factory=list)
    full_moves: int = 0
    move_started: bool = False
    gumbel: _GumbelRoot | None = None
    _gidx: int = -1


class BatchedSelfPlay:
    def __init__(
        self,
        model,
        cfg,
        mcts_cfg: MCTSConfig,
        *,
        device: str,
        max_parallel: int = 64,
    ) -> None:
        self.cfg = cfg
        self.mcts = MCTS(model, mcts_cfg, device=device)
        self.max_parallel = max(1, int(max_parallel))
        self.action_size = cfg.board_size * cfg.board_size
        self.gumbel = bool(getattr(mcts_cfg, "gumbel", False))
        self.gumbel_considered = int(getattr(mcts_cfg, "gumbel_considered", 16))
        self.gumbel_c_visit = float(getattr(mcts_cfg, "gumbel_c_visit", 50.0))
        self.gumbel_c_scale = float(getattr(mcts_cfg, "gumbel_c_scale", 1.0))

        # Aggregated outputs (mirror play_self_games_worker's return shape).
        self.examples: list = []
        self.kl_surprises: list = []
        self.lengths: list = []
        self._game_entropies: list = []
        self._winners: list = []
        self._full_rates: list = []

    # -- game lifecycle ---------------------------------------------------

    def _new_game_state(self) -> GomokuState:
        cfg = self.cfg
        state = GomokuState.new(size=cfg.board_size, win_length=cfg.win_length)
        if cfg.selfplay_opening_moves > 0 and random.random() < cfg.selfplay_opening_prob:
            n_open = random.randint(1, cfg.selfplay_opening_moves)
            for action in random_opening(cfg.board_size, cfg.win_length, n_open, random):
                if state.is_terminal:
                    break
                state = state.apply(action)
        return state

    def _begin_move(self, slot: _Slot) -> None:
        """Decide full/fast search and set up (or reuse) the root for this move."""
        cfg = self.cfg
        slot.is_full = (not cfg.playout_cap_randomization) or (
            random.random() < cfg.full_search_prob
        )
        slot.sims_target = cfg.simulations if slot.is_full else max(1, cfg.fast_simulations)
        slot.sims_done = 0
        slot.move_started = True

        if self.gumbel:
            # Gumbel runs a fresh Sequential-Halving budget per move, so tree reuse
            # (which carries stale visit allocation) is disabled; the root is
            # expanded with raw logits (no root temp, no Dirichlet) and the
            # _GumbelRoot is built once the network eval returns.
            slot.gumbel = None
            slot.root = Node(prior=1.0, raw_prior=1.0)
            slot.needs_root_eval = True
            return

        reuse = (
            cfg.selfplay_tree_reuse
            and slot.next_root is not None
            and slot.next_root.expanded
        )
        if reuse:
            slot.root = slot.next_root
            slot.needs_root_eval = False
            if slot.is_full:
                self.mcts._add_dirichlet_noise(slot.root)
        else:
            slot.root = Node(prior=1.0, raw_prior=1.0)
            slot.needs_root_eval = True

    def _finalize_move(self, slot: _Slot) -> None:
        """Port of play_self_game's per-move bookkeeping + move sampling."""
        cfg = self.cfg
        mcts = self.mcts
        root = slot.root
        state = slot.state
        gumbel_action = None
        if slot.gumbel is not None:
            # Gumbel improved-policy target + Sequential-Halving winning action.
            target = slot.gumbel.target(root, self.action_size, float(root.value))
            gumbel_action = slot.gumbel.winner(root)
        else:
            target = mcts.policy_target(root, self.action_size, pruned=slot.is_full)
        if target.sum() <= 0:
            legal = state.legal_actions()
            target = np.zeros(self.action_size, dtype=np.float32)
            target[legal] = 1.0 / len(legal)
        target = target.astype(np.float32)

        if slot.is_full:
            slot.full_moves += 1
            slot.entropies.append(_policy_entropy(target))
            actions = sorted(root.children)
            prior = np.array([root.children[a].raw_prior for a in actions], dtype=np.float32)
            mcts_p = np.array([target[a] for a in actions], dtype=np.float32)
            kl = float(np.sum(mcts_p * np.log((mcts_p + 1e-12) / (prior + 1e-12))))
            slot.kl_surprises.append(max(0.0, kl))
        else:
            slot.kl_surprises.append(0.0)

        mcts_value = float(root.value) if root.visit_count > 0 else 0.0
        slot.history.append(
            (state.encode(), target, state.current_player, mcts_value, 1.0 if slot.is_full else 0.0)
        )

        if gumbel_action is not None:
            # Gumbel noise already provides root exploration; play the SH winner.
            action = int(gumbel_action)
        else:
            temperature = 1.0 if state.moves_played < cfg.temperature_moves else 0.0
            move_policy = visit_count_policy(root, self.action_size, temperature)
            if move_policy.sum() <= 0:
                move_policy = target
            action = _sample_action(move_policy)
        slot.next_root = root.children.get(action)
        slot.state = state.apply(action)
        slot.move_started = False

    def _finish_game(self, slot: _Slot) -> None:
        """Port of play_self_game's terminal example construction + stats."""
        cfg = self.cfg
        state = slot.state
        w = cfg.mcts_value_weight
        final_board = state.board.reshape(-1).astype(np.float32)
        for encoded, policy, player, mcts_val, policy_weight in slot.history:
            if state.winner == 0:
                terminal_v = 0.0
            else:
                terminal_v = 1.0 if state.winner == player else -1.0
            value = (1.0 - w) * terminal_v + w * mcts_val if w > 0 else terminal_v
            if cfg.use_ownership:
                ownership = final_board * float(player)
                self.examples.append(
                    (encoded.astype(np.float32), policy, float(value), policy_weight, ownership)
                )
            else:
                self.examples.append((encoded.astype(np.float32), policy, float(value), policy_weight))

        self.kl_surprises.extend(slot.kl_surprises)
        self.lengths.append(len(slot.history))
        self._game_entropies.append(
            float(np.mean(slot.entropies)) if slot.entropies else 0.0
        )
        self._winners.append(float(state.winner if state.winner is not None else 0))
        self._full_rates.append(slot.full_moves / max(1, len(slot.history)))

    # -- search stepping --------------------------------------------------

    def _select_leaf(self, slot: _Slot):
        """Descend one path to a non-terminal leaf, backing up terminals inline.

        Returns the scratch state to evaluate, or None when this tick produced no
        network request for the slot (a terminal was reached, or the sim budget
        for the move was exhausted).
        """
        if slot.gumbel is not None:
            return self._select_leaf_gumbel(slot)
        mcts = self.mcts
        root = slot.root
        use_forced = slot.is_full and mcts.config.forced_playouts
        while slot.sims_done < slot.sims_target:
            node = root
            scratch = slot.state.clone()
            path = [node]
            while node.expanded:
                is_root = node is root
                action, node = mcts._select_child(
                    node, forced=use_forced and is_root, is_root=is_root
                )
                scratch = scratch.apply(action)
                path.append(node)
            if scratch.is_terminal:
                mcts._backpropagate(path, scratch.terminal_value_for_current_player())
                slot.sims_done += 1
                continue
            slot._pending = (path, node, scratch)
            return scratch
        slot._pending = None
        return None

    def _setup_gumbel_root(self, slot: _Slot) -> None:
        """Build the per-move Gumbel/Sequential-Halving plan for an expanded root."""
        root = slot.root
        actions = sorted(root.children)
        prior = np.clip(
            np.array([root.children[a].raw_prior for a in actions], dtype=np.float64), 1e-12, 1.0
        )
        logits = np.log(prior)
        m_req = min(self.gumbel_considered, len(actions))
        plan = gumbel_lib.sequential_halving_plan(slot.sims_target, m_req)
        considered_idx, g = gumbel_lib.gumbel_top_m(logits, plan[0][0], self.mcts.np_rng)
        slot.gumbel = _GumbelRoot(
            actions, logits, g, considered_idx, plan, self.gumbel_c_visit, self.gumbel_c_scale
        )

    def _select_leaf_gumbel(self, slot: _Slot):
        """Gumbel root: simulate the next Sequential-Halving candidate.

        The first ply from the root is forced to the current SH candidate; the
        descent below it uses ordinary PUCT. Phases advance (halving the alive
        set) when every alive candidate has spent its per-phase visits. When the
        plan is exhausted the budget is marked spent so the move finalizes.
        """
        mcts = self.mcts
        gr = slot.gumbel
        root = slot.root
        while not gr.done:
            nc = gr.next_candidate()
            if nc is None:
                if not gr.advance_phase(root):
                    break
                continue
            action, idx = nc
            child = root.children[action]
            scratch = slot.state.clone().apply(action)
            path = [root, child]
            node = child
            while node.expanded:
                a2, node = mcts._select_child(node, forced=False, is_root=False)
                scratch = scratch.apply(a2)
                path.append(node)
            if scratch.is_terminal:
                mcts._backpropagate(path, scratch.terminal_value_for_current_player())
                slot.sims_done += 1
                gr.record_visit(idx)
                continue
            slot._pending = (path, node, scratch)
            slot._gidx = idx
            return scratch
        # planning finished: force the move to finalize on the next loop turn
        slot.sims_done = slot.sims_target
        slot._pending = None
        return None

    def play(self, num_games: int) -> dict:
        """Run ``num_games`` self-play games and aggregate examples/stats."""
        n_parallel = min(self.max_parallel, num_games)
        slots: list[_Slot] = [_Slot(state=self._new_game_state()) for _ in range(n_parallel)]
        games_started = n_parallel
        games_finished = 0

        while games_finished < num_games:
            root_reqs: list[tuple[_Slot, GomokuState]] = []
            leaf_reqs: list[tuple[_Slot, GomokuState]] = []

            for slot in slots:
                if slot.state is None:
                    continue
                # Finalize a completed move / finish or refill terminal games,
                # then (re)start the move state machine until the slot either
                # needs a network request or has filled its sim budget.
                guard = 0
                while True:
                    guard += 1
                    if guard > 8:  # safety: never spin on one slot
                        break
                    if not slot.move_started:
                        if slot.state.is_terminal:
                            self._finish_game(slot)
                            games_finished += 1
                            if games_started < num_games:
                                slot.state = self._new_game_state()
                                slot.root = None
                                slot.next_root = None
                                slot.gumbel = None
                                slot.history = []
                                slot.entropies = []
                                slot.kl_surprises = []
                                slot.full_moves = 0
                                games_started += 1
                            else:
                                slot.state = None  # retire slot
                            break
                        self._begin_move(slot)
                        if slot.needs_root_eval:
                            root_reqs.append((slot, slot.state))
                            break
                    # move in progress with an expanded root: try to emit a leaf
                    if slot.sims_done >= slot.sims_target:
                        self._finalize_move(slot)
                        continue  # immediately begin the next move
                    scratch = self._select_leaf(slot)
                    if scratch is not None:
                        leaf_reqs.append((slot, scratch))
                    elif slot.sims_done >= slot.sims_target:
                        self._finalize_move(slot)
                        continue
                    break

            # One network forward per temperature group, spanning all games.
            # Gumbel uses raw root logits (no root policy temperature, no Dirichlet).
            if root_reqs:
                evals = self.mcts._evaluate_batch(
                    [s for _, s in root_reqs], apply_root_temp=not self.gumbel
                )
                for (slot, state), (policy, _value) in zip(root_reqs, evals):
                    self.mcts._expand_with_policy(slot.root, state, policy)
                    slot.needs_root_eval = False
                    if self.gumbel:
                        self._setup_gumbel_root(slot)
                    elif slot.is_full:
                        self.mcts._add_dirichlet_noise(slot.root)
            if leaf_reqs:
                evals = self.mcts._evaluate_batch(
                    [s for _, s in leaf_reqs], apply_root_temp=False
                )
                for (slot, _state), (policy, value) in zip(leaf_reqs, evals):
                    path, node, scratch = slot._pending
                    self.mcts._expand_with_policy(node, scratch, policy)
                    self.mcts._backpropagate(path, value)
                    slot.sims_done += 1
                    if slot.gumbel is not None:
                        slot.gumbel.record_visit(slot._gidx)
                    slot._pending = None

            if not root_reqs and not leaf_reqs and games_finished < num_games:
                # No work emitted but games remain: all live slots are mid-finalize
                # with reused roots; loop again (the per-slot guard handles them).
                if all(s.state is None for s in slots):
                    break

        stats = {
            "policy_entropy": float(np.mean(self._game_entropies)) if self._game_entropies else 0.0,
            "black_win_rate": float(np.mean([w == 1 for w in self._winners])) if self._winners else 0.0,
            "white_win_rate": float(np.mean([w == -1 for w in self._winners])) if self._winners else 0.0,
            "draw_rate": float(np.mean([w == 0 for w in self._winners])) if self._winners else 0.0,
            "full_search_rate": float(np.mean(self._full_rates)) if self._full_rates else 0.0,
            "games": float(len(self._winners)),
        }
        return stats


def play_self_games_batched(
    model,
    cfg,
    mcts_cfg: MCTSConfig,
    num_games: int,
    *,
    device: str,
    max_parallel: int = 64,
) -> tuple[list, list, list, dict]:
    """Drop-in batched replacement for the serial self-play worker.

    Returns ``(examples, kl_surprises, lengths, stats)`` matching
    ``play_self_games_worker`` (minus the trailing device string).
    """
    engine = BatchedSelfPlay(model, cfg, mcts_cfg, device=device, max_parallel=max_parallel)
    stats = engine.play(num_games)
    return engine.examples, engine.kl_surprises, engine.lengths, stats
