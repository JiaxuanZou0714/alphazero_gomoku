from __future__ import annotations

import math
import random
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
import torch

from .game import GomokuState
from .model import PolicyValueNet
from .torch_compat import tensor_from_array


@dataclass
class MCTSConfig:
    simulations: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    eval_batch_size: int = 1
    amp_dtype: str = "bf16"
    # KataGo improvements (all default to off for backward compat)
    root_policy_temp: float = 1.0    # >1 flattens root priors, improves early exploration
    shaped_dirichlet: bool = False   # per-action alpha based on prior rank
    dynamic_cpuct: bool = False      # scale c_puct by sqrt(empirical value variance)
    fpu_reduction: float = 0.0       # unvisited child Q = parent Q - fpu * sqrt(visited prior mass)
    forced_playouts: bool = False    # minimum playouts for root children during noised searches
    forced_playout_k: float = 2.0    # n_forced = sqrt(k * prior * root_visits)


@dataclass
class Node:
    prior: float
    raw_prior: float = 0.0          # prior before Dirichlet noise (for surprise weighting)
    visit_count: int = 0
    value_sum: float = 0.0
    value_sq_sum: float = 0.0       # for dynamic cPUCT variance estimation
    children: dict[int, "Node"] = field(default_factory=dict)

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def value_var(self) -> float:
        """Empirical variance of backed-up values."""
        if self.visit_count < 2:
            return 1.0
        mean = self.value_sum / self.visit_count
        return max(0.0, self.value_sq_sum / self.visit_count - mean * mean)


class MCTS:
    def __init__(
        self,
        model: PolicyValueNet,
        config: MCTSConfig | None = None,
        device: str | torch.device = "cpu",
        rng: random.Random | None = None,
    ) -> None:
        self.model = model
        self.config = config or MCTSConfig()
        self.device = torch.device(device)
        # Derive from the global random module so set_seed() makes searches
        # reproducible (tie-breaks and Dirichlet noise included).
        self.rng = rng or random.Random(random.getrandbits(64))
        self.np_rng = np.random.default_rng(self.rng.getrandbits(64))

    def search(
        self,
        state: GomokuState,
        add_exploration_noise: bool = False,
        reuse_root: Node | None = None,
        simulations: int | None = None,
    ) -> Node:
        """Run MCTS from `state`.

        reuse_root: subtree of a previous search rooted at `state` (tree reuse);
            its accumulated visits are kept and `simulations` new ones are added.
        simulations: override config.simulations for this call (playout cap
            randomization runs cheap and full searches with one config).
        """
        if state.is_terminal:
            raise ValueError("cannot search from a terminal state")

        sims = self.config.simulations if simulations is None else max(1, simulations)
        if reuse_root is not None and reuse_root.expanded:
            root = reuse_root
        else:
            root = Node(prior=1.0, raw_prior=1.0)
            self._expand(root, state, is_root=True)
        if add_exploration_noise:
            self._add_dirichlet_noise(root)
        use_forced = add_exploration_noise and self.config.forced_playouts

        simulations_done = 0
        while simulations_done < sims:
            batch_size = min(
                max(1, self.config.eval_batch_size),
                sims - simulations_done,
            )
            leaves = []
            seen_leaf_nodes: set[int] = set()
            attempts = 0
            terminal_backups = 0
            while len(leaves) < batch_size and attempts < batch_size * 4:
                attempts += 1
                node = root
                scratch = state.clone()
                search_path = [node]

                while node.expanded:
                    forced = use_forced and node is root
                    action, node = self._select_child(node, forced=forced)
                    scratch = scratch.apply(action)
                    search_path.append(node)

                if scratch.is_terminal:
                    # Terminal values are exact: backpropagate immediately and
                    # allow repeats within a batch instead of wasting attempts.
                    self._backpropagate(
                        search_path, scratch.terminal_value_for_current_player()
                    )
                    terminal_backups += 1
                    simulations_done += 1
                    if simulations_done >= sims:
                        break
                    continue
                if id(node) in seen_leaf_nodes:
                    continue
                seen_leaf_nodes.add(id(node))
                self._add_virtual_visits(search_path)
                leaves.append((search_path, node, scratch))

            if not leaves:
                if terminal_backups == 0:
                    break
                continue
            evaluations = self._evaluate_batch([scratch for _, _, scratch in leaves])
            for (search_path, node, scratch), (policy, value) in zip(leaves, evaluations):
                self._remove_virtual_visits(search_path)
                self._expand_with_policy(node, scratch, policy)
                self._backpropagate(search_path, value)
            simulations_done += len(leaves)

        return root

    def policy_target(self, root: Node, action_size: int, pruned: bool = True) -> np.ndarray:
        """τ=1 visit distribution for policy training.

        With forced playouts enabled and pruned=True, applies KataGo policy
        target pruning: forced visits that PUCT itself would not have spent are
        subtracted so the noise-driven exploration does not pollute the target.
        """
        visits = np.zeros(action_size, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        if pruned and self.config.forced_playouts and root.children:
            self._prune_forced_playouts(root, visits)
        total = visits.sum()
        if total <= 0:
            return visits
        return visits / total

    def _prune_forced_playouts(self, root: Node, visits: np.ndarray) -> None:
        items = [(a, c) for a, c in root.children.items() if c.visit_count > 0]
        if len(items) < 2:
            return
        parent_visits = max(1, root.visit_count)
        sqrt_parent = math.sqrt(parent_visits)
        c_val = self.config.c_puct
        if self.config.dynamic_cpuct:
            c_val *= math.sqrt(max(0.25, root.value_var))
        best_action, best_child = max(items, key=lambda kv: kv[1].visit_count)
        best_urgency = (
            -best_child.value
            + c_val * best_child.prior * sqrt_parent / (1 + best_child.visit_count)
        )
        k = self.config.forced_playout_k
        for action, child in items:
            if action == best_action:
                continue
            v = child.visit_count
            n_forced = math.sqrt(k * child.prior * parent_visits)
            q = -child.value
            if best_urgency <= q:
                continue  # visits justified by Q alone, nothing to prune
            # Smallest visit count at which this child's PUCT urgency still does
            # not exceed the most-visited child's: those visits are legitimate.
            min_v = c_val * child.prior * sqrt_parent / (best_urgency - q) - 1.0
            new_v = max(v - n_forced, min_v)
            if new_v < 1.0:
                new_v = 0.0
            visits[action] = min(float(v), new_v)

    def _expand(self, node: Node, state: GomokuState, is_root: bool = False) -> float:
        policy, value = self._evaluate(state, apply_root_temp=is_root)
        self._expand_with_policy(node, state, policy)
        return value

    def _expand_with_policy(self, node: Node, state: GomokuState, policy: np.ndarray) -> None:
        legal_actions = state.legal_actions()
        for action in legal_actions:
            prior = float(policy[action])
            node.children[int(action)] = Node(prior=prior, raw_prior=prior)

    def _evaluate(self, state: GomokuState, apply_root_temp: bool = False) -> tuple[np.ndarray, float]:
        return self._evaluate_batch([state], apply_root_temp=apply_root_temp)[0]

    def _evaluate_batch(
        self, states: list[GomokuState], apply_root_temp: bool = False
    ) -> list[tuple[np.ndarray, float]]:
        if not states:
            return []

        self.model.eval()
        encoded = tensor_from_array(
            np.stack([state.encode() for state in states]),
            dtype=torch.float32,
            device=self.device,
        )
        legal_masks_np = np.stack([state.legal_mask() for state in states])  # (N, A)

        autocast_ctx = self._autocast_context()
        with torch.inference_mode(), autocast_ctx:
            policy_logits, _, values_batch = self.model(encoded)
            logits_np = torch.nan_to_num(
                policy_logits.float(), nan=0.0, posinf=0.0, neginf=0.0
            ).cpu().numpy()  # (N, A)
            values = torch.nan_to_num(
                values_batch.float(), nan=0.0, posinf=1.0, neginf=-1.0
            ).clamp(-1.0, 1.0).cpu().tolist()

        # Apply root policy temperature to flatten overconfident priors
        temp = self.config.root_policy_temp
        if apply_root_temp and temp != 1.0 and temp > 0:
            logits_np /= temp

        # Masked softmax in numpy (vectorised)
        logits_np[~legal_masks_np] = -1.0e9
        logits_np -= logits_np.max(axis=1, keepdims=True)
        exp = np.exp(logits_np)
        exp[~legal_masks_np] = 0.0
        totals = exp.sum(axis=1, keepdims=True)
        uniform_fallback = legal_masks_np.sum(axis=1, keepdims=True).clip(min=1)
        probs_batch = np.where(totals > 0, exp / totals, legal_masks_np / uniform_fallback)

        return [(probs_batch[i].astype(np.float32), float(values[i])) for i in range(len(states))]

    def _autocast_context(self) -> object:
        if self.device.type != "cuda" or self.config.amp_dtype == "none":
            return nullcontext()
        if self.config.amp_dtype == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if self.config.amp_dtype == "fp16":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        raise ValueError(f"unknown amp_dtype: {self.config.amp_dtype}")

    @staticmethod
    def _add_virtual_visits(search_path: list[Node]) -> None:
        for node in search_path:
            node.visit_count += 1

    @staticmethod
    def _remove_virtual_visits(search_path: list[Node]) -> None:
        for node in search_path:
            node.visit_count -= 1

    def _select_child(self, node: Node, forced: bool = False) -> tuple[int, Node]:
        parent_visits = max(1, node.visit_count)

        if forced:
            # KataGo forced playouts: root children below their minimum playout
            # count are searched first (pick the largest deficit).
            k = self.config.forced_playout_k
            best_deficit = 1.0e-9
            candidates: list[tuple[int, Node]] = []
            for action, child in node.children.items():
                deficit = math.sqrt(k * child.prior * parent_visits) - child.visit_count
                if deficit > best_deficit:
                    best_deficit = deficit
                    candidates = [(action, child)]
                elif candidates and deficit == best_deficit:
                    candidates.append((action, child))
            if candidates:
                return self.rng.choice(candidates)

        best_score = -float("inf")
        best: list[tuple[int, Node]] = []
        sqrt_parent = math.sqrt(parent_visits)

        # Dynamic cPUCT: scale by sqrt of empirical value variance to adapt exploration
        c = self.config.c_puct
        if self.config.dynamic_cpuct:
            c = c * math.sqrt(max(0.25, node.value_var))

        # KataGo first-play urgency: estimate unvisited children slightly below
        # the parent's value instead of a flat 0, which is too optimistic for
        # the losing side and too pessimistic for the winning side.
        fpu = self.config.fpu_reduction
        if fpu > 0:
            visited_mass = sum(
                child.prior for child in node.children.values() if child.visit_count > 0
            )
            fpu_value = node.value - fpu * math.sqrt(visited_mass)
        else:
            fpu_value = 0.0

        for action, child in node.children.items():
            prior_score = c * child.prior * sqrt_parent / (child.visit_count + 1)
            q = -child.value if child.visit_count > 0 else fpu_value
            score = q + prior_score
            if score > best_score:
                best_score = score
                best = [(action, child)]
            elif score == best_score:
                best.append((action, child))

        return self.rng.choice(best)

    def _add_dirichlet_noise(self, root: Node) -> None:
        if not root.children:
            return
        actions = list(root.children)

        if self.config.shaped_dirichlet:
            # KataGo shaped Dirichlet: actions above median prior get higher alpha
            # (explore plausible moves more), below-median get lower alpha (less noise dilution)
            priors = np.array([root.children[a].prior for a in actions])
            median = float(np.median(priors))
            alphas = np.where(
                priors >= median,
                self.config.dirichlet_alpha * 2.0,
                self.config.dirichlet_alpha * 0.5,
            )
            noise = self.np_rng.dirichlet(alphas)
        else:
            noise = self.np_rng.dirichlet([self.config.dirichlet_alpha] * len(actions))

        frac = self.config.dirichlet_fraction
        for action, sample in zip(actions, noise):
            child = root.children[action]
            child.prior = child.prior * (1.0 - frac) + float(sample) * frac

    @staticmethod
    def _backpropagate(search_path: list[Node], value: float) -> None:
        for node in reversed(search_path):
            node.value_sum += value
            node.value_sq_sum += value * value
            node.visit_count += 1
            value = -value


def visit_count_policy(root: Node, action_size: int, temperature: float) -> np.ndarray:
    visits = np.zeros(action_size, dtype=np.float32)
    for action, child in root.children.items():
        visits[action] = child.visit_count

    if visits.sum() <= 0:
        return visits

    if temperature <= 1.0e-6:
        policy = np.zeros_like(visits)
        policy[int(np.argmax(visits))] = 1.0
        return policy

    visits = np.power(visits, 1.0 / temperature)
    return visits / visits.sum()
