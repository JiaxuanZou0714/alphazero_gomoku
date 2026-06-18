"""v0 — a pure hand-written Gomoku heuristic (no neural network, no search).

This is the deliberate "rules baseline" for the strength ladder: a classic
threat/shape evaluator that an experienced human might code in an afternoon.
It only ever looks one ply ahead (plus the immediate win/block reflexes), so it
has no notion of read-ahead tactics — exactly what the learned MCTS models are
meant to tower over on the leaderboard.

The same scoring table is mirrored byte-for-byte in ``docs/engine.worker.js`` so
the browser opponent plays identically to the rated engine here.
"""

from __future__ import annotations

import random

import numpy as np

from .game import GomokuState

_DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))


def _shape_score(count: int, opens: int) -> int:
    """Score a single direction's run after a stone is placed.

    ``count`` is the length of the consecutive same-colour run through the new
    stone; ``opens`` is how many of the two ends are empty (extendable).
    """
    if count >= 5:
        return 100_000          # five-in-a-row (handled as a hard win too)
    if count == 4:
        if opens == 2:
            return 15_000       # open four — unstoppable
        if opens == 1:
            return 1_200        # four — forces a block
        return 0                # dead four
    if count == 3:
        if opens == 2:
            return 1_000        # open three — forces a response
        if opens == 1:
            return 120
        return 0
    if count == 2:
        if opens == 2:
            return 100
        if opens == 1:
            return 12
        return 0
    if count == 1:
        return 8 if opens == 2 else 1
    return 0


def _placement_score(board: np.ndarray, r: int, c: int, player: int, size: int) -> int:
    """Sum the shape scores over all 4 directions for ``player`` playing (r, c).

    ``board`` must already contain ``player`` at ``(r, c)``. Summing across
    directions naturally rewards forks (e.g. two simultaneous open threes).
    """
    total = 0
    for dr, dc in _DIRS:
        count = 1
        rr, cc = r + dr, c + dc
        while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == player:
            count += 1
            rr += dr
            cc += dc
        fwd_open = 0 <= rr < size and 0 <= cc < size and board[rr, cc] == 0
        rr, cc = r - dr, c - dc
        while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == player:
            count += 1
            rr -= dr
            cc -= dc
        bwd_open = 0 <= rr < size and 0 <= cc < size and board[rr, cc] == 0
        total += _shape_score(count, (1 if fwd_open else 0) + (1 if bwd_open else 0))
    return total


def _makes_five(board: np.ndarray, r: int, c: int, player: int, size: int, win: int) -> bool:
    for dr, dc in _DIRS:
        count = 1
        rr, cc = r + dr, c + dc
        while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == player:
            count += 1
            rr += dr
            cc += dc
        rr, cc = r - dr, c - dc
        while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == player:
            count += 1
            rr -= dr
            cc -= dc
        if count >= win:
            return True
    return False


def candidate_cells(board: np.ndarray, size: int) -> list[int]:
    """Empty cells within Chebyshev distance 2 of any stone (whole board if empty)."""
    occupied = np.argwhere(board != 0)
    if occupied.size == 0:
        return [(size // 2) * size + (size // 2)]
    near = np.zeros((size, size), dtype=bool)
    for r, c in occupied:
        r0, r1 = max(0, r - 2), min(size, r + 3)
        c0, c1 = max(0, c - 2), min(size, c + 3)
        near[r0:r1, c0:c1] = True
    near &= board == 0
    return [int(r * size + c) for r, c in np.argwhere(near)]


def score_moves(state: GomokuState) -> dict[int, float]:
    """Return a {action: heuristic value} map over the candidate cells.

    Used by the browser to render the hint/candidate panel; the chosen move is
    simply the arg-max (with the hard win/block reflexes applied first).
    """
    board = state.board
    size = state.size
    player = state.current_player
    opponent = -player
    center = (size - 1) / 2.0
    scores: dict[int, float] = {}
    for action in candidate_cells(board, size):
        r, c = divmod(action, size)
        board[r, c] = player
        attack = _placement_score(board, r, c, player, size)
        board[r, c] = opponent
        defense = _placement_score(board, r, c, opponent, size)
        board[r, c] = 0
        # Slight attack bias; central squares break ties.
        proximity = -(abs(r - center) + abs(c - center))
        scores[action] = attack + 0.9 * defense + 0.05 * proximity
    return scores


def select_move(state: GomokuState, rng: random.Random | None = None) -> int:
    """Pick v0's move: win > block-loss > best threat/shape score."""
    board = state.board
    size = state.size
    win = state.win_length
    player = state.current_player
    opponent = -player
    cands = candidate_cells(board, size)

    # 1. Take an immediate win.
    for action in cands:
        r, c = divmod(action, size)
        board[r, c] = player
        won = _makes_five(board, r, c, player, size, win)
        board[r, c] = 0
        if won:
            return action

    # 2. Block an immediate opponent win (prefer the block with the best follow-up).
    blocks = []
    for action in cands:
        r, c = divmod(action, size)
        board[r, c] = opponent
        threat = _makes_five(board, r, c, opponent, size, win)
        board[r, c] = 0
        if threat:
            blocks.append(action)
    if blocks:
        sub = {a: score_moves_single(state, a) for a in blocks}
        return max(sub, key=sub.get)

    # 3. Otherwise maximise the threat/shape score.
    scores = score_moves(state)
    best = max(scores.values())
    top = [a for a, s in scores.items() if s >= best - 1e-9]
    if rng is not None and len(top) > 1:
        return rng.choice(top)
    return top[0]


def score_moves_single(state: GomokuState, action: int) -> float:
    board = state.board
    size = state.size
    player = state.current_player
    opponent = -player
    r, c = divmod(action, size)
    board[r, c] = player
    attack = _placement_score(board, r, c, player, size)
    board[r, c] = opponent
    defense = _placement_score(board, r, c, opponent, size)
    board[r, c] = 0
    return attack + 0.9 * defense
