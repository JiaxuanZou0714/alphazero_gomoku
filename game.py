from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GomokuState:
    """Immutable 10x10 Gomoku state.

    The board stores 1 for black, -1 for white, and 0 for empty. The only game
    knowledge here is the environment rule: legal moves and five-in-a-row wins.
    """

    board: np.ndarray
    current_player: int = 1
    last_move: int | None = None
    winner: int | None = None
    moves_played: int = 0
    win_length: int = 5

    @classmethod
    def new(cls, size: int = 10, win_length: int = 5) -> "GomokuState":
        return cls(
            board=np.zeros((size, size), dtype=np.int8),
            current_player=1,
            last_move=None,
            winner=None,
            moves_played=0,
            win_length=win_length,
        )

    @property
    def size(self) -> int:
        return int(self.board.shape[0])

    @property
    def action_size(self) -> int:
        return self.size * self.size

    @property
    def is_terminal(self) -> bool:
        return self.winner is not None

    def clone(self) -> "GomokuState":
        return GomokuState(
            board=self.board.copy(),
            current_player=self.current_player,
            last_move=self.last_move,
            winner=self.winner,
            moves_played=self.moves_played,
            win_length=self.win_length,
        )

    def action_to_coord(self, action: int) -> tuple[int, int]:
        if action < 0 or action >= self.action_size:
            raise ValueError(f"action must be in [0, {self.action_size}), got {action}")
        return divmod(action, self.size)

    def coord_to_action(self, row: int, col: int) -> int:
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError(f"coordinate out of bounds: ({row}, {col})")
        return row * self.size + col

    def legal_mask(self) -> np.ndarray:
        if self.is_terminal:
            return np.zeros(self.action_size, dtype=bool)
        return (self.board.reshape(-1) == 0)

    def legal_actions(self) -> np.ndarray:
        return np.flatnonzero(self.legal_mask())

    def apply(self, action: int) -> "GomokuState":
        if self.is_terminal:
            raise ValueError("cannot apply a move to a terminal state")

        row, col = self.action_to_coord(action)
        if self.board[row, col] != 0:
            raise ValueError(f"illegal move at occupied coordinate ({row}, {col})")

        board = self.board.copy()
        board[row, col] = self.current_player
        moves_played = self.moves_played + 1

        winner: int | None
        if self._is_winning_move(board, row, col, self.current_player):
            winner = self.current_player
        elif moves_played == self.action_size:
            winner = 0
        else:
            winner = None

        return GomokuState(
            board=board,
            current_player=-self.current_player,
            last_move=action,
            winner=winner,
            moves_played=moves_played,
            win_length=self.win_length,
        )

    def encode(self) -> np.ndarray:
        """Return two planes from the current player's perspective."""

        own = (self.board == self.current_player).astype(np.float32)
        opponent = (self.board == -self.current_player).astype(np.float32)
        return np.stack([own, opponent], axis=0)

    def terminal_value_for_current_player(self) -> float:
        if not self.is_terminal:
            raise ValueError("state is not terminal")
        if self.winner == 0:
            return 0.0
        return 1.0 if self.winner == self.current_player else -1.0

    def _is_winning_move(
        self, board: np.ndarray, row: int, col: int, player: int
    ) -> bool:
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        for dr, dc in directions:
            count = 1
            count += self._count_direction(board, row, col, player, dr, dc)
            count += self._count_direction(board, row, col, player, -dr, -dc)
            if count >= self.win_length:
                return True
        return False

    def _count_direction(
        self, board: np.ndarray, row: int, col: int, player: int, dr: int, dc: int
    ) -> int:
        count = 0
        row += dr
        col += dc
        while 0 <= row < self.size and 0 <= col < self.size and board[row, col] == player:
            count += 1
            row += dr
            col += dc
        return count

    def render(self) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        header = "   " + " ".join(f"{i + 1:2d}" for i in range(self.size))
        rows = []
        for i, row in enumerate(self.board):
            cells = " ".join(f"{symbols[int(cell)]:>2}" for cell in row)
            rows.append(f"{i + 1:2d} {cells}")
        return "\n".join([header, *rows])
