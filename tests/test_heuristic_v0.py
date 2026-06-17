import unittest

import numpy as np

from alphazero_gomoku.game import GomokuState
from alphazero_gomoku.heuristic_v0 import candidate_cells, select_move


def _state(board: np.ndarray, player: int = 1) -> GomokuState:
    return GomokuState(
        board=board.astype(np.int8),
        current_player=player,
        last_move=None,
        winner=None,
        moves_played=int((board != 0).sum()),
        win_length=5,
    )


class TestHeuristicV0(unittest.TestCase):
    def test_opens_in_the_center(self):
        s = GomokuState.new()
        self.assertEqual(select_move(s, None), 5 * 10 + 5)

    def test_takes_the_immediate_win(self):
        b = np.zeros((10, 10))
        b[4, 2:6] = 1  # four in a row, both ends open
        move = select_move(_state(b, player=1), None)
        self.assertIn(move, (41, 46))  # complete the five

    def test_blocks_the_immediate_loss(self):
        b = np.zeros((10, 10))
        b[4, 2:6] = -1  # opponent has an open four
        move = select_move(_state(b, player=1), None)
        self.assertIn(move, (41, 46))  # must block one of the open ends

    def test_winning_beats_blocking(self):
        b = np.zeros((10, 10))
        b[4, 2:6] = 1   # our own open four -> we should win, not block
        b[6, 2:6] = -1  # opponent also threatens
        move = select_move(_state(b, player=1), None)
        self.assertIn(move, (41, 46))

    def test_candidates_stay_local(self):
        b = np.zeros((10, 10))
        b[5, 5] = 1
        cells = candidate_cells(b.astype(np.int8), 10)
        # only cells within Chebyshev distance 2 of the lone stone are considered
        for a in cells:
            r, c = divmod(a, 10)
            self.assertLessEqual(max(abs(r - 5), abs(c - 5)), 2)
        self.assertNotIn(5 * 10 + 5, cells)  # the occupied cell is excluded


if __name__ == "__main__":
    unittest.main()
