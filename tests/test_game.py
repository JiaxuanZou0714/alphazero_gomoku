from __future__ import annotations

import unittest

import numpy as np

from alphazero_gomoku.game import GomokuState


class GomokuStateTest(unittest.TestCase):
    def test_horizontal_win(self) -> None:
        state = GomokuState.new(size=10)
        # black: (0,0)..(0,4); white: (1,0)..(1,3)
        for col in range(4):
            state = state.apply(state.coord_to_action(0, col))
            state = state.apply(state.coord_to_action(1, col))
        state = state.apply(state.coord_to_action(0, 4))
        self.assertEqual(state.winner, 1)
        self.assertTrue(state.is_terminal)

    def test_vertical_win(self) -> None:
        state = GomokuState.new(size=10)
        for row in range(4):
            state = state.apply(state.coord_to_action(row, 0))
            state = state.apply(state.coord_to_action(row, 1))
        state = state.apply(state.coord_to_action(4, 0))
        self.assertEqual(state.winner, 1)

    def test_diagonal_win(self) -> None:
        state = GomokuState.new(size=10)
        for i in range(4):
            state = state.apply(state.coord_to_action(i, i))
            state = state.apply(state.coord_to_action(i, 9))
        state = state.apply(state.coord_to_action(4, 4))
        self.assertEqual(state.winner, 1)

    def test_anti_diagonal_win(self) -> None:
        state = GomokuState.new(size=10)
        for i in range(4):
            state = state.apply(state.coord_to_action(i, 9 - i))
            state = state.apply(state.coord_to_action(9, i))
        state = state.apply(state.coord_to_action(4, 5))
        self.assertEqual(state.winner, 1)

    def test_four_in_a_row_is_not_a_win(self) -> None:
        state = GomokuState.new(size=10)
        for col in range(3):
            state = state.apply(state.coord_to_action(0, col))
            state = state.apply(state.coord_to_action(1, col))
        state = state.apply(state.coord_to_action(0, 3))
        self.assertIsNone(state.winner)

    def test_draw_on_full_board(self) -> None:
        state = GomokuState.new(size=2, win_length=3)
        for action in range(4):
            state = state.apply(action)
        self.assertEqual(state.winner, 0)
        self.assertEqual(state.terminal_value_for_current_player(), 0.0)

    def test_illegal_moves_raise(self) -> None:
        state = GomokuState.new(size=10)
        state = state.apply(0)
        with self.assertRaises(ValueError):
            state.apply(0)

    def test_encode_perspective(self) -> None:
        state = GomokuState.new(size=10)
        state = state.apply(0)  # black plays, white to move
        planes = state.encode()
        self.assertEqual(planes.shape, (2, 10, 10))
        # plane 0 = current player's (white) stones, plane 1 = opponent's (black)
        self.assertEqual(planes[0].sum(), 0.0)
        self.assertEqual(planes[1].reshape(-1)[0], 1.0)

    def test_legal_mask_terminal(self) -> None:
        state = GomokuState.new(size=10)
        for col in range(4):
            state = state.apply(state.coord_to_action(0, col))
            state = state.apply(state.coord_to_action(1, col))
        state = state.apply(state.coord_to_action(0, 4))
        self.assertFalse(state.legal_mask().any())
        self.assertEqual(len(np.flatnonzero(state.legal_mask())), 0)


if __name__ == "__main__":
    unittest.main()
