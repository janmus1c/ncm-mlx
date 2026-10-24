import numpy as np
import pytest

from ncm_mlx.eval.metrics import sudoku_report
from ncm_mlx.text.sudoku import make_puzzles


def test_sudoku_report_scores_every_board():
    puzzles, solutions = make_puzzles(2, 30, 40, seed=0)
    predictions = solutions.copy()
    predictions[1] = 0
    report = sudoku_report(puzzles, predictions, solutions)
    assert report["n"] == 2
    assert report["solved"] == report["cell_acc"] == 0.5
    assert report["respects_givens"] == 0.5


@pytest.mark.parametrize("field", range(3))
@pytest.mark.parametrize("shape", [(1, 81), (3, 81), (2, 3), (2, 9, 9), (0, 81)])
def test_sudoku_report_rejects_missing_extra_or_malformed_boards(field, shape):
    puzzles, solutions = make_puzzles(2, 30, 40, seed=0)
    arrays = [puzzles, solutions.copy(), solutions]
    arrays[field] = np.zeros(shape, dtype=np.int32)
    with pytest.raises(ValueError, match="Sudoku"):
        sudoku_report(*arrays)


def test_sudoku_report_rejects_empty_evaluation():
    empty = np.empty((0, 81), dtype=np.int32)
    with pytest.raises(ValueError, match="nonempty"):
        sudoku_report(empty, empty, empty)
