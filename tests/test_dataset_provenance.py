import pickle

import numpy as np
import pytest

from ncm_mlx.text.upstream_sudoku import _read, load


def test_upstream_loader_refuses_pickle_globals(tmp_path):
    path = tmp_path / "untrusted.pickle"
    path.write_bytes(pickle.dumps(eval))
    with pytest.raises(pickle.UnpicklingError, match="refusing to load global"):
        _read(path)


def test_upstream_loader_decodes_published_token_layout(tmp_path):
    directory = tmp_path / "easy"
    directory.mkdir()
    cells = [1, 2, 3, 4, 5, 6, 7, 8, 9] * 9
    row = [11] + [token for offset in range(0, 81, 9) for token in cells[offset:offset + 9] + [10]]
    for name in ("puzzles", "solutions"):
        (directory / f"{name}.pickle").write_bytes(pickle.dumps([row]))
    puzzles, solutions = load(tmp_path, "easy")
    np.testing.assert_array_equal(puzzles, np.array([cells], dtype=np.int32))
    np.testing.assert_array_equal(solutions, puzzles)
