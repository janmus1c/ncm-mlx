"""Load the published NCM Sudoku test boards (hazyresearch/ncm-datasets, MIT).

The files are pickled lists of token-id lists in upstream's Sudoku tokenizer:
``0`` blank, ``1-9`` digits, ``10`` row separator, ``11`` BOS. They are read
with an unpickler that refuses every global, so a modified file cannot execute
code. Download them with ``scripts/fetch_upstream_sudoku.sh``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

SEPARATOR, BOS = 10, 11


class _NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load global {module}.{name}")


def _read(path: Path) -> list[list[int]]:
    with open(path, "rb") as f:
        data = _NoGlobals(f).load()
    if not isinstance(data, list) or not all(isinstance(r, list) for r in data):
        raise ValueError(f"{path}: expected a list of lists")
    return data


def load_tokens(path: str | Path) -> list[list[int]]:
    """Raw token rows: 91-token puzzle prompts (BOS, board, BOS) or 89-token solutions."""
    rows = _read(Path(path))
    if not all(all(isinstance(t, int) and 0 <= t <= BOS for t in r) for r in rows):
        raise ValueError(f"{path}: tokens outside the Sudoku vocabulary")
    return rows


def _cells(row: list[int]) -> np.ndarray:
    cells = [t for t in row if t not in (SEPARATOR, BOS)]
    if len(cells) != 81 or not all(0 <= t <= 9 for t in cells):
        raise ValueError(f"unexpected board encoding (len {len(cells)})")
    return np.array(cells, dtype=np.int32)


def load(root: str | Path, difficulty: str) -> tuple[np.ndarray, np.ndarray]:
    """``(puzzles, solutions)`` as ``(N, 81)`` int32 arrays, blanks 0."""
    d = Path(root) / difficulty
    puzzles = np.stack([_cells(r) for r in _read(d / "puzzles.pickle")])
    solutions = np.stack([_cells(r) for r in _read(d / "solutions.pickle")])
    return puzzles, solutions
