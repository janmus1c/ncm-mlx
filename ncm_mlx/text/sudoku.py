"""Synthetic Sudoku puzzles for conditional generation.

A puzzle is 81 characters, row-major, ``0`` for a blank; its target is the
81-digit solution it was cut from. Solved grids are filled by randomized
backtracking (random digit order at every cell), so they are spread across the
space of valid grids rather than being symmetry variants of one grid. Puzzles
keep a uniformly random number of givens in ``[min_givens, max_givens]``.

Uniqueness of the solution is **not** enforced (checking it per puzzle would
dominate data generation time). Scoring therefore accepts any valid grid that
agrees with every given, which is the standard "solved" criterion; agreement
with the reference solution is reported separately as cell accuracy.
"""

from __future__ import annotations

import multiprocessing
import os
import random

import numpy as np


def random_grid(rng: random.Random) -> list[int]:
    """One uniformly-ordered backtracking fill of an empty grid."""
    grid = [0] * 81
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9

    def fill(pos: int) -> bool:
        if pos == 81:
            return True
        r, c = divmod(pos, 9)
        b = (r // 3) * 3 + c // 3
        used = rows[r] | cols[c] | boxes[b]
        digits = [d for d in range(1, 10) if not used >> d & 1]
        rng.shuffle(digits)
        for d in digits:
            bit = 1 << d
            grid[pos] = d
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            if fill(pos + 1):
                return True
            rows[r] ^= bit
            cols[c] ^= bit
            boxes[b] ^= bit
        grid[pos] = 0
        return False

    fill(0)
    return grid


def _grid_chunk(args: tuple[int, int]) -> np.ndarray:
    n, seed = args
    rng = random.Random(seed)
    return np.array([random_grid(rng) for _ in range(n)], dtype=np.int8)


def solved_grids(n: int, seed: int, workers: int | None = None) -> np.ndarray:
    """``(n, 81)`` solved grids, deterministic in ``seed`` regardless of worker count."""
    chunk = 2000
    jobs = [(min(chunk, n - i), seed * 1_000_003 + i) for i in range(0, n, chunk)]
    if len(jobs) == 1 or workers == 1:
        return np.concatenate([_grid_chunk(j) for j in jobs])
    workers = workers or min(4, max(1, (os.cpu_count() or 2) - 2))
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        return np.concatenate(pool.map(_grid_chunk, jobs))


def make_puzzles(n: int, min_givens: int, max_givens: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``(puzzles, solutions)``, each ``(n, 81)`` int8; blanks are 0."""
    rng = np.random.default_rng(seed)
    solutions = solved_grids(n, seed)
    givens = rng.integers(min_givens, max_givens + 1, size=n)
    scores = rng.random((n, 81))
    keep = scores < np.sort(scores, axis=1)[np.arange(n), givens - 1][:, None] + 1e-12
    puzzles = np.where(keep, solutions, 0).astype(np.int8)
    return puzzles, solutions


def is_valid_grid(grid: np.ndarray) -> bool:
    g = np.asarray(grid).reshape(9, 9)
    target = set(range(1, 10))
    for i in range(9):
        if set(g[i].tolist()) != target or set(g[:, i].tolist()) != target:
            return False
    for br in range(3):
        for bc in range(3):
            if set(g[3 * br : 3 * br + 3, 3 * bc : 3 * bc + 3].ravel().tolist()) != target:
                return False
    return True


def score(puzzle: np.ndarray, prediction: np.ndarray, solution: np.ndarray) -> dict:
    puzzle, prediction, solution = (np.asarray(a).reshape(81) for a in (puzzle, prediction, solution))
    given = puzzle != 0
    valid = is_valid_grid(prediction)
    respects = bool((prediction[given] == puzzle[given]).all())
    return {
        "solved": valid and respects,
        "valid_grid": valid,
        "respects_givens": respects,
        "cell_acc": float((prediction == solution).mean()),
        "blank_cell_acc": float((prediction[~given] == solution[~given]).mean()) if (~given).any() else 1.0,
    }


def to_text(grid: np.ndarray) -> str:
    return "".join(str(int(x)) for x in np.asarray(grid).reshape(81))


def count_solutions(puzzle: list[int], limit: int = 2) -> int:
    """Number of solutions of ``puzzle`` (0 = blank), counting up to ``limit``."""
    grid = list(puzzle)
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    for pos, d in enumerate(grid):
        if d:
            r, c = divmod(pos, 9)
            bit = 1 << d
            b = (r // 3) * 3 + c // 3
            if rows[r] & bit or cols[c] & bit or boxes[b] & bit:
                return 0
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
    blanks = [p for p in range(81) if not grid[p]]
    found = 0

    def solve() -> bool:
        nonlocal found
        best, best_free = -1, None
        for p in blanks:
            if grid[p]:
                continue
            r, c = divmod(p, 9)
            free = ~(rows[r] | cols[c] | boxes[(r // 3) * 3 + c // 3]) & 0x3FE
            if not free:
                return False
            if best_free is None or free.bit_count() < best_free.bit_count():
                best, best_free = p, free
                if free.bit_count() == 1:
                    break
        if best < 0:
            found += 1
            return found >= limit
        r, c = divmod(best, 9)
        b = (r // 3) * 3 + c // 3
        free = best_free
        while free:
            bit = free & -free
            free ^= bit
            grid[best] = bit.bit_length() - 1
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            done = solve()
            rows[r] ^= bit
            cols[c] ^= bit
            boxes[b] ^= bit
            grid[best] = 0
            if done:
                return True
        return False

    solve()
    return found


def unique_puzzle(solution: list[int], givens: int, rng: random.Random) -> list[int] | None:
    """Remove cells from ``solution`` in random order while the solution stays unique.

    Returns a puzzle with exactly ``givens`` clues, or None if the random order reached a
    minimal puzzle with more clues first.
    """
    puzzle = list(solution)
    order = list(range(81))
    rng.shuffle(order)
    clues = 81
    for pos in order:
        if clues == givens:
            return puzzle
        keep = puzzle[pos]
        puzzle[pos] = 0
        if count_solutions(puzzle) == 1:
            clues -= 1
        else:
            puzzle[pos] = keep
    return puzzle if clues == givens else None


def _unique_chunk(args: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    n, min_givens, max_givens, seed = args
    rng = random.Random(seed)
    puzzles, solutions = [], []
    while len(puzzles) < n:
        solution = random_grid(rng)
        puzzle = unique_puzzle(solution, rng.randint(min_givens, max_givens), rng)
        if puzzle is not None:
            puzzles.append(puzzle)
            solutions.append(solution)
    return np.array(puzzles, dtype=np.int8), np.array(solutions, dtype=np.int8)


def make_unique_puzzles(n: int, min_givens: int, max_givens: int, seed: int,
                        workers: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Like :func:`make_puzzles`, but every puzzle has exactly one solution."""
    chunk = 500
    jobs = [(min(chunk, n - i), min_givens, max_givens, seed * 1_000_003 + i) for i in range(0, n, chunk)]
    if len(jobs) == 1 or workers == 1:
        parts = [_unique_chunk(j) for j in jobs]
    else:
        workers = workers or min(4, max(1, (os.cpu_count() or 2) - 2))
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            parts = pool.map(_unique_chunk, jobs)
    return np.concatenate([p for p, _ in parts]), np.concatenate([s for _, s in parts])
