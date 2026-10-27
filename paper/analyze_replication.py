"""Derive the two-seed Sudoku assessment from the committed aggregate reports."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/research/completed-sudoku-results.json"
DESTINATION = ROOT / "docs/research/e20-seed-assessment.json"


def wilson(solved: int, n: int) -> list[float]:
    if n <= 0 or not 0 <= solved <= n:
        raise ValueError("Expected 0 <= solved <= n and n > 0")
    p, z = solved / n, 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [round(100 * (center - half), 6), round(100 * (center + half), 6)]


def count(rate: float, n: int) -> int:
    if not 0 <= rate <= 1 or not math.isclose(rate * n, round(rate * n), abs_tol=1e-8):
        raise ValueError("Aggregate rate does not identify an integer board count")
    return round(rate * n)


def analyze(data: dict) -> dict:
    runs = [data["experiments"][name] for name in ("sudoku-e19-funded", "sudoku-e20-seed1")]
    if [r["training_seed"] for r in runs] != [0, 1]:
        raise ValueError("Expected distinct configured training seeds 0 and 1")
    if any(r["sequence_presentations"] != 16_384_000 for r in runs):
        raise ValueError("Expected equal generator exposure")
    if any(r["generation_temperature"] != 0 for r in runs):
        raise ValueError("Expected greedy evaluation")
    result = {
        "source": SOURCE.relative_to(ROOT).as_posix(),
        "status": "descriptive analysis of retained aggregates; no new model evaluation",
        "training_seeds": [0, 1],
        "generator_presentations_per_seed": 16_384_000,
        "original_ncm_source": "https://arxiv.org/html/2610.14162v1",
        "interval_method": "Wilson 95%, z=1.959963984540054; board independence assumed",
        "limitations": [
            "Two seeds do not estimate population training variability reliably.",
            "The same benchmark boards recur across runs; do not pool them as independent boards.",
            "No paired significance, board-overlap or causal mechanism is inferred from aggregates.",
            "Historical remote evaluation/source identity gaps remain; see the run evidence.",
            "Repeated benchmark-guided recipe selection is not accounted for by these intervals.",
        ],
        "difficulties": {},
    }
    for difficulty, reference in zip(("easy", "medium", "hard"), (99.5, 96.5, 86.1)):
        scores = [r["quality"]["upstream-" + difficulty] for r in runs]
        if any(s["n"] != 2000 for s in scores):
            raise ValueError("Expected 2,000 reported boards per difficulty")
        rows = []
        for score in scores:
            n = int(score["n"])
            solved = count(score["solved"], n)
            valid = count(score["valid_grid"], n)
            preserves = count(score["respects_givens"], n)
            if not max(0, valid + preserves - n) <= solved <= min(valid, preserves):
                raise ValueError("Solved must be the intersection of valid grids and preserved givens")
            rows.append({
                "n": n, "solved_boards": solved, "solved_percent": round(100 * score["solved"], 6),
                "wilson95_percent": wilson(solved, n), "invalid_grids": n - valid,
                "valid_grids_violating_givens": valid - solved,
                "boards_violating_givens": n - preserves,
                "blank_cell_accuracy_percent": round(100 * score["blank_cell_acc"], 6),
                "gap_below_original_ncm_pp": round(reference - 100 * score["solved"], 6),
            })
        values = [row["solved_percent"] for row in rows]
        result["difficulties"][difficulty] = {
            "seed_0": rows[0], "seed_1": rows[1],
            "seed_1_minus_seed_0_pp": round(values[1] - values[0], 6),
            "descriptive_two_seed_mean_percent": round(sum(values) / 2, 6),
            "observed_range_percent": [min(values), max(values)],
            "quoted_original_ncm_percent": reference,
        }
    return result


def main() -> None:
    result = analyze(json.loads(SOURCE.read_text()))
    DESTINATION.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {DESTINATION.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
