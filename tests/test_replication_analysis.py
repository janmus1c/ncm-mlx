import copy
import json

import pytest

from paper.analyze_replication import ROOT, SOURCE, analyze, count, wilson


def test_wilson_includes_extreme_observations_without_leaving_probability_range():
    assert wilson(0, 2000)[0] == 0
    assert wilson(2000, 2000)[1] == 100
    assert wilson(0, 2000)[1] == pytest.approx(0.191705, abs=1e-6)
    with pytest.raises(ValueError):
        wilson(1, 0)


def test_board_count_rejects_fractional_or_out_of_range_results():
    assert count(0.3325, 2000) == 665
    for rate in (0.33251, -0.1, 1.1):
        with pytest.raises(ValueError):
            count(rate, 2000)


def test_hard_failures_partition_and_seed_denominators_are_preserved():
    result = analyze(json.loads(SOURCE.read_text()))
    hard = result["difficulties"]["hard"]
    second = hard["seed_1"]
    assert second["n"] == 2000
    assert second["solved_boards"] + second["invalid_grids"] + second["valid_grids_violating_givens"] == 2000
    assert second["boards_violating_givens"] == 14
    assert hard["seed_1_minus_seed_0_pp"] == -7.05
    assert hard["descriptive_two_seed_mean_percent"] == 36.775
    assert result["source"] == SOURCE.relative_to(ROOT).as_posix()


def test_assessment_rejects_incompatible_exposure_and_impossible_metric_intersection():
    data = json.loads(SOURCE.read_text())
    mismatched = copy.deepcopy(data)
    mismatched["experiments"]["sudoku-e20-seed1"]["sequence_presentations"] = 8_192_000
    with pytest.raises(ValueError, match="equal generator exposure"):
        analyze(mismatched)
    impossible = copy.deepcopy(data)
    impossible["experiments"]["sudoku-e20-seed1"]["quality"]["upstream-hard"]["valid_grid"] = 0.3
    with pytest.raises(ValueError, match="intersection"):
        analyze(impossible)
