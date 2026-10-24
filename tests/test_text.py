import pytest

from ncm_mlx.config import Config
from ncm_mlx.latent.resample import area_matrix, linear_matrix
from ncm_mlx.text import toygrammar
from ncm_mlx.text.tokenizer import CharTokenizer


def test_language_size_and_agreement():
    sentences = toygrammar.all_sentences()
    assert len(sentences) == 8 * 6 * 5 * 8 * 8
    assert all(toygrammar.check(s).valid for s in sentences)
    assert toygrammar.max_sentence_length() <= 32


@pytest.mark.parametrize(
    "text,well_formed,valid",
    [
        ("ann sees one red cat.", True, True),
        ("ann sees one red cats.", True, False),
        ("ann sees two red cat.", True, False),
        ("ann sees two red foxes.____", True, True),
        ("ann sees two red foxs.", False, False),
        ("zed sees two red cats.", False, False),
        ("ann sees two red cats", False, False),
        ("", False, False),
    ],
)
def test_checker(text, well_formed, valid):
    c = toygrammar.check(text)
    assert (c.well_formed, c.valid) == (well_formed, valid)


def test_split_is_disjoint_and_deterministic():
    a = toygrammar.split(0, 100, 20)
    b = toygrammar.split(0, 100, 20)
    assert a == b
    train, val, unseen = a
    assert not set(train) & set(val) and not set(train) & set(unseen) and not set(val) & set(unseen)
    assert len(train) + len(val) + len(unseen) == len(toygrammar.all_sentences())


def test_tokenizer_roundtrip_and_padding():
    tok = CharTokenizer.for_corpus(toygrammar.all_sentences(), "_", 32)
    ids = tok.encode("ann sees one red cat.")
    assert len(ids) == 32 and ids[-1] == tok.pad_id
    assert tok.decode(ids).rstrip("_") == "ann sees one red cat."
    with pytest.raises(ValueError):
        tok.encode("x" * 33)
    assert CharTokenizer.from_dict(tok.to_dict()).chars == tok.chars


def test_area_matrix_matches_adaptive_average_pool():
    m = area_matrix(32, 1)
    assert m.shape == (1, 32) and abs(m.sum() - 1) < 1e-6
    m = area_matrix(5, 3)
    expected = [[0.5, 0.5, 0, 0, 0], [0, 1 / 3, 1 / 3, 1 / 3, 0], [0, 0, 0, 0.5, 0.5]]
    assert (abs(m - expected) < 1e-6).all()


def test_linear_matrix_align_corners_false():
    m = linear_matrix(2, 4)
    expected = [[1, 0], [0.75, 0.25], [0.25, 0.75], [0, 1]]
    assert (abs(m - expected) < 1e-6).all()
    assert (abs(linear_matrix(1, 8) - 1) < 1e-6).all()
    assert (abs(linear_matrix(7, 19).sum(axis=1) - 1) < 1e-6).all()


def test_config_roundtrip_and_validation(tmp_path):
    cfg = Config()
    cfg.save(tmp_path / "c.json")
    assert Config.load(tmp_path / "c.json") == cfg
    bad = cfg.to_dict()
    bad["codec"]["levels"] = [1, 2, 4, 8, 16, 31]
    with pytest.raises(ValueError):
        Config.from_dict(bad)
    bad = cfg.to_dict()
    bad["codec"]["typo"] = 1
    with pytest.raises(ValueError):
        Config.from_dict(bad)


def test_sudoku_data_and_scoring():
    from ncm_mlx.text import sudoku

    puzzles, solutions = sudoku.make_puzzles(50, 30, 40, seed=0)
    assert puzzles.shape == solutions.shape == (50, 81)
    givens = (puzzles != 0).sum(1)
    assert givens.min() >= 30 and givens.max() <= 40
    assert all(sudoku.is_valid_grid(s) for s in solutions)
    assert ((puzzles == 0) | (puzzles == solutions)).all()
    perfect = sudoku.score(puzzles[0], solutions[0], solutions[0])
    assert perfect["solved"] and perfect["cell_acc"] == 1.0
    broken = solutions[0].copy()
    broken[[0, 1]] = broken[[1, 0]]
    assert not sudoku.score(puzzles[0], broken, solutions[0])["valid_grid"]
    assert len({sudoku.to_text(s) for s in solutions}) == 50


def test_llm_grid_parsing():
    from ncm_mlx.baselines.openai_llm import grid_text, parse_grid

    sol = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
    text = "<think>try 1 2 3</think>\n" + "\n".join(sol[r * 9 : r * 9 + 9] for r in range(9))
    assert "".join(map(str, parse_grid(text))) == sol
    assert parse_grid("I cannot solve this.") is None
    import numpy as np

    assert grid_text(np.array([0] * 80 + [5])).splitlines()[-1] == "........5"


def test_unique_puzzles_have_one_solution():
    import random

    from ncm_mlx.text import sudoku

    rng = random.Random(0)
    solution = sudoku.random_grid(rng)
    assert sudoku.count_solutions(solution) == 1
    assert sudoku.count_solutions([0] * 81) == 2
    puzzles, solutions = sudoku.make_unique_puzzles(6, 30, 34, seed=1, workers=1)
    for p, s in zip(puzzles, solutions):
        assert 30 <= (p != 0).sum() <= 34
        assert sudoku.count_solutions(p.tolist()) == 1
        assert ((p == 0) | (p == s)).all()
