from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from ncm_mlx.config import Config
from ncm_mlx.latent.codec import Codec
from ncm_mlx.models.generator import NextConceptGenerator
from ncm_mlx.text import toygrammar
from ncm_mlx.text.tokenizer import CharTokenizer


@dataclass
class Corpus:
    tokenizer: CharTokenizer
    train_text: list[str]
    val_text: list[str]
    unseen_text: list[str]
    train: np.ndarray
    val: np.ndarray
    train_prefix: np.ndarray | None = None
    val_prefix: np.ndarray | None = None


def load_corpus(cfg: Config) -> Corpus:
    if cfg.data.corpus == "sudoku":
        return _load_sudoku(cfg)
    if cfg.data.corpus != "toy-grammar":
        raise ValueError(f"unknown corpus {cfg.data.corpus!r}")
    if toygrammar.max_sentence_length() > cfg.data.seq_len:
        raise ValueError("data.seq_len is shorter than the longest sentence")
    train, val, unseen = toygrammar.split(cfg.data.split_seed, cfg.data.n_train, cfg.data.n_val)
    tok = CharTokenizer.for_corpus(toygrammar.all_sentences(), toygrammar.PAD, cfg.data.seq_len)
    return Corpus(tok, train, val, unseen, tok.encode_batch(train), tok.encode_batch(val))


def _load_sudoku(cfg: Config) -> Corpus:
    from ncm_mlx.text import sudoku

    if cfg.data.seq_len != 81:
        raise ValueError("sudoku needs data.seq_len = 81")
    d = cfg.data
    puzzles, solutions = sudoku_cache(d.n_train + d.n_val, d.min_givens, d.max_givens, d.split_seed)
    tok = CharTokenizer("0123456789", "0", 81)
    to_text = [sudoku.to_text(s) for s in solutions]
    t, v = slice(0, d.n_train), slice(d.n_train, d.n_train + d.n_val)
    sol = solutions.astype(np.int32)
    puz = puzzles.astype(np.int32)
    train_sol, train_puz = [sol[t]], [puz[t]]
    for seed in d.extra_train_seeds:
        extra = sudoku_cache(d.extra_train_per_seed, d.min_givens, d.max_givens, seed, d.extra_train_unique)
        if d.extra_train_keep_unique:
            extra = unique_subset(extra, f"{d.extra_train_per_seed}-{d.min_givens}-{d.max_givens}-{seed}")
        train_sol.append(extra[1].astype(np.int32))
        train_puz.append(extra[0].astype(np.int32))
    train_sol, train_puz = np.concatenate(train_sol), np.concatenate(train_puz)
    return Corpus(tok, to_text[t], to_text[v], [], train_sol, sol[v], train_puz, puz[v])


def unique_subset(data: tuple[np.ndarray, np.ndarray], key: str) -> tuple[np.ndarray, np.ndarray]:
    """The puzzles of ``data`` that happen to have exactly one solution (cached by ``key``).

    Keeps the random-removal construction of the input, unlike ``unique=True`` generation,
    which removes cells greedily and changes which clues remain.
    """
    from ncm_mlx.text import sudoku

    cache = Path("artifacts/cache") / f"sudoku-keepunique-{key}.npy"
    if cache.exists():
        keep = np.load(cache)
    else:
        keep = np.array([sudoku.count_solutions(p.tolist()) == 1 for p in data[0]])
        np.save(cache, keep)
    return data[0][keep], data[1][keep]


def sudoku_cache(n: int, min_givens: int, max_givens: int, seed: int, unique: bool = False,
                 workers: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(puzzles, solutions)`` for one seed, generated once and cached under artifacts/cache.

    ``unique`` makes every puzzle have exactly one solution (slower to generate).
    """
    from ncm_mlx.text import sudoku

    kind = "sudoku-unique" if unique else "sudoku"
    cache = Path("artifacts/cache") / f"{kind}-{n}-{min_givens}-{max_givens}-{seed}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["puzzles"], z["solutions"]
    if unique:
        puzzles, solutions = sudoku.make_unique_puzzles(n, min_givens, max_givens, seed, workers)
    else:
        puzzles, solutions = sudoku.make_puzzles(n, min_givens, max_givens, seed)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, puzzles=puzzles, solutions=solutions)
    tmp.rename(cache)
    return puzzles, solutions


def seed_all(seed: int) -> random.Random:
    mx.random.seed(seed)
    np.random.seed(seed)
    return random.Random(seed)


def schedule(lr: float, warmup: int, steps: int):
    warm = optim.linear_schedule(lr * 0.01, lr, warmup)
    decay = optim.cosine_decay(lr, max(steps - warmup, 1), lr * 0.05)
    return optim.join_schedules([warm, decay], [warmup])


class BatchStream:
    def __init__(self, data: np.ndarray, batch_size: int, rng: np.random.Generator):
        if not 0 < batch_size <= len(data):
            raise ValueError("batch_size must be between 1 and the dataset size")
        self.data, self.batch_size, self.rng = data, batch_size, rng
        self.order = None
        self.cursor = 0
        self.epoch_rng = copy.deepcopy(rng.bit_generator.state)

    def __iter__(self):
        return self

    def _indices(self):
        if self.order is None or self.cursor + self.batch_size > len(self.data):
            self.epoch_rng = copy.deepcopy(self.rng.bit_generator.state)
            self.order = self.rng.permutation(len(self.data))
            self.cursor = 0
        indices = self.order[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return indices

    def __next__(self):
        return mx.array(self.data[self._indices()])

    def advance(self, steps: int):
        for _ in range(steps):
            self._indices()

    def state_dict(self):
        return {"epoch_rng": copy.deepcopy(self.epoch_rng), "cursor": self.cursor}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["epoch_rng"]
        self.epoch_rng = copy.deepcopy(state["epoch_rng"])
        self.order = self.rng.permutation(len(self.data)) if state["cursor"] else None
        self.cursor = state["cursor"]


def batches(data: np.ndarray, batch_size: int, rng: np.random.Generator):
    return BatchStream(data, batch_size, rng)


def build_codec(cfg: Config, vocab_size: int) -> Codec:
    return Codec(vocab_size, cfg.data.seq_len, cfg.codec)


def build_generator(cfg: Config) -> NextConceptGenerator:
    return NextConceptGenerator(cfg.codec.dim, cfg.codec.levels, cfg.codec.codebook_sizes, cfg.generator)


def save_model(model, path: Path, meta: dict) -> None:
    from ncm_mlx.training.checkpoint import atomic_weights

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_weights(model, path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def load_model(model, path: Path):
    model.load_weights(str(path))
    model.eval()
    mx.eval(model.parameters())
    return model


def param_count(model) -> int:
    from mlx.utils import tree_flatten

    return sum(v.size for _, v in tree_flatten(model.parameters()))


def trainable_count(model) -> int:
    from mlx.utils import tree_flatten

    return sum(v.size for _, v in tree_flatten(model.trainable_parameters()))


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
