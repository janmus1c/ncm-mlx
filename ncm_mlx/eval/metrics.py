"""Codec and sample-quality metrics. Codec accuracy and generator quality are reported separately."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ncm_mlx.text import toygrammar


def codec_report(codec, tokens: np.ndarray, batch: int = 512) -> dict:
    """Held-out reconstruction from every level prefix, and codebook utilization per level."""
    import mlx.core as mx

    K = codec.vq.num_levels
    char_hits = np.zeros(K)
    seq_hits = np.zeros(K)
    usage = [Counter() for _ in range(K)]
    for i in range(0, len(tokens), batch):
        t = mx.array(tokens[i : i + batch])
        codes = codec.encode(t)
        for k, logits in enumerate(codec.decode_codes(codes)):
            pred = mx.argmax(logits, axis=-1)
            eq = np.array(pred == t)
            char_hits[k] += eq.mean(axis=1).sum()
            seq_hits[k] += eq.all(axis=1).sum()
        for k, c in enumerate(codes):
            usage[k].update(np.array(c).ravel().tolist())
    n = len(tokens)
    util, ppl = [], []
    for k, V in enumerate(codec.vq.codebook_sizes):
        total = sum(usage[k].values())
        p = np.array(list(usage[k].values()), dtype=np.float64) / total
        util.append(round(len(usage[k]) / V, 4))
        ppl.append(round(float(math.exp(-(p * np.log(p)).sum())), 2))
    return {
        "char_acc_by_levels_used": [round(float(x / n), 4) for x in char_hits],
        "seq_exact_by_levels_used": [round(float(x / n), 4) for x in seq_hits],
        "codebook_utilization": util,
        "code_perplexity": ppl,
    }


def sample_report(texts: list[str], train_text: list[str], val_text: list[str]) -> dict:
    """Grammar validity, agreement, memorization and diversity of generated sentences."""
    checks = [toygrammar.check(t) for t in texts]
    train, val = set(train_text), set(val_text)
    bodies = [t.rstrip(toygrammar.PAD) for t in texts]
    valid = [b for b, c in zip(bodies, checks) if c.valid]
    well_formed = [c for c in checks if c.well_formed]
    n = max(len(texts), 1)
    return {
        "n": len(texts),
        "valid": round(len(valid) / n, 4),
        "well_formed": round(len(well_formed) / n, 4),
        "agreement_given_well_formed": round(sum(c.agrees for c in well_formed) / max(len(well_formed), 1), 4),
        "valid_novel": round(sum(b not in train for b in valid) / n, 4),
        "valid_in_val": round(sum(b in val for b in valid) / n, 4),
        "unique": round(len(set(bodies)) / n, 4),
    }


def sudoku_report(puzzles: np.ndarray, predictions: np.ndarray, solutions: np.ndarray) -> dict:
    """Mean of :func:`ncm_mlx.text.sudoku.score` over a batch of puzzles."""
    from ncm_mlx.text import sudoku

    puzzles, predictions, solutions = (np.asarray(a) for a in (puzzles, predictions, solutions))
    if puzzles.ndim != 2 or puzzles.shape[1] != 81 or puzzles.shape[0] == 0:
        raise ValueError("Sudoku evaluation requires a nonempty (N, 81) puzzle batch")
    if predictions.shape != puzzles.shape or solutions.shape != puzzles.shape:
        raise ValueError("Sudoku puzzles, predictions and solutions must have matching (N, 81) shapes")
    rows = [sudoku.score(p, y, s) for p, y, s in zip(puzzles, predictions, solutions)]
    keys = rows[0].keys()
    out = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}
    out["n"] = len(rows)
    return out
