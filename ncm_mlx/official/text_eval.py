"""Held-out and free-running evaluation of TinyStories text models.

Reported separately, never mixed:

* Codec reconstruction on held-out test windows (token accuracy, exact windows,
  per-level cumulative accuracy, code usage).
* Generator teacher-forced **code** cross-entropy per level (nats per code);
  this is not a text perplexity.
* AR next-token cross-entropy on the same test windows (nats per BPE token).
* Free-running samples at fixed seeds: length, EOS rate, distinct-n,
  within-sample 4-gram repetition, per-sample unigram entropy (upstream's
  ``H`` over the 8192-token BPE re-encoding), duplicates and copied training
  openings.
* Generative perplexity under one fixed, locally run scorer (GPT-2 large in
  MLX through ``mlx-lm``, pinned revision). Following the upstream TinyStories
  protocol, each decoded sample is GPT-2-tokenized without a BOS token, every
  token after the first is scored, and perplexity is ``exp`` of the corpus mean
  NLL per seed; seeds are then averaged. Empty samples are unscorable and
  counted separately.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
import numpy as np

from ncm_mlx.official import tinystories as ts

SCORER_REPO = "openai-community/gpt2-large"
SCORER_REVISION = "32b71b12589c2f8d625668d2335a01cac3249519"
SCORER_DIR = Path("artifacts/scorers/gpt2-large")
SCORER_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                "merges.txt", "generation_config.json"]


def cut(ids) -> list[int]:
    """Upstream evaluation cleanup: stop at the first EOS and drop padding."""
    ids = [int(t) for t in ids]
    if ts.EOS_ID in ids:
        ids = ids[: ids.index(ts.EOS_ID)]
    return [t for t in ids if t != ts.PAD_ID]


def _ngrams(ids, n):
    return [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]


def entropy(ids) -> float:
    c, m = Counter(ids), len(ids)
    return -sum(v / m * math.log(v / m) for v in c.values()) if m else 0.0


def diagnostics(raw_rows: list[list[int]], texts: list[str], bpe, train_openings: set | None = None,
                opening: int = 32) -> dict:
    """Length, repetition and diversity diagnostics of decoded samples. Not quality proofs."""
    ids = [cut(r) for r in raw_rows]
    reencoded = [bpe.encode(t) for t in texts]
    out = {
        "samples": len(ids),
        "mean_tokens": round(float(np.mean([len(x) for x in ids])), 2),
        "mean_chars": round(float(np.mean([len(t) for t in texts])), 1),
        "eos_in_window_pct": round(100 * float(np.mean([ts.EOS_ID in [int(v) for v in r] for r in raw_rows])), 2),
        "empty_pct": round(100 * float(np.mean([not t.strip() for t in texts])), 2),
        "unigram_entropy": round(float(np.mean([entropy(x) for x in reencoded if x])), 4),
        "duplicate_samples_pct": round(100 * (1 - len(set(texts)) / max(len(texts), 1)), 2),
    }
    for n in (1, 2, 3):
        grams = [g for x in ids for g in _ngrams(x, n)]
        out[f"distinct_{n}"] = round(len(set(grams)) / max(len(grams), 1), 4)
    rep = []
    for x in ids:
        g = _ngrams(x, 4)
        if g:
            rep.append(1 - len(set(g)) / len(g))
    out["repeated_4gram_pct"] = round(100 * float(np.mean(rep)) if rep else 0.0, 2)
    if train_openings is not None:
        hits = [len(x) >= opening and bytes(np.array(x[:opening], np.uint16)) in train_openings for x in ids]
        out[f"copied_train_opening_{opening}_pct"] = round(100 * float(np.mean(hits)), 2)
    return out


def train_opening_index(train_rows: np.ndarray, opening: int = 32) -> set:
    return {r.tobytes() for r in np.ascontiguousarray(train_rows[:, :opening])}


# ---------------------------------------------------------------- scorer


def fetch_scorer(dest: Path = SCORER_DIR) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(SCORER_REPO, revision=SCORER_REVISION, allow_patterns=SCORER_FILES, local_dir=str(dest))
    return dest


class Scorer:
    """GPT-2 large in MLX (``mlx-lm``), loaded from a local snapshot of a pinned revision."""

    def __init__(self, path: Path = SCORER_DIR):
        from mlx_lm import load

        if not (path / "model.safetensors").exists():
            fetch_scorer(path)
        self.model, self.tokenizer = load(str(path))
        self.path = path

    def token_nll(self, texts: list[str], batch: int = 32, max_tokens: int = 1024,
                  skip: list[int] | None = None) -> list[tuple[float, int]]:
        """Per text: summed NLL of tokens 2..n given their prefix, and the number of scored tokens.

        ``skip[i]`` leading tokens of text ``i`` are read as context but not counted.
        """
        enc = [self.tokenizer.encode(t)[:max_tokens] if t else [] for t in texts]
        order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
        out: list[tuple[float, int]] = [(0.0, 0)] * len(enc)
        for s in range(0, len(order), batch):
            idx = [i for i in order[s : s + batch] if len(enc[i]) >= 2]
            if not idx:
                continue
            T = max(len(enc[i]) for i in idx)
            ids = np.zeros((len(idx), T), dtype=np.int32)
            mask = np.zeros((len(idx), T), dtype=np.float32)
            for r, i in enumerate(idx):
                ids[r, : len(enc[i])] = enc[i]
                mask[r, (skip[i] if skip is not None else 0) : len(enc[i])] = 1
            x = mx.array(ids)
            logits = self.model(x[:, :-1]).astype(mx.float32)
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            nll = -mx.take_along_axis(logp, x[:, 1:, None], axis=-1)[..., 0]
            m = mx.array(mask[:, 1:])
            sums = np.array((nll * m).sum(axis=1))
            counts = mask[:, 1:].sum(1)
            for r, i in enumerate(idx):
                out[i] = (float(sums[r]), int(counts[r]))
        return out


def gen_ppl(scorer: Scorer, texts_by_seed: dict[int, list[str]], prefixes: list[str] | None = None) -> dict:
    """Generative perplexity per seed. With ``prefixes`` (one per sample, the same list for every seed) the
    scorer reads ``prefix + continuation`` but only tokens after the prefix's own GPT-2 token count are scored."""
    per = {}
    unscorable = 0
    skip = None
    if prefixes is not None:
        skip = [len(scorer.tokenizer.encode(p)) if p else 0 for p in prefixes]
    for seed, texts in texts_by_seed.items():
        if prefixes is None:
            res = scorer.token_nll(texts)
        else:
            res = scorer.token_nll([p + t if t else "" for p, t in zip(prefixes, texts)], skip=skip)
        unscorable += sum(c == 0 for _, c in res)
        nll, cnt = sum(a for a, _ in res), sum(c for _, c in res)
        per[seed] = round(math.exp(nll / max(cnt, 1)), 3)
    vals = list(per.values())
    return {"scorer": f"{SCORER_REPO}@{SCORER_REVISION}", "per_seed": per,
            "mean": round(float(np.mean(vals)), 3),
            "sd": round(float(np.std(vals, ddof=1)), 3) if len(vals) > 1 else 0.0,
            "unscorable_samples": unscorable}


# ---------------------------------------------------------------- samplers


def ncm_sampler(weights: Path):
    from ncm_mlx.official import sample

    m = sample.load(weights)

    def run(batch: int, seed: int, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0):
        g = sample.generate(m, sample.unconditional_prefix(m, batch), temperature=temperature, top_k=top_k,
                            top_p=top_p, seed=seed)
        return g.tokens

    return m, run, len(m.codec.levels) - m.gen.first


def ar_sampler(cfg):
    from ncm_mlx.official.ar import load_ar

    model = load_ar(cfg)

    def run(batch: int, seed: int, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0):
        return model.generate(batch, temperature, top_k, top_p, seed)

    return model, run, model.seq_len


def draw(run, n: int, seeds: list[int], batch: int, **kw) -> dict[int, list[list[int]]]:
    out = {}
    for seed in seeds:
        rows = []
        for s in range(0, n, batch):
            b = min(batch, n - s)
            rows += np.array(run(b, seed * 100_003 + s, **kw)).tolist()
        out[seed] = rows
    return out


def latency(run, seq_len: int, forwards: int, batch_sizes=(1, 16), warmup: int = 3, iters: int = 20) -> list[dict]:
    from ncm_mlx.eval.benchmark import bench_sampler

    return bench_sampler(lambda b, s: run(b, s), seq_len, forwards, tuple(batch_sizes), warmup, iters)


def gpu_utilization() -> str:
    import subprocess

    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], check=False, capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    for line in out.splitlines():
        if '"Device Utilization %"' in line:
            i = line.index('"Device Utilization %"')
            return line[i : i + 40].split(",")[0]
    return "unavailable"


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_samples(path: Path, model: str, rows_by_seed: dict[int, list[list[int]]], texts_by_seed) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for seed, rows in rows_by_seed.items():
            for i, (r, t) in enumerate(zip(rows, texts_by_seed[seed])):
                f.write(json.dumps({"model": model, "seed": seed, "index": i, "text": t, "ids": r}) + "\n")
