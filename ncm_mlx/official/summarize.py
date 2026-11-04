"""XSum summarization with the released next-concept generator.

Follows the upstream evaluation for the ``xsum`` task: the article's GPT-2 token ids, truncated to
the first 512 and padded with ``<|pad|>``, are the prefix; level 0 comes from learned queries; the
codes of every level are the argmax of the (optionally guided) logits; and the final
reconstruction is read out token by token with a frequency and an adjacency penalty on the
decoder logits (:func:`penalized_readout`). Defaults are the ``generation:`` block of the released
``config.yaml`` (:class:`~ncm_mlx.official.spec.DecodingSpec`).
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, replace
from pathlib import Path

import mlx.core as mx
import numpy as np

from ncm_mlx.official import sample
from ncm_mlx.official.spec import DecodingSpec

_CLEANUPS = ((" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"), (" n't", "n't"), (" 'm", "'m"),
             (" 's", "'s"), (" 've", "'ve"), (" 're", "'re"))


def clean_up_tokenization(text: str) -> str:
    """Remove the spaces that word-level detokenizers leave before punctuation and clitics.

    Opt-in: the Hugging Face GPT-2 decode that the parity run compares with (``transformers`` 5.22)
    keeps the text as decoded, so that is the default here.
    """
    for old, new in _CLEANUPS:
        text = text.replace(old, new)
    return text


class _NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load global {module}.{name}")


def load_test_set(directory: str | Path) -> tuple[list[list[int]], list[str]]:
    """The published XSum test articles (GPT-2 token ids) and reference summaries."""
    d = Path(directory)
    out = []
    for name in ("passages.pickle", "reference_summaries.pickle"):
        path = d / name
        if not path.exists():
            raise SystemExit(f"missing {path}; run `python -m ncm_mlx official fetch --task xsum`")
        with open(path, "rb") as f:
            out.append(_NoGlobals(f).load())
    articles, refs = out
    if len(articles) != len(refs) or not all(isinstance(r, str) for r in refs):
        raise SystemExit(f"{d}: articles and references do not match")
    return articles, refs


def prefix_row(ids: list[int], length: int, pad_id: int) -> list[int]:
    """The first ``length`` article ids, padded on the right with ``pad_id``."""
    return (list(ids[:length]) + [pad_id] * length)[:length]


def prefix_rows(m: sample.Official, articles: list[list[int]]) -> mx.array:
    spec = m.task.generator
    rows = [prefix_row(a, spec.prefix_len, spec.prefix_pad_id) for a in articles]
    return mx.array(np.array(rows, dtype=np.int32))


def tokenize_article(m: sample.Official, text: str) -> list[int]:
    """GPT-2 ids of ``text`` without special tokens added, as the upstream evaluation does for raw strings."""
    return m.bpe.encode(text)


def penalized_readout(logits: np.ndarray, token_unigram: np.ndarray, dec: DecodingSpec) -> np.ndarray:
    """Token-by-token argmax over decoder logits with frequency and adjacency penalties.

    ``logits`` is ``(B, T, V)``. With ``count`` the tokens already chosen in a row and
    ``allow = token_unigram * T`` the number of times a token may appear in a sequence of ``T``
    tokens, step ``t`` subtracts ``frequency_penalty * max(count - allow, 0)``, adds
    ``frequency_deficit * (T - t) / T * max(allow - count, 0)`` and subtracts
    ``adjacency_penalty * adjacency_decay ** j`` for the token chosen ``j`` steps ago, for the last
    ``adjacency_window`` steps. All arithmetic is float64, like the reference.
    """
    B, T, V = logits.shape
    lg = logits.astype(np.float64)
    allow = token_unigram.astype(np.float64)[:V] * T
    count = np.zeros((B, V), dtype=np.float64)
    out = np.empty((B, T), dtype=np.int64)
    rows = np.arange(B)
    window = dec.adjacency_window if dec.adjacency_penalty else 0
    for t in range(T):
        step = lg[:, t, :].copy()
        if dec.frequency_penalty > 0:
            step -= dec.frequency_penalty * np.maximum(count - allow, 0.0)
            if dec.frequency_deficit > 0:
                step += dec.frequency_deficit * ((T - t) / T) * np.maximum(allow - count, 0.0)
        for j, back in enumerate(range(1, min(window, t) + 1)):
            step[rows, out[:, t - back]] -= dec.adjacency_penalty * dec.adjacency_decay**j
        out[:, t] = step.argmax(-1)
        count[rows, out[:, t]] += 1.0
    return out


def readout(m: sample.Official, z_logits: mx.array, dec: DecodingSpec) -> np.ndarray:
    if dec.frequency_penalty <= 0 and not dec.adjacency_penalty:
        return np.array(mx.argmax(z_logits, axis=-1))
    if m.token_unigram is None and dec.frequency_penalty > 0:
        raise ValueError("this checkpoint has no stored token unigram for the frequency penalty")
    unigram = np.array(m.token_unigram) if m.token_unigram is not None else np.zeros(z_logits.shape[-1])
    return penalized_readout(np.array(z_logits), unigram, dec)


@dataclass
class Summaries:
    texts: list[str]
    token_ids: np.ndarray
    codes: list[np.ndarray]
    generator_forwards: int


def summarize_prefixes(m: sample.Official, prefix: mx.array, dec: DecodingSpec | None = None, seed: int = 0,
                       clean_up: bool = False, **sampler) -> Summaries:
    """Summaries for a batch of ready prefixes (``(B, prefix_len)`` ids), with the released defaults."""
    dec = dec or m.task.decoding
    g = sample.generate(m, prefix, temperature=dec.temperature, top_k=dec.top_k, top_p=dec.top_p, seed=seed,
                        cfg_scale=dec.cfg_scale, **sampler)
    ids = readout(m, m.codec.decode_logits(g.final), dec)
    texts = [m.decode(row) if dec.cut_at_eos else _decode_all(m, row) for row in ids]
    if clean_up:
        texts = [clean_up_tokenization(t) for t in texts]
    return Summaries(texts, ids, [np.array(c) for c in g.codes], len(m.codec.levels) - m.gen.first)


def _decode_all(m: sample.Official, row) -> str:
    eos, pad = m.task.eos_id, m.task.pad_id
    return m.bpe.decode([int(t) for t in row if int(t) not in (eos, pad)], skip_special=True)


def summarize_articles(m: sample.Official, articles: list[list[int]], batch: int = 16,
                       dec: DecodingSpec | None = None, seed: int = 0, clean_up: bool = False,
                       **sampler) -> tuple[list[str], dict]:
    """Summaries for token-id articles in batches, with timing and peak memory in the returned stats."""
    texts: list[str] = []
    forwards = 0
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    for s in range(0, len(articles), batch):
        part = articles[s : s + batch]
        out = summarize_prefixes(m, prefix_rows(m, part), dec, seed + s, clean_up, **sampler)
        texts += out.texts
        forwards += out.generator_forwards
    seconds = time.perf_counter() - t0
    stats = {"articles": len(articles), "seconds": round(seconds, 3),
             "articles_per_second": round(len(articles) / seconds, 3) if seconds else None,
             "batch": batch, "generator_forwards": forwards,
             "peak_memory_gb": round(mx.get_peak_memory() / 2**30, 2)}
    return texts, stats


def with_overrides(dec: DecodingSpec, **kw) -> DecodingSpec:
    return replace(dec, **{k: v for k, v in kw.items() if v is not None})
