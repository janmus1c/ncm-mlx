"""Byte-level BPE tokenizer compatible with the released TinyStories vocabulary.

The released tokenizer stores an id -> bytes vocabulary and a ranked merge list.
Text is split by special tokens, then pre-tokenized with GPT-2-style word
boundaries (contractions, optional-space letter runs, number runs, other
symbol runs, whitespace), and each chunk's bytes are merged by lowest rank.
The pre-tokenizer is written with ``unicodedata`` so no ``regex`` dependency
is needed; parity with the upstream pattern is checked by
``scripts/official_parity.py``.
"""

from __future__ import annotations

import json
import unicodedata
from itertools import pairwise
from pathlib import Path

_CONTRACTIONS = ("ll", "ve", "re", "s", "d", "m", "t")


def _is_letter(c: str) -> bool:
    return unicodedata.category(c).startswith("L")


def _is_number(c: str) -> bool:
    return unicodedata.category(c).startswith("N")


def _is_other(c: str) -> bool:
    return not (c.isspace() or _is_letter(c) or _is_number(c))


def pre_tokenize(text: str) -> list[str]:
    """Split ``text`` into BPE chunks with the GPT-2 word-boundary rules."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == "'":
            hit = next((s for s in _CONTRACTIONS if text.startswith(s, i + 1)), None)
            if hit:
                out.append(text[i : i + 1 + len(hit)])
                i += 1 + len(hit)
                continue
        start = i + 1 if c == " " and i + 1 < n else i
        matched = False
        for test in (_is_letter, _is_number, _is_other):
            if start < n and test(text[start]) and (start == i or not text[start].isspace()):
                j = start
                while j < n and test(text[j]):
                    j += 1
                out.append(text[i:j])
                i, matched = j, True
                break
        if matched:
            continue
        j = i
        while j < n and text[j].isspace():
            j += 1
        if j < n and j - i >= 2:
            j -= 1
        out.append(text[i:j])
        i = j
    return out


class BPETokenizer:
    def __init__(self, vocab: list[bytes], merges: list[tuple[bytes, bytes]], special: list[str]):
        self.vocab = vocab
        self.ids = {b: i for i, b in enumerate(vocab)}
        if len(self.ids) != len(vocab):
            raise ValueError("BPE vocabulary contains duplicate byte strings")
        self.ranks = {m: r for r, m in enumerate(merges)}
        self.special = special
        self.special_ids = {self.ids[s.encode()] for s in special}

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        d = json.loads(Path(path).read_text())
        vocab = [bytes.fromhex(h) for h in d["vocab"]]
        merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in d["merges"]]
        return cls(vocab, merges, d["special_tokens"])

    def save(self, path: str | Path) -> None:
        d = {"vocab": [b.hex() for b in self.vocab],
             "merges": [[a.hex(), b.hex()] for a, b in sorted(self.ranks, key=self.ranks.get)],
             "special_tokens": self.special}
        Path(path).write_text(json.dumps(d))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _merge(self, word: bytes) -> list[bytes]:
        parts = [bytes([b]) for b in word]
        while len(parts) > 1:
            pairs = [(self.ranks.get((a, b)), k) for k, (a, b) in enumerate(pairwise(parts))]
            ranked = [p for p in pairs if p[0] is not None]
            if not ranked:
                break
            best = min(ranked)[0]
            merged, k = [], 0
            while k < len(parts):
                if k + 1 < len(parts) and self.ranks.get((parts[k], parts[k + 1])) == best:
                    merged.append(parts[k] + parts[k + 1])
                    k += 2
                else:
                    merged.append(parts[k])
                    k += 1
            parts = merged
        return parts

    def _split_special(self, text: str) -> list[tuple[str, bool]]:
        specials = sorted(self.special, key=len, reverse=True)
        out, i, buf = [], 0, 0
        while i < len(text):
            hit = next((s for s in specials if text.startswith(s, i)), None)
            if hit:
                if buf < i:
                    out.append((text[buf:i], False))
                out.append((hit, True))
                i += len(hit)
                buf = i
            else:
                i += 1
        if buf < len(text):
            out.append((text[buf:], False))
        return out

    def encode(self, text: str) -> list[int]:
        ids = []
        for chunk, is_special in self._split_special(text):
            if is_special:
                ids.append(self.ids[chunk.encode()])
                continue
            for word in pre_tokenize(chunk):
                ids += [self.ids[p] for p in self._merge(word.encode("utf-8"))]
        return ids

    def decode(self, ids, skip_special: bool = False) -> str:
        data = b"".join(self.vocab[i] for i in ids if not (skip_special and i in self.special_ids))
        return data.decode("utf-8", errors="replace")
