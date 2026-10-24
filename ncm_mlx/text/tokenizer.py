"""Fixed-length character tokenizer.

The first-stage "pre-tokenizer" in upstream NCM is BPE or a task tokenizer; this
project starts with characters so that reconstruction accuracy is directly
interpretable. Sequences are right-padded to a fixed length because the
multiscale quantizer operates on a fixed final level length.
"""

from __future__ import annotations

import numpy as np


class CharTokenizer:
    def __init__(self, alphabet: str, pad: str, seq_len: int):
        if pad not in alphabet:
            alphabet = pad + alphabet
        self.chars = sorted(set(alphabet))
        self.pad = pad
        self.seq_len = seq_len
        self.index = {c: i for i, c in enumerate(self.chars)}
        self.pad_id = self.index[pad]

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        if len(text) > self.seq_len:
            raise ValueError(f"text longer than {self.seq_len}: {text!r}")
        try:
            ids = [self.index[c] for c in text]
        except KeyError as e:
            raise ValueError(f"character {e.args[0]!r} not in alphabet") from None
        return ids + [self.pad_id] * (self.seq_len - len(ids))

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.encode(t) for t in texts], dtype=np.int32)

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)

    def to_dict(self) -> dict:
        return {"alphabet": "".join(self.chars), "pad": self.pad, "seq_len": self.seq_len}

    @classmethod
    def from_dict(cls, d: dict) -> CharTokenizer:
        return cls(d["alphabet"], d["pad"], d["seq_len"])

    @classmethod
    def for_corpus(cls, texts: list[str], pad: str, seq_len: int) -> CharTokenizer:
        return cls("".join(sorted(set("".join(texts)))), pad, seq_len)
