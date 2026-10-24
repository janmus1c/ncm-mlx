"""Stage 2: the next-concept generator.

One transformer predicts every code of level k in parallel from the
reconstruction of levels < k. Sequence layout during training (teacher forcing):

    [ prefix (P) | level 0 queries (l_0) | input for level 1 (l_1) | ... | level K-1 (l_{K-1}) ]

Attention is block diagonal, as in upstream's default ``attn_pattern``:
prefix positions see only the prefix; positions of level k see the prefix and
the other positions of level k. Coarser levels reach level k only through its
input (the pooled reconstruction), so training on the whole layout at once is
exactly equivalent to running one level at a time at inference.

Level 0 is predicted from learned query vectors (upstream ``query_level0``).
RoPE positions restart at 0 inside every level, and a learned level embedding
tells the levels apart.
"""

from __future__ import annotations

from itertools import pairwise

import mlx.core as mx
import numpy as np
from mlx import nn

from ncm_mlx.config import GeneratorConfig
from ncm_mlx.models.blocks import Block, rope_tables


def level_mask(prefix_len: int, levels: list[int], pattern: str = "block_diagonal") -> np.ndarray:
    """Boolean (S, S) mask, True where attention is allowed.

    Prefix positions attend to the prefix only. Positions of level k attend to
    the prefix and to level k (``block_diagonal``) or to every level <= k
    (``block_causal``).
    """
    if pattern not in ("block_diagonal", "block_causal"):
        raise ValueError(f"unknown attention pattern {pattern!r}")
    group = np.concatenate([np.full(prefix_len, -1)] + [np.full(l, k) for k, l in enumerate(levels)])
    S = len(group)
    kv_prefix = np.broadcast_to(group[None, :] == -1, (S, S))
    if pattern == "block_diagonal":
        visible = group[:, None] == group[None, :]
    else:
        visible = (group[None, :] <= group[:, None]) & (group[None, :] >= 0)
    q_prefix = np.broadcast_to(group[:, None] == -1, (S, S))
    return np.where(q_prefix, kv_prefix, kv_prefix | visible)


def block_diagonal_mask(prefix_len: int, levels: list[int]) -> np.ndarray:
    return level_mask(prefix_len, levels, "block_diagonal")


class NextConceptGenerator(nn.Module):
    def __init__(self, codec_dim: int, levels: list[int], codebook_sizes: list[int], cfg: GeneratorConfig):
        super().__init__()
        self.cfg = cfg
        self.levels = list(levels)
        self.codebook_sizes = list(codebook_sizes)
        self.causal = cfg.attn_pattern == "block_causal"
        P, d = cfg.prefix_len, cfg.n_embd
        if cfg.prefix_vocab:
            self.prefix_emb = nn.Embedding(cfg.prefix_vocab, d)
        else:
            self.prefix = mx.random.normal((P, d)) * d**-0.5
        self.l0_queries = mx.random.normal((levels[0], codec_dim)) * codec_dim**-0.5
        self.wte = nn.Linear(codec_dim, d)
        self.wle = nn.Embedding(len(levels), d)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = [Block(d, cfg.n_head, cfg.dropout) for _ in range(cfg.n_layer)]
        self.ln_f = nn.LayerNorm(d)
        self.heads = [nn.Linear(d, V) for V in codebook_sizes]

        head_dim = d // cfg.n_head
        K = len(levels)
        self._rope, self._mask, self._level_ids = [], [], []
        for k in range(K):
            used = levels[: k + 1] if self.causal else [levels[k]]
            first = 0 if self.causal else k
            pos = np.concatenate([np.arange(P)] + [np.arange(l) for l in used])
            ids = np.concatenate([np.full(l, first + j) for j, l in enumerate(used)])
            self._rope.append(rope_tables(pos, head_dim, cfg.rope_base))
            self._mask.append(mx.array(level_mask(P, used, cfg.attn_pattern)))
            self._level_ids.append(mx.array(ids))
        pos = np.concatenate([np.arange(P)] + [np.arange(l) for l in levels])
        self._rope_train = rope_tables(pos, head_dim, cfg.rope_base)
        self._mask_train = mx.array(level_mask(P, levels, cfg.attn_pattern))
        self._level_ids_train = mx.array(np.concatenate([np.full(l, k) for k, l in enumerate(levels)]))
        bounds = np.cumsum([P, *levels])
        self._segments_train = (P, [(int(a), int(b)) for a, b in pairwise(bounds.tolist())])

    def _stack(self, x: mx.array, mask: mx.array | None, rope, segments=None) -> mx.array:
        for blk in self.blocks:
            x = blk(x, mask, rope, segments)
        return self.ln_f(x)

    def _with_prefix(self, h: mx.array, prefix_tokens: mx.array | None) -> mx.array:
        B = h.shape[0]
        if self.cfg.prefix_vocab:
            if prefix_tokens is None or prefix_tokens.shape != (B, self.cfg.prefix_len):
                raise ValueError(f"conditional generator needs prefix_tokens of shape ({B}, {self.cfg.prefix_len})")
            prefix = self.prefix_emb(prefix_tokens)
        else:
            prefix = mx.broadcast_to(self.prefix[None], (B, *self.prefix.shape))
        return mx.concatenate([prefix, h], axis=1)

    def __call__(self, level_inputs: list[mx.array], prefix_tokens: mx.array | None = None) -> list[mx.array]:
        """Teacher-forced logits for all levels. ``level_inputs`` covers levels 1..K-1."""
        B = level_inputs[0].shape[0]
        x = mx.concatenate([self.level0_input(B), *level_inputs], axis=1)
        h = self.drop(self.wte(x) + self.wle(self._level_ids_train))
        h = self._with_prefix(h, prefix_tokens)
        if self.causal:
            h = self._stack(h, self._mask_train, self._rope_train)
        else:
            h = self._stack(h, None, self._rope_train, self._segments_train)
        h = h[:, self.cfg.prefix_len :]
        logits, offset = [], 0
        for k, l in enumerate(self.levels):
            logits.append(self.heads[k](h[:, offset : offset + l]))
            offset += l
        return logits

    def level_logits(self, inputs: list[mx.array], k: int, prefix_tokens: mx.array | None = None) -> mx.array:
        """Logits for level ``k`` from the inputs of levels ``0..k``: one forward.

        ``block_diagonal`` reads only ``inputs[k]`` (prefix + l_k positions);
        ``block_causal`` runs over prefix + levels 0..k. Either way the mask is the
        corresponding sub-block of the training mask, so logits match training.
        """
        x = mx.concatenate(inputs[: k + 1], axis=1) if self.causal else inputs[k]
        h = self.wte(x) + self.wle(self._level_ids[k])
        h = self._stack(self._with_prefix(h, prefix_tokens), self._mask[k], self._rope[k])
        return self.heads[k](h[:, -self.levels[k] :])

    def level0_input(self, batch: int) -> mx.array:
        return mx.broadcast_to(self.l0_queries[None], (batch, *self.l0_queries.shape))
