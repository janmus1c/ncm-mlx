"""Pre-norm transformer block shared by the codec and the generator."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn


def rope_tables(positions: np.ndarray, head_dim: int, base: float) -> tuple[mx.array, mx.array]:
    """cos/sin tables for arbitrary (possibly repeating) position ids, half-split layout."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (np.arange(half, dtype=np.float32) / half))
    freqs = positions.astype(np.float32)[:, None] * inv_freq[None, :]
    cos = np.concatenate([np.cos(freqs), np.cos(freqs)], axis=-1)
    sin = np.concatenate([np.sin(freqs), np.sin(freqs)], axis=-1)
    return mx.array(cos), mx.array(sin)


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    rotated = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, dim: int, n_head: int, dropout: float):
        super().__init__()
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.drop = nn.Dropout(dropout)

    def __call__(self, x: mx.array, mask: mx.array | None = None, rope=None,
                 segments: tuple[int, list[tuple[int, int]]] | None = None) -> mx.array:
        """``segments = (P, [(start, end), ...])`` replaces a block-diagonal mask.

        The first ``P`` positions attend to themselves; each segment attends to
        the first ``P`` positions and to itself. This computes the same result as
        the equivalent boolean mask without forming the full (T, T) score matrix.
        """
        B, T, C = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            cos, sin = rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        scale = self.head_dim**-0.5
        if segments is None:
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        else:
            P, spans = segments
            kp, vp = k[:, :, :P], v[:, :, :P]
            parts = [mx.fast.scaled_dot_product_attention(q[:, :, :P], kp, vp, scale=scale)] if P else []
            for a, b in spans:
                ks = mx.concatenate([kp, k[:, :, a:b]], axis=2) if P else k[:, :, a:b]
                vs = mx.concatenate([vp, v[:, :, a:b]], axis=2) if P else v[:, :, a:b]
                parts.append(mx.fast.scaled_dot_product_attention(q[:, :, a:b], ks, vs, scale=scale))
            y = mx.concatenate(parts, axis=2)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, dim: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_head, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, 4 * dim)
        self.out = nn.Linear(4 * dim, dim)
        self.drop = nn.Dropout(dropout)

    def __call__(self, x: mx.array, mask: mx.array | None = None, rope=None, segments=None) -> mx.array:
        x = x + self.attn(self.ln1(x), mask, rope, segments)
        return x + self.drop(self.out(nn.gelu(self.fc(self.ln2(x)))))
