"""MLX inference modules matching the released NCM checkpoints.

Written from the upstream architecture description (HazyResearch/ncm, MIT,
Copyright (c) 2025 Hermann Kumbong); no upstream code is copied. Semantics that
matter for numerical parity:

* LayerNorm and RMSNorm use eps 1e-6; the codec's pre-quantisation GroupNorm
  uses eps 1e-5 without affine parameters and pools each group over all
  positions.
* Level 0 is resampled by a cascade of strided convolutions (kernel = stride)
  each way; other levels use area pooling down and linear interpolation
  (``align_corners=False``) up. The last level is quantised at full length.
* Every level's upsampled code passes through its own ``Phi`` blend
  ``(1 - r) h + r Conv1d_k3(h)``.
* The generator reads, for level ``k + 1``, the cumulative reconstruction after
  levels ``0..k`` area-pooled to ``l_{k+1}``. Each forward sees the prefix and
  one level; the prefix attends only to itself. RoPE positions restart at 0 for
  the prefix and for the level.
* Codebook-tied heads score ``(2 z.E - |E|^2) / mean|E|^2 * exp(log_inv_T)``.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
from mlx import nn

from ncm_mlx.latent.resample import area_matrix, linear_matrix
from ncm_mlx.official.spec import CodecSpec, GeneratorSpec

LN_EPS = 1e-6
GROUP_NORM_EPS = 1e-5


def rope_tables(positions: np.ndarray, head_dim: int, base: float,
                dtype=mx.float32) -> tuple[mx.array, mx.array]:
    """Half-split RoPE tables, computed in float32 like upstream unless ``dtype`` is float64."""
    np_dtype = np.float64 if dtype == mx.float64 else np.float32
    half = head_dim // 2
    inv_freq = (1.0 / (base ** (np.arange(half, dtype=np_dtype) / half))).astype(np_dtype)
    freqs = positions.astype(np_dtype)[:, None] * inv_freq[None, :]
    return (mx.array(np.concatenate([np.cos(freqs)] * 2, axis=-1), dtype=dtype),
            mx.array(np.concatenate([np.sin(freqs)] * 2, axis=-1), dtype=dtype))


def _rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return x * cos + mx.concatenate([-x[..., half:], x[..., :half]], axis=-1) * sin


class Attention(nn.Module):
    def __init__(self, dim: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=LN_EPS)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=LN_EPS)

    def __call__(self, x: mx.array, mask: mx.array | None = None, rope=None) -> mx.array:
        B, T, C = x.shape
        q, k, v = (t.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
                   for t in mx.split(self.qkv(x), 3, axis=-1))
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = _rotate(q, *rope), _rotate(k, *rope)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)
        return self.proj(y.transpose(0, 2, 1, 3).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, dim: int, n_head: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, eps=LN_EPS)
        self.attn = Attention(dim, n_head)
        self.ln2 = nn.LayerNorm(dim, eps=LN_EPS)
        self.fc = nn.Linear(dim, 4 * dim)
        self.out = nn.Linear(4 * dim, dim)

    def __call__(self, x: mx.array, mask: mx.array | None = None, rope=None) -> mx.array:
        x = x + self.attn(self.ln1(x), mask, rope)
        return x + self.out(nn.gelu(self.fc(self.ln2(x))))


class StridedDown(nn.Module):
    """``Conv1d(kernel=stride)`` as one matmul; weight stored as ``(stride * C_in, C_out)``."""

    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.stride = stride
        self.weight = mx.zeros((stride * dim, dim))
        self.bias = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        B, L, C = x.shape
        return x.reshape(B, L // self.stride, self.stride * C) @ self.weight + self.bias


class StridedUp(nn.Module):
    """``ConvTranspose1d(kernel=stride)``; weight stored as ``(C_in, stride * C_out)``."""

    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.stride = stride
        self.weight = mx.zeros((dim, stride * dim))
        self.bias = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        B, L, _ = x.shape
        return (x @ self.weight).reshape(B, L * self.stride, -1) + self.bias


class Phi(nn.Module):
    def __init__(self, dim: int, ratio: float):
        super().__init__()
        self.ratio = ratio
        self.weight = mx.zeros((dim, 3, dim))
        self.bias = mx.zeros((dim,))

    def __call__(self, h: mx.array) -> mx.array:
        conv = mx.conv1d(h, self.weight, padding=1) + self.bias
        return h * (1 - self.ratio) + conv * self.ratio


def phi_index(level: int, n_levels: int) -> int:
    """Index of the ``Phi`` used at ``level`` when every level owns one (upstream ``share_mode=0``)."""
    k = n_levels
    ticks = (np.linspace(1 / 3 / k, 1 - 1 / 3 / k, k) if k == 4
             else np.linspace(1 / 2 / k, 1 - 1 / 2 / k, k))
    return int(np.argmin(np.abs(ticks - level / max(n_levels - 1, 1))))


class Codec(nn.Module):
    """Frozen tokenizer: token lookup encoder, multiscale residual VQ and one-block decoder."""

    def __init__(self, spec: CodecSpec):
        super().__init__()
        self.spec = spec
        D, L, K = spec.dim, spec.seq_len, len(spec.levels)
        self.levels = list(spec.levels)
        self.tok = nn.Embedding(spec.vocab_size, D)
        if spec.encoder_positional:
            self.pos = nn.Embedding(L, D)
        self.codebooks = [mx.zeros((v, D)) for v in spec.codebook_sizes]
        self.phi = [Phi(D, spec.phi_ratio) for _ in range(K)]
        self.down0 = [StridedDown(D, s) for s in spec.l0_strides]
        self.up0 = [StridedUp(D, s) for s in reversed(spec.l0_strides)]
        self.dec_block = Block(D, spec.n_head)
        self.dec_ln = nn.LayerNorm(D, eps=LN_EPS, bias=False)
        self.dec_head = nn.Linear(D, spec.vocab_size, bias=False)
        self._phi_of = [phi_index(i, K) for i in range(K)]
        self._tables(mx.float32)
        if math.prod(spec.l0_strides) * spec.levels[0] != L:
            raise ValueError("level-0 strides do not map the sequence length to the first level")

    def _tables(self, dtype) -> None:
        np_dtype = np.float64 if dtype == mx.float64 else np.float32
        L = self.spec.seq_len
        self._area_full = [mx.array(area_matrix(L, n, np_dtype), dtype=dtype) for n in self.levels]
        self._linear_up = [mx.array(linear_matrix(n, L, np_dtype), dtype=dtype) for n in self.levels]

    def astype(self, dtype) -> Codec:
        """Cast parameters and rebuild resampling tables; float64 runs only on ``mx.cpu``."""
        self.set_dtype(dtype)
        self._tables(dtype)
        return self

    def embed(self, tokens: mx.array) -> mx.array:
        z = self.tok(tokens)
        if self.spec.encoder_positional:
            z = z + self.pos.weight[: tokens.shape[1]]
        B, L, D = z.shape
        G = self.spec.group_norm_groups
        g = z.reshape(B, L, G, D // G)
        mean = g.mean(axis=(1, 3), keepdims=True)
        var = ((g - mean) ** 2).mean(axis=(1, 3), keepdims=True)
        return ((g - mean) * mx.rsqrt(var + GROUP_NORM_EPS)).reshape(B, L, D)

    def _down(self, z: mx.array, level: int) -> mx.array:
        if level == 0:
            for m in self.down0:
                z = m(z)
            return z
        return self._area_full[level] @ z

    def level_contribution(self, codes: mx.array, level: int) -> mx.array:
        """Full-length contribution of one level's codes (after upsampling and ``Phi``)."""
        q = self.codebooks[level][codes]
        if level == 0:
            for m in self.up0:
                q = m(q)
        elif level != len(self.levels) - 1:
            q = self._linear_up[level] @ q
        return self.phi[self._phi_of[level]](q)

    def encode(self, tokens: mx.array) -> list[mx.array]:
        """Greedy residual quantisation of ``(B, L)`` tokens into one ``(B, l_k)`` code array per level."""
        rest = self.embed(tokens)
        codes = []
        for k, cb in enumerate(self.codebooks):
            x = self._down(rest, k) if k != len(self.levels) - 1 else rest
            d = (x * x).sum(-1, keepdims=True) + (cb * cb).sum(-1) - 2 * (x @ cb.T)
            idx = mx.argmin(d, axis=-1)
            codes.append(idx)
            rest = rest - self.level_contribution(idx, k)
        return codes

    def cumulative(self, codes: list[mx.array]) -> list[mx.array]:
        """Full-length reconstruction after each level: ``z_hat_0, z_hat_0 + z_hat_1, ...``."""
        out, z = [], None
        for k, c in enumerate(codes):
            q = self.level_contribution(c, k)
            z = q if z is None else z + q
            out.append(z)
        return out

    def pool(self, z: mx.array, level: int) -> mx.array:
        return self._area_full[level] @ z

    def decode_logits(self, z: mx.array) -> mx.array:
        return self.dec_head(self.dec_ln(self.dec_block(z)))


class TiedHead(nn.Module):
    def __init__(self, dim: int, code_dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, code_dim, bias=False)
        self.log_inv_t = mx.zeros(())


class Generator(nn.Module):
    """Next-concept transformer; ``level_logits`` runs one forward for one level."""

    def __init__(self, spec: GeneratorSpec, codec: Codec):
        super().__init__()
        self.spec = spec
        D, cd = spec.dim, codec.spec.dim
        levels = codec.levels
        self.first = 0 if spec.query_level0 else 1
        self.target_levels = levels[self.first:]
        n_heads = len(self.target_levels)
        self.in_proj = nn.Linear(cd, D)
        self.level_emb = nn.Embedding(n_heads, D)
        self.blocks = [Block(D, spec.n_head) for _ in range(spec.n_layer)]
        self.ln_f = nn.LayerNorm(D, eps=LN_EPS)
        if spec.tied_heads:
            self.heads = [TiedHead(D, cd) for _ in range(n_heads)]
        else:
            self.heads = [nn.Linear(D, v) for v in codec.spec.codebook_sizes[self.first:]]
        if spec.query_level0:
            self.l0_queries = mx.zeros((1, levels[0], cd))
        self._rope_tables(mx.float32)
        object.__setattr__(self, "_codec_ref", codec)

    def _rope_tables(self, dtype) -> None:
        P, hd = self.spec.prefix_len, self.spec.dim // self.spec.n_head
        self._rope = [rope_tables(np.concatenate([np.arange(P), np.arange(n)]), hd, self.spec.rope_base, dtype)
                      for n in self.target_levels]

    def astype(self, dtype) -> Generator:
        """Cast parameters and rebuild RoPE tables at the target precision."""
        self.set_dtype(dtype)
        self._rope_tables(dtype)
        return self

    @property
    def codec(self) -> Codec:
        return self._codec_ref

    def head_logits(self, j: int, h: mx.array) -> mx.array:
        head = self.heads[j]
        if not self.spec.tied_heads:
            return head(h)
        E = self.codec.codebooks[j + self.first]
        e_sq = (E * E).sum(-1)
        scale = mx.maximum(e_sq.mean(), 1e-12)
        z = head.proj(h)
        return (2.0 * (z @ E.T) - e_sq) / scale * mx.exp(head.log_inv_t)

    def prefix_embedding(self, prefix_tokens: mx.array) -> mx.array:
        return self.in_proj(self.codec.tok(prefix_tokens[:, : self.spec.prefix_len]))

    def _mask(self, prefix_tokens: mx.array, n: int) -> mx.array:
        P = self.spec.prefix_len
        T = P + n
        rows = np.arange(T)[:, None]
        cols = np.arange(T)[None, :]
        allowed = mx.array(~((rows < P) & (cols >= P)))[None, None]
        if self.spec.prefix_pad_id is not None:
            keep = prefix_tokens[:, :P] != self.spec.prefix_pad_id
            keep = mx.concatenate([keep, mx.ones((keep.shape[0], n), dtype=mx.bool_)], axis=1)
            allowed = allowed & keep[:, None, None, :]
        return allowed

    def level_logits(self, j: int, level_input: mx.array, prefix_emb: mx.array,
                     prefix_tokens: mx.array) -> mx.array:
        """Logits for target level ``j`` given its ``(B, l, code_dim)`` input."""
        x = self.in_proj(level_input) + self.level_emb.weight[j]
        P = self.spec.prefix_len
        h = mx.concatenate([mx.broadcast_to(prefix_emb, (x.shape[0], *prefix_emb.shape[1:])), x], axis=1)
        mask = self._mask(prefix_tokens, x.shape[1])
        for blk in self.blocks:
            h = blk(h, mask, self._rope[j])
        return self.head_logits(j, self.ln_f(h)[:, P:])
