"""Resampling between level lengths as constant matrices.

Upstream uses ``F.interpolate(mode="area")`` to downsample the residual to a
level's length and ``mode="linear"`` (``align_corners=False``) to upsample a
level's codes back to the full length. Both are linear maps over the sequence
axis, so here they are precomputed once as dense matrices and applied with a
single batched matmul: ``(l, L) @ (B, L, C) -> (B, l, C)``.
"""

from __future__ import annotations

import math

import numpy as np


def area_matrix(src_len: int, dst_len: int) -> np.ndarray:
    """Adaptive average pooling, identical to PyTorch's area interpolation."""
    m = np.zeros((dst_len, src_len), dtype=np.float32)
    for i in range(dst_len):
        start = (i * src_len) // dst_len
        end = math.ceil((i + 1) * src_len / dst_len)
        m[i, start:end] = 1.0 / (end - start)
    return m


def linear_matrix(src_len: int, dst_len: int) -> np.ndarray:
    """1-D linear interpolation with ``align_corners=False`` semantics."""
    m = np.zeros((dst_len, src_len), dtype=np.float32)
    if src_len == 1:
        m[:, 0] = 1.0
        return m
    scale = src_len / dst_len
    for j in range(dst_len):
        src = max((j + 0.5) * scale - 0.5, 0.0)
        i0 = min(math.floor(src), src_len - 1)
        i1 = min(i0 + 1, src_len - 1)
        w = src - i0
        m[j, i0] += 1.0 - w
        m[j, i1] += w
    return m
