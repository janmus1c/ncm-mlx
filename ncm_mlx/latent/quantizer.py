"""Multiscale residual vector quantizer (1-D, EMA codebooks).

Shape contract, with level lengths ``l_0 < l_1 < ... < l_{K-1} = L``:

    z        (B, L, C)        continuous encoder output
    codes[k] (B, l_k)         int32 indices into codebook k (size V_k)
    z_hat_k  (B, L, C)        sum over j <= k of phi_j(up_j(E_j[codes[j]]))

Level k quantizes the *remaining residual* after levels < k, downsampled to l_k
positions by area pooling. Its codes are looked up, linearly upsampled back to L
positions, refined by a per-level blended convolution ``phi`` and subtracted from
the residual. The decoder reads ``z_hat_{K-1}``; any prefix ``z_hat_k`` is a
coarser reconstruction of the same sequence.

With ``learned_l0`` the level-0 down- and upsampling are learned instead
(upstream's ``learned_l1_sampling``): one strided convolution from ``L`` to
``l_0`` positions and one transposed convolution back, initialised to area
pooling and nearest repetition. Upstream's released Sudoku codec has L = 89, a
prime, so its ``learned_l1_cascade`` also reduces to a single layer each way. A
four-layer cascade (L = 81 factored by 3) diverged within 100 steps here. A plain
average over the whole sequence is nearly constant, so without this the first
level carries almost no information. Level 0 then passes gradients to these
layers through a straight-through estimator.

Codebooks are not trained by gradients: they follow an exponential moving
average of the inputs assigned to each code, computed from the statistics the
forward pass returns and applied by :meth:`ema_update` outside autodiff.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from ncm_mlx.latent.resample import area_matrix, linear_matrix


def _identity_conv(conv: nn.Module, value: float) -> nn.Module:
    """Channel-identity weights (``value`` at every tap) and zero bias."""
    out_ch, kernel, in_ch = conv.weight.shape
    conv.weight = mx.broadcast_to(mx.eye(out_ch, in_ch)[:, None, :] * value, (out_ch, kernel, in_ch))
    conv.bias = mx.zeros_like(conv.bias)
    return conv


class MultiscaleResidualVQ(nn.Module):
    def __init__(
        self,
        dim: int,
        levels: list[int],
        codebook_sizes: list[int],
        decay: float = 0.95,
        epsilon: float = 1e-5,
        commitment: float = 0.25,
        quant_resi_ratio: float = 0.5,
        restart_threshold: float = 0.0,
        learned_l0: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.levels = list(levels)
        self.codebook_sizes = list(codebook_sizes)
        self.decay = decay
        self.epsilon = epsilon
        self.commitment = commitment
        self.ratio = quant_resi_ratio
        self.restart_threshold = restart_threshold
        L = self.levels[-1]

        self.codebooks = [mx.random.normal((V, dim)) for V in self.codebook_sizes]
        self.ema_w = [mx.array(e) for e in self.codebooks]
        self.ema_count = [mx.zeros((V,)) for V in self.codebook_sizes]
        self.phi = [nn.Conv1d(dim, dim, kernel_size=3, padding=1) for _ in self.levels]
        self.freeze(recurse=False, keys=["codebooks", "ema_w", "ema_count"])
        self.learned_l0 = learned_l0
        if learned_l0:
            if L % self.levels[0]:
                raise ValueError(f"learned_l0 needs L={L} divisible by l_0={self.levels[0]}")
            s = L // self.levels[0]
            self.l0_down = [_identity_conv(nn.Conv1d(dim, dim, s, stride=s), 1.0 / s)]
            self.l0_up = [_identity_conv(nn.ConvTranspose1d(dim, dim, s, stride=s), 1.0)]

        self._down = [mx.array(area_matrix(L, l)) for l in self.levels]
        self._up = [mx.array(linear_matrix(l, L)) for l in self.levels]
        self._down_next = [mx.array(area_matrix(L, l)) for l in self.levels[1:]]

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    @property
    def seq_len(self) -> int:
        return self.levels[-1]

    def down(self, x: mx.array, k: int) -> mx.array:
        if k == 0 and self.learned_l0:
            for conv in self.l0_down:
                x = conv(x)
            return x
        return x if self.levels[k] == self.seq_len else self._down[k] @ x

    def up(self, q: mx.array, k: int) -> mx.array:
        if k == 0 and self.learned_l0:
            for conv in self.l0_up:
                q = conv(q)
            return q
        return q if self.levels[k] == self.seq_len else self._up[k] @ q

    def refine(self, h: mx.array, k: int) -> mx.array:
        if self.ratio == 0:
            return h
        return (1.0 - self.ratio) * h + self.ratio * self.phi[k](h)

    def assign(self, x: mx.array, k: int) -> mx.array:
        """Nearest code by squared distance; ``||x||^2`` is constant per row so it is dropped."""
        e = self.codebooks[k]
        scores = 2.0 * (x @ e.T) - mx.sum(e * e, axis=-1)
        return mx.argmax(scores, axis=-1).astype(mx.int32)

    def level_contribution(self, codes_k: mx.array, k: int) -> mx.array:
        return self.refine(self.up(self.codebooks[k][codes_k], k), k)

    def quantize(self, z: mx.array) -> tuple[list[mx.array], list[mx.array], list[mx.array]]:
        """Encode ``z`` into codes. Returns codes, the cumulative z_hats and the per-level inputs."""
        rest = mx.stop_gradient(z)
        z_hat = mx.zeros_like(z)
        codes, z_hats, inputs = [], [], []
        for k in range(self.num_levels):
            x = self.down(rest, k)
            idx = self.assign(mx.stop_gradient(x), k)
            if k == 0 and self.learned_l0:
                q = self.codebooks[0][idx]
                contrib = self.refine(self.up(x + mx.stop_gradient(q - x), 0), 0)
            else:
                contrib = self.level_contribution(idx, k)
            z_hat = z_hat + contrib
            rest = rest - mx.stop_gradient(contrib)
            codes.append(idx)
            z_hats.append(z_hat)
            inputs.append(x)
        return codes, z_hats, inputs

    def loss(self, z: mx.array, z_hats: list[mx.array]) -> mx.array:
        """Commitment on the encoder plus a fit term that trains ``phi``, averaged over levels."""
        z_sg = mx.stop_gradient(z)
        total = mx.array(0.0)
        for z_hat in z_hats:
            total = total + self.commitment * mx.mean((mx.stop_gradient(z_hat) - z) ** 2)
            total = total + mx.mean((z_hat - z_sg) ** 2)
        return total / len(z_hats)

    def codes_to_z_hats(self, codes: list[mx.array]) -> list[mx.array]:
        B = codes[0].shape[0]
        z_hat = mx.zeros((B, self.seq_len, self.dim))
        out = []
        for k, c in enumerate(codes):
            z_hat = z_hat + self.level_contribution(c, k)
            out.append(z_hat)
        return out

    def next_level_input(self, z_hat: mx.array, k: int) -> mx.array:
        """Generator input for level ``k + 1``: the reconstruction so far, pooled to ``l_{k+1}``."""
        return z_hat if self.levels[k + 1] == self.seq_len else self._down_next[k] @ z_hat

    def teacher_inputs(self, codes: list[mx.array]) -> list[mx.array]:
        """Inputs the generator sees for levels 1..K-1 when the true coarser codes are known."""
        z_hats = self.codes_to_z_hats(codes[:-1])
        return [self.next_level_input(z_hat, k) for k, z_hat in enumerate(z_hats)]

    def ema_update(self, codes: list[mx.array], inputs: list[mx.array], key: mx.array | None = None) -> None:
        d = self.decay
        for k, (idx, x) in enumerate(zip(codes, inputs)):
            V = self.codebook_sizes[k]
            flat = mx.stop_gradient(x).reshape(-1, self.dim)
            onehot = (idx.reshape(-1, 1) == mx.arange(V)[None, :]).astype(flat.dtype)
            counts = onehot.sum(axis=0)
            dw = onehot.T @ flat
            ema_count = d * self.ema_count[k] + (1 - d) * counts
            ema_w = d * self.ema_w[k] + (1 - d) * dw
            n = ema_count.sum()
            smoothed = (ema_count + self.epsilon) / (n + V * self.epsilon) * n
            codebook = ema_w / smoothed[:, None]
            if self.restart_threshold > 0:
                key, sub = mx.random.split(key) if key is not None else (None, None)
                pick = mx.random.randint(0, flat.shape[0], (V,), key=sub)
                dead = (ema_count < self.restart_threshold)[:, None]
                fresh = flat[pick]
                codebook = mx.where(dead, fresh, codebook)
                ema_w = mx.where(dead, fresh * self.restart_threshold, ema_w)
                ema_count = mx.where(dead[:, 0], self.restart_threshold, ema_count)
            self.codebooks[k] = codebook
            self.ema_w[k] = ema_w
            self.ema_count[k] = ema_count
