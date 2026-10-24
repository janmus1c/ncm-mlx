"""Stage 1: text <-> multiscale latent codes.

    tokens (B, L) -> Encoder -> z (B, L, C) -> MultiscaleResidualVQ -> codes per level
    codes -> z_hat (B, L, C) -> Decoder -> logits (B, L, vocab)

Upstream's released configurations use a lookup-only encoder (token plus
position embedding) and a one-block decoder. This codec makes encoder and
decoder depth configurable; with ``enc_layers > 0`` every position of ``z`` can
carry sentence-level information, which the coarse levels need to represent
anything beyond local averages. The pre-quantization norm (LayerNorm, or
upstream's GroupNorm with ``pre_quant_groups``) has no affine parameters because
squared distances are not scale invariant.
"""

from __future__ import annotations

import random

import mlx.core as mx
from mlx import nn

from ncm_mlx.config import CodecConfig
from ncm_mlx.latent.quantizer import MultiscaleResidualVQ
from ncm_mlx.models.blocks import Block


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, cfg: CodecConfig):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, cfg.dim)
        self.wpe = nn.Embedding(seq_len, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = [Block(cfg.dim, cfg.n_head, cfg.dropout) for _ in range(cfg.enc_layers)]
        if cfg.pre_quant_groups:
            # Statistics over a whole sequence per channel group, as torch's GroupNorm on (B, C, L).
            self.norm = nn.GroupNorm(cfg.pre_quant_groups, cfg.dim, affine=False, pytorch_compatible=True)
        else:
            self.norm = nn.LayerNorm(cfg.dim, affine=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        x = self.drop(self.wte(tokens) + self.wpe(mx.arange(tokens.shape[1])))
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, cfg: CodecConfig):
        super().__init__()
        self.wpe = nn.Embedding(seq_len, cfg.dim)
        self.blocks = [Block(cfg.dim, cfg.n_head, cfg.dropout) for _ in range(max(cfg.dec_layers, 1))]
        self.ln = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, vocab_size, bias=False)

    def __call__(self, z_hat: mx.array) -> mx.array:
        x = z_hat + self.wpe(mx.arange(z_hat.shape[1]))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))


class Codec(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, cfg: CodecConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(vocab_size, seq_len, cfg)
        self.vq = MultiscaleResidualVQ(
            cfg.dim, cfg.levels, cfg.codebook_sizes, cfg.decay, cfg.epsilon, cfg.commitment,
            cfg.quant_resi_ratio, cfg.restart_threshold, cfg.learned_l0,
        )
        self.decoder = Decoder(vocab_size, seq_len, cfg)

    def encode(self, tokens: mx.array) -> list[mx.array]:
        codes, _, _ = self.vq.quantize(self.encoder(tokens))
        return codes

    def decode_codes(self, codes: list[mx.array]) -> list[mx.array]:
        """Logits from every prefix of levels: element k decodes levels 0..k."""
        return [self.decoder(z) for z in self.vq.codes_to_z_hats(codes)]

    def loss(self, tokens: mx.array, rng: random.Random | None = None):
        """Reconstruction CE through a straight-through z_hat plus the VQ loss.

        With probability ``fine_drop_prob`` the decoder receives a truncated
        reconstruction (levels 0..k for a random k), as in upstream's fine-level
        dropout, so coarse prefixes decode to plausible text.
        """
        z = self.encoder(tokens)
        codes, z_hats, inputs = self.vq.quantize(z)
        vq_loss = self.vq.loss(z, z_hats)
        z_hat = z_hats[-1]
        rng = rng or random
        K = len(z_hats)
        if self.training and K >= 2 and rng.random() < self.cfg.fine_drop_prob:
            lo = max(self.cfg.fine_drop_min_keep - 1, 0)
            z_hat = z_hats[rng.randint(lo, K - 2)]
        z_dec = z + mx.stop_gradient(z_hat - z)
        logits = self.decoder(z_dec)
        ce = nn.losses.cross_entropy(logits, tokens, reduction="mean")
        return ce + vq_loss, (ce, vq_loss, codes, inputs)
