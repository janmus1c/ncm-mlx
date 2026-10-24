"""Coarse-to-fine sampling: levels in sequence, positions within a level in parallel.

For K levels this performs exactly K generator forwards and one decoder forward,
independent of sequence length:

    level 0:  learned queries            -> codes_0 (l_0 positions at once)
    level k:  pool(z_hat_{<k}) to l_k     -> codes_k (l_k positions at once)
    decode:   z_hat_{K-1} -> decoder      -> characters
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from ncm_mlx.latent.codec import Codec
from ncm_mlx.models.generator import NextConceptGenerator


@dataclass
class SampleOptions:
    temperature: float = 1.0
    top_k: int = 0
    seed: int = 0
    level_temperatures: list[float] | None = None

    def temperature_at(self, k: int) -> float:
        return self.level_temperatures[k] if self.level_temperatures else self.temperature


@dataclass
class Generation:
    codes: list[mx.array]
    tokens: mx.array
    per_level_tokens: list[mx.array] | None = None


def sample_logits(logits: mx.array, temperature: float, top_k: int, key: mx.array) -> mx.array:
    if temperature == 0:
        return mx.argmax(logits, axis=-1).astype(mx.int32)
    logits = logits / temperature
    if top_k and top_k < logits.shape[-1]:
        kth = mx.sort(logits, axis=-1)[..., -top_k][..., None]
        logits = mx.where(logits < kth, -mx.inf, logits)
    return mx.random.categorical(logits, axis=-1, key=key).astype(mx.int32)


def generate_codes(
    gen: NextConceptGenerator, codec: Codec, batch: int, opts: SampleOptions,
    forced: list[mx.array] | None = None, prefix_tokens: mx.array | None = None,
) -> tuple[list[mx.array], mx.array]:
    """Sampled codes per level and the final reconstruction ``z_hat``.

    ``forced`` supplies known codes for the first ``len(forced)`` levels, which
    are used instead of sampling (for error attribution experiments).
    """
    forced = forced or []
    vq = codec.vq
    key = mx.random.key(opts.seed)
    keys = mx.random.split(key, vq.num_levels)
    inputs = [gen.level0_input(batch)]
    z_hat = mx.zeros((batch, vq.seq_len, vq.dim))
    codes = []
    for k in range(vq.num_levels):
        if k < len(forced):
            c = forced[k]
        else:
            logits = gen.level_logits(inputs, k, prefix_tokens)
            c = sample_logits(logits, opts.temperature_at(k), opts.top_k, keys[k])
        codes.append(c)
        z_hat = z_hat + vq.level_contribution(c, k)
        if k + 1 < vq.num_levels:
            inputs.append(vq.next_level_input(z_hat, k))
    return codes, z_hat


def generate(
    gen: NextConceptGenerator, codec: Codec, batch: int, opts: SampleOptions, keep_levels: bool = False,
    forced: list[mx.array] | None = None, prefix_tokens: mx.array | None = None,
) -> Generation:
    codes, z_hat = generate_codes(gen, codec, batch, opts, forced, prefix_tokens)
    if keep_levels:
        per_level = [mx.argmax(logits, axis=-1) for logits in codec.decode_codes(codes)]
        tokens = per_level[-1]
    else:
        tokens = mx.argmax(codec.decoder(z_hat), axis=-1)
        per_level = None
    mx.eval(tokens, codes)
    return Generation(codes=codes, tokens=tokens, per_level_tokens=per_level)
