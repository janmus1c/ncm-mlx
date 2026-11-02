"""Random tensors under upstream's names and PyTorch layouts for a codec spec.

Shared by ``tests/test_official_codec.py`` and ``scripts/official_codec_parity.py``. The script
loads this state into upstream's ``VQVAE`` with ``strict=True``, which checks every name and shape
here against the real module, and stores upstream's codes; the test converts the same state with
the port's rules and must reproduce those codes without PyTorch.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ncm_mlx.official.spec import CodecSpec

SMALL = {"vocab_size": 61, "dim": 24}


def small(codec: CodecSpec) -> CodecSpec:
    """The real level geometry and head count with a narrow width, vocabulary and codebooks."""
    return replace(codec, **SMALL, codebook_sizes=(16,) * len(codec.levels))


def stage_indices(count: int, between: bool) -> list[int]:
    """List index of each resampling stage once the parameter-free norms between them are counted."""
    return [i * 2 if between else i for i in range(count)]


def upstream_state(codec: CodecSpec, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    D, V, L = codec.dim, codec.vocab_size, codec.seq_len

    def w(*shape, scale=0.15, offset=0.0):
        return (rng.normal(0, scale, shape) + offset).astype(np.float32)

    s = {"encoder.wte.weight": w(V, D), "encoder.wpe.weight": w(L, D)}
    for i, n in enumerate(codec.codebook_sizes):
        s[f"vq.embeddings.{i}.weight"] = w(n, D, scale=1.0)
        s[f"vq.ema_cluster_size_{i}"] = np.ones(n, dtype=np.float32)
        s[f"vq.ema_w_{i}"] = w(n, D)
        s[f"vq.codebook_hit_{i}"] = np.zeros(n, dtype=np.int64)
    for i in range(len(codec.levels)):
        s[f"vq.quant_resi.{i}.weight"], s[f"vq.quant_resi.{i}.bias"] = w(D, D, 3, scale=0.05), w(D, scale=0.05)
    strides = codec.l0_strides
    for i, k in zip(stage_indices(len(strides), codec.cascade_norm), strides):
        s[f"vq.learned_down_l1.{i}.weight"], s[f"vq.learned_down_l1.{i}.bias"] = w(D, D, k, scale=0.1), w(D, scale=0.1)
    for i, k in zip(stage_indices(len(strides), codec.cascade_norm), reversed(strides)):
        s[f"vq.learned_up_l1.{i}.weight"], s[f"vq.learned_up_l1.{i}.bias"] = w(D, D, k, scale=0.1), w(D, scale=0.1)
    hd = D // codec.n_head
    block = {"ln_1.weight": (D,), "ln_1.bias": (D,), "attn.c_attn.weight": (3 * D, D), "attn.c_attn.bias": (3 * D,),
             "attn.c_proj.weight": (D, D), "attn.c_proj.bias": (D,), "attn.q_norm.weight": (hd,),
             "attn.k_norm.weight": (hd,), "ln_2.weight": (D,), "ln_2.bias": (D,),
             "mlp.c_fc.weight": (4 * D, D), "mlp.c_fc.bias": (4 * D,),
             "mlp.c_proj.weight": (D, 4 * D), "mlp.c_proj.bias": (D,)}
    for name, shape in block.items():
        s[f"decoder.middle_block.{name}"] = w(*shape, offset=1.0 if name.endswith("norm.weight") or
                                              name in ("ln_1.weight", "ln_2.weight") else 0.0)
    s["decoder.ln_f.weight"] = w(D, offset=1.0)
    s["decoder.output_layer.weight"] = w(V, D)
    return s
