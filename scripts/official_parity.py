"""Compare the MLX port with upstream PyTorch on the released NCM checkpoints.

Runs both implementations on CPU in fp32 with identical inputs:

1. BPE pre-tokenization and ids (TinyStories) against upstream's ``regex`` tokenizer.
2. Codec: pre-quantisation embedding, per-level codes, decoder logits.
3. Generator: per-level logits with the same teacher-forced codes, routed
   through each implementation's own generation loop.
4. Greedy (temperature 0) end-to-end generation: codes and decoded text.

Needs PyTorch and upstream's dependencies only for this run, e.g.::

    uv run --with torch --with omegaconf --with hydra-core --with regex \\
        python scripts/official_parity.py --upstream /path/to/HazyResearch-ncm --task tinystories

``--fixture`` writes the small upstream-side reference used by
``tests/test_official.py``. Upstream code is imported from ``--upstream`` and
is not part of this repository.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TEXTS = [
    "Once upon a time, there was a little girl named Lily. She loved to play outside in the park.",
    "Tom had a red ball. He threw it high and it didn't come back! \"Oh no,\" said Tom.",
    "The cat sat on the mat. It was 3 o'clock and the sun was warm.\n\nThey'll go home soon.",
    "One day, a big dog named Max found a shiny stone. Max was very happy and he ran to show his friend.",
]
PRETOKENIZE_CASES = TEXTS + [
    "  leading spaces and trailing   ", "tabs\tand\nnewlines\n\n  x", "I'm we've you're it's I'd we'll",
    "numbers 12345 and 3.14 and x2", "emoji 😀 café naïve — dash", "'quoted' ''double'' end'",
    "a<|endoftext|>b<|pad|>", "!!! ??? ... ,,,", "MiXeD CaSe'S",
]


class _NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing global {module}.{name}")


def _plain_pickle(path):
    with open(path, "rb") as f:
        return _NoGlobals(f).load()


def build_upstream(upstream: Path, src: Path):
    import torch
    from omegaconf import OmegaConf

    sys.path.insert(0, str(upstream))
    from models.next_concept import NextConceptTransformer
    from models.vqvae import VQVAE

    cfg = OmegaConf.load(src / "config.yaml")
    vqvae = VQVAE(**cfg.tokenizer.vqvae)
    vqvae.load_state_dict(torch.load(src / "tokenizer.pt", map_location="cpu", weights_only=True)["model"],
                          strict=True)
    vqvae.eval()
    gen_ck = torch.load(src / "generator.pt", map_location="cpu", weights_only=True)
    net = NextConceptTransformer(vqvae_local=vqvae, config=cfg.model)
    net.load_state_dict(gen_ck["model"], strict=True)
    net.eval()
    bpe = None
    if cfg.tokenizer.pre_tokenizer.name == "bpe":
        from tokenization.bpe_tokenizer import BPETokenizer

        bpe = BPETokenizer(_plain_pickle(src / "bpe_vocab_8192.pickle"),
                           _plain_pickle(src / "bpe_merges_8192.pickle"),
                           dict(cfg.tokenizer.pre_tokenizer.special_tokens))
    return cfg, vqvae, net, bpe, gen_ck.get("inference", {})


def upstream_generate(net, level0, prefix, forced=None):
    """Upstream ``generate`` at temperature 0; ``forced`` replaces each sampled level and logits are recorded."""
    import torch

    first = 0 if net.query_level0 else 1
    seen = []
    original = net._sample_level

    def hook(logits, temperature, top_k, top_p):
        seen.append(logits.detach().clone())
        if forced is not None:
            return forced[len(seen) - 1 + first]
        return original(logits, temperature, top_k, top_p)

    net._sample_level = hook
    try:
        with torch.no_grad():
            out = net.generate(level0, prefix, temperature=0.0, top_k=0, top_p=1.0)
    finally:
        del net._sample_level
    return out, seen


def split_levels(flat, levels):
    out, o = [], 0
    for n in levels:
        out.append(flat[:, o : o + n])
        o += n
    return out


@contextlib.contextmanager
def exact_float64_ops():
    """Swap MLX ops that compute float64 inputs at float32 precision on CPU for exact versions.

    MLX's CPU ``conv1d``, fused layer/RMS norm, fused attention, ``exp`` and
    ``erf`` round float64 work to roughly float32 accuracy; matmul, sums and
    ``rsqrt`` are exact. Inside this context float64 inputs use composed exact
    ops; ``exp`` and ``erf`` are evaluated on the host (NumPy, ``torch.special.erf``). Float32 inputs are
    unchanged, so the float32 comparison above is unaffected.
    """
    import mlx.core as mx
    import torch
    from mlx import nn

    def f64(x):
        return x.dtype == mx.float64

    def host(fn, x):
        return mx.array(fn(np.array(x)), dtype=mx.float64)

    orig = (mx.fast.layer_norm, mx.fast.rms_norm, mx.fast.scaled_dot_product_attention, mx.conv1d, nn.gelu,
            mx.exp)

    def exp(x):
        return host(np.exp, x) if f64(x) else orig[5](x)

    def layer_norm(x, weight, bias, eps):
        if not f64(x):
            return orig[0](x, weight, bias, eps)
        mu = x.mean(-1, keepdims=True)
        y = (x - mu) * mx.rsqrt(((x - mu) ** 2).mean(-1, keepdims=True) + eps)
        y = y * weight if weight is not None else y
        return y + bias if bias is not None else y

    def rms_norm(x, weight, eps):
        if not f64(x):
            return orig[1](x, weight, eps)
        y = x * mx.rsqrt((x * x).mean(-1, keepdims=True) + eps)
        return y * weight if weight is not None else y

    def sdpa(q, k, v, scale, mask=None, **kw):
        if not f64(q):
            return orig[2](q, k, v, scale=scale, mask=mask, **kw)
        s = (q @ k.swapaxes(-1, -2)) * scale
        if mask is not None:
            s = mx.where(mask, s, -mx.inf) if mask.dtype == mx.bool_ else s + mask
        s = s - s.max(-1, keepdims=True)
        e = host(np.exp, s)
        return (e / e.sum(-1, keepdims=True)) @ v

    def conv1d(x, w, stride=1, padding=0, **kw):
        if not f64(x):
            return orig[3](x, w, stride=stride, padding=padding, **kw)
        K, L = w.shape[1], x.shape[1]
        xp = mx.pad(x, [(0, 0), (padding, padding), (0, 0)])
        return sum(xp[:, j : j + L + 2 * padding - K + 1] @ w[:, j, :].T for j in range(K))

    def gelu(x):
        if not f64(x):
            return orig[4](x)
        return x * 0.5 * (1 + host(lambda a: torch.special.erf(torch.from_numpy(a) / math.sqrt(2)).numpy(), x))

    mx.fast.layer_norm, mx.fast.rms_norm, mx.fast.scaled_dot_product_attention = layer_norm, rms_norm, sdpa
    mx.conv1d, nn.gelu, mx.exp = conv1d, gelu, exp
    try:
        yield
    finally:
        (mx.fast.layer_norm, mx.fast.rms_norm, mx.fast.scaled_dot_product_attention, mx.conv1d, nn.gelu,
         mx.exp) = orig


def rope64(positions, head_dim, base):
    half = head_dim // 2
    inv = 1.0 / (base ** (np.arange(half, dtype=np.float64) / half))
    f = positions.astype(np.float64)[:, None] * inv[None]
    return np.concatenate([np.cos(f)] * 2, -1), np.concatenate([np.sin(f)] * 2, -1)


def max_abs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def run(args) -> dict:
    import mlx.core as mx
    import torch

    from ncm_mlx.official import sample, spec

    mx.set_default_device(mx.cpu)
    torch.manual_seed(0)
    task = spec.get(args.task)
    src = Path(args.src or ROOT / "artifacts/official" / task.remote_dir)
    _, vqvae, net, up_bpe, inference = build_upstream(Path(args.upstream), src)
    m = sample.load(src / "mlx")
    levels = list(task.codec.levels)
    L = task.codec.seq_len
    report = {"task": task.name, "torch": torch.__version__, "mlx": mx.__version__,
              "device": "cpu", "dtype": "float32", "upstream_code_revision": spec.UPSTREAM_CODE_REVISION,
              "checkpoint_revision": spec.REVISION}
    fixture = {"task": task.name}

    if task.tokenizer == "bpe":
        import regex

        from ncm_mlx.official.bpe import pre_tokenize

        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        cases = [c for c in PRETOKENIZE_CASES if "<|" not in c]
        pre_ok = all(pre_tokenize(c) == regex.findall(pattern, c) for c in cases)
        ids_ok = all(m.bpe.encode(c) == up_bpe.encode(c) for c in PRETOKENIZE_CASES)
        dec_ok = all(m.bpe.decode(m.bpe.encode(c)) == up_bpe.decode(up_bpe.encode(c)) for c in PRETOKENIZE_CASES)
        report["bpe"] = {"cases": len(PRETOKENIZE_CASES), "pretokenize_equal": pre_ok, "ids_equal": ids_ok,
                         "roundtrip_equal": dec_ok}
        fixture["pretokenize"] = [[c, regex.findall(pattern, c)] for c in cases]
        fixture["bpe_ids"] = [[c, up_bpe.encode(c)] for c in PRETOKENIZE_CASES]
        rows = [(up_bpe.encode(t) + [task.pad_id] * L)[:L] for t in TEXTS]
        prefix_np = np.full((len(rows), 1), task.eos_id, dtype=np.int64)
    else:
        from ncm_mlx.text import upstream_sudoku

        boards = Path(args.boards) / task.name.split("-")[1]
        puzzles = upstream_sudoku.load_tokens(boards / "puzzles.pickle")[: args.boards_n]
        solutions = upstream_sudoku.load_tokens(boards / "solutions.pickle")[: args.boards_n]
        rows = solutions
        prefix_np = np.array(puzzles, dtype=np.int64)
        fixture["boards"] = {"puzzles": puzzles[:4], "solutions": solutions[:4]}

    x_t = torch.tensor(rows, dtype=torch.long)
    x_m = mx.array(np.array(rows, dtype=np.int32))
    prefix_t = torch.tensor(prefix_np)
    prefix_m = mx.array(prefix_np.astype(np.int32))

    with torch.no_grad():
        z_t = vqvae._encode_pre_quant(x_t)
        idx_t = vqvae.vq.z_to_idx(z_t)
        rec_t = vqvae.decode(idx_t)
    z_m = m.codec.embed(x_m)
    codes_m = m.codec.encode(x_m)
    up_codes = split_levels(idx_t.numpy(), levels)
    agree = [float((np.array(c) == u).mean()) for c, u in zip(codes_m, up_codes)]
    forced_m = [mx.array(c.astype(np.int32)) for c in up_codes]
    rec_m = m.codec.decode_logits(m.codec.cumulative(forced_m)[-1])
    first_gaps = []
    with torch.no_grad():
        rest = z_t.clone()
        for i, n in enumerate(levels):
            flat = vqvae.vq._down_for_quant(rest, i, n) if i != len(levels) - 1 else rest
            d = vqvae.vq._distances(flat.reshape(-1, flat.shape[-1]), vqvae.vq.embeddings[i].weight)
            top2 = torch.topk(-d, 2, dim=1).values
            gap = (top2[:, 0] - top2[:, 1]).view(len(rows), n).numpy()
            for b in range(len(rows)):
                diff = np.array(codes_m[i][b]) != up_codes[i][b]
                earlier = any((np.array(codes_m[j][b]) != up_codes[j][b]).any() for j in range(i))
                if diff.any() and not earlier:
                    first_gaps.append({"item": b, "level": i, "distance_gap": float(gap[b][diff].max()),
                                       "distance": float(-top2.view(len(rows), n, 2)[b, :, 0].max())})
            q = vqvae.vq.embeddings[i](torch.tensor(up_codes[i]))
            q = vqvae.vq._up_to_full(q, i, levels[-1]) if i != len(levels) - 1 else q
            rest = rest - vqvae.vq._apply_quant_resi(q, i)
    report["codec"] = {
        "items": len(rows), "embed_max_abs_diff": max_abs(z_m, z_t.numpy()),
        "code_agreement_per_level": agree,
        "first_divergence_per_item": first_gaps,
        "decoder_logits_max_abs_diff": max_abs(rec_m, rec_t.numpy()),
        "decoder_argmax_equal": bool(np.array_equal(np.array(mx.argmax(rec_m, -1)), rec_t.argmax(-1).numpy())),
    }
    fixture["codec"] = {"tokens": rows[:2], "codes": [c[:2].tolist() for c in up_codes],
                        "decoder_logits_row0": np.round(rec_t[0, :4, :16].numpy(), 6).tolist()}

    forced_t = [torch.tensor(c) for c in up_codes]
    _, logits_t = upstream_generate(net, forced_t[0] if not task.generator.query_level0 else None,
                                    prefix_t, forced=forced_t)
    g = sample.generate(m, prefix_m, temperature=0.0, keep_logits=True, forced=forced_m)
    per_level = []
    for lt, lm in zip(logits_t, g.level_logits):
        lt = lt.numpy()
        lm = np.array(lm)
        per_level.append({"max_abs_diff": max_abs(lm, lt), "max_abs_logit": float(np.abs(lt).max()),
                          "argmax_equal": float((lm.argmax(-1) == lt.argmax(-1)).mean())})
    report["forced_level_logits"] = per_level
    report["forced_level_logits_max_abs_diff"] = max(r["max_abs_diff"] for r in per_level)

    # fp64: the same forced pass with both implementations in float64 separates
    # semantic differences from float32 rounding. Upstream computes RMSNorm, tied-head
    # norms and RoPE tables in float32 even for float64 inputs; for this run only, they
    # are recomputed in float64 on both sides (a numerics-only change).
    import models.common as up_common
    import models.vae as up_vae

    def rms64(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight

    saved = (up_common.RMSNorm.forward, up_vae.RMSNorm.forward)
    up_common.RMSNorm.forward = up_vae.RMSNorm.forward = rms64
    torch.set_default_dtype(torch.float64)
    vq64 = copy.deepcopy(vqvae).double()
    net64 = copy.deepcopy(net).double()
    net64.vqvae_local = (vq64,)
    hd = task.generator.dim // task.generator.n_head
    target_levels = levels[0 if task.generator.query_level0 else 1:]
    for j, n in enumerate(target_levels):
        pos = np.concatenate([np.arange(task.generator.prefix_len), np.arange(n)])
        cos, sin = rope64(pos, hd, task.generator.rope_base)
        setattr(net64, f"rope_cos_gen_{j}", torch.tensor(cos))
        setattr(net64, f"rope_sin_gen_{j}", torch.tensor(sin))
    for head in net64.lm_heads:
        if hasattr(head, "E_sq"):
            head.E_sq = (head.E ** 2).sum(-1)
            head.d_scale = head.E_sq.mean().clamp(min=1e-12)
    _, logits64_t = upstream_generate(net64, forced_t[0] if not task.generator.query_level0 else None,
                                      prefix_t, forced=forced_t)
    torch.set_default_dtype(torch.float32)
    up_common.RMSNorm.forward, up_vae.RMSNorm.forward = saved
    m.codec.astype(mx.float64)
    m.gen.astype(mx.float64)
    with exact_float64_ops():
        g64 = sample.generate(m, prefix_m, temperature=0.0, keep_logits=True, forced=forced_m)
    m.codec.astype(mx.float32)
    m.gen.astype(mx.float32)
    report["fp64_forced_level_logits_max_abs_diff"] = max(
        max_abs(a, b.numpy()) for a, b in zip(g64.level_logits, logits64_t))
    report["fp32_rounding_reference"] = {
        "torch_fp32_vs_torch_fp64": max(max_abs(a.numpy(), b.numpy()) for a, b in zip(logits_t, logits64_t)),
        "mlx_fp32_vs_torch_fp64": max(max_abs(a, b.numpy()) for a, b in zip(g.level_logits, logits64_t)),
    }
    fixture["forced_logits"] = [{"row0_head": np.round(lt[0, :2, :16].numpy(), 6).tolist(),
                                 "row0_argmax": lt[0].argmax(-1).tolist(),
                                 "row0_sum": float(lt[0].double().sum()),
                                 "row0_abs_sum": float(lt[0].double().abs().sum())} for lt in logits_t]

    if task.generator.query_level0:
        l0_t, l0_m = None, None
        gp_t, gp_m = prefix_t, prefix_m
    else:
        p = inference["level0_unigram"].numpy()
        rng = np.random.default_rng(0)
        l0 = np.concatenate([np.argsort(-p)[:4], rng.choice(len(p), 4, p=p)])[:, None]
        l0_t, l0_m = torch.tensor(l0), mx.array(l0.astype(np.int32))
        gp_t = torch.full((len(l0), 1), task.eos_id, dtype=torch.long)
        gp_m = mx.array(gp_t.numpy().astype(np.int32))
    out_t, _ = upstream_generate(net, l0_t, gp_t)
    with torch.no_grad():
        _, zl = vqvae.vq.idx_to_zhats(out_t, return_zlevels=True)
        tok_t = vqvae.decoder(zl[-1]).argmax(-1).numpy()
    g = sample.generate(m, gp_m, temperature=0.0, level0=l0_m)
    codes_greedy_m = np.concatenate([np.array(c) for c in g.codes], axis=1)
    texts_t = [m.decode(r) for r in tok_t]
    texts_m = [m.decode(r) for r in np.array(g.tokens)]
    report["greedy"] = {"samples": len(texts_t),
                        "codes_equal_per_level": [bool(np.array_equal(a, b)) for a, b in
                                                  zip(split_levels(codes_greedy_m, levels),
                                                      split_levels(out_t.numpy(), levels))],
                        "text_equal": [a == b for a, b in zip(texts_m, texts_t)]}
    if task.tokenizer == "sudoku":
        sols = ["".join(sample.SUDOKU_CHARS[t] for t in s) for s in rows]
        report["greedy"]["solved_upstream"] = sum(a == b for a, b in zip(texts_t, sols))
        report["greedy"]["solved_mlx"] = sum(a == b for a, b in zip(texts_m, sols))
    report["greedy_examples"] = texts_t[:2]
    k = 4
    fixture["greedy"] = {"level0": None if l0_t is None else l0[:k].tolist(),
                         "prefix": gp_t[:k].tolist(),
                         "codes": out_t[:k].tolist(), "texts": texts_t[:k]}
    if args.fixture:
        path = Path(args.fixture)
        data = json.loads(path.read_text()) if path.exists() else {}
        data[task.name] = fixture
        data["_meta"] = {k_: report[k_] for k_ in ("torch", "upstream_code_revision", "checkpoint_revision")}
        path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", required=True, help="checkout of HazyResearch/ncm")
    ap.add_argument("--task", default="tinystories")
    ap.add_argument("--src", help="release files (default artifacts/official/<task dir>)")
    ap.add_argument("--boards", default=str(ROOT / "artifacts/upstream-sudoku"))
    ap.add_argument("--boards-n", type=int, default=32)
    ap.add_argument("--fixture", help="update this JSON fixture with upstream reference values")
    ap.add_argument("--out", help="write the report JSON here")
    args = ap.parse_args()
    report = run(args)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
