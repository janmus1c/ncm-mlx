"""Autoregressive baseline on the same BPE windows as the TinyStories NCM.

It uses the official generator ``Block`` (QK RMSNorm, RoPE base from the config,
LayerNorm eps 1e-6, exact GELU) at the generator's width and depth, a tied
8192-token embedding, the same single-EOS null prefix (as the first input
token) and next-token cross-entropy over all 128 window positions, padding
included, exactly the positions the NCM codec reconstructs. Sampling uses a KV
cache, one forward per token, and the same temperature/top-k/top-p semantics as
the NCM sampler.
"""

from __future__ import annotations

import json

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten

from ncm_mlx.official import tinystories as ts
from ncm_mlx.official.model import LN_EPS, Block, _rotate, rope_tables
from ncm_mlx.official.sample import sample_codes
from ncm_mlx.official.train_config import TextConfig
from ncm_mlx.official.trainer import (
    Run,
    block_forward,
    compiled,
    export_dir,
    load_rows,
    scaled_residual_init,
    to_batch,
)
from ncm_mlx.training.checkpoint import atomic_arrays
from ncm_mlx.training.common import BatchStream


class TextAR(nn.Module):
    def __init__(self, vocab: int, seq_len: int, dim: int, n_layer: int, n_head: int, rope_base: float,
                 bos_id: int = ts.EOS_ID, prefix_len: int = 0):
        super().__init__()
        self.seq_len, self.bos_id, self.n_head, self.prefix_len = seq_len, bos_id, n_head, prefix_len
        total = prefix_len + seq_len
        self.wte = nn.Embedding(vocab, dim)
        self.blocks = [Block(dim, n_head) for _ in range(n_layer)]
        self.ln_f = nn.LayerNorm(dim, eps=LN_EPS)
        self._rope = rope_tables(np.arange(total), dim // n_head, rope_base)
        self._mask = mx.array(np.tril(np.ones((total, total), dtype=bool)))

    def __call__(self, tokens: mx.array, p: float = 0.0) -> mx.array:
        """Next-token logits for ``[BOS] + tokens[:, :-1]``, aligned with ``tokens``.

        With a prefix, ``tokens`` are ``[prefix, target]`` rows and the logits are those that predict the target.
        """
        P = self.prefix_len
        if P:
            inp = tokens[:, :-1]
        else:
            inp = mx.concatenate([mx.full((tokens.shape[0], 1), self.bos_id, dtype=tokens.dtype), tokens[:, :-1]],
                                 axis=1)
        h = self.wte(inp)
        T = h.shape[1]
        rope = (self._rope[0][:T], self._rope[1][:T])
        for blk in self.blocks:
            h = block_forward(blk, h, self._mask[:T, :T], rope, p)
        return self.wte.as_linear(self.ln_f(h))[:, max(P - 1, 0):]

    def target(self, tokens: mx.array) -> mx.array:
        return tokens[:, self.prefix_len :]

    def _cached(self, tok: mx.array, pos: int, cache: list) -> mx.array:
        """Logits for one new token at ``pos``, extending the per-block key/value cache."""
        h = self.wte(tok)
        cos, sin = self._rope[0][pos : pos + 1], self._rope[1][pos : pos + 1]
        B = h.shape[0]
        for i, blk in enumerate(self.blocks):
            a = blk.attn
            q, k, v = (t.reshape(B, 1, a.n_head, a.head_dim).transpose(0, 2, 1, 3)
                       for t in mx.split(a.qkv(blk.ln1(h)), 3, axis=-1))
            q, k = _rotate(a.q_norm(q), cos, sin), _rotate(a.k_norm(k), cos, sin)
            if cache[i] is not None:
                k = mx.concatenate([cache[i][0], k], axis=2)
                v = mx.concatenate([cache[i][1], v], axis=2)
            cache[i] = (k, v)
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=a.head_dim**-0.5)
            h = h + a.proj(y.transpose(0, 2, 1, 3).reshape(B, 1, -1))
            h = h + blk.out(nn.gelu(blk.fc(blk.ln2(h))))
        return self.wte.as_linear(self.ln_f(h))[:, -1]

    def generate(self, batch: int, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
                 seed: int = 0, keep_logits: bool = False, prefix: mx.array | None = None):
        """Sample ``seq_len`` tokens; with ``prefix_len > 0`` a ``(batch, prefix_len)`` prompt is required."""
        P = self.prefix_len
        if P and (prefix is None or prefix.shape != (batch, P)):
            raise ValueError(f"this model needs a ({batch}, {P}) prefix")
        keys = mx.random.split(mx.random.key(seed), self.seq_len)
        cache = [None] * len(self.blocks)
        out, seen = [], []
        if P:
            for i in range(P - 1):
                self._cached(prefix[:, i : i + 1].astype(mx.int32), i, cache)
            tok = prefix[:, P - 1 :].astype(mx.int32)
        else:
            tok = mx.full((batch, 1), self.bos_id, dtype=mx.int32)
        start = max(P - 1, 0)
        for pos in range(start, start + self.seq_len):
            logits = self._cached(tok, pos, cache)
            if keep_logits:
                seen.append(logits)
            tok = sample_codes(logits, temperature, top_k, top_p, keys[pos - start]).astype(mx.int32)[:, None]
            out.append(tok)
        tokens = mx.concatenate(out, axis=1)
        return (tokens, seen) if keep_logits else tokens


def build_ar(cfg: TextConfig, seed: int) -> TextAR:
    g = cfg.generator
    mx.random.seed(seed)
    m = TextAR(8192, cfg.data.seq_len, g.dim, g.n_layer, g.n_head, g.rope_base, prefix_len=cfg.data.prefix_len)
    scaled_residual_init(m.blocks, g.n_layer, seed + 7)
    return m


def ar_loss(m: TextAR, tokens: mx.array, p: float):
    return nn.losses.cross_entropy(m(tokens, p), m.target(tokens), reduction="mean")


def ar_report(m: TextAR, rows: np.ndarray, batch: int = 256) -> dict:
    """Held-out next-token CE over the window (all positions, and non-pad targets only)."""
    tot = tot_np = n = n_np = 0.0
    for i in range(0, len(rows), batch):
        t = to_batch(rows[i : i + batch])
        logits, t = m(t), m.target(t)
        ce = nn.losses.cross_entropy(logits, t, reduction="none")
        mask = t != ts.PAD_ID
        tot += ce.sum().item()
        tot_np += (ce * mask).sum().item()
        n += t.size
        n_np += mask.sum().item()
    return {"token_ce": round(tot / n, 5), "nonpad_token_ce": round(tot_np / n_np, 5)}


def train_ar(cfg: TextConfig, resume: bool = False, log=print) -> dict:
    st, g = cfg.train.ar, cfg.generator
    train_rows, val_rows, manifest = load_rows(cfg, log)
    model = build_ar(cfg, cfg.train.seed + 2)
    model.train()
    data = BatchStream(train_rows, st.batch_size, np.random.default_rng(cfg.train.seed + 2))
    run = Run(cfg, "ar", st, model, data, log)
    if resume:
        run.resume()
    log(f"ar: {sum(v.size for _, v in tree_flatten(model.parameters())):,} values; train rows {len(train_rows):,}")
    grad_fn = nn.value_and_grad(model, lambda m, t: ar_loss(m, t, g.dropout))

    def step(tokens):
        loss, grads = grad_fn(model, tokens)
        grads, norm = optim.clip_grad_norm(grads, st.grad_clip)
        run.opt.update(model, grads)
        return loss, {"grad_norm": norm}

    step = compiled(model, run.opt, step, cfg.train.compile)

    def evaluate():
        model.eval()
        r = ar_report(model, val_rows)
        model.train()
        return r

    last = run.loop(step, lambda out: None, evaluate)
    model.eval()
    out = export_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    atomic_arrays(dict(tree_flatten(model.parameters())), out / "ar.safetensors")
    report = ar_report(model, val_rows)
    meta = {"steps": last, "planned_steps": st.steps, "time_limited": run.time_limited,
            "train_seconds": round(run.elapsed_before, 2), "sequences_seen": last * st.batch_size,
            "parameters": sum(v.size for _, v in tree_flatten(model.parameters())),
            "data_manifest": manifest, "val": report, "history": run.history}
    (out / "ar.json").write_text(json.dumps({"config": cfg.to_dict(), **meta}, indent=2) + "\n")
    log(f"ar done: {json.dumps(report)}")
    return meta


def load_ar(cfg: TextConfig) -> TextAR:
    path = export_dir(cfg) / "ar.safetensors"
    if not path.exists():
        raise SystemExit(f"no trained AR baseline at {path}; run `official train-ar` first")
    g = cfg.generator
    m = TextAR(8192, cfg.data.seq_len, g.dim, g.n_layer, g.n_head, g.rope_base, prefix_len=cfg.data.prefix_len)
    m.load_weights(str(path), strict=True)
    m.eval()
    return m
