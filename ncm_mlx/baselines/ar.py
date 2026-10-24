"""Character-level autoregressive baseline on the same corpus and hardware.

Same block, width and depth as the generator; it predicts one character per
forward with a KV cache, so sampling a sequence of length L takes L forwards.
It exists to compare quality and latency against the coarse-to-fine model on
identical data, not as a tuned competitor.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn

from ncm_mlx.config import Config
from ncm_mlx.models.blocks import Block, apply_rope, rope_tables
from ncm_mlx.training.common import Timer, batches, save_model, schedule, seed_all


class CharAR(nn.Module):
    """Optionally conditioned on ``prefix_len`` prompt tokens, which attend to each other bidirectionally."""

    def __init__(self, vocab_size: int, seq_len: int, n_embd: int, n_layer: int, n_head: int, dropout: float,
                 prefix_len: int = 0, prefix_vocab: int = 0):
        super().__init__()
        self.seq_len = seq_len
        self.prefix_len = prefix_len if prefix_vocab else 0
        self.bos = vocab_size
        self.wte = nn.Embedding(vocab_size + 1, n_embd)
        if self.prefix_len:
            self.prefix_emb = nn.Embedding(prefix_vocab, n_embd)
        self.blocks = [Block(n_embd, n_head, dropout) for _ in range(n_layer)]
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        P, S = self.prefix_len, self.prefix_len + seq_len
        self._rope = rope_tables(np.arange(S), n_embd // n_head, 10000.0)
        mask = np.tril(np.ones((S, S), dtype=bool))
        mask[:P, :P] = True
        self._mask = mx.array(mask)

    def _embed(self, inp: mx.array, prefix: mx.array | None) -> mx.array:
        h = self.wte(inp)
        if self.prefix_len:
            h = mx.concatenate([self.prefix_emb(prefix), h], axis=1)
        return h

    def __call__(self, tokens: mx.array, prefix: mx.array | None = None) -> mx.array:
        """Next-token logits for ``[prefix] + [BOS] + tokens[:, :-1]`` at the target positions."""
        B = tokens.shape[0]
        inp = mx.concatenate([mx.full((B, 1), self.bos, dtype=tokens.dtype), tokens[:, :-1]], axis=1)
        h = self._embed(inp, prefix)
        for blk in self.blocks:
            h = blk(h, self._mask, self._rope)
        return self.head(self.ln_f(h[:, self.prefix_len :]))

    def _step(self, h: mx.array, pos: int, cache: list, mask: mx.array | None = None):
        """Run embeddings ``h`` at positions ``pos..pos+T-1`` through the stack, extending the KV cache."""
        T = h.shape[1]
        cos, sin = self._rope[0][pos : pos + T], self._rope[1][pos : pos + T]
        for i, blk in enumerate(self.blocks):
            a = blk.attn
            x = blk.ln1(h)
            B = x.shape[0]
            q, k, v = mx.split(a.qkv(x), 3, axis=-1)
            q = a.q_norm(q.reshape(B, T, a.n_head, a.head_dim).transpose(0, 2, 1, 3))
            k = a.k_norm(k.reshape(B, T, a.n_head, a.head_dim).transpose(0, 2, 1, 3))
            v = v.reshape(B, T, a.n_head, a.head_dim).transpose(0, 2, 1, 3)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            if cache[i] is not None:
                k = mx.concatenate([cache[i][0], k], axis=2)
                v = mx.concatenate([cache[i][1], v], axis=2)
            cache[i] = (k, v)
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=a.head_dim**-0.5, mask=mask)
            h = h + a.proj(y.transpose(0, 2, 1, 3).reshape(B, T, -1))
            h = h + blk.out(nn.gelu(blk.fc(blk.ln2(h))))
        return self.head(self.ln_f(h))[:, -1]

    def generate(self, batch: int, temperature: float = 1.0, seed: int = 0, prefix: mx.array | None = None) -> mx.array:
        key = mx.random.key(seed)
        cache = [None] * len(self.blocks)
        P = self.prefix_len
        tok = mx.full((batch, 1), self.bos, dtype=mx.int32)
        if P:
            first = self._embed(tok, prefix)
            logits = self._step(first, 0, cache, self._mask[: P + 1, : P + 1])
        out = []
        for pos in range(self.seq_len):
            if pos or not P:
                logits = self._step(self.wte(tok), P + pos, cache)
            key, sub = mx.random.split(key)
            if temperature == 0:
                nxt = mx.argmax(logits, axis=-1)
            else:
                nxt = mx.random.categorical(logits / temperature, key=sub)
            tok = nxt.astype(mx.int32)[:, None]
            out.append(tok)
        tokens = mx.concatenate(out, axis=1)
        mx.eval(tokens)
        return tokens


def build_ar(cfg: Config, vocab_size: int) -> CharAR:
    g = cfg.generator
    return CharAR(vocab_size, cfg.data.seq_len, g.n_embd, g.n_layer, g.n_head, g.dropout,
                  g.prefix_len, g.prefix_vocab)


def train_ar(cfg: Config, corpus, steps: int | None = None, log=print):
    steps = cfg.train.generator_steps if steps is None else steps
    seed_all(cfg.train.seed + 2)
    model = build_ar(cfg, corpus.tokenizer.vocab_size)
    model.train()
    opt = optim.AdamW(schedule(cfg.train.generator_lr, cfg.train.warmup, steps),
                      weight_decay=cfg.train.weight_decay)

    n = cfg.data.seq_len
    train = corpus.train if corpus.train_prefix is None else np.concatenate([corpus.train, corpus.train_prefix], 1)
    val = corpus.val if corpus.val_prefix is None else np.concatenate([corpus.val, corpus.val_prefix], 1)

    def loss_fn(m, rows):
        t, prefix = rows[:, :n], (rows[:, n:] if rows.shape[1] > n else None)
        return nn.losses.cross_entropy(m(t, prefix), t, reduction="mean")

    def val_loss():
        model.eval()
        total = sum(loss_fn(model, mx.array(val[i : i + 500])).item() * len(val[i : i + 500])
                    for i in range(0, len(val), 500))
        model.train()
        return total / len(val)

    grad_fn = nn.value_and_grad(model, loss_fn)
    data = batches(train, cfg.train.batch_size, np.random.default_rng(cfg.train.seed + 2))
    history = []
    with Timer() as t:
        for step in range(1, steps + 1):
            loss, grads = grad_fn(model, next(data))
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            if step % cfg.train.log_every == 0 or step == steps:
                row = {"step": step, "train_loss": round(loss.item(), 4), "val_loss": round(val_loss(), 4)}
                history.append(row)
                log(f"ar step {step:5d}  loss {row['train_loss']:.4f}  val {row['val_loss']:.4f}  "
                    f"peak {mx.get_peak_memory() / 2**30:.2f} GB")
    model.eval()
    meta = {"stage": "ar-baseline", "steps": steps, "train_seconds": round(t.seconds, 2), "history": history,
            "val_ce_per_char": round(val_loss(), 4)}
    save_model(model, Path(cfg.out_dir) / "ar.safetensors", meta)
    return model, meta
