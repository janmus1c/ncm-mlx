"""Measure steady-state training throughput of candidate TinyStories configurations.

Usage: ``python scripts/text_train_probe.py codec|gen|ar``. Each probe compiles
one training step exactly as the trainer does, runs 3 warmup and 12 timed steps
on cached training rows and prints sequences per second and milliseconds per
step (synchronized with ``mx.eval``). Requires ``official prepare-data``.
"""

from __future__ import annotations

import sys
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn

from ncm_mlx.official import ar as armod
from ncm_mlx.official import tinystories as ts
from ncm_mlx.official import trainer
from ncm_mlx.official.train_config import TextConfig

ROWS = np.asarray(ts.load_split(Path("artifacts/tinystories/v2-bpe8192-L128"), "train")[:20000])


def timeit(step, B, n=12):
    for i in range(3):
        mx.eval(step(trainer.to_batch(ROWS[i * B : (i + 1) * B])))
    t = time.perf_counter()
    for i in range(n):
        mx.eval(step(trainer.to_batch(ROWS[(i % 10) * B : (i % 10 + 1) * B])))
    dt = (time.perf_counter() - t) / n
    return {"seq_per_s": round(B / dt, 1), "ms_per_step": round(dt * 1000, 1), "batch": B,
            "peak_gb": round(mx.get_peak_memory() / 2**30, 2)}


def _compile(model, opt, fn):
    state = [model.state, opt.state, mx.random.state]
    return partial(mx.compile, inputs=state, outputs=state)(fn)


def codec_probe(dim, V, B):
    cfg = TextConfig.from_dict({"codec": {"dim": dim, "n_head": dim // 64, "codebook_sizes": [V] * 12,
                                          "restart_threshold": 0.01}})
    tr = trainer.CodecTrainer(cfg, 0)
    tr.train()
    opt = trainer.optimizer(cfg.train.codec)
    g = nn.value_and_grad(tr, lambda m, t: m.loss(t))

    def step(t):
        (loss, (_, _, codes, inputs)), grads = g(tr, t)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(tr, grads)
        return loss, codes, inputs

    cs = _compile(tr, opt, step)

    def full(t):
        loss, codes, inputs = cs(t)
        tr.ema_update(codes, inputs)
        return [loss, *tr.codec.codebooks]

    return trainer.parameter_count(tr.codec), timeit(full, B)


def gen_probe(dim, L, B, cdim=256):
    cfg = TextConfig.from_dict({"codec": {"dim": cdim, "n_head": 4},
                                "generator": {"dim": dim, "n_head": dim // 64, "n_layer": L}})
    codec = trainer.CodecTrainer(cfg, 0).codec
    codec.eval()
    codec.freeze()
    gen = trainer.build_generator(cfg, codec, 1)
    gen.train()
    tf = trainer.TeacherForced(gen)
    opt = trainer.optimizer(cfg.train.generator)
    rates = trainer.corruption_rates(0.75, 0.25, 11)
    g = nn.value_and_grad(gen, lambda m, t: trainer.generator_loss(tf, codec, t, rates, 0.0, 0.1))

    def step(t):
        (loss, _), grads = g(gen, t)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(gen, grads)
        return loss

    return trainer.parameter_count(gen), timeit(_compile(gen, opt, step), B)


def ar_probe(dim, L, B):
    cfg = TextConfig.from_dict({"generator": {"dim": dim, "n_head": dim // 64, "n_layer": L}})
    m = armod.build_ar(cfg, 0)
    m.train()
    opt = trainer.optimizer(cfg.train.ar)
    g = nn.value_and_grad(m, lambda mm, t: armod.ar_loss(mm, t, 0.1))

    def step(t):
        loss, grads = g(m, t)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(m, grads)
        return loss

    return trainer.parameter_count(m), timeit(_compile(m, opt, step), B)


if __name__ == "__main__":
    mx.set_memory_limit(24 * 2**30)
    which = sys.argv[1]
    shapes = [(int(a), int(b)) for a, b in (s.split("x") for s in sys.argv[2:])]
    if which == "codec":
        for dim, V in shapes or [(256, 2048), (384, 4096)]:
            print("codec", dim, V, codec_probe(dim, V, 256), flush=True)
    elif which == "gen":
        for dim, L in shapes or [(384, 6), (512, 8)]:
            print("gen", dim, L, gen_probe(dim, L, 128), flush=True)
    else:
        for dim, L in shapes or [(384, 6), (512, 8)]:
            print("ar", dim, L, ar_probe(dim, L, 128), flush=True)
