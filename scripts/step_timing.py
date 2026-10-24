"""Median training-step time of the Sudoku generator at a few widths, on the default device.

Cases come from the command line as WIDTHxBATCH (default: 256x64 256x128 512x128).
Memory is capped (NCM_MLX_MEMORY_LIMIT_GB; default 10 GB on macOS, where memory is shared with
the desktop, otherwise 85% of the GPU); a case that does not fit is skipped.
Do not run it on a machine that is already training.
"""

import os
import statistics
import sys
import time
from functools import partial

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn

from ncm_mlx.config import Config
from ncm_mlx.training.common import build_codec, build_generator
from ncm_mlx.training.generator import corruption_schedule, generator_loss

default_gb = 10.0 if sys.platform == "darwin" else 0.85 * mx.device_info().get("memory_size", 0) / 2**30
limit = float(os.environ.get("NCM_MLX_MEMORY_LIMIT_GB", default_gb))
mx.set_memory_limit(int(limit * 2**30))
mx.set_cache_limit(int(min(limit / 4, 2) * 2**30))
cfg = Config.load("configs/sudoku-e8.json")
codec = build_codec(cfg, 10)
codec.eval()
codec.freeze()
n_codes = sum(cfg.codec.levels)
rng = np.random.default_rng(0)
probs = corruption_schedule(0.3, 0.05, len(cfg.codec.levels))
print(f"device {mx.default_device()}")


def time_case(width: int, batch: int) -> None:
    cfg.generator.n_embd = width
    gen = build_generator(cfg)
    gen.train()
    opt = optim.AdamW(3e-4)
    grad_fn = nn.value_and_grad(gen, lambda m, f: generator_loss(m, codec, f, probs)[0])

    def step(f):
        loss, g = grad_fn(gen, f)
        opt.update(gen, g)
        return loss

    state = [gen.state, opt.state, mx.random.state]
    step = partial(mx.compile, inputs=state, outputs=state)(step)
    codes = rng.integers(0, 16, (batch, n_codes))
    prefix = rng.integers(0, 10, (batch, 81))
    flat = mx.array(np.concatenate([codes, prefix], axis=1).astype(np.int32))
    times = []
    for i in range(13):
        t = time.perf_counter()
        mx.eval(step(flat), gen.parameters(), opt.state)
        if i >= 3:
            times.append(time.perf_counter() - t)
    dt = statistics.median(times)
    print(f"width {width} batch {batch}: {dt * 1000:.0f} ms/step, {batch / dt:.0f} sequences/s, "
          f"peak {mx.get_peak_memory() / 2**30:.1f} GB")
    mx.reset_peak_memory()


cases = sys.argv[1:] or ["256x64", "256x128", "512x128"]
for width, batch in (tuple(int(v) for v in c.split("x")) for c in cases):
    try:
        time_case(width, batch)
    except RuntimeError as e:
        print(f"width {width} batch {batch}: skipped ({e})")
