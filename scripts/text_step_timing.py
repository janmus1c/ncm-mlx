"""Time compiled text-training steps (generator or AR) at one precision on the default device.

Uses the training code's own step functions with random token windows, a randomly
initialised frozen codec for the generator stage, and no data files, so it runs
on any machine that has MLX. Prints one JSON object: median/min/max step seconds,
tokens per second (window tokens, ``batch_size * seq_len``), peak memory and the
loss of the first and last timed step.

    python scripts/text_step_timing.py configs/tinystories-e24-seed1.json --stage generator --precision bfloat16

Memory is capped by NCM_MLX_MEMORY_LIMIT_GB (default 10 GB on macOS, otherwise 90% of
the device); a run that does not fit exits with an error. Do not run it next to a
training job: both would be timed under contention.
"""

import argparse
import json
import os
import statistics
import sys
import time

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from ncm_mlx.official import ar as ar_mod
from ncm_mlx.official import trainer
from ncm_mlx.official.model import Codec
from ncm_mlx.official.train_config import PRECISIONS, TextConfig

VOCAB = 8192


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", help="text training config JSON")
    ap.add_argument("--stage", choices=["generator", "ar"], default="generator")
    ap.add_argument("--precision", choices=PRECISIONS, default="float32")
    ap.add_argument("--steps", type=int, default=20, help="timed steps after the warm-up (default 20)")
    ap.add_argument("--warmup", type=int, default=5, help="untimed steps, including compilation (default 5)")
    ap.add_argument("--batch-size", type=int, default=0, help="override the stage batch size")
    ap.add_argument("--no-compile", action="store_true", help="time the uncompiled step")
    args = ap.parse_args()

    default_gb = 10.0 if sys.platform == "darwin" else 0.9 * mx.device_info().get("memory_size", 0) / 2**30
    limit = float(os.environ.get("NCM_MLX_MEMORY_LIMIT_GB", default_gb))
    if limit > 0:
        mx.set_memory_limit(int(limit * 2**30))

    cfg = TextConfig.load(args.config)
    cfg.train.precision = args.precision
    cfg.train.compile = not args.no_compile
    cfg.validate()
    st = getattr(cfg.train, args.stage)
    batch = args.batch_size or st.batch_size
    seq_len = cfg.data.seq_len
    seed = cfg.train.seed

    if args.stage == "generator":
        codec = Codec(cfg.codec_spec())
        trainer._init_codec(codec, seed)
        codec.eval()
        codec.freeze()
        model = trainer.build_generator(cfg, codec, seed + 1)
        model.train()
        tf = trainer.TeacherForced(model, trainer.compute_dtype(cfg))
        rates = trainer.corruption_rates(cfg.generator.corrupt_first, cfg.generator.corrupt_last,
                                         len(codec.levels) - 1)
        opt = trainer.optimizer(st)
        step = trainer.generator_step(cfg, model, codec, tf, opt, rates if any(rates) else None)
        positions = model.spec.prefix_len + sum(model.target_levels)
    else:
        model = ar_mod.build_ar(cfg, seed + 2)
        model.train()
        opt = trainer.optimizer(st)
        step = ar_mod.ar_step(cfg, model, opt)
        positions = seq_len

    rng = np.random.default_rng(seed)
    batches = [mx.array(rng.integers(2, VOCAB, (batch, seq_len)).astype(np.int32))
               for _ in range(args.warmup + args.steps)]
    mx.eval(batches)
    times, losses = [], []
    for i, tokens in enumerate(batches):
        if i == args.warmup:
            mx.reset_peak_memory()
        t0 = time.perf_counter()
        out = step(tokens)
        mx.eval(model.parameters(), opt.state, out[0])
        dt = time.perf_counter() - t0
        if i >= args.warmup:
            times.append(dt)
            losses.append(out[0].item())

    median = statistics.median(times)
    print(json.dumps({
        "device": str(mx.default_device()),
        "device_info": {k: v for k, v in mx.device_info().items() if isinstance(v, (str, int, float))},
        "mlx": mx.__version__,
        "config": os.path.basename(args.config),
        "stage": args.stage,
        "precision": args.precision,
        "compiled": cfg.train.compile,
        "batch_size": batch,
        "seq_len": seq_len,
        "positions_per_sequence": positions,
        "parameters": sum(v.size for _, v in tree_flatten(model.parameters())),
        "timed_steps": len(times),
        "step_seconds_median": round(median, 5),
        "step_seconds_min": round(min(times), 5),
        "step_seconds_max": round(max(times), 5),
        "tokens_per_s": round(batch * seq_len / median, 1),
        "peak_memory_gb": round(mx.get_peak_memory() / 2**30, 3),
        "loss_first": round(losses[0], 4),
        "loss_last": round(losses[-1], 4),
        "loss_finite": bool(np.isfinite(losses).all()),
    }, indent=2))


if __name__ == "__main__":
    main()
