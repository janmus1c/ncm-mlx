"""Stage 1 training: the codec learns to reconstruct text through the code hierarchy."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn

from ncm_mlx.config import Config
from ncm_mlx.eval.metrics import codec_report
from ncm_mlx.training.checkpoint import (
    StopRequest,
    TrainingPaused,
    read_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
from ncm_mlx.training.common import (
    Timer,
    batches,
    build_codec,
    load_corpus,
    save_model,
    schedule,
    seed_all,
)


def train_codec(cfg: Config, steps: int | None = None, log=print, *, resume: bool = False):
    steps = cfg.train.codec_steps if steps is None else steps
    rng = seed_all(cfg.train.seed)
    corpus = load_corpus(cfg)
    codec = build_codec(cfg, corpus.tokenizer.vocab_size)
    codec.train()
    opt = optim.AdamW(schedule(cfg.train.codec_lr, cfg.train.warmup, steps), weight_decay=cfg.train.weight_decay)

    def loss_fn(model, tokens):
        return model.loss(tokens, rng)

    grad_fn = nn.value_and_grad(codec, loss_fn)
    data = batches(corpus.train, cfg.train.batch_size, np.random.default_rng(cfg.train.seed))
    key = mx.random.key(cfg.train.seed)
    history = []
    start_step, elapsed_before = 0, 0.0
    training_path = Path(cfg.out_dir) / "codec-training.safetensors"
    if resume:
        state, saved = read_checkpoint(training_path, cfg, "codec", steps, log)
        restore_checkpoint(state, codec, opt)
        data.load_state_dict(saved["data"])
        key = state["vq_key"]
        python_state = saved["python_rng"]
        rng.setstate((python_state[0], tuple(python_state[1]), python_state[2]))
        start_step, elapsed_before, history = saved["step"], saved["train_seconds"], saved["history"]
        log(f"resuming full codec state at step {start_step}/{steps}")
    with Timer() as t, StopRequest() as stop:
        for step in range(start_step + 1, steps + 1):
            tokens = next(data)
            (loss, (ce, vq_loss, codes, inputs)), grads = grad_fn(codec, tokens)
            opt.update(codec, grads)
            key, sub = mx.random.split(key)
            codec.vq.ema_update(codes, inputs, sub)
            mx.eval(codec.parameters(), opt.state, loss, ce, vq_loss)
            if step % cfg.train.log_every == 0 or step == steps or stop.requested:
                row = {"step": step, "loss": loss.item(), "ce": ce.item(), "vq": vq_loss.item()}
                history.append(row)
                log(f"codec step {step:5d}  loss {row['loss']:.4f}  ce {row['ce']:.4f}  vq {row['vq']:.4f}  "
                    f"peak {mx.get_peak_memory() / 2**30:.2f} GB")
                save_checkpoint(training_path, codec, opt, {
                    "stage": "codec", "config": cfg.to_dict(), "step": step, "total_steps": steps,
                    "data": data.state_dict(), "python_rng": rng.getstate(), "history": history,
                    "train_seconds": elapsed_before + t.elapsed(),
                }, vq_key=key)
                if stop.requested:
                    log(f"paused codec at step {step}; full training state saved to {training_path}")
                    raise TrainingPaused(step)
    codec.eval()
    report = codec_report(codec, corpus.val)
    meta = {
        "stage": "codec",
        "steps": steps,
        "train_seconds": round(elapsed_before + t.seconds, 2),
        "tokenizer": corpus.tokenizer.to_dict(),
        "history": history,
        "val": report,
    }
    save_model(codec, Path(cfg.out_dir) / "codec.safetensors", meta)
    return codec, corpus, meta
