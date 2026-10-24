"""Stage 2 training: the codec is frozen, the generator learns to predict its codes level by level."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from ncm_mlx.config import Config
from ncm_mlx.latent.codec import Codec
from ncm_mlx.training.checkpoint import (
    StopRequest,
    TrainingPaused,
    atomic_arrays,
    atomic_weights,
    read_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
from ncm_mlx.training.common import (
    Corpus,
    Timer,
    batches,
    build_generator,
    save_model,
    schedule,
    seed_all,
)


def encode_dataset(codec: Codec, tokens: np.ndarray, batch: int = 512) -> np.ndarray:
    """All level codes of a token array, concatenated level by level: (N, sum(levels))."""
    out = []
    for i in range(0, len(tokens), batch):
        codes = codec.encode(mx.array(tokens[i : i + batch]))
        out.append(np.array(mx.concatenate(codes, axis=1)))
    return np.concatenate(out, axis=0)


def with_prefix(codes: np.ndarray, prefix: np.ndarray | None) -> np.ndarray:
    return codes if prefix is None else np.concatenate([codes, prefix.astype(codes.dtype)], axis=1)


def split_levels(flat: mx.array, levels: list[int]) -> list[mx.array]:
    out, offset = [], 0
    for l in levels:
        out.append(flat[:, offset : offset + l])
        offset += l
    return out


def corruption_schedule(first: float, last: float, n_levels: int) -> list[float]:
    """Replacement probability for each conditioning level 0..K-2, linear from ``first`` to ``last``."""
    n = n_levels - 1
    if n <= 1:
        return [first] * n
    return [first + (last - first) * k / (n - 1) for k in range(n)]


def corrupt(codes: list[mx.array], probs: list[float], sizes: list[int],
            replacements: list[mx.array] | None = None) -> list[mx.array]:
    """Replace each conditioning code with probability ``probs[k]``.

    Replacements are uniformly random codes of the same level, or the codes in
    ``replacements`` (for example the generator's own samples) when given.
    """
    out = []
    for k, c in enumerate(codes):
        if k < len(probs) and probs[k] > 0:
            flip = mx.random.uniform(shape=c.shape) < probs[k]
            new = replacements[k] if replacements is not None else mx.random.randint(0, sizes[k], c.shape)
            c = mx.where(flip, new.astype(c.dtype), c)
        out.append(c)
    return out


def model_samples(gen, codec: Codec, codes: list[mx.array], prefix: mx.array | None) -> list[mx.array]:
    """One sample per code from the generator, each level conditioned on the true coarser levels."""
    logits = gen([mx.stop_gradient(x) for x in codec.vq.teacher_inputs(codes)], prefix)
    return [mx.stop_gradient(mx.random.categorical(lg)) for lg in logits]


def generator_loss(gen, codec: Codec, flat_codes: mx.array, corrupt_probs: list[float] | None = None,
                   corrupt_source: str = "uniform"):
    """``flat_codes`` holds all level codes, followed by prefix tokens for a conditional generator.

    With ``corrupt_probs``, the generator reads partly corrupted coarser levels
    (upstream's conditioning corruption) while the labels stay the true codes,
    so it learns to predict a level from imperfect coarser levels, as it must
    at sampling time. ``corrupt_source="model"`` draws the replacements from the
    generator's own predictions instead of uniformly, so the corrupted inputs
    resemble the errors it actually makes (one extra forward, no gradient).
    """
    n_codes = sum(codec.vq.levels)
    prefix = flat_codes[:, n_codes:] if flat_codes.shape[1] > n_codes else None
    codes = split_levels(flat_codes[:, :n_codes], codec.vq.levels)
    seen = codes
    if corrupt_probs:
        own = model_samples(gen, codec, codes, prefix) if corrupt_source == "model" else None
        seen = corrupt(codes, corrupt_probs, codec.vq.codebook_sizes, own)
    inputs = codec.vq.teacher_inputs(seen)
    logits = gen([mx.stop_gradient(x) for x in inputs], prefix)
    per_level = [nn.losses.cross_entropy(lg, c, reduction="mean") for lg, c in zip(logits, codes)]
    weights = mx.array(codec.vq.levels, dtype=mx.float32)
    total = sum(w * l for w, l in zip(weights, per_level)) / weights.sum()
    acc = [mx.mean(mx.argmax(lg, axis=-1) == c) for lg, c in zip(logits, codes)]
    return total, (per_level, acc)


def evaluate_generator(gen, codec: Codec, codes: np.ndarray, batch: int = 512) -> dict:
    gen.eval()
    sums, n = None, 0
    for i in range(0, len(codes), batch):
        chunk = mx.array(codes[i : i + batch])
        _, (ce, acc) = generator_loss(gen, codec, chunk)
        vals = np.array([*[c.item() for c in ce], *[a.item() for a in acc]]) * chunk.shape[0]
        sums = vals if sums is None else sums + vals
        n += chunk.shape[0]
    K = len(codec.vq.levels)
    avg = sums / n
    return {"ce_per_level": [round(float(x), 4) for x in avg[:K]],
            "acc_per_level": [round(float(x), 4) for x in avg[K:]]}


def weighted_ce(report: dict, levels: list[int]) -> float:
    """Validation CE averaged over positions, i.e. weighted by level length (the training objective)."""
    return sum(l * c for l, c in zip(levels, report["ce_per_level"])) / sum(levels)


def train_generator(cfg: Config, codec: Codec, corpus: Corpus, steps: int | None = None, log=print,
                    *, resume: bool = False, checkpoint: str | None = None, start_step: int = 0):
    steps = cfg.train.generator_steps if steps is None else steps
    if not 0 <= start_step < steps or (start_step and not checkpoint) or (resume and checkpoint):
        raise ValueError("invalid generator continuation: use --resume or weights with a valid start step")
    path = Path(cfg.out_dir) / "generator.safetensors"
    training_path = Path(cfg.out_dir) / "generator-training.safetensors"
    if checkpoint and Path(checkpoint).resolve() == path.resolve():
        raise ValueError("continue weights into a separate out_dir to preserve the source checkpoint")
    seed_all(cfg.train.seed + 1)
    codec.eval()
    codec.freeze()
    train_codes = with_prefix(encode_dataset(codec, corpus.train), corpus.train_prefix)
    val_codes = with_prefix(encode_dataset(codec, corpus.val), corpus.val_prefix)
    mx.clear_cache()
    gen = build_generator(cfg)
    if checkpoint:
        gen.load_weights(checkpoint)
    gen.train()
    lr_schedule = schedule(cfg.train.generator_lr, cfg.train.warmup, steps)
    opt = optim.AdamW(
        lambda step: lr_schedule(step + start_step), weight_decay=cfg.train.weight_decay
    )
    data = batches(train_codes, cfg.train.batch_size, np.random.default_rng(cfg.train.seed + 1))
    history, elapsed_before, continuation = [], 0.0, None
    best = (float("inf"), 0)
    best_weights = None
    loop_end = steps
    if checkpoint:
        initial = evaluate_generator(gen, codec, val_codes)
        gen.train()
        best = (weighted_ce(initial, codec.vq.levels), start_step)
        atomic_weights(gen, path)
        best_weights = mx.load(str(path))
        data.advance(start_step)
        continuation = {"checkpoint": checkpoint, "start_step": start_step,
                        "optimizer_reset": True, "random_state_reset": True,
                        "initial_val": initial}
        log(f"continuing weights from {checkpoint} at step {start_step}; "
            "optimizer/random state reset; original learning-rate and batch positions retained")
    schedule_offset = start_step
    if resume:
        state, saved = read_checkpoint(training_path, cfg, "generator", steps, log)
        data.load_state_dict(saved["data"])
        start_step = saved["step"]
        schedule_offset = saved["schedule_offset"]
        opt = optim.AdamW(lambda step: lr_schedule(step + schedule_offset),
                          weight_decay=cfg.train.weight_decay)
        restore_checkpoint(state, gen, opt)
        history, elapsed_before = saved["history"], saved["train_seconds"]
        best = tuple(saved["best"])
        best_weights = dict(tree_flatten(state["best_model"]))
        atomic_arrays(best_weights, path)
        continuation = saved.get("continuation")
        if saved.get("time_limit_reached"):
            loop_end = steps = start_step
        log(f"resuming full generator state at step {start_step}/{steps}")

    probs = corruption_schedule(cfg.generator.corrupt_first, cfg.generator.corrupt_last, len(codec.vq.levels))
    probs = probs if any(probs) else None

    def loss_fn(model, flat):
        return generator_loss(model, codec, flat, probs, cfg.generator.corrupt_source)

    grad_fn = nn.value_and_grad(gen, loss_fn)

    def train_step(flat):
        (loss, _), grads = grad_fn(gen, flat)
        opt.update(gen, grads)
        return loss

    if cfg.train.compile:
        state = [gen.state, opt.state, mx.random.state]
        train_step = partial(mx.compile, inputs=state, outputs=state)(train_step)
    with Timer() as t, StopRequest() as stop:
        for step in range(start_step + 1, loop_end + 1):
            loss = train_step(next(data))
            mx.eval(gen.parameters(), opt.state, loss)
            out_of_time = cfg.train.max_minutes and elapsed_before + t.elapsed() > 60 * cfg.train.max_minutes
            if step % cfg.train.log_every == 0 or step == steps or out_of_time or stop.requested:
                val = evaluate_generator(gen, codec, val_codes)
                gen.train()
                score = weighted_ce(val, codec.vq.levels)
                row = {"step": step, "train_loss": round(loss.item(), 4), "val_weighted_ce": round(score, 4), **val}
                history.append(row)
                log(f"gen step {step:5d}  loss {loss.item():.4f}  val {score:.4f}  ce/level {val['ce_per_level']}  "
                    f"peak {mx.get_peak_memory() / 2**30:.2f} GB")
                if score < best[0]:
                    best = (score, step)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_weights(gen, path)
                    best_weights = mx.load(str(path))
                save_checkpoint(training_path, gen, opt, {
                    "stage": "generator", "config": cfg.to_dict(), "step": step,
                    "total_steps": steps, "schedule_offset": schedule_offset,
                    "data": data.state_dict(), "best": best, "history": history,
                    "train_seconds": elapsed_before + t.elapsed(), "continuation": continuation,
                    "time_limit_reached": bool(out_of_time),
                }, best_model=tree_unflatten(list(best_weights.items())))
                if stop.requested:
                    log(f"paused generator at step {step}; full training state saved to {training_path}")
                    raise TrainingPaused(step)
                if out_of_time:
                    log(f"stopping at step {step}: max_minutes={cfg.train.max_minutes} reached")
                    steps = step
                    break
    gen.load_weights(str(path))
    gen.eval()
    meta = {
        "stage": "generator",
        "steps": steps,
        "best_step": best[1],
        "train_seconds": round(elapsed_before + t.seconds, 2),
        "history": history,
        "train": evaluate_generator(gen, codec, train_codes[:2000]),
        "val": evaluate_generator(gen, codec, val_codes),
    }
    if continuation:
        meta["continuation"] = continuation
    save_model(gen, path, meta)
    return gen, meta
