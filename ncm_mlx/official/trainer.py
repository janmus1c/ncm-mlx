"""Two-stage training for the released TinyStories architecture family.

Upstream released inference code only, so the objectives here are inferred from
the paper (arXiv:2610.14162, Sections 3.3-3.4 and Appendices C, F) and from the
upstream quantizer and model code (HazyResearch/ncm, MIT, Copyright (c) 2025
Hermann Kumbong). No upstream code is copied. The models are the inference
modules of :mod:`ncm_mlx.official.model`; training only adds dropout, the VQ
losses, EMA codebook updates and the teacher-forced generator forward, so
exported weights load with ``strict=True`` into the ``official`` sampler.

Stage 1 (codec), per batch:

* ``z = GroupNorm(dropout(tok + pos))``; residual VQ over all levels on
  ``stop_gradient(z)`` with nearest-code assignment; level 0 uses a
  straight-through estimator into the learned strided cascade.
* ``vq = mean_k [beta * mse(sg(z_hat_k), z) + mse(z_hat_k, sg(z))]`` over all
  levels (upstream code; paper eq. 9 states the cutoff level instead).
* Nested ("fine-level") dropout: with probability ``fine_drop_prob`` the
  decoder reads ``z_hat`` truncated after a uniformly drawn level in
  ``[min_keep - 1, K - 2]``. Drawn per sequence here, per batch upstream.
* Decoder input ``z + sg(z_hat - z)`` (straight-through to the encoder), one
  bidirectional block with residual dropout, LayerNorm, linear head; token CE
  over all ``L`` positions, padding included.
* Codebooks follow an EMA of assigned inputs (Laplace-smoothed counts) applied
  after the optimizer step from the pre-update assignments, with optional
  dead-code restart.

Stage 2 (generator), with the codec frozen:

* Targets are the codec's greedy codes of the clean sequence. Conditioning codes
  of levels ``0..K-2`` are replaced uniformly at random with a probability that
  falls linearly from ``corrupt_first`` to ``corrupt_last``; labels stay clean.
* One forward over ``[prefix, level_1, ..., level_{K-1}]`` with block-diagonal
  attention (each level sees the prefix and itself; the prefix sees itself),
  RoPE restarting per segment and a learned level embedding. Level ``k`` reads
  the cumulative reconstruction of the (corrupted) levels ``< k`` area-pooled
  to ``l_k``. Heads are tied to the frozen codebooks.
* Loss ``sum_k l_k^(1-alpha) CE_k / sum_k l_k^(1-alpha)`` (paper eq. 10).
* ``train.precision = "bfloat16"`` runs the generator forward and backward in
  bfloat16 on a per-step cast of float32 master weights (see :func:`mixed_value_and_grad`);
  the codec stage and every frozen-codec computation stay float32.
* Options (defaults reproduce the behaviour above exactly): ``train.<stage>.optimizer = "muon"`` puts the
  2-D block weights on Muon and everything else on AdamW; ``generator.corruption_mode = "neighbor"``
  replaces a corrupted code by one of its ``corruption_neighbors`` nearest codebook entries, and
  ``generator.corruption_rate_jitter`` scales each sequence's rates by a factor in ``[1 - j, 1 + j]``.
* Level 0 is not predicted: its empirical unigram over training sequences is
  stored and sampled at generation time, as in the released checkpoint.

Prefix-conditioned variant (``data.prefix_len = P > 0``): rows are ``P`` prefix
tokens followed by the ``seq_len`` target tokens. The codec encodes only the
target. The generator's prefix is the ``P`` real text tokens (pad-masked, RoPE
from 0, attending only to itself) instead of the single EOS, and level 0 is
predicted from the prefix with learned queries (``query_level0``), as in the
upstream prefix tasks; the stored unigram is unused. The loss covers target
codes only. ``P = 0`` takes exactly the code paths above.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from ncm_mlx.official import tinystories as ts
from ncm_mlx.official.model import Codec, Generator, TiedHead, _rotate, rope_tables
from ncm_mlx.official.train_config import Stage, TextConfig
from ncm_mlx.training.checkpoint import (
    StopRequest,
    TrainingPaused,
    atomic_arrays,
    read_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
from ncm_mlx.training.common import BatchStream

# ---------------------------------------------------------------- shared pieces


def dropout(x: mx.array, p: float) -> mx.array:
    if p <= 0:
        return x
    keep = mx.random.bernoulli(1.0 - p, x.shape)
    return mx.where(keep, x / (1.0 - p), mx.zeros_like(x))


def compute_dtype(cfg: TextConfig) -> mx.Dtype:
    return mx.bfloat16 if cfg.train.precision == "bfloat16" else mx.float32


def fp32_norm(norm: nn.Module, x: mx.array) -> mx.array:
    """Apply a LayerNorm/RMSNorm with its statistics computed in float32; a no-op wrapper for float32 input."""
    if x.dtype == mx.float32:
        return norm(x)
    return norm(x.astype(mx.float32)).astype(x.dtype)


def rotated_qk(attn, q: mx.array, k: mx.array, rope, dtype) -> tuple[mx.array, mx.array]:
    """QK RMSNorm and RoPE in float32, returned in ``dtype``; the official computation when ``dtype`` is float32."""
    if q.dtype != mx.float32:
        q, k = q.astype(mx.float32), k.astype(mx.float32)
    q, k = attn.q_norm(q), attn.k_norm(k)
    if rope is not None:
        q, k = _rotate(q, *rope), _rotate(k, *rope)
    return q.astype(dtype), k.astype(dtype)


def attention_forward(attn, x: mx.array, mask, rope) -> mx.array:
    """The official ``Attention`` forward with float32 QK norm and RoPE under reduced-precision input."""
    B, T, _ = x.shape
    q, k, v = (t.reshape(B, T, attn.n_head, attn.head_dim).transpose(0, 2, 1, 3)
               for t in mx.split(attn.qkv(x), 3, axis=-1))
    q, k = rotated_qk(attn, q, k, rope, v.dtype)
    y = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.head_dim**-0.5, mask=mask)
    return attn.proj(y.transpose(0, 2, 1, 3).reshape(B, T, -1))


def block_forward(blk, x: mx.array, mask, rope, p: float) -> mx.array:
    """The official ``Block`` with residual dropout; identical to ``blk(x, mask, rope)`` at ``p = 0`` in float32."""
    x = x + dropout(attention_forward(blk.attn, fp32_norm(blk.ln1, x), mask, rope), p)
    return x + dropout(blk.out(nn.gelu(blk.fc(fp32_norm(blk.ln2, x)))), p)


def float32_prefixes(model: nn.Module) -> tuple[str, ...]:
    """Parameter-name prefixes that stay float32 under autocast: norm layers and the tied level heads."""
    keep = (nn.LayerNorm, nn.RMSNorm, TiedHead)
    return tuple(f"{name}." for name, m in model.named_modules() if name and isinstance(m, keep))


def cast_parameters(params: dict, keep: tuple[str, ...], dtype) -> dict:
    flat = [(k, v if v.dtype != mx.float32 or k.startswith(keep) else v.astype(dtype))
            for k, v in tree_flatten(params)]
    return tree_unflatten(flat)


def mixed_value_and_grad(model: nn.Module, fn, dtype):
    """``nn.value_and_grad`` over float32 master parameters with the forward run on a ``dtype`` cast of them.

    The cast sits inside the differentiated function, so gradients come back in
    float32 for the float32 master weights, and the optimizer, gradient clipping
    and checkpoints never see a reduced-precision value. Norm layers and tied
    heads are not cast. The model holds the original arrays again on return.
    """
    keep = float32_prefixes(model)

    def wrapped(m, *args):
        master = m.trainable_parameters()
        m.update(cast_parameters(master, keep, dtype))
        try:
            return fn(m, *args)
        finally:
            m.update(master)

    return nn.value_and_grad(model, wrapped)


def value_and_grad_for(cfg: TextConfig, model: nn.Module, fn):
    """The float32 path is exactly ``nn.value_and_grad``; bfloat16 adds the autocast above."""
    dtype = compute_dtype(cfg)
    if dtype == mx.float32:
        return nn.value_and_grad(model, fn)
    return mixed_value_and_grad(model, fn, dtype)


def schedule(st: Stage):
    decay = optim.cosine_decay(st.lr, max(st.steps - st.warmup, 1), st.lr_floor)
    if st.warmup == 0:
        return decay
    warm = optim.linear_schedule(st.lr / st.warmup, st.lr, st.warmup)
    return optim.join_schedules([warm, decay], [st.warmup])


MUON_PATH = re.compile(r"(^|\.)(blocks\.\d+|dec_block|in_proj)\.")


def muon_matrix(key: str, value: mx.array) -> bool:
    """Weights Muon orthogonalises: 2-D projections inside the transformer blocks (generator, AR and the codec's decoder block) and the generator input projection.

    Embeddings, tied level heads, codebooks, norms and biases stay on AdamW.
    """
    return value.ndim == 2 and key.endswith(".weight") and MUON_PATH.search(key) is not None


def optimizer(st: Stage, offset: int = 0):
    lr = schedule(st)
    shifted = (lambda step: lr(step + offset)) if offset else lr
    adamw = optim.AdamW(learning_rate=shifted, betas=[st.beta1, st.beta2], weight_decay=st.weight_decay)
    if st.optimizer == "adamw":
        return adamw
    ratio = st.muon_lr / st.lr
    muon = optim.Muon(learning_rate=lambda step: shifted(step) * ratio, momentum=st.muon_momentum,
                      weight_decay=st.muon_weight_decay, ns_steps=st.muon_ns_steps)
    return optim.MultiOptimizer([muon, adamw], [muon_matrix])


def scaled_residual_init(blocks, n_layer: int, seed: int) -> None:
    """Upstream generator init for residual projections: N(0, 0.02 / sqrt(2 n_layer))."""
    std = 0.02 / (2 * n_layer) ** 0.5
    key = mx.random.key(seed)
    for blk in blocks:
        for lin in (blk.attn.proj, blk.out):
            key, sub = mx.random.split(key)
            lin.weight = mx.random.normal(lin.weight.shape, key=sub) * std


def to_batch(rows: np.ndarray) -> mx.array:
    return mx.array(np.asarray(rows, dtype=np.int32))


def load_rows(cfg: TextConfig, log=print) -> tuple[np.ndarray, np.ndarray, dict]:
    d = cfg.data
    manifest = ts.prepare(Path(d.raw_dir), Path(d.cache_root), Path(d.bpe), d.seq_len, limit=d.limit, log=log,
                          prefix_len=d.prefix_len)
    directory = ts.split_dir(Path(d.cache_root), d.seq_len, d.limit, d.prefix_len)
    train = ts.load_split(directory, "train")
    val = np.asarray(ts.load_split(directory, "val")[: d.val_rows])
    return train, val, manifest


def split_prefix(cfg: TextConfig, rows: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    """``(prefix, target)`` views of continuation rows; ``(None, rows)`` for the unconditional task."""
    P = cfg.data.prefix_len
    return (rows[:, :P], rows[:, P:]) if P else (None, rows)


def export_dir(cfg: TextConfig) -> Path:
    return Path(cfg.out_dir) / "export"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Run:
    """Generic resumable loop: full-state checkpoints at ``log_every``, pause on signals, time limit."""

    def __init__(self, cfg: TextConfig, stage: str, st: Stage, model: nn.Module, data: BatchStream, log):
        self.cfg, self.stage, self.st, self.model, self.data, self.log = cfg, stage, st, model, data, log
        self.path = Path(cfg.out_dir) / f"{stage}-training.safetensors"
        self.opt = optimizer(st)
        self.history: list[dict] = []
        self.start_step, self.elapsed_before, self.time_limited = 0, 0.0, False
        self.extra_arrays: dict = {}
        self.extra_meta: dict = {}

    def resume(self) -> dict:
        state, saved = read_checkpoint(self.path, self.cfg, self.stage, self.st.steps, self.log)
        restore_checkpoint(state, self.model, self.opt)
        self.data.load_state_dict(saved["data"])
        self.start_step, self.elapsed_before = saved["step"], saved["train_seconds"]
        self.history, self.time_limited = saved["history"], saved.get("time_limit_reached", False)
        self.log(f"{_now()} resuming full {self.stage} state at step {self.start_step}/{self.st.steps}")
        return state

    def save(self, step: int, elapsed: float) -> None:
        save_checkpoint(self.path, self.model, self.opt, {
            "stage": self.stage, "config": self.cfg.to_dict(), "step": step, "total_steps": self.st.steps,
            "data": self.data.state_dict(), "history": self.history, "train_seconds": elapsed,
            "time_limit_reached": self.time_limited, **self.extra_meta,
        }, **self.extra_arrays)

    def loop(self, train_step, after_step, evaluate) -> int:
        """Run to ``st.steps`` (or the time limit); return the last completed step."""
        st, cfg = self.st, self.cfg
        if self.time_limited:
            return self.start_step
        last = self.start_step
        with StopRequest() as stop:
            t0 = time.perf_counter()
            tick, tick_step = t0, self.start_step
            for step in range(self.start_step + 1, st.steps + 1):
                batch = to_batch(next(self.data))
                out = train_step(batch)
                after_step(out)
                mx.eval(self.model.parameters(), self.opt.state, out[0])
                last = step
                elapsed = self.elapsed_before + time.perf_counter() - t0
                self.time_limited = bool(st.max_minutes and elapsed > 60 * st.max_minutes)
                boundary = step % cfg.train.log_every == 0 or step == st.steps
                if boundary or self.time_limited or stop.requested:
                    now = time.perf_counter()
                    rate = (step - tick_step) * st.batch_size / max(now - tick, 1e-9)
                    tick, tick_step = now, step
                    row = {"step": step, "loss": round(out[0].item(), 5), "lr": round(self.opt.learning_rate.item(), 8),
                           "seq_per_s": round(rate, 1), "elapsed_s": round(elapsed, 1)}
                    row.update({k: round(v.item(), 5) for k, v in out[1].items()})
                    if step % cfg.train.eval_every == 0 or step == st.steps or self.time_limited:
                        row["val"] = evaluate()
                    self.history.append(row)
                    self.log(f"{_now()} {self.stage} {json.dumps(row)}  peak {mx.get_peak_memory() / 2**30:.2f} GB")
                    self.save(step, elapsed)
                    if stop.requested:
                        self.log(f"paused {self.stage} at step {step}; full training state saved to {self.path}")
                        raise TrainingPaused(step)
                    if self.time_limited:
                        self.log(f"stopping {self.stage} at step {step}: max_minutes={st.max_minutes} reached")
                        break
        self.elapsed_before += time.perf_counter() - t0
        return last


def compiled(model, opt, step_fn, enabled: bool):
    if not enabled:
        return step_fn
    state = [model.state, opt.state, mx.random.state]
    return partial(mx.compile, inputs=state, outputs=state)(step_fn)


# ---------------------------------------------------------------- stage 1: codec


def _init_codec(codec: Codec, seed: int) -> None:
    """Upstream initialisation of the parameters that the inference module creates as zeros."""
    key = mx.random.key(seed)
    D = codec.spec.dim
    keys = mx.random.split(key, len(codec.codebooks) + 2 * len(codec.phi))
    codec.codebooks = [mx.random.normal(cb.shape, key=keys[i]) for i, cb in enumerate(codec.codebooks)]
    bound = 1.0 / (3 * D) ** 0.5
    for i, phi in enumerate(codec.phi):
        kw, kb = keys[len(codec.codebooks) + 2 * i], keys[len(codec.codebooks) + 2 * i + 1]
        phi.weight = mx.random.uniform(-bound, bound, phi.weight.shape, key=kw)
        phi.bias = mx.random.uniform(-bound, bound, phi.bias.shape, key=kb)
    eye = mx.eye(D)
    for m in codec.down0:
        m.weight = mx.concatenate([eye / m.stride] * m.stride, axis=0)
        m.bias = mx.zeros_like(m.bias)
    for m in codec.up0:
        m.weight = mx.concatenate([eye] * m.stride, axis=1)
        m.bias = mx.zeros_like(m.bias)


class CodecTrainer(nn.Module):
    """The official ``Codec`` plus its EMA codebook state; only ``codec`` is exported."""

    def __init__(self, cfg: TextConfig, seed: int = 0):
        super().__init__()
        self.codec = Codec(cfg.codec_spec())
        _init_codec(self.codec, seed)
        self.ema_w = [mx.array(cb) for cb in self.codec.codebooks]
        self.ema_count = [mx.zeros((cb.shape[0],)) for cb in self.codec.codebooks]
        self.codec.freeze(recurse=False, keys=["codebooks"])
        self.freeze(recurse=False, keys=["ema_w", "ema_count"])
        self.c = cfg.codec

    def quantize(self, z: mx.array, ste_level0: bool = True):
        """Residual VQ of ``z``; returns codes, cumulative reconstructions and the assigned inputs."""
        codec = self.codec
        K = len(codec.levels)
        rest = mx.stop_gradient(z)
        z_hat, codes, z_hats, inputs = None, [], [], []
        for k, cb in enumerate(codec.codebooks):
            x = codec._down(rest, k) if k != K - 1 else rest
            xs = mx.stop_gradient(x)
            d = (xs * xs).sum(-1, keepdims=True) + (cb * cb).sum(-1) - 2 * (xs @ cb.T)
            idx = mx.argmin(d, axis=-1)
            q = cb[idx]
            if k == 0 and ste_level0:
                q = x + mx.stop_gradient(q - x)
            contrib = codec.contribution(q, k)
            z_hat = contrib if z_hat is None else z_hat + contrib
            rest = rest - mx.stop_gradient(contrib)
            codes.append(idx)
            z_hats.append(z_hat)
            inputs.append(xs)
        return codes, z_hats, inputs

    def decode_logits(self, z: mx.array, p: float) -> mx.array:
        codec = self.codec
        return codec.dec_head(codec.dec_ln(block_forward(codec.dec_block, z, None, None, p)))

    def loss(self, tokens: mx.array, train: bool = True):
        c, codec = self.c, self.codec
        K = len(codec.levels)
        p = c.dropout if train else 0.0
        z = codec.normalize(dropout(codec.lookup(tokens), p))
        codes, z_hats, inputs = self.quantize(z)
        z_sg = mx.stop_gradient(z)
        vq = sum(c.commitment * mx.mean((mx.stop_gradient(zh) - z) ** 2) + mx.mean((zh - z_sg) ** 2)
                 for zh in z_hats) / K
        z_dec = z_hats[-1]
        if train and c.fine_drop_prob > 0:
            B = tokens.shape[0]
            cut = mx.random.randint(c.fine_drop_min_keep - 1, K - 1, (B,))
            drop = mx.random.bernoulli(c.fine_drop_prob, (B,))
            level = mx.where(drop, cut, K - 1)
            z_dec = mx.stack(z_hats, axis=0)[level, mx.arange(B)]
        z_dec = z + mx.stop_gradient(z_dec - z)
        logits = self.decode_logits(z_dec, p)
        ce = nn.losses.cross_entropy(logits, tokens, reduction="mean")
        return ce + vq, (ce, vq, codes, inputs)

    def ema_update(self, codes, inputs) -> None:
        c = self.c
        d = c.decay
        for k, (idx, x) in enumerate(zip(codes, inputs)):
            V, D = self.ema_w[k].shape
            flat = x.reshape(-1, D)
            idx = idx.reshape(-1)
            counts = mx.zeros((V,)).at[idx].add(1.0)
            dw = mx.zeros((V, D)).at[idx].add(flat)
            ema_count = d * self.ema_count[k] + (1 - d) * counts
            ema_w = d * self.ema_w[k] + (1 - d) * dw
            n = ema_count.sum()
            smoothed = (ema_count + c.epsilon) / (n + V * c.epsilon) * n
            codebook = ema_w / smoothed[:, None]
            if c.restart_threshold > 0:
                dead = ema_count < c.restart_threshold
                fresh = flat[mx.random.randint(0, flat.shape[0], (V,))]
                codebook = mx.where(dead[:, None], fresh, codebook)
                ema_w = mx.where(dead[:, None], fresh * c.restart_threshold, ema_w)
                ema_count = mx.where(dead, c.restart_threshold, ema_count)
            self.codec.codebooks[k] = codebook
            self.ema_w[k] = ema_w
            self.ema_count[k] = ema_count


def codec_report(codec: Codec, rows: np.ndarray, batch: int = 256) -> dict:
    """Held-out reconstruction through the final level and after each level (eval mode, greedy codes)."""
    K = len(codec.levels)
    n = tot = nonpad = nonpad_ok = exact = ce_sum = 0.0
    level_ok = np.zeros(K)
    used = [np.zeros(v, dtype=bool) for v in codec.spec.codebook_sizes]
    for i in range(0, len(rows), batch):
        t = to_batch(rows[i : i + batch])
        codes = codec.encode(t)
        cum = codec.cumulative(codes)
        logits = codec.decode_logits(cum[-1])
        pred = mx.argmax(logits, axis=-1)
        ce_sum += nn.losses.cross_entropy(logits, t, reduction="sum").item()
        mask = t != ts.PAD_ID
        ok = pred == t
        tot += ok.sum().item()
        nonpad += mask.sum().item()
        nonpad_ok += (ok & mask).sum().item()
        exact += mx.all(ok, axis=1).sum().item()
        for k in range(K):
            pk = mx.argmax(codec.decode_logits(cum[k]), axis=-1)
            level_ok[k] += ((pk == t) & mask).sum().item()
            used[k][np.unique(np.array(codes[k]))] = True
        n += t.shape[0]
    L = rows.shape[1]
    return {
        "sequences": int(n),
        "token_acc": round(tot / (n * L), 5),
        "nonpad_token_acc": round(nonpad_ok / nonpad, 5),
        "exact_window_pct": round(100 * exact / n, 3),
        "recon_ce": round(ce_sum / (n * L), 5),
        "cumulative_nonpad_acc": [round(x / nonpad, 4) for x in level_ok],
        "code_usage": [round(float(u.mean()), 4) for u in used],
    }


def level0_unigram(codec: Codec, rows: np.ndarray, batch: int = 1024) -> np.ndarray:
    V = codec.spec.codebook_sizes[0]
    counts = np.zeros(V, dtype=np.float64)
    for i in range(0, len(rows), batch):
        c0 = np.array(codec.encode(to_batch(rows[i : i + batch]))[0]).reshape(-1)
        counts += np.bincount(c0, minlength=V)
    return (counts / counts.sum()).astype(np.float32)


def _write_spec(cfg: TextConfig, out: Path, extra: dict) -> None:
    path = out / "spec.json"
    meta = json.loads(path.read_text()) if path.exists() else {}
    meta.update({"task": cfg.task_spec().to_json(),
                 "source": {"trained_by": "ncm-mlx", "config_name": cfg.name, "config": cfg.to_dict()}})
    meta.setdefault("training", {}).update(extra)
    path.write_text(json.dumps(meta, indent=2) + "\n")


def train_codec(cfg: TextConfig, resume: bool = False, log=print) -> dict:
    st = cfg.train.codec
    mx.random.seed(cfg.train.seed)
    train_rows, val_rows, manifest = load_rows(cfg, log)
    if cfg.data.prefix_len:
        train_rows, val_rows = (np.ascontiguousarray(split_prefix(cfg, r)[1]) for r in (train_rows, val_rows))
    if cfg.train.precision != "float32":
        log(f"codec: precision {cfg.train.precision} applies to the generator and AR stages; the codec trains in float32")
    tr = CodecTrainer(cfg, cfg.train.seed)
    tr.train()
    data = BatchStream(train_rows, st.batch_size, np.random.default_rng(cfg.train.seed))
    run = Run(cfg, "codec", st, tr, data, log)
    if resume:
        run.resume()
    log(f"codec: {sum(v.size for _, v in tree_flatten(tr.codec.parameters())):,} values, "
        f"{sum(v.size for _, v in tree_flatten(tr.trainable_parameters())):,} gradient-trained; "
        f"train rows {len(train_rows):,}")
    grad_fn = nn.value_and_grad(tr, lambda m, t: m.loss(t))

    def step(tokens):
        (loss, (ce, vq, codes, inputs)), grads = grad_fn(tr, tokens)
        grads, _ = optim.clip_grad_norm(grads, st.grad_clip)
        run.opt.update(tr, grads)
        return loss, {"ce": ce, "vq": vq}, codes, inputs

    step = compiled(tr, run.opt, step, cfg.train.compile)

    def after(out):
        tr.ema_update(out[2], out[3])

    def evaluate():
        tr.eval()
        r = codec_report(tr.codec, val_rows)
        tr.train()
        return r

    last = run.loop(step, after, evaluate)
    tr.eval()
    out = export_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    atomic_arrays(dict(tree_flatten(tr.codec.parameters())), out / "codec.safetensors")
    shutil.copyfile(cfg.data.bpe, out / "bpe.json")
    unigram = level0_unigram(tr.codec, np.asarray(train_rows[: cfg.data.unigram_rows]))
    atomic_arrays({"level0_unigram": mx.array(unigram)}, out / "inference.safetensors")
    report = codec_report(tr.codec, val_rows)
    meta = {"steps": last, "planned_steps": st.steps, "time_limited": run.time_limited,
            "train_seconds": round(run.elapsed_before, 2), "sequences_seen": last * st.batch_size,
            "data_manifest": manifest, "val": report,
            "level0_codes_used": int((unigram > 0).sum()), "history": run.history}
    _write_spec(cfg, out, {"codec": meta})
    log(f"codec done: {json.dumps(report)}")
    return meta


def load_codec(cfg: TextConfig) -> Codec:
    path = export_dir(cfg) / "codec.safetensors"
    if not path.exists():
        raise SystemExit(f"no trained codec at {path}; run `official train-codec` first")
    codec = Codec(cfg.codec_spec())
    codec.load_weights(str(path), strict=True)
    codec.eval()
    codec.freeze()
    return codec


# ---------------------------------------------------------------- stage 2: generator


def block_diagonal_mask(prefix_len: int, lengths: list[int]) -> np.ndarray:
    """``(T, T)`` boolean mask: prefix rows see the prefix; level rows see the prefix and their own level."""
    seg = np.concatenate([np.full(prefix_len, -1)] + [np.full(n, j) for j, n in enumerate(lengths)])
    cols = np.arange(len(seg))
    return (cols[None, :] < prefix_len) | ((seg[:, None] == seg[None, :]) & (seg[:, None] >= 0))


def segmented_attention(attn, x: mx.array, prefix_len: int, lengths: list[int], rope, keep) -> mx.array:
    """Block-diagonal attention of an official ``Attention`` without forming the full ``T x T`` scores.

    Projections, QK norms and RoPE run once over the whole sequence; each level's
    queries then attend to the prefix keys and that level's keys only, which is
    the per-level inference computation. ``keep`` masks padded prefix keys.
    """
    B, T, C = x.shape
    H, hd = attn.n_head, attn.head_dim
    q, k, v = (t.reshape(B, T, H, hd).transpose(0, 2, 1, 3) for t in mx.split(attn.qkv(x), 3, axis=-1))
    q, k = rotated_qk(attn, q, k, rope, v.dtype)
    P = prefix_len
    kp, vp = k[:, :, :P], v[:, :, :P]
    scale = hd**-0.5
    pmask = None if keep is None else keep[:, None, None, :]
    outs = [mx.fast.scaled_dot_product_attention(q[:, :, :P], kp, vp, scale=scale, mask=pmask)]
    off = P
    for n in lengths:
        sl = slice(off, off + n)
        m = None
        if keep is not None:
            m = mx.concatenate([keep, mx.ones((B, n), dtype=mx.bool_)], axis=1)[:, None, None, :]
        outs.append(mx.fast.scaled_dot_product_attention(
            q[:, :, sl], mx.concatenate([kp, k[:, :, sl]], axis=2), mx.concatenate([vp, v[:, :, sl]], axis=2),
            scale=scale, mask=m))
        off += n
    y = mx.concatenate(outs, axis=2)
    return attn.proj(y.transpose(0, 2, 1, 3).reshape(B, T, C))


class TeacherForced:
    """Teacher-forced forward of an official ``Generator`` over all target levels at once."""

    def __init__(self, gen: Generator, dtype=mx.float32):
        self.gen, self.dtype = gen, dtype
        P, hd = gen.spec.prefix_len, gen.spec.dim // gen.spec.n_head
        self.lengths = list(gen.target_levels)
        pos = np.concatenate([np.arange(P)] + [np.arange(n) for n in self.lengths])
        self.rope = rope_tables(pos, hd, gen.spec.rope_base)

    def logits(self, level_inputs: list[mx.array], prefix_tokens: mx.array, p: float = 0.0) -> list[mx.array]:
        gen, dt = self.gen, self.dtype
        P = gen.spec.prefix_len
        prefix_emb = gen.in_proj(gen.codec.tok(prefix_tokens[:, :P]).astype(dt))
        xs = [gen.in_proj(x.astype(dt)) + gen.level_emb.weight[j] for j, x in enumerate(level_inputs)]
        h = mx.concatenate([prefix_emb, *xs], axis=1)
        keep = None
        if gen.spec.prefix_pad_id is not None:
            keep = prefix_tokens[:, :P] != gen.spec.prefix_pad_id
        for blk in gen.blocks:
            a = segmented_attention(blk.attn, fp32_norm(blk.ln1, h), P, self.lengths, self.rope, keep)
            h = h + dropout(a, p)
            h = h + dropout(blk.out(nn.gelu(blk.fc(fp32_norm(blk.ln2, h)))), p)
        h = fp32_norm(gen.ln_f, h).astype(mx.float32)
        out, off = [], P
        for j, n in enumerate(self.lengths):
            out.append(gen.head_logits(j, h[:, off : off + n]))
            off += n
        return out


def corruption_rates(first: float, last: float, n: int) -> list[float]:
    if n <= 1:
        return [first] * n
    return [first + (last - first) * k / (n - 1) for k in range(n)]


def teacher_inputs(codec: Codec, codes: list[mx.array]) -> list[mx.array]:
    """Input of target level ``k`` (1..K-1): reconstruction of levels ``< k``, pooled to ``l_k``."""
    out, z = [], None
    for k in range(1, len(codec.levels)):
        q = codec.level_contribution(codes[k - 1], k - 1)
        z = q if z is None else z + q
        out.append(codec.pool(z, k))
    return [mx.stop_gradient(x) for x in out]


def nearest_code_table(codebook: mx.array, k: int) -> mx.array:
    """``(V, k)`` indices of each code's ``k`` nearest other codes by Euclidean distance between codebook vectors."""
    cb = mx.stop_gradient(codebook).astype(mx.float32)
    sq = (cb * cb).sum(-1)
    d = sq[:, None] + sq[None, :] - 2 * (cb @ cb.T)
    d = d + mx.where(mx.eye(cb.shape[0], dtype=mx.bool_), mx.array(float("inf")), mx.array(0.0))
    return mx.argsort(d, axis=-1)[:, :k].astype(mx.int32)


class Corruption:
    """Non-default conditioning corruption: replacement from codebook neighbours and/or per-example rate scaling."""

    def __init__(self, tables: list[mx.array] | None, jitter: float):
        self.tables, self.jitter = tables, jitter


def make_corruption(cfg: TextConfig, codec: Codec) -> Corruption | None:
    """``None`` for the default uniform, unjittered corruption, which keeps the original random-number stream."""
    g = cfg.generator
    if g.corruption_mode == "uniform" and g.corruption_rate_jitter == 0:
        return None
    tables = None
    if g.corruption_mode == "neighbor":
        tables = [nearest_code_table(cb, g.corruption_neighbors) for cb in codec.codebooks[:-1]]
        mx.eval(tables)
    return Corruption(tables, g.corruption_rate_jitter)


def corrupt(codes: list[mx.array], rates: list[float], sizes, extra: Corruption | None = None) -> list[mx.array]:
    out = []
    scale = None
    if extra is not None and extra.jitter > 0:
        scale = 1.0 + extra.jitter * (2 * mx.random.uniform(shape=(codes[0].shape[0], 1)) - 1)
    for k, c in enumerate(codes):
        if k < len(rates) and rates[k] > 0:
            if extra is None:
                flip = mx.random.bernoulli(rates[k], c.shape)
                new = mx.random.randint(0, sizes[k], c.shape)
            else:
                rate = rates[k] if scale is None else mx.minimum(rates[k] * scale, 1.0)
                flip = mx.random.uniform(shape=c.shape) < rate
                if extra.tables is None:
                    new = mx.random.randint(0, sizes[k], c.shape)
                else:
                    pick = mx.random.randint(0, extra.tables[k].shape[1], c.shape)
                    new = extra.tables[k][c, pick]
            c = mx.where(flip, new.astype(c.dtype), c)
        out.append(c)
    return out


def level_weights(levels: list[int], alpha: float) -> list[float]:
    w = [n ** (1.0 - alpha) for n in levels]
    s = sum(w)
    return [x / s for x in w]


def generator_loss(tf: TeacherForced, codec: Codec, tokens: mx.array, rates, alpha: float, p: float,
                   prefix_len: int = 0, corruption: Corruption | None = None):
    """Weighted code cross-entropy. With ``prefix_len > 0``, ``tokens`` are ``[prefix, target]`` rows and
    only the target is encoded and predicted."""
    if prefix_len:
        prefix, tokens = tokens[:, :prefix_len], tokens[:, prefix_len:]
    else:
        prefix = mx.full((tokens.shape[0], 1), ts.EOS_ID, dtype=mx.int32)
    codes = [mx.stop_gradient(c).astype(mx.int32) for c in codec.encode(tokens)]
    seen = corrupt(codes[:-1], rates, codec.spec.codebook_sizes, corruption) if rates else codes[:-1]
    inputs = teacher_inputs(codec, seen)
    first = tf.gen.first
    if not first:
        inputs = [mx.broadcast_to(tf.gen.l0_queries, (tokens.shape[0], *tf.gen.l0_queries.shape[1:])), *inputs]
    logits = tf.logits(inputs, prefix, p)
    ce = [nn.losses.cross_entropy(lg, c, reduction="mean") for lg, c in zip(logits, codes[first:])]
    w = level_weights(tf.lengths, alpha)
    return sum(wi * c for wi, c in zip(w, ce)), ce


def generator_report(tf: TeacherForced, codec: Codec, rows: np.ndarray, batch: int = 256,
                     prefix_len: int = 0) -> dict:
    """Held-out teacher-forced code cross-entropy per level (clean conditioning). Not text perplexity."""
    sums, n = None, 0
    for i in range(0, len(rows), batch):
        t = to_batch(rows[i : i + batch])
        _, ce = generator_loss(tf, codec, t, None, 0.0, 0.0, prefix_len)
        v = np.array([c.item() for c in ce]) * t.shape[0]
        sums = v if sums is None else sums + v
        n += t.shape[0]
    per = sums / n
    w = level_weights(tf.lengths, 0.0)
    return {"code_ce_per_level": [round(float(x), 4) for x in per],
            "code_ce_position_weighted": round(float(np.dot(w, per)), 5)}


def build_generator(cfg: TextConfig, codec: Codec, seed: int) -> Generator:
    mx.random.seed(seed)
    gen = Generator(cfg.generator_spec(), codec)
    scaled_residual_init(gen.blocks, cfg.generator.n_layer, seed + 7)
    return gen


def precision_note(cfg: TextConfig) -> str:
    return "" if cfg.train.precision == "float32" else f"; {cfg.train.precision} compute, float32 master weights"


def generator_step(cfg: TextConfig, gen: Generator, codec: Codec, tf: TeacherForced, opt, rates, prefix_len: int = 0):
    """The (optionally compiled) generator training step; shared with ``scripts/text_step_timing.py``."""
    st, g = cfg.train.generator, cfg.generator
    extra = make_corruption(cfg, codec)
    grad_fn = value_and_grad_for(
        cfg, gen, lambda m, t: generator_loss(tf, codec, t, rates, g.loss_alpha, g.dropout, prefix_len, extra))

    def step(tokens):
        (loss, ce), grads = grad_fn(gen, tokens)
        grads, norm = optim.clip_grad_norm(grads, st.grad_clip)
        opt.update(gen, grads)
        return loss, {"grad_norm": norm, "ce_level0" if prefix_len else "ce_level1": ce[0],
                           "ce_last": ce[-1]}

    return compiled(gen, opt, step, cfg.train.compile)


def train_generator(cfg: TextConfig, resume: bool = False, log=print) -> dict:
    st, g = cfg.train.generator, cfg.generator
    codec = load_codec(cfg)
    train_rows, val_rows, _manifest = load_rows(cfg, log)
    gen = build_generator(cfg, codec, cfg.train.seed + 1)
    gen.train()
    tf, tf_eval = TeacherForced(gen, compute_dtype(cfg)), TeacherForced(gen)
    data = BatchStream(train_rows, st.batch_size, np.random.default_rng(cfg.train.seed + 1))
    run = Run(cfg, "generator", st, gen, data, log)
    if resume:
        run.resume()
    log(f"generator: {sum(v.size for _, v in tree_flatten(gen.parameters())):,} values; "
        f"train rows {len(train_rows):,}" + precision_note(cfg))
    rates = corruption_rates(g.corrupt_first, g.corrupt_last, len(codec.levels) - 1)
    rates = rates if any(rates) else None
    P = cfg.data.prefix_len
    step = generator_step(cfg, gen, codec, tf, run.opt, rates, P)

    def evaluate():
        gen.eval()
        r = generator_report(tf_eval, codec, val_rows, prefix_len=P)
        gen.train()
        return r

    last = run.loop(step, lambda out: None, evaluate)
    gen.eval()
    out = export_dir(cfg)
    atomic_arrays(dict(tree_flatten(gen.parameters())), out / "generator.safetensors")
    report = generator_report(tf_eval, codec, val_rows, prefix_len=P)
    meta = {"steps": last, "planned_steps": st.steps, "time_limited": run.time_limited,
            "train_seconds": round(run.elapsed_before, 2), "sequences_seen": last * st.batch_size,
            "corruption_rates": rates, "val": report, "history": run.history}
    _write_spec(cfg, out, {"generator": meta})
    log(f"generator done: {json.dumps(report)}")
    return meta


def parameter_count(model: nn.Module) -> int:
    return sum(v.size for _, v in tree_flatten(model.parameters()))

