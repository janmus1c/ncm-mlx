"""Training the released TinyStories architecture family: data contract, losses, equivalence, resume.

Everything runs on ``mx.cpu`` with tiny configurations. Data tests that need the
official BPE vocabulary skip when ``artifacts/official/tinystories/mlx/bpe.json``
is absent (``python -m ncm_mlx official fetch`` and ``convert``).
"""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from ncm_mlx.official import ar as ar_mod
from ncm_mlx.official import sample, trainer
from ncm_mlx.official import tinystories as ts
from ncm_mlx.official.model import Codec
from ncm_mlx.official.train_config import TextConfig
from ncm_mlx.training.checkpoint import TrainingPaused

pytestmark = pytest.mark.mlx
BPE = Path("artifacts/official/tinystories/mlx/bpe.json")
needs_bpe = pytest.mark.skipif(not BPE.exists(), reason="official TinyStories BPE not converted")


@pytest.fixture(autouse=True)
def cpu():
    with mx.stream(mx.cpu):
        mx.set_default_device(mx.cpu)
        yield
    mx.set_default_device(mx.gpu)


def tiny_cfg(tmp_path: Path, **train) -> TextConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    bpe = tmp_path / "bpe.json"
    vocab = [b"<|endoftext|>".hex(), b"<|pad|>".hex()] + [bytes([i]).hex() for i in range(256)]
    bpe.write_text(json.dumps({"vocab": vocab, "merges": [], "special_tokens": ["<|endoftext|>", "<|pad|>"]}))
    cfg = TextConfig.from_dict({
        "name": "tiny", "out_dir": str(tmp_path / "run"),
        "data": {"seq_len": 8, "bpe": str(bpe), "val_rows": 8, "unigram_rows": 32},
        "codec": {"dim": 16, "n_head": 2, "levels": [1, 2, 4, 8], "codebook_sizes": [8, 8, 8, 8],
                  "l0_strides": [4, 2], "group_norm_groups": 4, "dropout": 0.2, "restart_threshold": 0.01},
        "generator": {"n_layer": 1, "n_head": 2, "dim": 16, "dropout": 0.2,
                      "corrupt_first": 0.5, "corrupt_last": 0.1},
        "train": {"seed": 3, "log_every": 2, "eval_every": 4,
                  "codec": {"steps": 6, "batch_size": 4, "warmup": 1, **train},
                  "generator": {"steps": 6, "batch_size": 4, "warmup": 1, **train},
                  "ar": {"steps": 6, "batch_size": 4, "warmup": 1, **train}},
    })
    return cfg


def tiny_rows(n=32, L=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = rng.integers(2, 40, (n, L)).astype(np.uint16)
    rows[::3, 5] = ts.EOS_ID
    rows[::3, 6:] = ts.PAD_ID
    return rows


@pytest.fixture
def patched_rows(monkeypatch):
    rows = tiny_rows()

    def fake(cfg, log=print):
        return rows, rows[:8].copy(), {"fake": True}

    monkeypatch.setattr(trainer, "load_rows", fake)
    monkeypatch.setattr(ar_mod, "load_rows", fake)
    return rows


# ---------------------------------------------------------------- config and data


def test_config_rejects_unknown_keys_and_inconsistent_levels(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        TextConfig.from_dict({"codec": {"width": 3}})
    with pytest.raises(ValueError, match="l0_strides"):
        TextConfig.from_dict({"codec": {"l0_strides": [4, 4]}})
    cfg = tiny_cfg(tmp_path)
    assert cfg.task_spec().codec.levels == (1, 2, 4, 8)


def test_story_split_keeps_text_and_drops_empty_segments():
    text = "First story.\n<|endoftext|>\nSecond\n\n story.\n<|endoftext|>\n \n<|endoftext|>"
    assert ts.split_stories(text) == ["\nFirst story.\n", "\nSecond\n\n story.\n"]


@needs_bpe
def test_window_encoding_matches_full_encoding():
    from ncm_mlx.official.bpe import BPETokenizer

    bpe = BPETokenizer.load(BPE)
    long = "\n" + " ".join(f"Tom's dog ran {i} times, \"wow!\"  It was  fun." for i in range(200)) + "\n"
    for story in ["\nShort one.\n", "\nA café — naïve ünïcode test.\n", long, long[:1700], long[:1601]]:
        full = (bpe.encode(story) + [ts.EOS_ID])[:128]
        assert ts.encode_window(bpe, story, 128, {}) == full


@needs_bpe
def test_prepare_splits_by_document_and_removes_overlap(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    stories = [f"Once upon a time {i} cats sang a song number {i}." for i in range(60)]
    (raw / ts.TRAIN_FILE).write_text("\n<|endoftext|>\n".join(stories[:50]) + "\n<|endoftext|>\n")
    valid = stories[45:60]
    (raw / ts.VALID_FILE).write_text("\n<|endoftext|>\n".join(valid) + "\n<|endoftext|>\n")
    m = ts.prepare(raw, tmp_path / "cache", BPE, seq_len=32, workers=1, limit=50, log=lambda *_: None)
    d = ts.split_dir(tmp_path / "cache", 32, 50)
    train, val, test = (np.load(d / f"{s}.npy") for s in ("train", "val", "test"))
    assert m["heldout_overlap_removed"] == 5
    assert len(val) + len(test) == 10 and train.shape == (50, 32) and train.dtype == np.uint16
    assert not {r.tobytes() for r in train} & {r.tobytes() for r in np.concatenate([val, test])}
    from ncm_mlx.official.bpe import BPETokenizer

    bpe = BPETokenizer.load(BPE)
    row = train[0].tolist()
    assert bpe.decode(row[: row.index(ts.EOS_ID)]) == "\n" + stories[0] + "\n"
    assert set(row[row.index(ts.EOS_ID) + 1 :]) <= {ts.PAD_ID}
    assert ts.prepare(raw, tmp_path / "cache", BPE, seq_len=32, workers=1, limit=50) == m


# ---------------------------------------------------------------- codec


def test_codec_loss_shapes_and_eval_quantisation_matches_encode(tmp_path):
    cfg = tiny_cfg(tmp_path)
    tr = trainer.CodecTrainer(cfg, seed=0)
    tokens = mx.array(tiny_rows(4).astype(np.int32))
    loss, (ce, vq, codes, inputs) = tr.loss(tokens, train=True)
    assert loss.shape == ce.shape == vq.shape == ()
    assert [c.shape for c in codes] == [(4, n) for n in cfg.codec.levels]
    assert [x.shape for x in inputs] == [(4, n, 16) for n in cfg.codec.levels]
    assert float(loss) > 0 and np.isfinite(float(loss))
    tr.eval()
    z = tr.codec.embed(tokens)
    train_codes, z_hats, _ = tr.quantize(z, ste_level0=False)
    for a, b in zip(train_codes, tr.codec.encode(tokens)):
        np.testing.assert_array_equal(np.array(a), np.array(b))
    for a, b in zip(z_hats, tr.codec.cumulative(tr.codec.encode(tokens))):
        np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-6)
    np.testing.assert_array_equal(np.array(tr.decode_logits(z_hats[-1], 0.0)),
                                  np.array(tr.codec.decode_logits(z_hats[-1])))


def test_codec_gradients_reach_encoder_decoder_phi_and_level0_cascade(tmp_path):
    from mlx import nn
    from mlx.utils import tree_flatten

    cfg = tiny_cfg(tmp_path)
    tr = trainer.CodecTrainer(cfg, seed=0)
    tokens = mx.array(tiny_rows(4).astype(np.int32))
    _, grads = nn.value_and_grad(tr, lambda m, t: m.loss(t)[0])(tr, tokens)
    flat = dict(tree_flatten(grads))
    assert not any("codebooks" in k or "ema" in k for k in flat)
    for key in ("codec.tok.weight", "codec.dec_head.weight", "codec.phi.0.weight", "codec.down0.0.weight",
                "codec.up0.0.weight", "codec.dec_block.attn.qkv.weight"):
        assert float(mx.abs(flat[key]).sum()) > 0, key


def test_ema_update_matches_manual_formula(tmp_path):
    cfg = tiny_cfg(tmp_path)
    cfg.codec.restart_threshold = 0.0
    tr = trainer.CodecTrainer(cfg, seed=0)
    V, D = tr.ema_w[0].shape
    x = mx.random.normal((3, 1, D))
    idx = mx.array([[2], [2], [5]])
    w0, c0 = np.array(tr.ema_w[0]), np.array(tr.ema_count[0])
    tr.ema_update([idx], [x])
    d, eps = cfg.codec.decay, cfg.codec.epsilon
    counts = np.zeros(V)
    counts[[2, 5]] = [2, 1]
    dw = np.zeros((V, D))
    dw[2] = np.array(x[0, 0] + x[1, 0])
    dw[5] = np.array(x[2, 0])
    c = d * c0 + (1 - d) * counts
    w = d * w0 + (1 - d) * dw
    n = c.sum()
    expected = w / ((c + eps) / (n + V * eps) * n)[:, None]
    np.testing.assert_allclose(np.array(tr.codec.codebooks[0]), expected, rtol=1e-5)
    np.testing.assert_allclose(np.array(tr.ema_count[0]), c, rtol=1e-6)


def test_dead_code_restart_reinitialises_unused_codes(tmp_path):
    cfg = tiny_cfg(tmp_path)
    tr = trainer.CodecTrainer(cfg, seed=0)
    D = tr.ema_w[0].shape[1]
    x = mx.random.normal((4, 1, D))
    tr.ema_update([mx.zeros((4, 1), dtype=mx.int32)], [x])
    cb = np.array(tr.codec.codebooks[0])
    assert np.isfinite(cb).all() and np.abs(cb).max() < 100
    rows = {r.tobytes() for r in np.array(x).reshape(4, D)}
    assert sum(r.tobytes() in rows for r in cb[1:]) == len(cb) - 1


def test_codec_overfits_a_tiny_corpus(tmp_path, patched_rows, monkeypatch):
    cfg = tiny_cfg(tmp_path, steps=400, lr=3e-3, lr_floor=3e-4, warmup=10)
    cfg.codec.dropout, cfg.codec.fine_drop_prob = 0.0, 0.0
    cfg.codec.codebook_sizes = [16, 16, 32, 64]
    cfg.train.log_every, cfg.train.eval_every = 400, 400
    rows = patched_rows[:8]
    monkeypatch.setattr(trainer, "load_rows", lambda cfg, log=print: (rows, rows.copy(), {}))
    meta = trainer.train_codec(cfg, log=lambda *_: None)
    assert meta["val"]["token_acc"] >= 0.95, meta["val"]
    codec = trainer.load_codec(cfg)
    assert trainer.codec_report(codec, rows)["token_acc"] == meta["val"]["token_acc"]


# ---------------------------------------------------------------- generator


def _tiny_models(tmp_path, seed=0):
    cfg = tiny_cfg(tmp_path)
    tr = trainer.CodecTrainer(cfg, seed=seed)
    tr.eval()
    gen = trainer.build_generator(cfg, tr.codec, seed + 1)
    gen.eval()
    return cfg, tr.codec, gen


def test_teacher_forced_forward_equals_per_level_inference_on_cpu(tmp_path):
    """Training logits equal the logits of the official sampler's level-by-level forwards."""
    cfg, codec, gen = _tiny_models(tmp_path)
    tokens = mx.array(tiny_rows(4).astype(np.int32))
    codes = [c.astype(mx.int32) for c in codec.encode(tokens)]
    prefix = mx.full((4, 1), ts.EOS_ID, dtype=mx.int32)
    tf = trainer.TeacherForced(gen)
    full = tf.logits(trainer.teacher_inputs(codec, codes[:-1]), prefix)
    m = sample.Official(cfg.task_spec(), codec, gen, None, None)
    per_level = sample.generate(m, prefix, forced=codes, keep_logits=True).level_logits
    assert len(per_level) == len(full) == len(cfg.codec.levels) - 1
    errors = [float(mx.abs(a - b).max()) for a, b in zip(full, per_level)]
    assert max(errors) < 1e-5, errors


def test_block_diagonal_mask_layout():
    m = trainer.block_diagonal_mask(1, [1, 2])
    expected = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 1], [1, 0, 1, 1]], dtype=bool)
    np.testing.assert_array_equal(m, expected)


def test_generator_loss_shapes_weights_and_corruption(tmp_path):
    _cfg, codec, gen = _tiny_models(tmp_path)
    tf = trainer.TeacherForced(gen)
    tokens = mx.array(tiny_rows(4).astype(np.int32))
    loss, ce = trainer.generator_loss(tf, codec, tokens, None, 0.0, 0.0)
    assert loss.shape == () and len(ce) == 3
    w = trainer.level_weights([2, 4, 8], 0.0)
    assert np.isclose(float(loss), sum(a * float(b) for a, b in zip(w, ce)), rtol=1e-6)
    assert np.allclose(trainer.level_weights([2, 4, 8], 1.0), [1 / 3] * 3)
    rates = trainer.corruption_rates(0.75, 0.25, 3)
    assert rates == [0.75, 0.5, 0.25]
    codes = [mx.zeros((64, 32), dtype=mx.int32)] * 3
    out = trainer.corrupt(codes, rates, [1000] * 3)
    frac = [float((c != 0).mean()) for c in out]
    assert frac[0] > frac[1] > frac[2] > 0.1


def test_generation_runs_one_generator_forward_per_level(tmp_path, monkeypatch):
    cfg, codec, gen = _tiny_models(tmp_path)
    calls = []
    original = type(gen).level_logits

    def counting(self, j, *args):
        calls.append(j)
        return original(self, j, *args)

    monkeypatch.setattr(type(gen), "level_logits", counting)
    unigram = mx.ones((8,)) / 8
    m = sample.Official(cfg.task_spec(), codec, gen, None, unigram)
    sample.generate(m, sample.unconditional_prefix(m, 3), seed=0, prefix_cache=False)
    assert calls == list(range(len(cfg.codec.levels) - 1))


# ---------------------------------------------------------------- AR baseline


def test_ar_cached_generation_matches_full_forward(tmp_path):
    cfg = tiny_cfg(tmp_path)
    m = ar_mod.build_ar(cfg, 0)
    m.eval()
    tokens, logits = m.generate(3, temperature=1.0, seed=4, keep_logits=True)
    full = m(tokens)
    err = float(mx.abs(mx.stack(logits, axis=1) - full).max())
    assert err < 1e-4, err
    assert tokens.shape == (3, 8)


# ---------------------------------------------------------------- end to end and recovery


def pause_at_two(message):
    if '"step": 2,' in message:
        os.kill(os.getpid(), signal.SIGUSR1)


def checkpoint_arrays(path):
    arrays, metadata = mx.load(str(path), return_metadata=True)
    return {k: np.array(v) for k, v in arrays.items()}, json.loads(metadata["training"])


@pytest.mark.parametrize("stage", ["codec", "generator", "ar"])
def test_pause_resume_matches_uninterrupted_training(tmp_path, patched_rows, stage):
    runs = {}
    for name in ("whole", "split"):
        cfg = tiny_cfg(tmp_path / name)
        if stage != "codec":
            trainer.train_codec(cfg, log=lambda *_: None)
        fn = {"codec": trainer.train_codec, "generator": trainer.train_generator, "ar": ar_mod.train_ar}[stage]
        if name == "split":
            with pytest.raises(TrainingPaused) as paused:
                fn(cfg, log=pause_at_two)
            assert paused.value.code == 75 and paused.value.step == 2
            fn(cfg, resume=True, log=lambda *_: None)
        else:
            fn(cfg, log=lambda *_: None)
        runs[name] = checkpoint_arrays(Path(cfg.out_dir) / f"{stage}-training.safetensors")
    (whole, wmeta), (split, smeta) = runs["whole"], runs["split"]
    assert wmeta["step"] == smeta["step"] == 6
    assert whole.keys() == split.keys()
    for key in whole:
        np.testing.assert_array_equal(whole[key], split[key], err_msg=key)


def test_trained_checkpoint_loads_through_official_sampler(tmp_path, patched_rows):
    cfg = tiny_cfg(tmp_path)
    trainer.train_codec(cfg, log=lambda *_: None)
    trainer.train_generator(cfg, log=lambda *_: None)
    out = trainer.export_dir(cfg)
    m = sample.load(out)
    assert isinstance(m.codec, Codec) and m.level0_unigram is not None
    assert np.isclose(float(m.level0_unigram.sum()), 1.0, atol=1e-5)
    g = sample.generate(m, sample.unconditional_prefix(m, 2), seed=1)
    assert g.tokens.shape == (2, 8)
    spec = json.loads((out / "spec.json").read_text())
    assert spec["training"]["codec"]["steps"] == 6 and spec["training"]["generator"]["steps"] == 6


def test_segmented_attention_equals_dense_block_diagonal_attention(tmp_path):
    _, _, gen = _tiny_models(tmp_path)
    tf = trainer.TeacherForced(gen)
    T = 1 + sum(tf.lengths)
    x = mx.random.normal((3, T, gen.spec.dim))
    prefix = mx.array([[ts.EOS_ID], [ts.PAD_ID], [5]])
    keep = prefix != gen.spec.prefix_pad_id
    dense_mask = mx.array(trainer.block_diagonal_mask(1, tf.lengths))[None, None]
    dense_mask = dense_mask & mx.concatenate([keep, mx.ones((3, T - 1), dtype=mx.bool_)], 1)[:, None, None, :]
    attn = gen.blocks[0].attn
    dense = attn(x, dense_mask, tf.rope)
    seg = trainer.segmented_attention(attn, x, 1, tf.lengths, tf.rope, keep)
    np.testing.assert_allclose(np.array(seg)[:, 1:], np.array(dense)[:, 1:], atol=1e-6)
    np.testing.assert_allclose(np.array(seg)[[0, 2], :1], np.array(dense)[[0, 2], :1], atol=1e-6)
