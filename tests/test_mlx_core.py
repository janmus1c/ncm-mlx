import copy

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx import nn

from ncm_mlx.config import Config
from ncm_mlx.generation.sampler import SampleOptions, generate, generate_codes
from ncm_mlx.models.generator import block_diagonal_mask, level_mask
from ncm_mlx.training.common import build_codec, build_generator, load_corpus, load_model, save_model
from ncm_mlx.training.generator import generator_loss, split_levels

pytestmark = pytest.mark.mlx


def small_config() -> Config:
    cfg = Config()
    cfg.data.n_train, cfg.data.n_val = 512, 128
    cfg.codec.dim, cfg.codec.enc_layers, cfg.codec.dec_layers = 64, 1, 1
    cfg.codec.codebook_sizes = [32] * len(cfg.codec.levels)
    cfg.generator.n_embd, cfg.generator.n_layer = 64, 2
    cfg.train.batch_size = 64
    return cfg


@pytest.fixture(scope="module")
def setup():
    mx.random.seed(0)
    cfg = small_config()
    corpus = load_corpus(cfg)
    codec = build_codec(cfg, corpus.tokenizer.vocab_size)
    gen = build_generator(cfg)
    codec.eval()
    gen.eval()
    return cfg, corpus, codec, gen


def test_runs_on_gpu():
    import sys

    assert mx.default_device() == mx.gpu
    if sys.platform == "darwin":
        assert mx.metal.is_available()
    else:
        assert mx.cuda.is_available()


def test_code_shapes_and_ranges(setup):
    cfg, corpus, codec, _ = setup
    codes = codec.encode(mx.array(corpus.train[:16]))
    assert [tuple(c.shape) for c in codes] == [(16, l) for l in cfg.codec.levels]
    for c, V in zip(codes, cfg.codec.codebook_sizes):
        assert c.dtype == mx.int32
        assert int(c.min()) >= 0 and int(c.max()) < V


def test_assignment_is_nearest_code(setup):
    _, _, codec, _ = setup
    x = mx.random.normal((5, 7, codec.vq.dim))
    idx = np.array(codec.vq.assign(x, 2))
    e = np.array(codec.vq.codebooks[2])
    d = ((np.array(x)[..., None, :] - e[None, None]) ** 2).sum(-1)
    assert (idx == d.argmin(-1)).all()


def test_reconstruction_from_codes_matches_quantize(setup):
    _, corpus, codec, _ = setup
    z = codec.encoder(mx.array(corpus.train[:8]))
    codes, z_hats, _ = codec.vq.quantize(z)
    rebuilt = codec.vq.codes_to_z_hats(codes)
    for a, b in zip(z_hats, rebuilt):
        assert np.allclose(np.array(a), np.array(b), atol=1e-5)


def test_teacher_inputs_match_sampling_path(setup):
    _, corpus, codec, _ = setup
    codes = codec.encode(mx.array(corpus.train[:4]))
    teacher = codec.vq.teacher_inputs(codes)
    z_hat = mx.zeros((4, codec.vq.seq_len, codec.vq.dim))
    for k in range(codec.vq.num_levels - 1):
        z_hat = z_hat + codec.vq.level_contribution(codes[k], k)
        step = codec.vq.next_level_input(z_hat, k)
        assert step.shape == (4, codec.vq.levels[k + 1], codec.vq.dim)
        assert np.allclose(np.array(step), np.array(teacher[k]), atol=1e-5)


def test_block_diagonal_mask():
    m = block_diagonal_mask(2, [1, 2])
    expected = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
    ], dtype=bool)
    assert (m == expected).all()


def _level_vs_full_max_error(codec, gen, tokens, prefix=None):
    codes = codec.encode(tokens)
    full = gen(codec.vq.teacher_inputs(codes), prefix)
    inputs = [gen.level0_input(tokens.shape[0])] + codec.vq.teacher_inputs(codes)
    return [float(mx.abs(gen.level_logits(inputs, k, prefix) - full[k]).max()) for k in range(len(inputs))]


@pytest.mark.parametrize("pattern", ["block_diagonal", "block_causal"])
@pytest.mark.parametrize("conditional", [False, True])
def test_level_forward_equals_teacher_forced_forward_exactly_on_cpu(setup, pattern, conditional):
    """Generating one level at a time computes the training-time logits (fp32 CPU, ~exact)."""
    cfg, corpus, codec, _ = setup
    cfg = copy.deepcopy(cfg)
    cfg.generator.attn_pattern = pattern
    prefix = None
    if conditional:
        cfg.generator.prefix_len, cfg.generator.prefix_vocab = 5, 11
        prefix = mx.random.randint(0, 11, (4, 5))
    mx.set_default_device(mx.cpu)
    try:
        gen = build_generator(cfg)
        gen.eval()
        errors = _level_vs_full_max_error(codec, gen, mx.array(corpus.train[:4]), prefix)
    finally:
        mx.set_default_device(mx.gpu)
    assert max(errors) < 1e-5, errors


def test_block_causal_mask():
    m = level_mask(1, [1, 2], "block_causal")
    expected = np.array([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ], dtype=bool)
    assert (m == expected).all()


def test_conditional_generator_requires_prefix(setup):
    cfg, _, codec, _ = setup
    cfg = copy.deepcopy(cfg)
    cfg.generator.prefix_len, cfg.generator.prefix_vocab = 3, 7
    gen = build_generator(cfg)
    with pytest.raises(ValueError):
        generate(gen, codec, 2, SampleOptions(seed=0))
    out = generate(gen, codec, 2, SampleOptions(seed=0), prefix_tokens=mx.zeros((2, 3), dtype=mx.int32))
    assert out.tokens.shape == (2, codec.vq.seq_len)


def test_level_forward_matches_on_gpu_within_matmul_precision(setup):
    """On Metal, fp32 matmuls are reduced-precision (see docs/EXPERIMENTS.md), so allow ~1e-3 relative."""
    _, corpus, codec, gen = setup
    errors = _level_vs_full_max_error(codec, gen, mx.array(corpus.train[:4]))
    assert max(errors) < 5e-3, errors


def test_generation_uses_one_forward_per_level(setup, monkeypatch):
    _, _, codec, gen = setup
    calls = []
    original = type(gen).level_logits

    def counting(self, inputs, k, prefix_tokens=None):
        calls.append((k, inputs[k].shape[1]))
        return original(self, inputs, k, prefix_tokens)

    monkeypatch.setattr(type(gen), "level_logits", counting)
    generate_codes(gen, codec, 3, SampleOptions(seed=0))
    assert calls == list(enumerate(codec.vq.levels))


def test_seeded_generation_is_deterministic(setup):
    _, _, codec, gen = setup
    a = generate(gen, codec, 8, SampleOptions(seed=7))
    b = generate(gen, codec, 8, SampleOptions(seed=7))
    c = generate(gen, codec, 8, SampleOptions(seed=8))
    assert all(np.array_equal(np.array(x), np.array(y)) for x, y in zip(a.codes, b.codes))
    assert not all(np.array_equal(np.array(x), np.array(y)) for x, y in zip(a.codes, c.codes))
    assert a.tokens.shape == (8, codec.vq.seq_len)


def test_codec_training_step_updates_weights_and_codebooks(setup):
    import random

    import mlx.optimizers as optim

    cfg, corpus, _, _ = setup
    mx.random.seed(1)
    codec = build_codec(cfg, corpus.tokenizer.vocab_size)
    codec.train()
    trainable = {k for k, _ in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(
        codec.trainable_parameters())}
    assert not any("codebooks" in k or "ema_" in k for k in trainable)
    before_w = np.array(codec.encoder.wte.weight)
    before_cb = np.array(codec.vq.codebooks[0])
    opt = optim.Adam(1e-3)
    grad_fn = nn.value_and_grad(codec, lambda m, t: m.loss(t, random.Random(0)))
    (loss, (ce, vq, codes, inputs)), grads = grad_fn(codec, mx.array(corpus.train[:64]))
    opt.update(codec, grads)
    codec.vq.ema_update(codes, inputs, mx.random.key(0))
    mx.eval(codec.parameters())
    assert np.isfinite(loss.item()) and np.isfinite(ce.item()) and np.isfinite(vq.item())
    assert not np.allclose(before_w, np.array(codec.encoder.wte.weight))
    assert not np.allclose(before_cb, np.array(codec.vq.codebooks[0]))


def test_generator_loss_decreases_on_fixed_batch(setup):
    import mlx.optimizers as optim

    cfg, corpus, codec, _ = setup
    mx.random.seed(2)
    gen = build_generator(cfg)
    gen.train()
    flat = mx.concatenate(codec.encode(mx.array(corpus.train[:64])), axis=1)
    grad_fn = nn.value_and_grad(gen, lambda m, f: generator_loss(m, codec, f))
    opt = optim.Adam(1e-3)
    losses = []
    for _ in range(30):
        (loss, _), grads = grad_fn(gen, flat)
        opt.update(gen, grads)
        mx.eval(gen.parameters(), loss)
        losses.append(loss.item())
    assert all(np.isfinite(losses))
    assert losses[-1] < 0.7 * losses[0]
    assert [c.shape[1] for c in split_levels(flat, codec.vq.levels)] == codec.vq.levels


def test_save_and_reload_from_disk(setup, tmp_path):
    cfg, corpus, codec, gen = setup
    save_model(codec, tmp_path / "codec.safetensors", {"stage": "codec"})
    save_model(gen, tmp_path / "generator.safetensors", {"stage": "generator"})
    mx.random.seed(123)
    codec2 = load_model(build_codec(cfg, corpus.tokenizer.vocab_size), tmp_path / "codec.safetensors")
    gen2 = load_model(build_generator(copy.deepcopy(cfg)), tmp_path / "generator.safetensors")
    a = generate(gen, codec, 4, SampleOptions(seed=3))
    b = generate(gen2, codec2, 4, SampleOptions(seed=3))
    assert np.array_equal(np.array(a.tokens), np.array(b.tokens))
    t = mx.array(corpus.val[:8])
    assert all(np.array_equal(np.array(x), np.array(y)) for x, y in zip(codec.encode(t), codec2.encode(t)))


def test_ar_baseline_cached_step_matches_full_forward(setup):
    from ncm_mlx.baselines.ar import build_ar

    cfg, corpus, _, _ = setup
    model = build_ar(cfg, corpus.tokenizer.vocab_size)
    model.eval()
    tokens = mx.array(corpus.train[:2])
    mx.set_default_device(mx.cpu)
    try:
        full = model(tokens)
        cache = [None] * len(model.blocks)
        inp = mx.concatenate([mx.full((2, 1), model.bos, dtype=tokens.dtype), tokens[:, :-1]], axis=1)
        steps = [model._step(model.wte(inp[:, p : p + 1]), p, cache) for p in range(cfg.data.seq_len)]
        err = float(mx.abs(mx.stack(steps, axis=1) - full).max())
    finally:
        mx.set_default_device(mx.gpu)
    assert err < 1e-4, err
    assert model.generate(3, 1.0, seed=0).shape == (3, cfg.data.seq_len)


def test_conditional_ar_prefill_and_steps_match_full_forward(setup):
    from ncm_mlx.baselines.ar import CharAR

    mx.set_default_device(mx.cpu)
    try:
        model = CharAR(10, 12, 32, 2, 4, 0.0, prefix_len=6, prefix_vocab=10)
        model.eval()
        tokens = mx.random.randint(0, 10, (3, 12))
        prefix = mx.random.randint(0, 10, (3, 6))
        full = model(tokens, prefix)
        cache = [None] * len(model.blocks)
        bos = mx.full((3, 1), model.bos, dtype=tokens.dtype)
        steps = [model._step(model._embed(bos, prefix), 0, cache, model._mask[:7, :7])]
        steps += [model._step(model.wte(tokens[:, p - 1 : p]), 6 + p, cache) for p in range(1, 12)]
        err = float(mx.abs(mx.stack(steps, axis=1) - full).max())
    finally:
        mx.set_default_device(mx.gpu)
    assert err < 1e-4, err
    assert model.generate(3, 1.0, seed=0, prefix=prefix).shape == (3, 12)



def test_learned_level0_sampling_starts_as_area_pooling_and_trains(setup):
    _, corpus, _, _ = setup
    cfg = small_config()
    cfg.codec.learned_l0 = True
    cfg.codec.pre_quant_groups = 8
    with mx.stream(mx.cpu):
        codec = build_codec(cfg, corpus.tokenizer.vocab_size)
        z = mx.random.normal((4, cfg.data.seq_len, cfg.codec.dim))
        pooled = codec.vq.down(z, 0)
        assert pooled.shape == (4, cfg.codec.levels[0], cfg.codec.dim)
        assert np.allclose(np.array(pooled), np.array(z.mean(axis=1, keepdims=True)), atol=1e-5)
        repeated = codec.vq.up(pooled, 0)
        assert np.allclose(np.array(repeated), np.array(mx.broadcast_to(pooled, z.shape)), atol=1e-6)

        tokens = mx.array(corpus.train[:16])
        _, grads = nn.value_and_grad(codec, lambda t: codec.loss(t)[0])(tokens)
        flat = dict(nn.utils.tree_flatten(grads))
        assert float(mx.abs(flat["vq.l0_down.0.weight"]).sum()) > 0
        assert float(mx.abs(flat["vq.l0_up.0.weight"]).sum()) > 0


def test_corruption_schedule_and_rate():
    from ncm_mlx.training.generator import corrupt, corruption_schedule

    assert corruption_schedule(0.3, 0.1, 4) == pytest.approx([0.3, 0.2, 0.1])
    assert corruption_schedule(0.2, 0.0, 2) == [0.2]
    mx.random.seed(0)
    codes = [mx.zeros((64, 512), dtype=mx.int32), mx.zeros((64, 512), dtype=mx.int32)]
    out = corrupt(codes, [0.5], [1024, 1024])
    changed = float(mx.mean(out[0] != 0))
    assert 0.45 < changed < 0.55
    assert bool(mx.all(out[1] == 0))


def _three_training_losses(cfg, codec, flat, compiled: bool) -> list[float]:
    from functools import partial

    import mlx.optimizers as optim

    mx.random.seed(1)
    gen = build_generator(cfg)
    gen.eval()
    opt = optim.SGD(1e-2)
    grad_fn = nn.value_and_grad(gen, lambda m, f: generator_loss(m, codec, f)[0])

    def step(f):
        loss, g = grad_fn(gen, f)
        opt.update(gen, g)
        return loss

    if compiled:
        state = [gen.state, opt.state]
        step = partial(mx.compile, inputs=state, outputs=state)(step)
    losses = []
    for _ in range(3):
        loss = step(flat)
        mx.eval(gen.parameters(), opt.state, loss)
        losses.append(loss.item())
    return losses


def test_compiled_training_step_matches_eager(setup):
    from ncm_mlx.training.generator import encode_dataset

    cfg, corpus, codec, _ = setup
    flat = mx.array(encode_dataset(codec, corpus.train[:32]))
    eager = _three_training_losses(cfg, codec, flat, compiled=False)
    assert np.allclose(eager, _three_training_losses(cfg, codec, flat, compiled=True), rtol=1e-3)


def test_segmented_attention_equals_block_diagonal_mask():
    from ncm_mlx.models.blocks import Attention

    levels, P = [1, 2, 4, 8], 5
    with mx.stream(mx.cpu):
        mx.random.seed(3)
        attn = Attention(32, 4, 0.0)
        x = mx.random.normal((2, P + sum(levels), 32))
        masked = attn(x, mx.array(block_diagonal_mask(P, levels)))
        bounds = np.cumsum([P, *levels])
        segmented = attn(x, segments=(P, list(zip(bounds[:-1].tolist(), bounds[1:].tolist()))))
        assert np.allclose(np.array(masked), np.array(segmented), atol=1e-5)


def test_model_corruption_uses_generator_samples(setup):
    from ncm_mlx.training.generator import corrupt, encode_dataset, model_samples

    _, corpus, codec, gen = setup
    flat = mx.array(encode_dataset(codec, corpus.train[:16]))
    codes = split_levels(flat, codec.vq.levels)
    own = model_samples(gen, codec, codes, None)
    assert [o.shape for o in own] == [c.shape for c in codes]
    full = corrupt(codes, [1.0] * (len(codes) - 1), codec.vq.codebook_sizes, own)
    assert all(bool(mx.all(f == o)) for f, o in zip(full[:-1], own[:-1]))
    assert bool(mx.all(full[-1] == codes[-1]))
    loss, _ = generator_loss(gen, codec, flat, [0.5] * (len(codes) - 1), "model")
    assert np.isfinite(loss.item())


def test_training_stops_at_max_minutes(setup, tmp_path):
    from ncm_mlx.training.generator import train_generator

    cfg, corpus, codec, _ = setup
    cfg = copy.deepcopy(cfg)
    cfg.out_dir = str(tmp_path)
    cfg.train.max_minutes = 1e-9
    cfg.train.log_every = 1000
    _, meta = train_generator(cfg, codec, corpus, steps=50, log=lambda *_: None)
    assert meta["steps"] == 1 and meta["best_step"] == 1
