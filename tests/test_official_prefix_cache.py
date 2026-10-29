"""Prefix K/V cache and compiled per-level step of the official-checkpoint sampler.

The tiny-config tests run without downloaded weights. Weight-gated tests use the
converted release checkpoints under ``artifacts/official`` (and the published
Sudoku boards under ``artifacts/upstream-sudoku``) and skip when absent.
Exactness checks run on ``mx.cpu``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ncm_mlx.official import spec

ARTIFACTS = Path("artifacts/official")
BOARDS = Path("artifacts/upstream-sudoku")
LOGIT_RTOL = 2e-5


@pytest.fixture
def cpu():
    mx = pytest.importorskip("mlx.core")
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def tiny(query_level0: bool, tied: bool, pad_id: int | None, seed: int = 0):
    import mlx.core as mx
    from mlx.utils import tree_map

    from ncm_mlx.official.model import Codec, Generator
    from ncm_mlx.official.sample import Official

    codec_spec = spec.CodecSpec(vocab_size=12, seq_len=16, dim=16, n_head=2, levels=(1, 2, 4, 8, 16),
                                codebook_sizes=(8,) * 5, l0_strides=(16,))
    gen_spec = spec.GeneratorSpec(n_layer=3, n_head=2, dim=16, prefix_len=5, rope_base=10000.0,
                                  query_level0=query_level0, tied_heads=tied, prefix_pad_id=pad_id)
    task = spec.TaskSpec(name="tiny", remote_dir="tiny", tokenizer="sudoku", codec=codec_spec,
                         generator=gen_spec, eos_id=0, pad_id=1)
    mx.random.seed(seed)
    codec = Codec(codec_spec)
    gen = Generator(gen_spec, codec)
    for mod in (codec, gen):
        mod.update(tree_map(lambda a: mx.random.normal(a.shape) * 0.5 if a.ndim else mx.array(0.3),
                            mod.parameters()))
    mx.eval(codec.parameters(), gen.parameters())
    unigram = None if query_level0 else mx.softmax(mx.random.normal((8,)), axis=-1)
    return Official(task, codec, gen, None, unigram)


CONFIGS = [(True, False, None), (False, True, 1), (True, True, 1), (False, False, None)]


def prefixes(m, b, seed=1):
    import mlx.core as mx

    rng = np.random.default_rng(seed)
    p = rng.integers(0, 12, size=(b, m.task.generator.prefix_len)).astype(np.int32)
    pad = m.task.generator.prefix_pad_id
    if pad is not None:
        p[:, -2:] = pad
        p[0, 1] = pad
    return mx.array(p)


def assert_logits_close(la, lb):
    la, lb = np.array(la), np.array(lb)
    np.testing.assert_allclose(lb, la, rtol=0, atol=LOGIT_RTOL * max(1.0, float(np.abs(la).max())))


@pytest.mark.mlx
@pytest.mark.parametrize("query_level0,tied,pad", CONFIGS)
@pytest.mark.parametrize("batch", [1, 3, 8])
def test_cached_logits_match_uncached_at_every_level(cpu, query_level0, tied, pad, batch):
    from ncm_mlx.official import sample

    m = tiny(query_level0, tied, pad)
    p = prefixes(m, batch)
    a = sample.generate(m, p, temperature=1.0, seed=3, keep_logits=True, prefix_cache=False)
    b = sample.generate(m, p, temperature=1.0, seed=3, keep_logits=True, prefix_cache=True)
    assert len(a.level_logits) == len(b.level_logits) == len(m.codec.levels) - m.gen.first
    for la, lb in zip(a.level_logits, b.level_logits):
        assert_logits_close(la, lb)
    for ca, cb in zip(a.codes, b.codes):
        assert np.array_equal(np.array(ca), np.array(cb))
    assert np.array_equal(np.array(a.tokens), np.array(b.tokens))


@pytest.mark.mlx
@pytest.mark.parametrize("query_level0,tied,pad", CONFIGS)
def test_forced_cached_logits_match_uncached(cpu, query_level0, tied, pad):
    """Teacher forcing removes any dependence on sampled codes."""
    import mlx.core as mx

    from ncm_mlx.official import sample

    m = tiny(query_level0, tied, pad)
    p = prefixes(m, 4)
    rng = np.random.default_rng(5)
    forced = [mx.array(rng.integers(0, 8, size=(4, n)).astype(np.int32)) for n in m.codec.levels]
    a = sample.generate(m, p, keep_logits=True, forced=forced, prefix_cache=False)
    b = sample.generate(m, p, keep_logits=True, forced=forced, prefix_cache=True)
    for la, lb in zip(a.level_logits, b.level_logits):
        assert_logits_close(la, lb)


@pytest.mark.mlx
def test_cache_equals_prefix_rows_of_full_forward(cpu):
    """Cached prefix K/V are the keys and values the full forward computes for its prefix rows."""
    import mlx.core as mx

    m = tiny(True, False, None)
    gen = m.gen
    p = prefixes(m, 2)
    cache = gen.build_prefix_cache(p)
    h = mx.concatenate([gen.prefix_embedding(p), mx.zeros((2, 1, gen.spec.dim))], axis=1)
    mask = gen._mask(p, 1)
    P = gen.spec.prefix_len
    for blk, (pk, pv) in zip(gen.blocks, cache.kv):
        _, k, v = blk.attn.qkv_heads(blk.ln1(h), gen._rope[0])
        np.testing.assert_allclose(np.array(pk), np.array(k[:, :, :P]), atol=1e-6)
        np.testing.assert_allclose(np.array(pv), np.array(v[:, :, :P]), atol=1e-6)
        h = blk(h, mask, gen._rope[0])


@pytest.mark.mlx
@pytest.mark.parametrize("query_level0,tied,pad", CONFIGS)
def test_repeated_calls_do_not_leak_state(cpu, query_level0, tied, pad):
    from ncm_mlx.official import sample

    m = tiny(query_level0, tied, pad)
    pa, pb = prefixes(m, 3, seed=1), prefixes(m, 3, seed=2)
    first = np.array(sample.generate(m, pa, temperature=0.0).tokens)
    other = np.array(sample.generate(m, pb, temperature=0.0).tokens)
    again = np.array(sample.generate(m, pa, temperature=0.0).tokens)
    assert np.array_equal(first, again)
    ref = np.array(sample.generate(m, pb, temperature=0.0, prefix_cache=False).tokens)
    assert np.array_equal(other, ref)
    assert not [k for k in vars(m.gen) if "cache" in k.lower()]


@pytest.mark.mlx
def test_batch_permutation_permutes_outputs(cpu):
    import mlx.core as mx

    from ncm_mlx.official import sample

    m = tiny(True, False, None)
    p = prefixes(m, 6)
    perm = np.array([3, 0, 5, 1, 4, 2])
    a = np.array(sample.generate(m, p, temperature=0.0).tokens)
    b = np.array(sample.generate(m, p[mx.array(perm)], temperature=0.0).tokens)
    assert np.array_equal(a[perm], b)


@pytest.mark.mlx
def test_cached_generation_builds_prefix_once_and_runs_one_forward_per_level(cpu, monkeypatch):
    """Cached path: prefix built once, ``level_logits_cached`` once per level, no full per-level forward."""
    from ncm_mlx.official import sample
    from ncm_mlx.official.model import Generator

    m = tiny(True, False, None)
    levels, builds, full = [], [], []
    orig_level, orig_build, orig_full = (Generator.level_logits_cached, Generator.build_prefix_cache,
                                         Generator.level_logits)
    monkeypatch.setattr(Generator, "level_logits_cached",
                        lambda self, j, *a: levels.append(j) or orig_level(self, j, *a))
    monkeypatch.setattr(Generator, "build_prefix_cache", lambda self, *a: builds.append(1) or orig_build(self, *a))
    monkeypatch.setattr(Generator, "level_logits", lambda self, *a, **k: full.append(1) or orig_full(self, *a, **k))
    sample.generate(m, prefixes(m, 2), temperature=0.0)
    assert levels == list(range(len(m.codec.levels)))
    assert builds == [1]
    assert full == []


@pytest.mark.mlx
def test_sampled_codes_identical_with_and_without_cache(cpu):
    """Cache construction draws no random numbers: same seed, same sampled codes."""
    from ncm_mlx.official import sample

    m = tiny(False, True, 1)
    p = prefixes(m, 4)
    for seed in range(4):
        a = sample.generate(m, p, temperature=0.7, top_k=4, seed=seed, prefix_cache=False)
        b = sample.generate(m, p, temperature=0.7, top_k=4, seed=seed)
        assert [np.array(c).tolist() for c in a.codes] == [np.array(c).tolist() for c in b.codes]


@pytest.mark.mlx
@pytest.mark.parametrize("query_level0,tied,pad", CONFIGS)
def test_compiled_matches_eager_cached(cpu, query_level0, tied, pad):
    from ncm_mlx.official import sample

    m = tiny(query_level0, tied, pad)
    for batch in (2, 5, 2):
        p = prefixes(m, batch)
        a = sample.generate(m, p, temperature=1.0, seed=2, keep_logits=True)
        b = sample.generate(m, p, temperature=1.0, seed=2, keep_logits=True, compiled=True)
        for la, lb in zip(a.level_logits, b.level_logits):
            assert_logits_close(la, lb)
        assert np.array_equal(np.array(a.tokens), np.array(b.tokens))


@pytest.mark.mlx
def test_compiled_requires_prefix_cache(cpu):
    from ncm_mlx.official import sample

    m = tiny(True, False, None)
    with pytest.raises(ValueError):
        sample.generate(m, prefixes(m, 1), compiled=True, prefix_cache=False)


def _loaded(name):
    task = spec.get(name)
    d = ARTIFACTS / task.remote_dir / "mlx"
    if not (d / "spec.json").exists():
        pytest.skip("converted official weights absent")
    from ncm_mlx.official import sample

    return sample.load(d)


def _check_checkpoint(m, p, temperature=0.0):
    from ncm_mlx.official import sample

    a = sample.generate(m, p, temperature=temperature, keep_logits=True, prefix_cache=False)
    b = sample.generate(m, p, temperature=temperature, keep_logits=True)
    c = sample.generate(m, p, temperature=temperature, keep_logits=True, compiled=True)
    for other in (b, c):
        for la, lo in zip(a.level_logits, other.level_logits):
            assert_logits_close(la, lo)
        assert np.array_equal(np.array(a.tokens), np.array(other.tokens))
        for ca, co in zip(a.codes, other.codes):
            assert np.array_equal(np.array(ca), np.array(co))


@pytest.mark.mlx
def test_tinystories_checkpoint_cached_matches_uncached_on_8_samples(cpu):
    from ncm_mlx.official import sample

    m = _loaded("tinystories")
    _check_checkpoint(m, sample.unconditional_prefix(m, 8), temperature=0.0)


@pytest.mark.mlx
@pytest.mark.parametrize("name", ["sudoku-easy", "sudoku-medium", "sudoku-hard"])
def test_sudoku_checkpoints_cached_matches_uncached_on_64_boards(cpu, name):
    import mlx.core as mx

    from ncm_mlx.text import upstream_sudoku

    m = _loaded(name)
    d = BOARDS / name.split("-", 1)[1] / "puzzles.pickle"
    if not d.exists():
        pytest.skip("published Sudoku boards absent (scripts/fetch_upstream_sudoku.sh)")
    puzzles = upstream_sudoku.load_tokens(d)[:64]
    _check_checkpoint(m, mx.array(np.array(puzzles, dtype=np.int32)))
