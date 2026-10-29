"""Released-checkpoint port: reader, conversion rules, tokenizer and parity with upstream.

Parity tests compare against ``tests/fixtures/official_parity.json``, written by
``scripts/official_parity.py`` from upstream PyTorch on CPU. They skip when the
converted official weights are absent (``python -m ncm_mlx official fetch`` and
``convert``).
"""

from __future__ import annotations

import json
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest

from ncm_mlx.official import convert, spec, torch_reader
from ncm_mlx.official.bpe import pre_tokenize

FIXTURES = Path(__file__).parent / "fixtures"
PARITY = json.loads((FIXTURES / "official_parity.json").read_text())
ARTIFACTS = Path("artifacts/official")
PARITY_TASKS = [k for k in PARITY if not k.startswith("_")]


def test_reader_rebuilds_views_dtypes_and_nesting():
    ck = torch_reader.load(FIXTURES / "tiny_torch_checkpoint.pt")
    base = np.arange(24, dtype=np.float32).reshape(4, 6)
    m = ck["model"]
    assert list(m) == ["w", "t", "slice", "ids", "scalar"]
    np.testing.assert_array_equal(m["w"], base)
    np.testing.assert_array_equal(m["t"], base.T)
    np.testing.assert_array_equal(m["slice"], base[1:3, 2:5])
    assert m["ids"].dtype == np.int64 and m["ids"].tolist() == [-3, 4, 11, 18, 25]
    assert m["scalar"].shape == () and float(m["scalar"]) == 2.5
    np.testing.assert_allclose(ck["inference"]["p"], np.full(3, 1 / 3, dtype=np.float32))


class _Payload:
    def __reduce__(self):
        return (os.system, ("echo unsafe",))


def test_reader_refuses_foreign_globals(tmp_path):
    path = tmp_path / "bad.pt"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("bad/data.pkl", pickle.dumps({"x": _Payload()}, protocol=2))
    with pytest.raises(pickle.UnpicklingError, match="refusing"):
        torch_reader.load(path)


def test_reader_refuses_non_checkpoint_zip(tmp_path):
    path = tmp_path / "x.pt"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("a.txt", "no")
    with pytest.raises(ValueError):
        torch_reader.load(path)


def test_every_upstream_name_needs_exactly_one_rule():
    state = {"encoder.wte.weight": np.zeros((3, 2)), "vq.ema_w_0": np.zeros((4, 2))}
    out, dropped = convert.map_state(state, convert.CODEC_RULES)
    assert list(out) == ["tok.weight"] and dropped == ["vq.ema_w_0"]
    with pytest.raises(ValueError, match="matched 0"):
        convert.map_state({"encoder.unknown": np.zeros(1)}, convert.CODEC_RULES)


def test_generator_block_names_keep_layer_index():
    state = {"transformer.blocks.11.attn.q_norm.weight": np.ones(4),
             "transformer.blocks.3.mlp.c_proj.bias": np.ones(4)}
    out, _ = convert.map_state(state, convert.GENERATOR_RULES)
    assert set(out) == {"blocks.11.attn.q_norm.weight", "blocks.3.out.bias"}


def test_training_mask_is_block_diagonal_with_prefix():
    m = convert.expected_training_mask(1, [1, 2])[0, 0]
    allowed = np.isfinite(m)
    assert allowed.tolist() == [[True, False, False, False],
                                [True, True, False, False],
                                [True, False, True, True],
                                [True, False, True, True]]


def test_pre_tokenizer_matches_upstream_pattern():
    for text, expected in PARITY["tinystories"]["pretokenize"]:
        assert pre_tokenize(text) == expected, text


@pytest.mark.parametrize("name", list(spec.TASKS))
def test_spec_matches_downloaded_config(name):
    yaml = pytest.importorskip("yaml")
    task = spec.get(name)
    path = ARTIFACTS / task.remote_dir / "config.yaml"
    if not path.exists():
        pytest.skip("official config not downloaded")
    cfg = yaml.safe_load(path.read_text())
    vq, mdl = cfg["tokenizer"]["vqvae"], cfg["model"]
    qc = vq["vector_quantizer_config"]
    c, g = task.codec, task.generator
    assert (vq["in_vocab_size"], vq["context_length"], vq["embed_dim"], vq["n_layers"]) == \
        (c.vocab_size, c.seq_len, c.dim, c.n_head)
    assert tuple(qc["levels"]["manual_levels"]) == c.levels and tuple(vq["vqvae_vocab_size"]) == c.codebook_sizes
    assert qc["quant_resi"] == {"enabled": True, "ratio": c.phi_ratio, "share_mode": 0, "learnable_ratio": False}
    assert qc["learned_l1_sampling"] and qc["learned_l1_cascade"] and not qc["learned_all_sampling"]
    assert not qc.get("learned_l1_cascade_norm") and not qc.get("learned_l1_cascade_norm_final")
    assert vq["pre_quant_groupnorm"] == c.group_norm_groups and vq.get("pre_quant_norm_type", "group") == "group"
    assert vq.get("enc_layers", 0) == 0 and vq.get("dec_layers", 1) == 1 and vq["compression_factor"] == 1
    assert (mdl["n_layer"], mdl["n_head"], mdl["n_embd"], mdl["prefix_len"], mdl["rope_base"]) == \
        (g.n_layer, g.n_head, g.dim, g.prefix_len, g.rope_base)
    assert mdl["attn_pattern"] == "block_diagonal" and mdl["use_rope"] and mdl["use_qk_norm"]
    assert mdl["raw_token_prefix"] and mdl["use_level_encoding"]
    assert bool(mdl.get("query_level0")) == g.query_level0
    assert bool(mdl.get("tie_heads_to_codebook")) == g.tied_heads
    assert mdl.get("prefix_pad_token_id") == g.prefix_pad_id
    assert not mdl.get("predict_level0") and not (mdl.get("ctx_encoder") or {}).get("enabled")
    assert cfg["generation"]["temperature"] == task.temperature


@pytest.fixture
def cpu():
    mx = pytest.importorskip("mlx.core")
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _loaded(name):
    task = spec.get(name)
    d = ARTIFACTS / task.remote_dir / "mlx"
    if not (d / "spec.json").exists():
        pytest.skip("converted official weights absent")
    from ncm_mlx.official import sample

    return sample.load(d), PARITY[name]


@pytest.mark.mlx
def test_strided_resampling_layouts_match_convolution_definitions(cpu):
    import mlx.core as mx

    from ncm_mlx.official.model import StridedDown, StridedUp

    rng = np.random.default_rng(0)
    C, s, T = 3, 4, 2
    x = rng.standard_normal((1, T * s, C)).astype(np.float32)
    w = rng.standard_normal((C, C, s)).astype(np.float32)
    down = StridedDown(C, s)
    down.weight = mx.array(convert._strided_down(w))
    ref = np.einsum("ock,tkc->to", w, x[0].reshape(T, s, C))
    np.testing.assert_allclose(np.array(down(mx.array(x)))[0], ref, rtol=1e-5, atol=1e-5)
    up = StridedUp(C, s)
    up.weight = mx.array(convert._strided_up(w))
    y = rng.standard_normal((1, T, C)).astype(np.float32)
    ref = np.einsum("tc,cok->tko", y[0], w).reshape(T * s, C)
    np.testing.assert_allclose(np.array(up(mx.array(y)))[0], ref, rtol=1e-5, atol=1e-5)


@pytest.mark.mlx
def test_phi_index_is_one_per_level():
    from ncm_mlx.official.model import phi_index

    assert [phi_index(i, 12) for i in range(12)] == list(range(12))


@pytest.mark.mlx
def test_bpe_ids_match_upstream(cpu):
    m, ref = _loaded("tinystories")
    for text, ids in ref["bpe_ids"]:
        assert m.bpe.encode(text) == ids, text


def _forced(m, ref, prefix):
    import mlx.core as mx

    from ncm_mlx.official import sample

    codes = [mx.array(np.array(c, dtype=np.int32)) for c in ref["codec"]["codes"]]
    g = sample.generate(m, mx.array(np.array(prefix, dtype=np.int32)), temperature=0.0,
                        keep_logits=True, forced=codes)
    return codes, g.level_logits


@pytest.mark.mlx
@pytest.mark.parametrize("name", PARITY_TASKS)
def test_forced_level_logits_match_upstream_on_cpu(name, cpu):
    m, ref = _loaded(name)
    prefix = ([[m.task.eos_id]] * 2 if name == "tinystories" else ref["boards"]["puzzles"][:2])
    _, logits = _forced(m, ref, prefix)
    assert len(logits) == len(ref["forced_logits"]) == len(m.codec.levels) - m.gen.first
    for lm, lr in zip(logits, ref["forced_logits"]):
        row = np.array(lm[0], dtype=np.float64)
        scale = max(1.0, float(np.abs(row).max()))
        # Float32 rounding: on these checkpoints upstream fp32 itself differs from fp64 by ~1e-4 of the scale.
        np.testing.assert_allclose(row[:2, :16], np.array(lr["row0_head"]), rtol=0, atol=5e-4 * scale)
        assert row.argmax(-1).tolist() == lr["row0_argmax"]
        assert abs(row.sum() - lr["row0_sum"]) <= 1e-5 * max(1.0, lr["row0_abs_sum"])


@pytest.mark.mlx
@pytest.mark.parametrize("name", PARITY_TASKS)
def test_codec_matches_upstream_on_cpu(name, cpu):
    import mlx.core as mx

    m, ref = _loaded(name)
    tokens = mx.array(np.array(ref["codec"]["tokens"], dtype=np.int32))
    codes = [np.array(c) for c in m.codec.encode(tokens)]
    upstream = [np.array(r) for r in ref["codec"]["codes"]]
    # Greedy nearest-code search: before an item's first differing level the codes must be identical, and
    # where they first differ the two candidates must be a float32 near-tie under upstream's residual.
    rest = m.codec.embed(tokens)
    open_items = set(range(len(upstream[0])))
    for k, cb in enumerate(m.codec.codebooks):
        x = m.codec._down(rest, k) if k != len(m.codec.levels) - 1 else rest
        d = np.array((x * x).sum(-1, keepdims=True) + (cb * cb).sum(-1) - 2 * (x @ cb.T))
        for b in sorted(open_items):
            diff = np.nonzero(codes[k][b] != upstream[k][b])[0]
            for pos in diff:
                gap = abs(d[b, pos, codes[k][b, pos]] - d[b, pos, upstream[k][b, pos]])
                assert gap <= 1e-3 * abs(d[b, pos]).max(), (k, b, pos, gap)
            if len(diff):
                open_items.discard(b)
        rest = rest - m.codec.level_contribution(mx.array(upstream[k].astype(np.int32)), k)
    if name != "tinystories":
        assert all(np.array_equal(c, r) for c, r in zip(codes, upstream))
    forced = [mx.array(np.array(c, dtype=np.int32)) for c in ref["codec"]["codes"]]
    logits = np.array(m.codec.decode_logits(m.codec.cumulative(forced)[-1]))
    np.testing.assert_allclose(logits[0, :4, :16], np.array(ref["codec"]["decoder_logits_row0"]), atol=1e-3)


@pytest.mark.mlx
@pytest.mark.parametrize("name", PARITY_TASKS)
def test_greedy_generation_matches_upstream_exactly_on_cpu(name, cpu):
    import mlx.core as mx

    from ncm_mlx.official import sample

    m, ref = _loaded(name)
    gr = ref["greedy"]
    level0 = None if gr["level0"] is None else mx.array(np.array(gr["level0"], dtype=np.int32))
    g = sample.generate(m, mx.array(np.array(gr["prefix"], dtype=np.int32)), temperature=0.0, level0=level0)
    codes = np.concatenate([np.array(c) for c in g.codes], axis=1)
    assert codes.tolist() == gr["codes"]
    assert [m.decode(r) for r in np.array(g.tokens)] == gr["texts"]


@pytest.mark.mlx
def test_generation_uses_one_generator_forward_per_level(cpu):
    import mlx.core as mx

    from ncm_mlx.official import sample

    m, _ = _loaded("sudoku-easy")
    calls = []
    original = m.gen.level_logits
    m.gen.level_logits = lambda *a, **k: calls.append(a[0]) or original(*a, **k)
    try:
        sample.generate(m, mx.array(np.array(PARITY["sudoku-easy"]["boards"]["puzzles"][:1], dtype=np.int32)),
                        temperature=0.0)
    finally:
        del m.gen.level_logits
    assert calls == list(range(len(m.codec.levels)))


def test_fixture_records_reference_versions():
    meta = PARITY["_meta"]
    assert meta["checkpoint_revision"] == spec.REVISION
    assert meta["upstream_code_revision"] == spec.UPSTREAM_CODE_REVISION
