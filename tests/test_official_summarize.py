"""XSum summarization: configuration contract, strict conversion, readout, guidance and parity with upstream.

Everything outside the last block runs on CPU without downloaded weights.
``official_xsum_parity.json`` holds upstream PyTorch's output for fixed inputs, written by
``scripts/official_xsum_parity.py``; the weight-gated tests replay those inputs through the MLX port
and skip until ``official fetch`` and ``convert`` have produced ``artifacts/official/xsum/mlx``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from test_official_prefix_cache import prefixes, tiny

from ncm_mlx.official import convert, spec, summarize
from ncm_mlx.official.cli import register

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = json.loads((FIXTURES / "official_xsum_config.json").read_text())
ARTIFACTS = Path("artifacts/official")
TASK = spec.get("xsum")


@pytest.fixture
def cpu():
    mx = pytest.importorskip("mlx.core")
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_generator_and_decoding_spec_match_the_pinned_config():
    m, g, d = CONFIG["model"], TASK.generator, TASK.decoding
    assert TASK.files["config.yaml"] == CONFIG["config_sha256"]
    assert (g.n_layer, g.n_head, g.dim, g.prefix_len, g.rope_base) == \
        (m["n_layer"], m["n_head"], m["n_embd"], m["prefix_len"], m["rope_base"])
    assert (g.query_level0, g.tied_heads, g.prefix_pad_id) == (m["query_level0"], m["tie_heads_to_codebook"],
                                                              m["prefix_pad_token_id"])
    assert m["raw_token_prefix"] and m["attn_pattern"] == "block_diagonal" and m["use_qk_norm"] and m["use_rope"]
    assert not m["predict_level0"] and not m["ctx_encoder"]["enabled"] and m["tie_heads_learn_scale"]
    assert m["bias"] and m["use_level_encoding"] and m["use_positional_encoding"]
    assert len(m["vocab_size"]) == len(TASK.codec.levels) == 9
    gen = CONFIG["generation"]
    penalty, window, decay = (float(v) for v in gen["adjacency_penalty"].split(","))
    assert (d.temperature, d.top_k, d.top_p, d.cfg_scale) == (gen["temperature"], gen["top_k"], gen["top_p"],
                                                              gen["cfg_scale"])
    assert (d.frequency_penalty, d.frequency_deficit) == (gen["frequency_penalty"], gen["frequency_deficit"])
    assert (d.adjacency_penalty, d.adjacency_window, d.adjacency_decay) == (penalty, int(window), decay)
    assert gen["frequency_allowance"] == "generator.pt:inference.token_unigram"
    assert (CONFIG["eval"]["split"], CONFIG["eval"]["n_docs"], CONFIG["eval"]["metric"]) == ("test", 11334, "rouge")
    assert TASK.tokenizer == "gpt2" and (TASK.eos_id, TASK.pad_id) == (50256, 50257) and g.prefix_pad_id == 50257


def test_task_spec_with_decoding_and_aux_files_round_trips():
    assert spec.from_json(json.loads(json.dumps(TASK.to_json()))) == TASK
    assert spec.from_json(json.loads(json.dumps(spec.get("tinystories").to_json()))) == spec.get("tinystories")
    assert all(len(digest) == 64 for _, digest in TASK.aux_files.values())


def test_summarize_is_offered_for_xsum_only_and_old_commands_refuse_it():
    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(dest="cmd"))
    args = parser.parse_args(["official", "summarize", "--input", "a.txt"])
    assert args.task == "xsum" and args.batch == 16 and args.cfg_scale is None
    for command, task in (("summarize", "tinystories"), ("generate", "xsum"), ("sudoku", "xsum"), ("bench", "xsum")):
        with pytest.raises(SystemExit):
            parser.parse_args(["official", command, "--task", task])


def test_prefix_row_truncates_pads_and_keeps_short_articles():
    assert summarize.prefix_row(list(range(10)), 4, 99) == [0, 1, 2, 3]
    assert summarize.prefix_row([7, 8], 5, 99) == [7, 8, 99, 99, 99]
    assert summarize.prefix_row([], 3, 99) == [99, 99, 99]


def test_clean_up_matches_the_standard_detokenizer_rules():
    assert summarize.clean_up_tokenization("He said , \" it 's not ok . \" Do n't go !") == \
        "He said, \" it's not ok. \" Don't go!"
    assert summarize.clean_up_tokenization("we 've seen it , I 'm sure they 're there ?") == \
        "we've seen it, I'm sure they're there?"


def _naive_readout(logits, unigram, d):
    """Plain-Python restatement of the readout rules, one row at a time."""
    T, V = logits.shape
    allow = [float(unigram[v]) * T for v in range(V)]
    count = [0.0] * V
    chosen: list[int] = []
    for t in range(T):
        score = [float(x) for x in logits[t]]
        for v in range(V):
            score[v] -= d.frequency_penalty * max(count[v] - allow[v], 0.0)
            score[v] += d.frequency_deficit * (T - t) / T * max(allow[v] - count[v], 0.0)
        for j, prev in enumerate(reversed(chosen[-d.adjacency_window:])):
            score[prev] -= d.adjacency_penalty * d.adjacency_decay**j
        pick = max(range(V), key=lambda v: (score[v], -v))
        chosen.append(pick)
        count[pick] += 1.0
    return chosen


def test_penalized_readout_follows_the_rules_and_changes_plain_argmax():
    rng = np.random.default_rng(0)
    B, T, V = 3, 12, 9
    logits = rng.normal(size=(B, T, V)).astype(np.float32) * 2.0
    logits[:, :, 4] += 4.0
    unigram = rng.dirichlet(np.ones(V)).astype(np.float32)
    got = summarize.penalized_readout(logits, unigram, TASK.decoding)
    for b in range(B):
        assert got[b].tolist() == _naive_readout(logits[b], unigram, TASK.decoding)
    assert (got != logits.argmax(-1)).any()
    assert (got[:, 1:] != got[:, :-1]).all()


def test_readout_without_penalties_is_plain_argmax_and_adjacency_alone_blocks_repeats():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(2, 8, 6)).astype(np.float32)
    off = replace(TASK.decoding, frequency_penalty=0.0, frequency_deficit=0.0, adjacency_penalty=0.0)
    np.testing.assert_array_equal(summarize.penalized_readout(logits, np.full(6, 1 / 6), off), logits.argmax(-1))
    flat = np.zeros((1, 6, 4), dtype=np.float32)
    flat[:, :, 2] = 1.0
    adj = replace(off, adjacency_penalty=5.0, adjacency_window=1, adjacency_decay=0.5)
    ids = summarize.penalized_readout(flat, np.full(4, 0.25), adj)[0]
    assert ids[0] == 2 and ids[1] != 2 and ids[2] == 2


def test_byte_level_vocabulary_conversion(tmp_path):
    dec = convert._byte_decoder()
    assert len(dec) == 256 and sorted(dec.values()) == list(range(256))
    assert dec["Ġ"] == 0x20 and dec["Ċ"] == 0x0A and dec["a"] == ord("a")
    enc = {b: c for c, b in dec.items()}
    pieces = [enc[b] for b in range(256)] + [enc[ord("a")] + enc[ord("b")], "<|endoftext|>"]
    (tmp_path / "gpt2").mkdir()
    (tmp_path / "gpt2/vocab.json").write_text(json.dumps({p: i for i, p in enumerate(pieces)}))
    (tmp_path / "gpt2/merges.txt").write_text(f"#version: 0.2\n{enc[ord('a')]} {enc[ord('b')]}\n")
    task = replace(TASK, generator=replace(TASK.generator, prefix_pad_id=len(pieces)))
    convert.convert_gpt2(task, tmp_path, tmp_path)
    from ncm_mlx.official.bpe import BPETokenizer

    tok = BPETokenizer.load(tmp_path / "gpt2.json")
    assert tok.vocab_size == len(pieces) + 1 and tok.vocab[-1] == b"<|pad|>"
    ids = tok.encode("ab ab é<|endoftext|>")
    assert ids[0] == 256 and ids[1:3] == [0x20, 256] and ids[-1] == 257
    assert tok.decode(ids, skip_special=True) == "ab ab é"
    bad = replace(task, generator=replace(TASK.generator, prefix_pad_id=len(pieces) + 3))
    with pytest.raises(ValueError, match="pad token"):
        convert.convert_gpt2(bad, tmp_path, tmp_path)


def test_generator_conversion_rejects_unexpected_and_missing_tensors(cpu):
    m = tiny(True, True, 1)
    from mlx.utils import tree_flatten

    weights = {k: np.array(v) for k, v in tree_flatten(m.gen.parameters())}
    convert._check_shapes(m.gen, weights, "generator")
    with pytest.raises(ValueError, match="matched 0 conversion rules"):
        convert.map_state({"transformer.wte.weight": np.zeros((2, 2)), "transformer.extra.weight": np.zeros(1)},
                          convert.GENERATOR_RULES)
    with pytest.raises(ValueError, match="matched 0 conversion rules"):
        convert.map_state({"lm_heads.0.E": np.zeros((2, 2))}, convert.GENERATOR_RULES)
    dropped = dict(weights)
    dropped.pop("ln_f.bias")
    with pytest.raises(ValueError, match="missing"):
        convert._check_shapes(m.gen, dropped, "generator")
    with pytest.raises(ValueError, match="unexpected"):
        convert._check_shapes(m.gen, {**weights, "blocks.0.extra": np.zeros(1)}, "generator")
    wrong = dict(weights)
    wrong["ln_f.bias"] = np.zeros(weights["ln_f.bias"].shape[0] + 1, dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        convert._check_shapes(m.gen, wrong, "generator")


@pytest.mark.mlx
def test_unguided_default_is_unchanged_and_guidance_mixes_per_level(cpu):
    import mlx.core as mx

    from ncm_mlx.official import sample

    m = tiny(True, True, 1)
    p = prefixes(m, 3)
    plain = sample.generate(m, p, temperature=0.0, keep_logits=True)
    zero = sample.generate(m, p, temperature=0.0, keep_logits=True, cfg_scale=0.0)
    assert all(np.array_equal(np.array(a), np.array(b)) for a, b in zip(plain.codes, zero.codes))

    scale = 3.0
    g = sample.generate(m, p, temperature=0.0, keep_logits=True, cfg_scale=scale)
    null = np.full(p.shape, m.task.generator.prefix_pad_id, dtype=np.int32)
    null[:, 0] = m.task.eos_id
    ref = sample.generate(m, mx.array(null), temperature=0.0, keep_logits=True, forced=[mx.array(np.array(c)) for c in g.codes])
    cond = sample.generate(m, p, temperature=0.0, keep_logits=True, forced=[mx.array(np.array(c)) for c in g.codes])
    n = len(m.codec.levels)
    for k, raw in enumerate(g.level_logits):
        raw = np.array(raw)
        assert raw.shape[0] == 6
        np.testing.assert_allclose(raw[:3], np.array(cond.level_logits[k]), atol=1e-4)
        np.testing.assert_allclose(raw[3:], np.array(ref.level_logits[k]), atol=1e-4)
        t = scale * k / (n - 1)
        mixed = (1 + t) * raw[:3] - t * raw[3:]
        np.testing.assert_array_equal(np.array(g.codes[k]), mixed.argmax(-1))
    assert not all(np.array_equal(np.array(a), np.array(b)) for a, b in zip(plain.codes, g.codes))
    np.testing.assert_array_equal(np.array(plain.codes[0]), np.array(g.codes[0]))


@pytest.mark.mlx
def test_guidance_needs_a_padding_id(cpu):
    from ncm_mlx.official import sample

    m = tiny(True, True, None)
    with pytest.raises(ValueError, match="padding id"):
        sample.generate(m, prefixes(m, 1), temperature=0.0, cfg_scale=1.0)


@pytest.mark.mlx
def test_summarize_prefixes_reads_out_the_final_reconstruction_with_the_penalties(cpu):
    import mlx.core as mx

    from ncm_mlx.official import sample

    m = tiny(True, True, 1)
    dec = replace(TASK.decoding, top_k=1, cfg_scale=0.0)
    unigram = mx.softmax(mx.random.normal((12,)), axis=-1)
    m = replace(m, task=replace(m.task, decoding=dec), token_unigram=unigram)
    p = prefixes(m, 4)
    out = summarize.summarize_prefixes(m, p)
    g = sample.generate(m, p, temperature=1.0, top_k=1, top_p=0.0)
    logits = np.array(m.codec.decode_logits(g.final))
    expect = summarize.penalized_readout(logits, np.array(unigram), dec)
    np.testing.assert_array_equal(out.token_ids, expect)
    assert [np.array_equal(a, np.array(b)) for a, b in zip(out.codes, g.codes)] == [True] * len(g.codes)
    assert len(out.texts) == 4 and out.generator_forwards == len(m.codec.levels)
    # Batch size changes nothing for greedy decoding.
    texts, stats = summarize.summarize_articles(
        replace(m, task=replace(m.task, generator=m.task.generator)), [list(map(int, r)) for r in np.array(p)], batch=3)
    assert texts == out.texts and stats["articles"] == 4 and stats["generator_forwards"] == 2 * len(m.codec.levels)


def _parity():
    path = FIXTURES / "official_xsum_parity.json"
    return json.loads(path.read_text())


def _real():
    d = ARTIFACTS / "xsum" / "mlx"
    if not (d / "spec.json").exists() or not (d / "generator.safetensors").exists():
        pytest.skip("converted official XSum generator absent")
    from ncm_mlx.official import sample

    return sample.load(d)


@pytest.mark.mlx
def test_gpt2_tokenizer_matches_hugging_face_ids(cpu):
    m = _real()
    for text, ids in _parity()["tokenizer"]["encode"]:
        assert m.bpe.encode(text) == ids, text


@pytest.mark.mlx
def test_greedy_summaries_match_upstream_pytorch(cpu):
    import mlx.core as mx

    from ncm_mlx.official import sample

    ref = _parity()
    m = _real()
    prefix = mx.array(np.array(ref["rows"], dtype=np.int32))
    g = sample.generate(m, prefix, temperature=1.0, top_k=1, top_p=0.0, keep_logits=True)
    flat = np.concatenate([np.array(c) for c in g.codes], axis=1)
    np.testing.assert_array_equal(flat, np.array(ref["codes"]))
    forced = [mx.array(c.astype(np.int32)) for c in np.split(np.array(ref["codes"]), np.cumsum(m.codec.levels)[:-1], 1)]
    f = sample.generate(m, prefix, keep_logits=True, forced=forced)
    for k, expected in enumerate(ref["forced_logits"]):
        got = np.array(f.level_logits[k])[0]
        np.testing.assert_allclose(got[:2, :16], np.array(expected["row0_head"]), atol=2e-4 * max(1.0, np.abs(got).max()))
        assert got.argmax(-1).tolist() == expected["row0_argmax"]
    out = summarize.summarize_prefixes(m, prefix)
    np.testing.assert_array_equal(out.token_ids, np.array(ref["readout_ids"]))
    assert out.texts == ref["texts"]


@pytest.mark.mlx
def test_guided_codes_match_upstream_pytorch(cpu):
    import mlx.core as mx

    from ncm_mlx.official import sample

    ref = _parity()
    m = _real()
    prefix = mx.array(np.array(ref["rows"], dtype=np.int32))
    g = sample.generate(m, prefix, temperature=0.0, cfg_scale=1.0)
    np.testing.assert_array_equal(np.concatenate([np.array(c) for c in g.codes], axis=1), np.array(ref["cfg1_codes"]))
