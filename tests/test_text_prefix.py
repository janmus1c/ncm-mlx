"""Prompt-conditioned (prefix) text training and evaluation, on CPU with tiny configurations."""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten
from test_text_training import pause_at_two, tiny_cfg, tiny_rows

from ncm_mlx.official import ar as ar_mod
from ncm_mlx.official import prefix_eval, sample, text_eval, trainer
from ncm_mlx.official import tinystories as ts
from ncm_mlx.official.model import Generator
from ncm_mlx.official.train_config import TextConfig
from ncm_mlx.training.checkpoint import TrainingPaused

pytestmark = pytest.mark.mlx
P, L = 4, 8


@pytest.fixture(autouse=True)
def cpu():
    with mx.stream(mx.cpu):
        mx.set_default_device(mx.cpu)
        yield
    mx.set_default_device(mx.gpu)


def prefix_cfg(tmp_path: Path, prefix_len: int = P) -> TextConfig:
    cfg = tiny_cfg(tmp_path)
    cfg.data.prefix_len = prefix_len
    cfg.validate()
    return cfg


def prefix_rows(n=32, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = rng.integers(2, 40, (n, P + L)).astype(np.uint16)
    rows[::3, P + 5] = ts.EOS_ID
    rows[::3, P + 6 :] = ts.PAD_ID
    return rows


@pytest.fixture
def patched_prefix_rows(monkeypatch):
    rows = prefix_rows()

    def fake(cfg, log=print):
        return rows, rows[:8].copy(), {"fake": True}

    monkeypatch.setattr(trainer, "load_rows", fake)
    monkeypatch.setattr(ar_mod, "load_rows", fake)
    return rows


def models(tmp_path, prefix_len=P, seed=0):
    cfg = prefix_cfg(tmp_path, prefix_len)
    tr = trainer.CodecTrainer(cfg, seed=seed)
    tr.eval()
    gen = trainer.build_generator(cfg, tr.codec, seed + 1)
    gen.eval()
    return cfg, tr.codec, gen


# ---------------------------------------------------------------- config


def test_default_config_serialisation_is_unchanged(tmp_path):
    cfg = tiny_cfg(tmp_path)
    assert cfg.data.prefix_len == 0
    assert "prefix_len" not in cfg.to_dict()["data"]
    assert TextConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    g = cfg.generator_spec()
    assert (g.prefix_len, g.query_level0, g.prefix_pad_id) == (1, False, 1)


def test_prefix_config_changes_hash_and_generator_spec(tmp_path):
    base, cfg = tiny_cfg(tmp_path), prefix_cfg(tmp_path)
    assert cfg.to_dict()["data"]["prefix_len"] == P
    assert cfg.to_dict() != base.to_dict()
    assert TextConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    g = cfg.generator_spec()
    assert (g.prefix_len, g.query_level0, g.prefix_pad_id) == (P, True, 1)
    with pytest.raises(ValueError, match="prefix_len"):
        prefix_cfg(tmp_path, -1)


def test_shipped_prefix_configs_load():
    for name, p in (("tinystories-prefix-p32", 32), ("tinystories-prefix-smoke", 16)):
        cfg = TextConfig.load(Path("configs") / f"{name}.json")
        assert cfg.data.prefix_len == p and cfg.data.seq_len == 128
    ref, mine = (TextConfig.load(Path("configs") / f"{n}.json") for n in ("tinystories-e21", "tinystories-prefix-p32"))
    a, b = ref.to_dict(), mine.to_dict()
    for key in ("name", "out_dir"):
        a.pop(key), b.pop(key)
    b["data"].pop("prefix_len")
    assert a == b


# ---------------------------------------------------------------- data


def _tiny_bpe(tmp_path: Path) -> Path:
    cfg = tiny_cfg(tmp_path)
    return Path(cfg.data.bpe)


def test_continuation_rows_split_a_story_window_and_drop_short_stories(tmp_path):
    bpe_path = _tiny_bpe(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    stories = ["x" * 40 + str(i) for i in range(30)] + ["a"]
    (raw / ts.TRAIN_FILE).write_text("\n<|endoftext|>\n".join(stories[:25]) + "\n<|endoftext|>\n")
    (raw / ts.VALID_FILE).write_text("\n<|endoftext|>\n".join(stories[20:]) + "\n<|endoftext|>\n")

    def quiet(*_):
        return None

    m0 = ts.prepare(raw, tmp_path / "c", bpe_path, seq_len=P + L, workers=1, limit=25, log=quiet)
    m1 = ts.prepare(raw, tmp_path / "c", bpe_path, seq_len=L, workers=1, limit=25, log=quiet, prefix_len=P)
    d0 = ts.split_dir(tmp_path / "c", P + L, 25)
    d1 = ts.split_dir(tmp_path / "c", L, 25, P)
    assert d0 != d1 and f"-P{P}" in d1.name and m1["prefix_len"] == P and "prefix_len" not in m0
    for name in ("train", "val", "test"):
        flat, cont = np.load(d0 / f"{name}.npy"), np.load(d1 / f"{name}.npy")
        assert cont.shape[1] == P + L
        keep = (flat[:, :P] > 1).all(1)
        np.testing.assert_array_equal(cont, flat[keep])
        assert (cont[:, :P] > 1).all()
    assert ts.prepare(raw, tmp_path / "c", bpe_path, seq_len=L, workers=1, limit=25, prefix_len=P) == m1
    assert m1["splits"]["test"]["dropped_shorter_than_prefix"] + m1["splits"]["val"]["dropped_shorter_than_prefix"] >= 1


# ---------------------------------------------------------------- generator


def test_prefix_zero_matches_the_original_loss_formula(tmp_path):
    from mlx import nn

    cfg = tiny_cfg(tmp_path)
    tr = trainer.CodecTrainer(cfg, seed=0)
    tr.eval()
    gen = trainer.build_generator(cfg, tr.codec, 1)
    gen.eval()
    tf = trainer.TeacherForced(gen)
    tokens = mx.array(tiny_rows(4).astype(np.int32))
    loss, ce = trainer.generator_loss(tf, tr.codec, tokens, None, 0.0, 0.0)
    assert gen.first == 1 and gen.spec.prefix_len == 1 and not gen.spec.query_level0
    codes = [c.astype(mx.int32) for c in tr.codec.encode(tokens)]
    prefix = mx.full((4, 1), ts.EOS_ID, dtype=mx.int32)
    logits = tf.logits(trainer.teacher_inputs(tr.codec, codes[:-1]), prefix)
    ref = [nn.losses.cross_entropy(lg, c, reduction="mean") for lg, c in zip(logits, codes[1:])]
    np.testing.assert_array_equal(np.array(mx.stack(ce)), np.array(mx.stack(ref)))
    assert float(loss) == pytest.approx(sum(w * float(c) for w, c in zip(trainer.level_weights(tf.lengths, 0.0), ref)))


def test_prefix_loss_encodes_only_the_target_and_reads_the_prefix(tmp_path, monkeypatch):
    _cfg, codec, gen = models(tmp_path)
    assert gen.first == 0 and gen.spec.prefix_len == P and len(gen.heads) == len(codec.levels)
    tf = trainer.TeacherForced(gen)
    rows = mx.array(prefix_rows(4).astype(np.int32))
    seen = []
    original = type(codec).encode

    def spy(self, tokens):
        seen.append(tokens.shape)
        return original(self, tokens)

    monkeypatch.setattr(type(codec), "encode", spy)
    loss, ce = trainer.generator_loss(tf, codec, rows, None, 0.0, 0.0, P)
    assert seen == [(4, L)]
    assert len(ce) == len(codec.levels) and np.isfinite(float(loss))
    other = mx.concatenate([(rows[:, :P] + 1) % 40 + 2, rows[:, P:]], axis=1)
    loss2, _ = trainer.generator_loss(tf, codec, other, None, 0.0, 0.0, P)
    assert float(loss) != float(loss2)
    # every target code position is weighted, none of the prefix tokens is a label
    w = trainer.level_weights(tf.lengths, 0.0)
    assert float(loss) == pytest.approx(sum(a * float(b) for a, b in zip(w, ce)), rel=1e-6)
    assert tf.lengths == list(codec.levels)


def test_padded_prefix_positions_are_masked_like_a_shorter_prefix(tmp_path):
    _cfg, codec, gen = models(tmp_path)
    short_spec = type(gen.spec)(**{**gen.spec.__dict__, "prefix_len": 2})
    short = Generator(short_spec, codec)
    short.load_weights(list(tree_flatten(gen.parameters())))
    short.eval()
    tokens = mx.array(prefix_rows(3).astype(np.int32)[:, P:])
    codes = [c.astype(mx.int32) for c in codec.encode(tokens)]
    inputs = [mx.broadcast_to(gen.l0_queries, (3, *gen.l0_queries.shape[1:])),
              *trainer.teacher_inputs(codec, codes[:-1])]
    padded = mx.array([[5, 6, ts.PAD_ID, ts.PAD_ID], [7, 8, ts.PAD_ID, ts.PAD_ID], [9, 10, ts.PAD_ID, ts.PAD_ID]])
    a = trainer.TeacherForced(gen).logits(inputs, padded)
    b = trainer.TeacherForced(short).logits(inputs, padded[:, :2])
    for x, y in zip(a, b):
        np.testing.assert_allclose(np.array(x), np.array(y), atol=1e-5)
    differs = trainer.TeacherForced(gen).logits(inputs, mx.array([[5, 6, 11, 12]] * 3))
    assert float(mx.abs(differs[1] - a[1]).max()) > 1e-6


def test_prefix_training_logits_equal_the_official_sampler_levels(tmp_path):
    cfg, codec, gen = models(tmp_path)
    tokens = mx.array(prefix_rows(4).astype(np.int32)[:, P:])
    prefix = mx.array([[5, 6, 7, 8], [5, 6, 7, 8], [9, 10, ts.PAD_ID, ts.PAD_ID], [11, 12, 13, ts.PAD_ID]])
    codes = [c.astype(mx.int32) for c in codec.encode(tokens)]
    tf = trainer.TeacherForced(gen)
    inputs = [mx.broadcast_to(gen.l0_queries, (4, *gen.l0_queries.shape[1:])),
              *trainer.teacher_inputs(codec, codes[:-1])]
    full = tf.logits(inputs, prefix)
    m = sample.Official(cfg.task_spec(), codec, gen, None, None)
    for cached in (False, True):
        per_level = sample.generate(m, prefix, forced=codes, keep_logits=True, prefix_cache=cached).level_logits
        assert len(per_level) == len(full) == len(codec.levels)
        errors = [float(mx.abs(a - b).max()) for a, b in zip(full, per_level)]
        assert max(errors) < 1e-5, (cached, errors)


def test_unconditional_prefix_of_a_prompt_model_is_an_empty_prompt(tmp_path):
    cfg, codec, gen = models(tmp_path)
    m = sample.Official(cfg.task_spec(), codec, gen, None, None)
    prefix = np.array(sample.unconditional_prefix(m, 2))
    assert prefix.tolist() == [[ts.EOS_ID] + [ts.PAD_ID] * (P - 1)] * 2
    plain = tiny_cfg(tmp_path / "plain")
    m1 = sample.Official(plain.task_spec(), codec, gen, None, None)
    assert np.array(sample.unconditional_prefix(m1, 2)).tolist() == [[ts.EOS_ID]] * 2


# ---------------------------------------------------------------- AR baseline


def test_ar_prefix_zero_equals_the_bos_formulation(tmp_path):
    cfg = tiny_cfg(tmp_path)
    m = ar_mod.build_ar(cfg, 0)
    m.eval()
    tokens = mx.array(tiny_rows(3).astype(np.int32))
    inp = mx.concatenate([mx.full((3, 1), m.bos_id, dtype=tokens.dtype), tokens[:, :-1]], axis=1)
    h = m.wte(inp)
    for blk in m.blocks:
        h = trainer.block_forward(blk, h, m._mask, m._rope, 0.0)
    ref = m.wte.as_linear(m.ln_f(h))
    np.testing.assert_array_equal(np.array(m(tokens)), np.array(ref))


def test_ar_prefix_loss_covers_target_positions_only(tmp_path):
    from mlx import nn

    cfg = prefix_cfg(tmp_path)
    m = ar_mod.build_ar(cfg, 0)
    m.eval()
    rows = mx.array(prefix_rows(4).astype(np.int32))
    logits = m(rows)
    assert logits.shape == (4, L, 8192)
    manual = nn.losses.cross_entropy(logits, rows[:, P:], reduction="mean")
    assert float(ar_mod.ar_loss(m, rows, 0.0)) == pytest.approx(float(manual))
    report = ar_mod.ar_report(m, np.array(rows))
    assert set(report) == {"token_ce", "nonpad_token_ce"}
    # the prefix is context only: changing it changes the loss, but it is never a label
    other = mx.concatenate([(rows[:, :P] + 1) % 40 + 2, rows[:, P:]], axis=1)
    assert float(ar_mod.ar_loss(m, other, 0.0)) != float(ar_mod.ar_loss(m, rows, 0.0))


def test_ar_cached_generation_after_a_prompt_matches_full_forward(tmp_path):
    cfg = prefix_cfg(tmp_path)
    m = ar_mod.build_ar(cfg, 0)
    m.eval()
    prompt = mx.array(prefix_rows(3).astype(np.int32)[:, :P])
    tokens, logits = m.generate(3, temperature=1.0, seed=4, keep_logits=True, prefix=prompt)
    assert tokens.shape == (3, L)
    full = m(mx.concatenate([prompt, tokens], axis=1))
    assert float(mx.abs(mx.stack(logits, axis=1) - full).max()) < 1e-4
    with pytest.raises(ValueError, match="prefix"):
        m.generate(3)


# ---------------------------------------------------------------- training, resume, sampling


def checkpoint_arrays(path):
    arrays, metadata = mx.load(str(path), return_metadata=True)
    return {k: np.array(v) for k, v in arrays.items()}, json.loads(metadata["training"])


@pytest.mark.parametrize("stage", ["codec", "generator", "ar"])
def test_prefix_pause_resume_matches_uninterrupted_training(tmp_path, patched_prefix_rows, stage):
    runs = {}
    for name in ("whole", "split"):
        cfg = prefix_cfg(tmp_path / name)
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
    assert wmeta["config"]["data"]["prefix_len"] == P
    assert whole.keys() == split.keys()
    for key in whole:
        np.testing.assert_array_equal(whole[key], split[key], err_msg=key)


def test_resume_rejects_a_checkpoint_from_another_prefix_length(tmp_path, patched_prefix_rows):
    cfg = prefix_cfg(tmp_path)
    trainer.train_codec(cfg, log=lambda *_: None)
    with pytest.raises(TrainingPaused):
        trainer.train_generator(cfg, log=pause_at_two)
    cfg.data.prefix_len = P + 1
    with pytest.raises(ValueError, match="mismatch"):
        trainer.train_generator(cfg, resume=True, log=lambda *_: None)


def trained_prefix_run(tmp_path):
    cfg = prefix_cfg(tmp_path)
    trainer.train_codec(cfg, log=lambda *_: None)
    meta = trainer.train_generator(cfg, log=lambda *_: None)
    ar_mod.train_ar(cfg, log=lambda *_: None)
    return cfg, meta


def test_trained_prefix_checkpoint_continues_prompts_through_the_official_sampler(tmp_path, patched_prefix_rows):
    cfg, meta = trained_prefix_run(tmp_path)
    out = trainer.export_dir(cfg)
    spec = json.loads((out / "spec.json").read_text())
    assert spec["task"]["generator"]["prefix_len"] == P and spec["task"]["generator"]["query_level0"] is True
    assert len(meta["val"]["code_ce_per_level"]) == len(cfg.codec.levels)
    m = sample.load(out)
    prompts = mx.array(patched_prefix_rows[:3, :P].astype(np.int32))
    g = sample.generate(m, prompts, seed=1, keep_logits=True)
    assert g.tokens.shape == (3, L) and len(g.level_logits) == len(cfg.codec.levels)
    other = sample.generate(m, mx.array(patched_prefix_rows[3:6, :P].astype(np.int32)), seed=1, keep_logits=True)
    assert float(mx.abs(g.level_logits[0] - other.level_logits[0]).max()) > 0
    compiled = sample.generate(m, prompts, seed=1, compiled=True)
    np.testing.assert_array_equal(np.array(compiled.tokens), np.array(g.tokens))
    assert sample.generate(m, sample.unconditional_prefix(m, 2), seed=1).tokens.shape == (2, L)


# ---------------------------------------------------------------- evaluation


def test_prompt_names_skip_sentence_starts():
    text = "\nOne day, a girl named Lily met Ben. Then Tom said, \"Hi Mia.\" The dog ran."
    assert prefix_eval.prompt_names(text) == ["Lily", "Ben", "Tom", "Mia"]
    assert prefix_eval.prompt_names("\nTom went home. Sue ran.\nAnn sat") == []


def test_faithfulness_counts_carry_over_and_has_a_control():
    prompts = ["\nA girl named Lily had a red ball.", "\nA boy named Max had a green kite."]
    conts = ["Lily kicked the red ball high.", "Max flew the green kite today."]
    ids = [[1, 2, 3, 4, 5, 6, 7], [11, 12, 13, 14, 15, 16, 17]]
    own = prefix_eval.faithfulness(prompts, ids, conts, ids, 0)
    ctrl = prefix_eval.faithfulness(prompts, ids, conts, ids, 1)
    assert own["name_mention_pct"] == 100.0 and ctrl["name_mention_pct"] == 0.0
    assert own["content_word_overlap_pct"] > ctrl["content_word_overlap_pct"]
    assert own["copied_4gram_pct"] == 100.0 and ctrl["copied_4gram_pct"] == 0.0
    rep = prefix_eval.faithfulness_report(prompts, ids, conts, ids)
    assert "diagnostic" in rep["label"]


class _FakeTokenizer:
    def encode(self, text):
        return [ord(c) for c in text]


class _UniformModel:
    def __call__(self, x):
        return mx.zeros((*x.shape, 300))


def test_scorer_counts_only_continuation_tokens():
    scorer = text_eval.Scorer.__new__(text_eval.Scorer)
    scorer.model, scorer.tokenizer = _UniformModel(), _FakeTokenizer()
    res = scorer.token_nll(["abcdefgh", "abcd"], skip=[5, 0])
    assert res[0][1] == 3 and res[1][1] == 3
    assert res[0][0] == pytest.approx(3 * np.log(300), rel=1e-5)
    plain = scorer.token_nll(["abcdefgh"])
    assert plain[0][1] == 7
    ppl = text_eval.gen_ppl(scorer, {0: ["fgh", ""], 1: ["fgh", "x"]}, ["abcde", "abc"])
    assert ppl["per_seed"][0] == pytest.approx(300.0, rel=1e-4) and ppl["unscorable_samples"] == 1


def test_prefix_evaluation_runs_end_to_end_on_tiny_models(tmp_path, patched_prefix_rows):
    cfg, _ = trained_prefix_run(tmp_path)
    d = cfg.data
    vocab = [b"<|endoftext|>".hex(), b"<|pad|>".hex()] + [bytes([i]).hex() for i in range(256)] + [f"<{i}>".encode().hex() for i in range(8192 - 258)]
    Path(d.bpe).write_text(json.dumps({"vocab": vocab, "merges": [], "special_tokens": ["<|endoftext|>", "<|pad|>"]}))
    d.cache_root = str(tmp_path / "cache")
    directory = ts.split_dir(Path(d.cache_root), d.seq_len, d.limit, P)
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({"fake": True}))
    for name in ("train", "test"):
        np.save(directory / f"{name}.npy", patched_prefix_rows[:16])
    args = types.SimpleNamespace(models=["data", "official", "ncm", "ar"], n=6, seeds=[0, 1], batch=4,
                                 test_rows=16, no_scorer=True, out=str(tmp_path / "eval" / "report.json"))
    report = prefix_eval.run_prefix_eval(cfg, args, {"device": "cpu"}, log=lambda *_: None)
    assert set(report["models"]) == {"data", "ncm", "ar"}
    ncm, ar = report["models"]["ncm"], report["models"]["ar"]
    assert len(ncm["generator_test_code_ce"]["code_ce_per_level"]) == len(cfg.codec.levels)
    assert "generator_test_code_ce_shuffled_prompt" in ncm and "ar_test_shuffled_prompt" in ar
    for entry in report["models"].values():
        assert entry["prompt_faithfulness"]["label"].startswith("diagnostic")
        assert len(entry["examples_seed_first"]) == 6
    rows = [json.loads(line) for line in (tmp_path / "eval" / "continuations-ncm.jsonl").read_text().splitlines()]
    assert len(rows) == 12 and all(len(r["ids"]) == L for r in rows)


def test_continue_text_cli_continues_a_prompt_on_cpu(tmp_path, patched_prefix_rows, capsys):
    """The documented command: ``python -m ncm_mlx official continue-text --weights EXPORT --prompt TEXT --cpu``."""
    from ncm_mlx.cli import main

    cfg, _ = trained_prefix_run(tmp_path)
    bpe = trainer.export_dir(cfg) / "bpe.json"
    vocab = json.loads(bpe.read_text())
    vocab["vocab"] += [f"w{i}".encode().hex() for i in range(8192 - len(vocab["vocab"]))]
    bpe.write_text(json.dumps(vocab))
    export = str(trainer.export_dir(cfg))
    argv = ["official", "continue-text", "--weights", export, "--prompt", "ab", "--n", "2", "--seed", "3", "--cpu"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert out.count("--- continuation") == 2 and "[|]" in out
    assert main(argv) == 0 and capsys.readouterr().out == out
