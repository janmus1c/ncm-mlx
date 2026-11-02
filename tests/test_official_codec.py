"""Codec-only releases (XSum, WMT14): configuration contract, strict conversion and parity with upstream.

Everything above the opt-in block runs on CPU without checkpoints. ``official_codec_parity.json``
holds upstream PyTorch's codes, written by ``scripts/official_codec_parity.py``: ``synthetic`` for
seeded narrow tensors with the real level geometry (reproduced here from ``codec_state.py``) and
``inputs`` for the real checkpoints. The real-checkpoint tests skip until ``official fetch`` and
``convert`` have produced ``artifacts/official/<task>/mlx``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from codec_state import small, stage_indices, upstream_state

from ncm_mlx.official import convert, spec
from ncm_mlx.official.cli import register

FIXTURES = Path(__file__).parent / "fixtures"
PARITY = json.loads((FIXTURES / "official_codec_parity.json").read_text())
CONFIGS = json.loads((FIXTURES / "official_codec_configs.json").read_text())
ARTIFACTS = Path("artifacts/official")
CODEC_TASKS = [k for k in CONFIGS if not k.startswith("_")]


def _upstream_levels(flat, levels):
    return np.split(np.array(flat), np.cumsum(levels)[:-1], axis=1)


@pytest.fixture
def cpu():
    mx = pytest.importorskip("mlx.core")
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.mark.parametrize("name", CODEC_TASKS)
def test_spec_matches_pinned_config_contract(name):
    task, cfg = spec.get(name), CONFIGS[name]
    vq, c = cfg["vqvae"], task.codec
    qc = vq["vector_quantizer_config"]
    assert task.generator is None and name not in spec.GENERATOR_TASKS
    assert (vq["in_vocab_size"], vq["context_length"], vq["embed_dim"], vq["n_layers"]) == \
        (c.vocab_size, c.seq_len, c.dim, c.n_head)
    assert tuple(qc["levels"]["manual_levels"]) == c.levels and tuple(vq["vqvae_vocab_size"]) == c.codebook_sizes
    assert c.levels[-1] == c.seq_len and np.prod(c.l0_strides) * c.levels[0] == c.seq_len
    assert max(c.l0_strides) <= qc["learned_l1_cascade_base"]
    assert (qc["learned_l1_cascade_norm"], qc["learned_l1_cascade_norm_final"]) == (True, True)
    assert (c.cascade_norm, c.cascade_norm_final) == (True, True)
    assert qc["quant_resi"] == {"enabled": True, "ratio": c.phi_ratio, "share_mode": 0, "learnable_ratio": False}
    assert vq["pre_quant_groupnorm"] == c.group_norm_groups and vq["use_qk_norm"] and vq["compression_factor"] == 1
    # Six attention heads, not six layers; no contextual encoder block and the default single decoder block.
    assert {"enc_layers", "dec_layers", "extra_blocks"}.isdisjoint(vq)
    assert cfg["generation"]["cfg_scale"] == 0.0
    assert task.files["config.yaml"] == cfg["config_sha256"]
    hf_id, _ = spec.SOURCE_TOKENIZERS[task.tokenizer]
    assert cfg["pre_tokenizer"]["model_id"] in (hf_id, hf_id.split("/")[-1])
    assert PARITY["_meta"]["checkpoint_revision"] == spec.REVISION
    assert PARITY["_meta"]["upstream_code_revision"] == spec.UPSTREAM_CODE_REVISION


def test_existing_codecs_keep_plain_cascades():
    for name in spec.GENERATOR_TASKS:
        c = spec.get(name).codec
        assert not c.cascade_norm and not c.cascade_norm_final


@pytest.mark.parametrize("name", CODEC_TASKS)
def test_task_spec_round_trips_without_generator(name):
    task = spec.get(name)
    assert spec.from_json(json.loads(json.dumps(task.to_json()))) == task


def test_generator_commands_refuse_codec_only_tasks():
    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(dest="cmd"))
    for command in ("generate", "sudoku", "bench"):
        with pytest.raises(SystemExit):
            parser.parse_args(["official", command, "--task", "xsum"])
    for command in ("fetch", "convert", "codec"):
        extra = ["--ids", "x.json"] if command == "codec" else []
        assert parser.parse_args(["official", command, "--task", "wmt14", *extra]).task == "wmt14"


@pytest.mark.parametrize("name", CODEC_TASKS)
def test_every_upstream_tensor_has_one_rule_and_training_state_is_dropped(name):
    codec = small(spec.get(name).codec)
    state = upstream_state(codec, 0)
    weights, dropped = convert.map_state(state, convert.CODEC_RULES)
    assert len(dropped) == 3 * len(codec.levels) and all(d.startswith("vq.") for d in dropped)
    assert len(weights) + len(dropped) == len(state)
    # The parameter-free norms occupy list slots, so the convolutions keep upstream's indices.
    stages = stage_indices(len(codec.l0_strides), True)
    assert stages == [2 * i for i in range(len(stages))]
    assert {f"down0.{i}.weight" for i in stages} <= set(weights) and {f"up0.{i}.bias" for i in stages} <= set(weights)


def _codec(codec_spec, state):
    from ncm_mlx.official.model import Codec

    weights, _ = convert.map_state(state, convert.CODEC_RULES)
    model = Codec(codec_spec)
    convert._check_shapes(model, weights, "codec")
    import mlx.core as mx

    model.load_weights([(k, mx.array(v)) for k, v in weights.items()], strict=True)
    return model.eval()


@pytest.mark.mlx
@pytest.mark.parametrize("fault", ["missing", "unknown", "shape", "layout"])
@pytest.mark.parametrize("name", CODEC_TASKS)
def test_conversion_rejects_incomplete_or_unmapped_state(name, fault, cpu):
    codec = small(spec.get(name).codec)
    state = upstream_state(codec, 0)
    if fault == "missing":
        del state["decoder.middle_block.attn.q_norm.weight"]
    elif fault == "unknown":
        state["decoder.extra_blocks.0.ln_1.weight"] = np.zeros(codec.dim, dtype=np.float32)
    elif fault == "shape":
        state["vq.learned_down_l1.0.weight"] = state["vq.learned_down_l1.0.weight"][:, :, :-1]
    else:
        state = upstream_state(replace(codec, cascade_norm=False, cascade_norm_final=False), 0)
    with pytest.raises(ValueError):
        _codec(codec, state)


@pytest.mark.mlx
@pytest.mark.parametrize("name", CODEC_TASKS)
def test_codec_matches_upstream_codes_on_synthetic_weights(name, cpu):
    import mlx.core as mx

    ref = PARITY[name]["synthetic"]
    codec = small(spec.get(name).codec)
    model = _codec(codec, upstream_state(codec, ref["seed"]))
    codes = [np.array(c) for c in model.encode(mx.array(np.array(ref["token_ids"], dtype=np.int32)))]
    upstream = _upstream_levels(ref["codes"], codec.levels)
    assert len(codes) == len(codec.levels)
    for k, (a, b) in enumerate(zip(codes, upstream)):
        np.testing.assert_array_equal(a, b, err_msg=f"level {k}")
    forced = [mx.array(u.astype(np.int32)) for u in upstream]
    for k, expected in enumerate(ref["decoded"]):
        ids = mx.argmax(model.decode_logits(model.cumulative(forced[: k + 1])[-1]), axis=-1)
        np.testing.assert_array_equal(np.array(ids), np.array(expected), err_msg=f"level {k}")


@pytest.mark.mlx
@pytest.mark.parametrize("name", CODEC_TASKS)
def test_dropping_the_cascade_norms_would_fail_the_parity_fixture(name, cpu):
    import mlx.core as mx

    ref = PARITY[name]["synthetic"]
    plain = replace(small(spec.get(name).codec), cascade_norm=False, cascade_norm_final=False)
    model = _codec(plain, upstream_state(plain, ref["seed"]))
    codes = [np.array(c) for c in model.encode(mx.array(np.array(ref["token_ids"], dtype=np.int32)))]
    upstream = _upstream_levels(ref["codes"], plain.levels)
    assert sum(int((a != b).sum()) for a, b in zip(codes, upstream)) > 0


@pytest.mark.mlx
@pytest.mark.parametrize("name", CODEC_TASKS)
def test_converted_codec_directory_loads_strictly(name, tmp_path, cpu):
    import mlx.core as mx

    from ncm_mlx.official import sample

    task = spec.get(name)
    codec = small(task.codec)
    weights, _ = convert.map_state(upstream_state(codec, 0), convert.CODEC_RULES)
    mx.save_safetensors(str(tmp_path / "codec.safetensors"), {k: mx.array(v) for k, v in weights.items()})
    (tmp_path / "spec.json").write_text(json.dumps({"task": replace(task, codec=codec).to_json()}))
    loaded, model = sample.load_codec(tmp_path)
    assert loaded.generator is None and loaded.codec == codec
    assert [type(m).__name__ for m in model.down0] == ["StridedDown", "CascadeNorm"] * (len(codec.l0_strides) - 1) + \
        ["StridedDown", "CascadeNorm"]
    assert [type(m).__name__ for m in model.up0] == ["StridedUp", "CascadeNorm"] * (len(codec.l0_strides) - 1) + \
        ["StridedUp"]
    with pytest.raises(SystemExit, match="codec-only"):
        sample.load(tmp_path)


def _real(name):
    task = spec.get(name)
    d = ARTIFACTS / task.remote_dir / "mlx"
    if not (d / "spec.json").exists():
        pytest.skip("converted official weights absent")
    from ncm_mlx.official import sample

    return sample.load_codec(d)[1], PARITY[name]


@pytest.mark.mlx
@pytest.mark.parametrize("name", CODEC_TASKS)
def test_real_codec_codes_match_upstream_on_cpu(name, cpu):
    """Opt-in: needs ``official fetch`` and ``convert`` for the task. Requires zero code disagreements."""
    import mlx.core as mx

    model, ref = _real(name)
    rows = np.array(ref["inputs"]["token_ids"], dtype=np.int32)
    codes = [np.array(c) for c in model.encode(mx.array(rows))]
    upstream = _upstream_levels(ref["codes"], model.levels)
    for k, (a, b) in enumerate(zip(codes, upstream)):
        np.testing.assert_array_equal(a, b, err_msg=f"level {k}")
    forced = [mx.array(u.astype(np.int32)) for u in upstream]
    for k, expected in enumerate(ref["decoded"]):
        ids = mx.argmax(model.decode_logits(model.cumulative(forced[: k + 1])[-1]), axis=-1)
        np.testing.assert_array_equal(np.array(ids), np.array(expected), err_msg=f"level {k}")


@pytest.mark.mlx
def test_codec_command_encodes_integer_rows_and_rejects_other_input(tmp_path, cpu, capsys):
    import mlx.core as mx

    from ncm_mlx.official.cli import cmd_codec

    task = spec.get("xsum")
    codec = small(task.codec)
    weights, _ = convert.map_state(upstream_state(codec, 0), convert.CODEC_RULES)
    mx.save_safetensors(str(tmp_path / "codec.safetensors"), {k: mx.array(v) for k, v in weights.items()})
    (tmp_path / "spec.json").write_text(json.dumps({"task": replace(task, codec=codec).to_json()}))
    ids = tmp_path / "ids.json"

    def run(rows):
        ids.write_text(json.dumps(rows))
        return cmd_codec(argparse.Namespace(task="xsum", weights=str(tmp_path), ids=str(ids), cpu=True, out=None))

    assert run([[3] * codec.seq_len]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [len(c[0]) for c in out["codes"]] == list(codec.levels) and len(out["decoded"][0]) == codec.seq_len
    for bad in ([], [[3] * (codec.seq_len - 1)], [[codec.vocab_size] * codec.seq_len], [[1.5] * codec.seq_len],
                [[True] * codec.seq_len], {"rows": 1}):
        with pytest.raises(SystemExit, match="integer token ids"):
            run(bad)
