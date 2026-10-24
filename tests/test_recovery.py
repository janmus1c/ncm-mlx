import copy
import json
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from ncm_mlx.config import Config
from ncm_mlx.text.tokenizer import CharTokenizer
from ncm_mlx.training.checkpoint import TrainingPaused, read_checkpoint
from ncm_mlx.training.codec import train_codec
from ncm_mlx.training.common import batches, build_codec, schedule
from ncm_mlx.training.generator import train_generator

pytestmark = pytest.mark.mlx


def inputs(tmp_path):
    cfg = Config(out_dir=str(tmp_path))
    cfg.data.seq_len = 4
    cfg.codec.dim, cfg.codec.n_head = 8, 2
    cfg.codec.enc_layers, cfg.codec.dec_layers = 0, 1
    cfg.codec.levels, cfg.codec.codebook_sizes = [1, 2, 4], [4, 4, 4]
    cfg.codec.dropout = 0.2
    cfg.generator.n_embd, cfg.generator.n_head, cfg.generator.n_layer = 8, 2, 1
    cfg.generator.dropout, cfg.generator.corrupt_first, cfg.generator.corrupt_last = 0.2, 0.5, 0.1
    cfg.train.batch_size, cfg.train.warmup, cfg.train.log_every = 2, 1, 2
    cfg.train.codec_steps, cfg.train.generator_steps = 6, 6
    tokens = np.arange(32).reshape(8, 4).astype(np.int32) % 10
    corpus = SimpleNamespace(train=tokens, val=tokens[:4], train_prefix=None, val_prefix=None,
                             tokenizer=CharTokenizer("0123456789", "0", 4))
    return cfg, corpus


def pause_at_two(message):
    if "step     2 " in message:
        os.kill(os.getpid(), signal.SIGUSR1)


def checkpoint_arrays(path):
    arrays, metadata = mx.load(str(path), return_metadata=True)
    return {k: np.array(v) for k, v in arrays.items()}, json.loads(metadata["training"])


@pytest.mark.parametrize("compile", [False, True])
def test_generator_pause_resume_matches_uninterrupted_training(tmp_path, compile):
    with mx.stream(mx.cpu):
        cfg, corpus = inputs(tmp_path / "whole")
        cfg.train.compile = compile
        mx.random.seed(17)
        codec = build_codec(cfg, 10)
        train_generator(cfg, codec, corpus, log=lambda _: None)
        whole, _ = checkpoint_arrays(Path(cfg.out_dir) / "generator-training.safetensors")
        cfg.out_dir = str(tmp_path / "split")
        with pytest.raises(TrainingPaused) as paused:
            train_generator(cfg, codec, corpus, log=pause_at_two)
        assert paused.value.code == 75 and paused.value.step == 2
        train_generator(cfg, codec, corpus, resume=True, log=lambda _: None)
        split, meta = checkpoint_arrays(Path(cfg.out_dir) / "generator-training.safetensors")
        assert meta["step"] == 6
        assert whole.keys() == split.keys()
        for key in whole:
            np.testing.assert_array_equal(whole[key], split[key], err_msg=key)


def test_codec_pause_resume_preserves_ema_and_random_state(tmp_path, monkeypatch):
    with mx.stream(mx.cpu):
        cfg, corpus = inputs(tmp_path / "whole")
        monkeypatch.setattr("ncm_mlx.training.codec.load_corpus", lambda cfg: corpus)
        train_codec(cfg, log=lambda _: None)
        whole, _ = checkpoint_arrays(Path(cfg.out_dir) / "codec-training.safetensors")
        cfg.out_dir = str(tmp_path / "split")
        with pytest.raises(TrainingPaused):
            train_codec(cfg, log=pause_at_two)
        train_codec(cfg, resume=True, log=lambda _: None)
        split, _ = checkpoint_arrays(Path(cfg.out_dir) / "codec-training.safetensors")
        for key in whole:
            np.testing.assert_array_equal(whole[key], split[key], err_msg=key)


def test_weights_only_recovery_keeps_schedule_batch_position_and_source(tmp_path):
    with mx.stream(mx.cpu):
        cfg, corpus = inputs(tmp_path / "original")
        mx.random.seed(17)
        codec = build_codec(cfg, 10)
        with pytest.raises(TrainingPaused):
            train_generator(cfg, codec, corpus, log=pause_at_two)
        source = Path(cfg.out_dir) / "generator.safetensors"
        original = source.read_bytes()
        cfg.out_dir = str(tmp_path / "recovered")
        train_generator(cfg, codec, corpus, checkpoint=str(source), start_step=2, log=lambda _: None)
        arrays, meta = checkpoint_arrays(Path(cfg.out_dir) / "generator-training.safetensors")
        assert source.read_bytes() == original
        assert meta["continuation"]["optimizer_reset"]
        assert meta["continuation"]["start_step"] == 2
        assert [r["step"] for r in meta["history"]] == [4, 6]
        assert arrays["optimizer.step"].item() == 4
        expected_lr = schedule(cfg.train.generator_lr, cfg.train.warmup, 6)(mx.array(5)).item()
        assert arrays["optimizer.learning_rate"].item() == expected_lr
        expected = batches(corpus.train, 2, np.random.default_rng(1))
        expected.advance(6)
        assert meta["data"] == expected.state_dict()


def test_checkpoint_falls_back_and_rejects_changed_config(tmp_path):
    with mx.stream(mx.cpu):
        cfg, corpus = inputs(tmp_path)
        codec = build_codec(cfg, 10)
        train_generator(cfg, codec, corpus, log=lambda _: None)
        path = tmp_path / "generator-training.safetensors"
        path.write_bytes(b"incomplete checkpoint")
        _, meta = read_checkpoint(path, cfg, "generator", 6, log=lambda _: None)
        assert meta["step"] == 4
        changed = copy.deepcopy(cfg)
        changed.train.generator_lr *= 2
        with pytest.raises(ValueError, match="configuration/version mismatch"):
            read_checkpoint(path, changed, "generator", 6, log=lambda _: None)
        train_generator(cfg, codec, corpus, resume=True, log=lambda _: None)
        _, meta = read_checkpoint(path, cfg, "generator", 6)
        assert meta["step"] == 6


def test_batch_stream_restore_across_epoch_boundary():
    with mx.stream(mx.cpu):
        data = np.arange(21).reshape(7, 3)
        first = batches(data, 2, np.random.default_rng(3))
        first.advance(8)
        second = batches(data, 2, np.random.default_rng(999))
        second.load_state_dict(first.state_dict())
        for _ in range(12):
            np.testing.assert_array_equal(np.array(next(first)), np.array(next(second)))


def test_resume_exhausted_time_budget_keeps_completed_step_count(tmp_path):
    with mx.stream(mx.cpu):
        cfg, corpus = inputs(tmp_path)
        cfg.train.max_minutes = 1e-12
        codec = build_codec(cfg, 10)
        _, first = train_generator(cfg, codec, corpus, log=lambda _: None)
        assert first["steps"] == 1
        path = tmp_path / "generator-training.safetensors"
        before = path.read_bytes()
        _, resumed = train_generator(cfg, codec, corpus, resume=True, log=lambda _: None)
        assert resumed["steps"] == 1
        assert before == path.read_bytes()
