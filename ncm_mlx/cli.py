"""Command line: ``python -m ncm_mlx <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from ncm_mlx.config import Config
from ncm_mlx.device import require_gpu

DEFAULT_CONFIG = "configs/toy-grammar-12k.json"


def _load(cfg: Config):
    from ncm_mlx.text.tokenizer import CharTokenizer
    from ncm_mlx.training.common import build_codec, build_generator, load_corpus, load_model

    out = Path(cfg.out_dir)
    codec_path = Path(cfg.codec_checkpoint) if cfg.codec_checkpoint else out / "codec.safetensors"
    gen_path = out / "generator.safetensors"
    if not codec_path.exists() or not gen_path.exists():
        raise SystemExit(f"no checkpoint in {out}; run `python -m ncm_mlx train --config ...` first")
    corpus = load_corpus(cfg)
    meta = json.loads(codec_path.with_suffix(".json").read_text())
    if CharTokenizer.from_dict(meta["tokenizer"]).chars != corpus.tokenizer.chars:
        raise SystemExit("checkpoint tokenizer does not match the corpus")
    codec = load_model(build_codec(cfg, corpus.tokenizer.vocab_size), codec_path)
    gen = load_model(build_generator(cfg), gen_path)
    return codec, gen, corpus


def cmd_smoke(args) -> int:
    from ncm_mlx.generation.sampler import SampleOptions, generate
    from ncm_mlx.training.common import build_codec, build_generator, load_corpus

    info = require_gpu()
    print(json.dumps(info))
    cfg = Config.load(args.config)
    corpus = load_corpus(cfg)
    codec = build_codec(cfg, corpus.tokenizer.vocab_size)
    gen = build_generator(cfg)
    codec.eval()
    gen.eval()
    tokens = mx.array(corpus.train[:8])
    codes = codec.encode(tokens)
    shapes = [tuple(c.shape) for c in codes]
    expected = [(8, l) for l in cfg.codec.levels]
    assert shapes == expected, (shapes, expected)
    out = generate(gen, codec, 2, SampleOptions(seed=0))
    assert out.tokens.shape == (2, cfg.data.seq_len)
    print(f"smoke ok: code shapes {shapes}, untrained sample {corpus.tokenizer.decode(out.tokens[0].tolist())!r}")
    return 0


def cmd_train(args) -> int:
    from ncm_mlx.training.codec import train_codec
    from ncm_mlx.training.generator import train_generator

    info = require_gpu()
    cfg = Config.load(args.config)
    out = Path(cfg.out_dir)
    resume = getattr(args, "resume", False)
    checkpoint = getattr(args, "generator_checkpoint", None)
    start_step = getattr(args, "generator_start_step", 0)
    if resume and checkpoint:
        raise SystemExit("choose --resume or --generator-checkpoint, not both")
    has_generator_state = any(out.glob("generator-training*.safetensors")) if out.exists() else False
    if not resume and not checkpoint and has_generator_state:
        raise SystemExit("training state exists; use --resume or choose a new out_dir")
    if resume and cfg.codec_checkpoint and not has_generator_state:
        raise SystemExit("no full generator state; use --generator-checkpoint for a weights-only recovery")
    out.mkdir(parents=True, exist_ok=True)
    if not resume:
        cfg.save(out / "config.json")
    print(json.dumps(info))
    if cfg.codec_checkpoint or (resume and (out / "codec.safetensors").exists()
                                and (out / "codec.json").exists()):
        from ncm_mlx.training.common import build_codec, load_corpus, load_model

        corpus = load_corpus(cfg)
        codec_path = Path(cfg.codec_checkpoint) if cfg.codec_checkpoint else out / "codec.safetensors"
        codec = load_model(build_codec(cfg, corpus.tokenizer.vocab_size), codec_path)
        cmeta = ({"train_seconds": 0.0, "reused": cfg.codec_checkpoint} if cfg.codec_checkpoint
                 else json.loads(codec_path.with_suffix(".json").read_text()))
        print(f"reusing frozen codec {codec_path}")
    else:
        codec, corpus, cmeta = train_codec(cfg, args.codec_steps, resume=resume)
        print("codec val:", json.dumps(cmeta["val"]))
    _, gmeta = train_generator(cfg, codec, corpus, args.generator_steps,
                              resume=resume and has_generator_state,
                              checkpoint=checkpoint, start_step=start_step)
    print("generator val:", json.dumps(gmeta["val"]))
    print(f"train seconds: codec {cmeta['train_seconds']}, generator {gmeta['train_seconds']}")
    return 0


def _show(corpus, gen_out, n: int) -> None:
    from ncm_mlx.text import toygrammar

    tok = corpus.tokenizer
    K = len(gen_out.per_level_tokens)
    for i in range(n):
        print(f"--- sample {i}")
        for k in range(K):
            text = tok.decode(gen_out.per_level_tokens[k][i].tolist())
            print(f"  levels 0..{k}  {text}")
        final = tok.decode(gen_out.tokens[i].tolist())
        c = toygrammar.check(final)
        print(f"  check: {'valid' if c.valid else 'well-formed, agreement wrong' if c.well_formed else 'invalid'}")


def cmd_demo(args) -> int:
    from ncm_mlx.generation.sampler import SampleOptions, generate

    info = require_gpu()
    cfg = Config.load(args.config)
    out = Path(cfg.out_dir)
    if not (out / "generator.safetensors").exists():
        print(f"no checkpoint in {out}; training the quickstart model first")
        cmd_train(argparse.Namespace(config=args.config, codec_steps=None, generator_steps=None))
    codec, gen, corpus = _load(cfg)
    print(f"device: {info['device']} on {info.get('chip')}, MLX {info['mlx']}")
    if cfg.data.corpus == "sudoku":
        return _demo_sudoku(gen, codec, corpus, args)
    print(f"{codec.vq.num_levels} levels {codec.vq.levels}: one generator forward per level\n")
    t0 = time.perf_counter()
    g = generate(gen, codec, args.n, SampleOptions(temperature=args.temperature, top_k=args.top_k,
                                                   seed=args.seed), keep_levels=True)
    np.array(g.tokens)
    elapsed = time.perf_counter() - t0
    _show(corpus, g, args.n)
    print(f"\ngenerated {args.n} sequences in {elapsed * 1e3:.1f} ms (first call, includes graph setup)")
    return 0


def _grid(cells) -> list[str]:
    s = "".join("." if c == "0" else c for c in cells)
    return [" ".join(s[r * 9 : r * 9 + 9]) for r in range(9)]


def _demo_sudoku(gen, codec, corpus, args) -> int:
    from ncm_mlx.generation.sampler import SampleOptions, generate
    from ncm_mlx.text import sudoku

    tok = corpus.tokenizer
    for i in range(args.n):
        j = args.seed + i
        prefix = mx.array(corpus.val_prefix[j : j + 1])
        t0 = time.perf_counter()
        g = generate(gen, codec, 1, SampleOptions(temperature=args.temperature, seed=args.seed),
                     keep_levels=True, prefix_tokens=prefix)
        elapsed = (time.perf_counter() - t0) * 1e3
        puzzle, solution = corpus.val_prefix[j], corpus.val[j]
        pred = np.array(g.tokens[0])
        print(f"--- puzzle {j} ({int((puzzle != 0).sum())} givens)")
        print("  per-level cell accuracy vs reference: " + "  ".join(
            f"L{k}:{float((np.array(t[0]) == solution).mean()):.2f}" for k, t in enumerate(g.per_level_tokens)))
        for a, b in zip(_grid(tok.decode(puzzle)), _grid(tok.decode(pred))):
            print(f"  {a}    →    {b}")
        r = sudoku.score(puzzle, pred, solution)
        print(f"  solved: {r['solved']}  valid grid: {r['valid_grid']}  givens kept: {r['respects_givens']}"
              f"  ({elapsed:.1f} ms incl. first-call setup)")
    return 0


def cmd_eval(args) -> int:
    from ncm_mlx.eval.metrics import codec_report, sample_report
    from ncm_mlx.generation.sampler import SampleOptions, generate
    from ncm_mlx.training.common import param_count
    from ncm_mlx.training.generator import encode_dataset, evaluate_generator, with_prefix

    info = require_gpu()
    cfg = Config.load(args.config)
    codec, gen, corpus = _load(cfg)
    tok = corpus.tokenizer
    results = {"runtime": info, "params": {"codec": param_count(codec), "generator": param_count(gen)}}
    results["codec_val"] = codec_report(codec, corpus.val)
    val_codes = with_prefix(encode_dataset(codec, corpus.val), corpus.val_prefix)
    results["generator_val_teacher_forced"] = evaluate_generator(gen, codec, val_codes)
    if cfg.data.corpus == "sudoku":
        results["sudoku"] = {str(t): _sudoku_eval(gen, codec, corpus, t, args.seeds) for t in args.temperatures}
        print(json.dumps(results, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
        return 0
    samples = {}
    for temp in args.temperatures:
        texts = []
        for s in range(args.seeds):
            g = generate(gen, codec, args.n, SampleOptions(temperature=temp, seed=s))
            texts += [tok.decode(row) for row in np.array(g.tokens).tolist()]
        samples[str(temp)] = sample_report(texts, corpus.train_text, corpus.val_text)
    results["samples"] = samples
    results["reference_train_data"] = sample_report(corpus.train_text, corpus.train_text, corpus.val_text)
    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    return 0


def _sudoku_predict(gen, codec, prefixes: np.ndarray, temperature: float, seed: int, batch: int = 500):
    from ncm_mlx.generation.sampler import SampleOptions, generate

    preds = []
    for i in range(0, len(prefixes), batch):
        p = mx.array(prefixes[i : i + batch])
        g = generate(gen, codec, p.shape[0], SampleOptions(temperature=temperature, seed=seed + i), prefix_tokens=p)
        preds.append(np.array(g.tokens))
    return np.concatenate(preds)


UPSTREAM_SUDOKU = Path("artifacts/upstream-sudoku")


def _sudoku_sets(corpus) -> dict:
    """Own validation boards plus the published NCM test boards when downloaded."""
    from ncm_mlx.text import upstream_sudoku

    sets = {"val": (corpus.val_prefix, corpus.val)}
    if UPSTREAM_SUDOKU.exists():
        for level in ("easy", "medium", "hard"):
            sets[f"upstream-{level}"] = upstream_sudoku.load(UPSTREAM_SUDOKU, level)
    return sets


def _sudoku_scores(predict, corpus, temperature: float, seeds: int) -> dict:
    """``predict(prefixes, temperature, seed) -> (N, 81)`` scored on every board set."""
    from ncm_mlx.eval.metrics import sudoku_report

    out = {}
    for name, (puzzles, solutions) in _sudoku_sets(corpus).items():
        reports = [sudoku_report(puzzles, predict(puzzles, temperature, s), solutions)
                   for s in range(1 if temperature == 0 else seeds)]
        out[name] = {k: round(float(np.mean([r[k] for r in reports])), 4) for k in reports[0]}
    return out


def _sudoku_eval(gen, codec, corpus, temperature: float, seeds: int) -> dict:
    return _sudoku_scores(lambda p, t, s: _sudoku_predict(gen, codec, p, t, s), corpus, temperature, seeds)


def cmd_bench(args) -> int:
    from ncm_mlx.eval.benchmark import bench_generation

    info = require_gpu()
    cfg = Config.load(args.config)
    codec, gen, corpus = _load(cfg)
    warmup = 3
    rows = bench_generation(gen, codec, tuple(args.batch_sizes), warmup=warmup, iters=args.iters,
                            temperature=args.temperature, prefixes=corpus.val_prefix)
    result = {"runtime": info, "seq_len": cfg.data.seq_len, "levels": cfg.codec.levels,
              "temperature": args.temperature, "warmup": warmup, "iterations": args.iters,
              "sampler_compiled": False, "synchronization": "numpy_host_copy", "rows": rows}
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0


def cmd_baseline(args) -> int:
    from ncm_mlx.baselines.ar import build_ar, train_ar
    from ncm_mlx.eval.benchmark import bench_sampler
    from ncm_mlx.eval.metrics import sample_report
    from ncm_mlx.training.common import load_corpus, load_model, param_count

    info = require_gpu()
    cfg = Config.load(args.config)
    corpus = load_corpus(cfg)
    path = Path(cfg.out_dir) / "ar.safetensors"
    if path.exists() and not args.retrain:
        model = load_model(build_ar(cfg, corpus.tokenizer.vocab_size), path)
        meta = json.loads(path.with_suffix(".json").read_text())
    else:
        model, meta = train_ar(cfg, corpus, args.steps)
    tok = corpus.tokenizer
    samples = {}
    for temp in args.temperatures:
        if cfg.data.corpus == "sudoku":
            def predict(prefixes, t, seed):
                chunks = [prefixes[i : i + 500] for i in range(0, len(prefixes), 500)]
                return np.concatenate([np.array(model.generate(len(c), t, seed=seed + i, prefix=mx.array(c)))
                                       for i, c in enumerate(chunks)])

            samples[str(temp)] = _sudoku_scores(predict, corpus, temp, args.seeds)
            continue
        texts = []
        for s in range(args.seeds):
            texts += [tok.decode(r) for r in np.array(model.generate(args.n, temp, seed=s)).tolist()]
        samples[str(temp)] = sample_report(texts, corpus.train_text, corpus.val_text)
    result = {"runtime": info, "params": param_count(model), "train_seconds": meta["train_seconds"],
              "val_ce_per_char": meta["val_ce_per_char"], "samples": samples}
    if args.bench:
        bench_prefix = corpus.val_prefix

        def sample(b, seed):
            p = None if bench_prefix is None else mx.array(np.resize(bench_prefix, (b, bench_prefix.shape[1])))
            return model.generate(b, 1.0, seed, prefix=p)

        result["bench"] = bench_sampler(sample, cfg.data.seq_len,
                                        cfg.data.seq_len, tuple(args.batch_sizes), iters=args.iters)
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ncm_mlx", description=__doc__,
        epilog="NCM_MLX_MEMORY_LIMIT_GB (default 16, or 85%% of a smaller GPU) bounds MLX memory; the buffer cache is capped at an eighth of it, at most 2 GB.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", default=DEFAULT_CONFIG)
        sp.set_defaults(fn=fn)
        return sp

    add("smoke", cmd_smoke, "verify GPU execution, shapes and one untrained generation")
    sp = add("train", cmd_train, "train codec then generator")
    sp.add_argument("--codec-steps", type=int)
    sp.add_argument("--generator-steps", type=int)
    sp.add_argument("--resume", action="store_true", help="restore full training state from out_dir")
    sp.add_argument("--generator-checkpoint", help="continue weights with a fresh optimizer in a new out_dir")
    sp.add_argument("--generator-start-step", type=int, default=0,
                    help="completed steps for a weights-only continuation; retain the original schedule")
    sp = add("demo", cmd_demo, "show coarse-to-fine samples")
    sp.add_argument("--n", type=int, default=4)
    sp.add_argument("--temperature", type=float, default=1.0)
    sp.add_argument("--top-k", type=int, default=0)
    sp.add_argument("--seed", type=int, default=0)
    sp = add("eval", cmd_eval, "codec reconstruction and sample quality")
    sp.add_argument("--n", type=int, default=512)
    sp.add_argument("--seeds", type=int, default=2)
    sp.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 0.7, 0.0])
    sp.add_argument("--out")
    sp = add("bench", cmd_bench, "synchronized generation latency")
    sp.add_argument("--temperature", type=float, default=1.0,
                    help="sampling temperature; 0 selects greedy decoding (default: 1.0)")
    sp.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 64, 256])
    sp.add_argument("--iters", type=int, default=30)
    sp.add_argument("--out")
    sp = add("baseline", cmd_baseline, "train/evaluate/benchmark the autoregressive baseline")
    sp.add_argument("--steps", type=int)
    sp.add_argument("--retrain", action="store_true")
    sp.add_argument("--n", type=int, default=512)
    sp.add_argument("--seeds", type=int, default=2)
    sp.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 0.7, 0.0])
    sp.add_argument("--bench", action="store_true")
    sp.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 64, 256])
    sp.add_argument("--iters", type=int, default=30)
    sp.add_argument("--out")
    from ncm_mlx.official.cli import register, resolve_defaults

    register(sub)
    args = p.parse_args(argv)
    if args.cmd == "official":
        resolve_defaults(args)
    device_gb = mx.device_info().get("memory_size", 0) / 2**30
    default_gb = min(16.0, 0.85 * device_gb) if device_gb else 16.0
    limit_gb = float(os.environ.get("NCM_MLX_MEMORY_LIMIT_GB", default_gb))
    mx.set_memory_limit(int(limit_gb * 2**30))
    mx.set_cache_limit(int(min(limit_gb / 8, 2) * 2**30))
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
