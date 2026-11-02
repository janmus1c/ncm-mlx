"""``python -m ncm_mlx official ...``: fetch, convert and run the released NCM checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ncm_mlx.official import spec as specs

ARTIFACTS = Path("artifacts/official")


def _src(task: specs.TaskSpec, args) -> Path:
    return Path(args.src) if getattr(args, "src", None) else ARTIFACTS / task.remote_dir


def _converted(task: specs.TaskSpec, args) -> Path:
    return Path(args.weights) if getattr(args, "weights", None) else ARTIFACTS / task.remote_dir / "mlx"


def _device(args) -> dict:
    import mlx.core as mx

    from ncm_mlx.device import require_gpu, runtime_info

    if args.cpu:
        mx.set_default_device(mx.cpu)
        return runtime_info()
    return require_gpu()


def cmd_fetch(args) -> int:
    from ncm_mlx.official.convert import fetch

    task = specs.get(args.task)
    fetch(task, _src(task, args))
    return 0


def cmd_convert(args) -> int:
    from ncm_mlx.official.convert import convert

    task = specs.get(args.task)
    print(json.dumps(convert(task, _src(task, args), _converted(task, args)), indent=2))
    return 0


def cmd_codec(args) -> int:
    import mlx.core as mx

    from ncm_mlx.official import sample

    info = _device(args)
    task = specs.get(args.task)
    task, codec = sample.load_codec(_converted(task, args))
    rows = json.loads(Path(args.ids).read_text())
    if not isinstance(rows, list) or not rows or any(
            not isinstance(r, list) or len(r) != task.codec.seq_len
            or not all(type(t) is int and 0 <= t < task.codec.vocab_size for t in r) for r in rows):
        raise SystemExit(f"{args.ids}: expected rows of {task.codec.seq_len} integer token ids below "
                         f"{task.codec.vocab_size}")
    codes = codec.encode(mx.array(np.array(rows, dtype=np.int32)))
    decoded = mx.argmax(codec.decode_logits(codec.cumulative(codes)[-1]), axis=-1)
    result = {"task": task.name, "levels": list(codec.levels), "runtime": info,
              "codes": [np.array(c).tolist() for c in codes], "decoded": np.array(decoded).tolist()}
    print(json.dumps(result))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0


def _flat(text: str, limit: int) -> str:
    text = text.replace("\n", " ⏎ ")
    return text if not limit or len(text) <= limit else text[: limit - 1] + "…"


def _sampler_kwargs(args) -> dict:
    return {"prefix_cache": not args.no_prefix_cache, "compiled": args.compile}


def _gpu_utilization() -> str | None:
    """Metal ``Device Utilization %`` from ioreg, recorded next to benchmark results; None when unavailable."""
    import re
    import subprocess

    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True,
                             check=False, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r'"Device Utilization %"=(\d+)', out)
    return m.group(1) if m else None


def cmd_generate(args) -> int:
    import mlx.core as mx

    from ncm_mlx.official import sample

    info = _device(args)
    task = specs.get(args.task)
    if task.tokenizer != "bpe":
        raise SystemExit("generate is for unconditional text checkpoints; use `official sudoku` for Sudoku")
    m = sample.load(_converted(task, args))
    level0 = mx.full((args.n, 1), args.level0, dtype=mx.int32) if args.level0 is not None else None
    t0 = time.perf_counter()
    g = sample.generate(m, sample.unconditional_prefix(m, args.n), temperature=args.temperature,
                        top_k=args.top_k, top_p=args.top_p, seed=args.seed, level0=level0,
                        keep_levels=args.levels, **_sampler_kwargs(args))
    tokens = np.array(g.tokens)
    elapsed = time.perf_counter() - t0
    print(f"{task.name}: {len(m.codec.levels)} levels {m.codec.levels}, "
          f"{len(m.codec.levels) - m.gen.first} generator forwards, device {info['device']}")
    for i in range(args.n):
        print(f"--- sample {i} (level-0 code {int(np.array(g.codes[0])[i, 0])})")
        if args.levels:
            for k, lt in enumerate(g.per_level_tokens):
                print(f"  levels 0..{k:<2d} {_flat(m.decode(np.array(lt[i])), args.width)}")
            print("  final:")
        print(m.decode(tokens[i]))
    print(f"\n{args.n} samples in {elapsed * 1e3:.0f} ms (first call, includes setup)")
    return 0


def sudoku_rows(task: specs.TaskSpec, boards: Path, n: int) -> tuple[list[list[int]], list[list[int]]]:
    from ncm_mlx.text import upstream_sudoku

    d = boards / task.name.split("-", 1)[1]
    puzzles = upstream_sudoku.load_tokens(d / "puzzles.pickle")
    solutions = upstream_sudoku.load_tokens(d / "solutions.pickle")
    if len(puzzles) != len(solutions) or not puzzles:
        raise SystemExit(f"{d}: puzzle and solution counts differ or are empty")
    n = n or len(puzzles)
    return puzzles[:n], solutions[:n]


def sudoku_valid(text: str) -> bool:
    g = text.replace("|", "")
    if len(g) != 81 or not g.isdigit() or "0" in g:
        return False
    rows = [g[i * 9 : i * 9 + 9] for i in range(9)]
    cols = ["".join(r[j] for r in rows) for j in range(9)]
    boxes = ["".join(rows[r][c] for r in range(br, br + 3) for c in range(bc, bc + 3))
             for br in (0, 3, 6) for bc in (0, 3, 6)]
    return all(sorted(u) == list("123456789") for u in rows + cols + boxes)


def cmd_sudoku(args) -> int:
    import mlx.core as mx

    from ncm_mlx.official import sample

    info = _device(args)
    task = specs.get(args.task)
    m = sample.load(_converted(task, args))
    puzzles, solutions = sudoku_rows(task, Path(args.boards), args.n)
    hyps = []
    t0 = time.perf_counter()
    for i in range(0, len(puzzles), args.batch):
        p = mx.array(np.array(puzzles[i : i + args.batch], dtype=np.int32))
        g = sample.generate(m, p, temperature=args.temperature, seed=args.seed + i, **_sampler_kwargs(args))
        hyps += [m.decode(r) for r in np.array(g.tokens)]
    elapsed = time.perf_counter() - t0
    refs = [m.decode(r) for r in solutions]
    solved = sum(h == r for h, r in zip(hyps, refs))
    cells = float(np.mean([np.mean([a == b for a, b in zip(h.replace("|", ""), r.replace("|", ""))])
                           for h, r in zip(hyps, refs)]))
    result = {"task": task.name, "boards": len(refs), "sample": f"first {len(refs)} of the published test set",
              "temperature": args.temperature, "solved": solved, "solved_pct": round(100 * solved / len(refs), 2),
              "cell_pct": round(100 * cells, 2),
              "valid_grid_pct": round(100 * sum(map(sudoku_valid, hyps)) / len(refs), 2),
              "seconds": round(elapsed, 2), "runtime": info}
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0


def cmd_bench(args) -> int:
    import mlx.core as mx

    from ncm_mlx.eval.benchmark import bench_sampler
    from ncm_mlx.official import sample

    info = _device(args)
    task = specs.get(args.task)
    m = sample.load(_converted(task, args))
    if task.tokenizer == "sudoku":
        puzzles, _ = sudoku_rows(task, Path(args.boards), 0)
        base = np.array(puzzles, dtype=np.int32)

        def prefix(b):
            return mx.array(base[:b])
    else:
        def prefix(b):
            return sample.unconditional_prefix(m, b)

    forwards = len(m.codec.levels) - m.gen.first
    modes = {"eager": {"prefix_cache": False, "compiled": False},
             "cached": {"prefix_cache": True, "compiled": False},
             "cached+compiled": {"prefix_cache": True, "compiled": True}}
    def runner(kwargs):
        def run(b, seed):
            return sample.generate(m, prefix(b), temperature=args.temperature, seed=seed, **kwargs).tokens
        return run

    result = {"task": task.name, "runtime": info, "temperature": args.temperature, "warmup": args.warmup,
              "iterations": args.iters, "synchronization": "numpy_host_copy",
              "gpu_utilization_pct_before": _gpu_utilization(), "modes": {}}
    for mode in args.modes:
        kwargs = modes[mode]
        rows = bench_sampler(runner(kwargs), task.codec.seq_len, forwards, tuple(args.batch_sizes), warmup=args.warmup,
                             iters=args.iters)
        result["modes"][mode] = {**kwargs, "rows": rows}
        print(f"{mode}: " + ", ".join(f"b={r['batch']} p50 {r['p50_ms']} ms (first call {r['first_call_ms']} ms)"
                                       for r in rows), file=sys.stderr)
    result["gpu_utilization_pct_after"] = _gpu_utilization()
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0


def _text_cfg(args):
    from ncm_mlx.official.train_config import TextConfig

    return TextConfig.load(args.config)


def cmd_prepare_data(args) -> int:
    from ncm_mlx.official import tinystories

    cfg = _text_cfg(args)
    d = cfg.data
    if d.limit == 0:
        tinystories.fetch(Path(d.raw_dir))
    m = tinystories.prepare(Path(d.raw_dir), Path(d.cache_root), Path(d.bpe), d.seq_len, args.workers, d.limit,
                            prefix_len=d.prefix_len)
    print(json.dumps(m, indent=2))
    return 0


def cmd_train_text(args) -> int:
    from ncm_mlx.official import ar, trainer

    info = _device(args)
    cfg = _text_cfg(args)
    stage = {"train-codec": "codec", "train-generator": "generator", "train-ar": "ar"}[args.official_cmd]
    out = Path(cfg.out_dir)
    state = out / f"{stage}-training.safetensors"
    if state.exists() and not args.resume:
        raise SystemExit(f"{state} exists; pass --resume or choose a new out_dir")
    if args.resume and not state.exists():
        raise SystemExit(f"no {state} to resume")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stage}-config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    print(json.dumps({"stage": stage, "config": args.config, "runtime": info}), flush=True)
    fn = {"codec": trainer.train_codec, "generator": trainer.train_generator, "ar": ar.train_ar}[stage]
    fn(cfg, resume=args.resume, log=lambda m: print(m, flush=True))
    return 0


def cmd_continue_text(args) -> int:
    import mlx.core as mx

    from ncm_mlx.official import sample, tinystories

    _device(args)
    m = sample.load(args.weights)
    P = m.task.generator.prefix_len
    if P <= 1:
        raise SystemExit("this checkpoint has a one-token prefix; train with data.prefix_len > 0 to continue prompts")
    text = args.prompt if args.prompt.startswith("\n") else "\n" + args.prompt
    ids = m.bpe.encode(text)
    if len(ids) > P:
        print(f"prompt has {len(ids)} tokens; the model reads the first {P}", file=sys.stderr)
    if len(ids) < P:
        print(f"prompt has {len(ids)} tokens, fewer than the {P} used in training; the rest is masked padding",
              file=sys.stderr)
    row = (ids[:P] + [tinystories.PAD_ID] * P)[:P]
    prefix = mx.broadcast_to(mx.array(row, dtype=mx.int32), (args.n, P))
    g = sample.generate(m, prefix, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, seed=args.seed)
    shown = m.bpe.decode([t for t in row if t != tinystories.PAD_ID], skip_special=True)
    for i, tokens in enumerate(np.array(g.tokens)):
        print(f"--- continuation {i}\n{shown}[|]{m.decode(tokens)}")
    return 0


def cmd_eval_text(args) -> int:
    from ncm_mlx.official.text_run import run_text_eval

    info = _device(args)
    cfg = _text_cfg(args)
    report = run_text_eval(cfg, args, info)
    out = Path(args.out) if args.out else Path("runs") / f"{cfg.name}-eval" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report written to {out}")
    return 0


def register_text(osub) -> None:
    sp = osub.add_parser("prepare-data", help="download and tokenize the pinned TinyStories files")
    sp.add_argument("--config", required=True)
    sp.add_argument("--workers", type=int, default=0)
    sp.set_defaults(fn=cmd_prepare_data)
    for name, help_ in (("train-codec", "train the codec (stage 1)"),
                        ("train-generator", "train the next-concept generator on the frozen codec (stage 2)"),
                        ("train-ar", "train the autoregressive baseline")):
        sp = osub.add_parser(name, help=help_)
        sp.add_argument("--config", required=True)
        sp.add_argument("--resume", action="store_true", help="restore the full training state from out_dir")
        sp.add_argument("--cpu", action="store_true")
        sp.set_defaults(fn=cmd_train_text)
    sp = osub.add_parser("continue-text", help="continue a prompt with a prefix-conditioned text checkpoint")
    sp.add_argument("--weights", required=True, help="<out_dir>/export of a run with data.prefix_len > 0")
    sp.add_argument("--prompt", required=True)
    sp.add_argument("--n", type=int, default=2)
    sp.add_argument("--temperature", type=float, default=1.0)
    sp.add_argument("--top-k", type=int, default=0)
    sp.add_argument("--top-p", type=float, default=1.0)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--cpu", action="store_true")
    sp.set_defaults(fn=cmd_continue_text)
    sp = osub.add_parser("eval-text", help="held-out, free-running and scorer evaluation of text models")
    sp.add_argument("--config", required=True)
    sp.add_argument("--models", nargs="+", default=["data", "official", "ncm", "ar"],
                    choices=["data", "official", "ncm", "ar"])
    sp.add_argument("--n", type=int, default=256, help="samples per seed")
    sp.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    sp.add_argument("--batch", type=int, default=128)
    sp.add_argument("--test-rows", type=int, default=4096)
    sp.add_argument("--no-scorer", action="store_true", help="skip generative perplexity")
    sp.add_argument("--no-bench", action="store_true", help="skip latency")
    sp.add_argument("--official-weights", default=str(ARTIFACTS / "tinystories" / "mlx"))
    sp.add_argument("--cpu", action="store_true")
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_eval_text)


def register(sub) -> None:
    p = sub.add_parser("official", help="fetch, convert and run the released NCM checkpoints")
    osub = p.add_subparsers(dest="official_cmd", required=True)
    tasks = list(specs.TASKS)
    register_text(osub)

    def add(name, fn, help_, default_task="tinystories", choices=tasks):
        sp = osub.add_parser(name, help=help_)
        sp.add_argument("--task", choices=choices, default=default_task)
        sp.set_defaults(fn=fn)
        return sp

    sp = add("fetch", cmd_fetch, "download the pinned release files and verify SHA-256")
    sp.add_argument("--src", help="destination (default artifacts/official/<task dir>)")
    sp = add("convert", cmd_convert, "convert release .pt files to MLX safetensors")
    sp.add_argument("--src", help="release files (default artifacts/official/<task dir>)")
    sp.add_argument("--weights", help="output directory (default <src>/mlx)")
    sp = add("codec", cmd_codec, "encode token-id rows with a converted codec (the only command for XSum and WMT14)",
             "xsum")
    sp.add_argument("--ids", required=True, help="JSON list of token-id rows, one row of seq_len ids per input")
    sp.add_argument("--weights", help="converted checkpoint directory")
    sp.add_argument("--cpu", action="store_true", help="run on mx.cpu instead of the GPU")
    sp.add_argument("--out")
    for name, fn, help_, task in (
            ("generate", cmd_generate, "sample text and show per-level refinement", "tinystories"),
            ("sudoku", cmd_sudoku, "solve the published Sudoku test boards", "sudoku-easy"),
            ("bench", cmd_bench, "synchronized generation latency", "tinystories")):
        sp = add(name, fn, help_, task, specs.GENERATOR_TASKS)
        sp.add_argument("--weights", help="converted checkpoint directory")
        sp.add_argument("--cpu", action="store_true", help="run on mx.cpu instead of the GPU")
        sp.add_argument("--seed", type=int, default=0)
        if name != "bench":
            sp.add_argument("--no-prefix-cache", action="store_true",
                            help="recompute the prefix in every level (reference path)")
            sp.add_argument("--compile", action="store_true", help="mx.compile the per-level step")
        sp.add_argument("--temperature", type=float, default=None,
                        help="default: the released config (1.0 TinyStories, 0 Sudoku)")
        if name == "generate":
            sp.add_argument("--n", type=int, default=2)
            sp.add_argument("--top-k", type=int, default=0)
            sp.add_argument("--top-p", type=float, default=1.0)
            sp.add_argument("--level0", type=int, help="fix the level-0 code instead of sampling the unigram")
            sp.add_argument("--levels", action="store_true", help="decode after every level")
            sp.add_argument("--width", type=int, default=150, help="truncate per-level lines (0: no limit)")
        if name in ("sudoku", "bench"):
            sp.add_argument("--boards", default="artifacts/upstream-sudoku")
        if name == "sudoku":
            sp.add_argument("--n", type=int, default=200, help="first N boards (0: all)")
            sp.add_argument("--batch", type=int, default=100)
            sp.add_argument("--out")
        if name == "bench":
            sp.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
            sp.add_argument("--modes", nargs="+", choices=["eager", "cached", "cached+compiled"],
                            default=["eager", "cached", "cached+compiled"])
            sp.add_argument("--iters", type=int, default=20)
            sp.add_argument("--warmup", type=int, default=3)
            sp.add_argument("--out")


def resolve_defaults(args: argparse.Namespace) -> None:
    if getattr(args, "temperature", 0) is None:
        args.temperature = specs.get(args.task).temperature
