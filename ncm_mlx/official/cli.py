"""``python -m ncm_mlx official ...``: fetch, convert and run the released NCM checkpoints."""

from __future__ import annotations

import argparse
import json
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


def _flat(text: str, limit: int) -> str:
    text = text.replace("\n", " ⏎ ")
    return text if not limit or len(text) <= limit else text[: limit - 1] + "…"


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
                        keep_levels=args.levels)
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
        g = sample.generate(m, p, temperature=args.temperature, seed=args.seed + i)
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

    def run(b, seed):
        return sample.generate(m, prefix(b), temperature=args.temperature, seed=seed).tokens

    forwards = len(m.codec.levels) - m.gen.first
    rows = bench_sampler(run, task.codec.seq_len, forwards, tuple(args.batch_sizes), warmup=args.warmup,
                         iters=args.iters)
    result = {"task": task.name, "runtime": info, "temperature": args.temperature, "warmup": args.warmup,
              "iterations": args.iters, "synchronization": "numpy_host_copy", "sampler_compiled": False,
              "rows": rows}
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0


def register(sub) -> None:
    p = sub.add_parser("official", help="fetch, convert and run the released NCM checkpoints")
    osub = p.add_subparsers(dest="official_cmd", required=True)
    tasks = list(specs.TASKS)

    def add(name, fn, help_, default_task="tinystories"):
        sp = osub.add_parser(name, help=help_)
        sp.add_argument("--task", choices=tasks, default=default_task)
        sp.set_defaults(fn=fn)
        return sp

    sp = add("fetch", cmd_fetch, "download the pinned release files and verify SHA-256")
    sp.add_argument("--src", help="destination (default artifacts/official/<task dir>)")
    sp = add("convert", cmd_convert, "convert release .pt files to MLX safetensors")
    sp.add_argument("--src", help="release files (default artifacts/official/<task dir>)")
    sp.add_argument("--weights", help="output directory (default <src>/mlx)")
    for name, fn, help_, task in (
            ("generate", cmd_generate, "sample text and show per-level refinement", "tinystories"),
            ("sudoku", cmd_sudoku, "solve the published Sudoku test boards", "sudoku-easy"),
            ("bench", cmd_bench, "synchronized generation latency", "tinystories")):
        sp = add(name, fn, help_, task)
        sp.add_argument("--weights", help="converted checkpoint directory")
        sp.add_argument("--cpu", action="store_true", help="run on mx.cpu instead of the GPU")
        sp.add_argument("--seed", type=int, default=0)
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
            sp.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16])
            sp.add_argument("--iters", type=int, default=20)
            sp.add_argument("--warmup", type=int, default=3)
            sp.add_argument("--out")


def resolve_defaults(args: argparse.Namespace) -> None:
    if getattr(args, "temperature", 0) is None:
        args.temperature = specs.get(args.task).temperature
