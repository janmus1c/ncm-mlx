"""Upstream PyTorch reference runs for the released NCM checkpoints.

``sudoku`` solves published boards with upstream ``generate`` (greedy) and
scores them like the upstream evaluation. ``bench`` times generation plus
decoding with a host copy of the tokens as the synchronization point, matching
``python -m ncm_mlx official bench``. Run with ephemeral dependencies, e.g.::

    uv run --with torch --with omegaconf --with hydra-core --with regex \\
        python scripts/official_torch_reference.py bench --upstream PATH --task tinystories --device cpu
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _setup(args):
    import torch
    from official_parity import build_upstream

    from ncm_mlx.official import spec

    task = spec.get(args.task)
    src = ROOT / "artifacts/official" / task.remote_dir
    _, vqvae, net, _, inference = build_upstream(Path(args.upstream), src)
    dev = torch.device(args.device)
    vqvae.to(dev)
    net.to(dev)
    if args.device == "mps":
        for name, buf in list(net.named_buffers()):
            if buf.dtype == torch.float64:
                raise SystemExit(f"float64 buffer {name} is unsupported on MPS")
    return task, vqvae, net, inference, dev


def _runner(task, vqvae, net, inference, dev, temperature):
    import torch

    unigram = inference.get("level0_unigram")

    def run(prefix, seed):
        torch.manual_seed(seed)
        l0 = None
        if unigram is not None:
            l0 = torch.multinomial(unigram, prefix.shape[0], replacement=True).view(-1, 1).to(dev)
        with torch.no_grad():
            g = net.generate(l0, prefix, temperature=temperature, top_k=0, top_p=1.0)
            _, zl = vqvae.vq.idx_to_zhats(g, return_zlevels=True)
            return vqvae.decoder(zl[-1]).argmax(-1).cpu().numpy()

    return run


def cmd_sudoku(args) -> dict:
    import torch

    from ncm_mlx.official.cli import sudoku_rows, sudoku_valid
    from ncm_mlx.official.sample import SUDOKU_CHARS

    task, vqvae, net, inference, dev = _setup(args)
    run = _runner(task, vqvae, net, inference, dev, 0.0)
    puzzles, solutions = sudoku_rows(task, Path(args.boards), args.n)

    def decode(r):
        return "".join(SUDOKU_CHARS[int(t)] for t in r if int(t) != 11)

    hyps = []
    t0 = time.perf_counter()
    for i in range(0, len(puzzles), args.batch):
        hyps += [decode(r) for r in run(torch.tensor(puzzles[i : i + args.batch], device=dev), 0)]
    refs = [decode(r) for r in solutions]
    solved = sum(h == r for h, r in zip(hyps, refs))
    return {"task": task.name, "implementation": "upstream PyTorch", "device": args.device,
            "torch": torch.__version__, "boards": len(refs), "solved": solved,
            "solved_pct": round(100 * solved / len(refs), 2),
            "valid_grid_pct": round(100 * sum(map(sudoku_valid, hyps)) / len(refs), 2),
            "seconds": round(time.perf_counter() - t0, 2), "hypotheses": hyps if args.keep else None}


def cmd_bench(args) -> dict:
    import torch

    from ncm_mlx.official.cli import sudoku_rows

    task, vqvae, net, inference, dev = _setup(args)
    temperature = task.temperature if args.temperature is None else args.temperature
    run = _runner(task, vqvae, net, inference, dev, temperature)
    if task.tokenizer == "sudoku":
        base = torch.tensor(sudoku_rows(task, Path(args.boards), 0)[0], device=dev)

        def prefix(b):
            return base[:b]
    else:
        def prefix(b):
            return torch.full((b, 1), task.eos_id, dtype=torch.long, device=dev)

    rows = []
    for b in args.batch_sizes:
        t0 = time.perf_counter()
        run(prefix(b), 0)
        cold = time.perf_counter() - t0
        for i in range(args.warmup):
            run(prefix(b), i + 1)
        times = []
        for i in range(args.iters):
            t0 = time.perf_counter()
            run(prefix(b), 1000 + i)
            times.append(time.perf_counter() - t0)
        p50 = float(np.percentile(times, 50))
        rows.append({"batch": b, "first_call_ms": round(cold * 1e3, 2), "p50_ms": round(p50 * 1e3, 3),
                     "p95_ms": round(float(np.percentile(times, 95)) * 1e3, 3),
                     "mean_ms": round(statistics.mean(times) * 1e3, 3),
                     "stdev_ms": round(statistics.stdev(times) * 1e3, 3), "sequences_per_s": round(b / p50, 1)})
    return {"task": task.name, "implementation": "upstream PyTorch", "device": args.device,
            "torch": torch.__version__, "threads": torch.get_num_threads(), "machine": platform.machine(),
            "temperature": temperature, "warmup": args.warmup, "iterations": args.iters,
            "synchronization": "tokens copied to host (.cpu().numpy())", "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["sudoku", "bench"])
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--task", default="tinystories")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--boards", default=str(ROOT / "artifacts/upstream-sudoku"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--keep", action="store_true", help="include every hypothesis in the output")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out")
    args = ap.parse_args()
    result = (cmd_sudoku if args.mode == "sudoku" else cmd_bench)(args)
    text = json.dumps(result, indent=2)
    print(text if not args.keep else json.dumps({k: v for k, v in result.items() if k != "hypotheses"}, indent=2))
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
