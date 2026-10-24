"""Synchronized generation latency and throughput.

Every timed iteration ends in ``mx.eval`` on the decoded tokens and a host copy
of the result, so the numbers include all GPU work and the device-to-host sync;
lazy graph construction alone is never timed.
"""

from __future__ import annotations

import statistics
import time

import mlx.core as mx
import numpy as np

from ncm_mlx.generation.sampler import SampleOptions, generate


def _percentile(xs: list[float], q: float) -> float:
    return float(np.percentile(np.array(xs), q))


def bench_generation(gen, codec, batch_sizes=(1, 4, 8, 16, 64), warmup: int = 3, iters: int = 20,
                     temperature: float = 1.0, prefixes: np.ndarray | None = None) -> list[dict]:
    def sample(b, seed):
        p = None if prefixes is None else mx.array(np.resize(prefixes, (b, prefixes.shape[1])))
        return generate(gen, codec, b, SampleOptions(temperature=temperature, seed=seed), prefix_tokens=p).tokens

    return bench_sampler(sample, codec.vq.seq_len, codec.vq.num_levels, batch_sizes, warmup, iters)


def bench_sampler(sample, seq_len: int, forwards: int, batch_sizes, warmup: int = 3, iters: int = 20) -> list[dict]:
    """Time ``sample(batch, seed) -> tokens`` with a host copy of the result as the sync point."""
    rows = []
    for b in batch_sizes:
        mx.clear_cache()
        cold_start = time.perf_counter()
        np.array(sample(b, 0))
        cold = time.perf_counter() - cold_start
        for i in range(warmup):
            np.array(sample(b, i + 1))
        mx.reset_peak_memory()
        times = []
        for i in range(iters):
            t0 = time.perf_counter()
            np.array(sample(b, 1000 + i))
            times.append(time.perf_counter() - t0)
        p50 = _percentile(times, 50)
        rows.append({
            "batch": b,
            "first_call_ms": round(cold * 1e3, 2),
            "p50_ms": round(p50 * 1e3, 3),
            "p95_ms": round(_percentile(times, 95) * 1e3, 3),
            "mean_ms": round(statistics.mean(times) * 1e3, 3),
            "stdev_ms": round(statistics.stdev(times) * 1e3, 3),
            "sequences_per_s": round(b / p50, 1),
            "chars_per_s": round(b * seq_len / p50, 0),
            "peak_memory_mb": round(mx.get_peak_memory() / 2**20, 1),
            "model_forwards": forwards,
        })
    return rows
