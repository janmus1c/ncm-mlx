---
title: Status
status: current
updated: 2026-10-24
---

# Status

## Works today (implemented and tested)

- Multiscale residual VQ codec in MLX: 6 levels, EMA codebooks, dead-code restart, fine-level dropout. 100% held-out sentence reconstruction on TinyWorld, 100% codebook utilisation.
- Next-concept generator with block-diagonal attention: one forward per level. Per-level inference proven equal to the training forward (exact on CPU).
- Level-by-level sampling with seeded determinism; per-level decodes show the refinement.
- Two-stage training from one command, best-validation checkpoint, weights saved and reloaded from disk.
- Character-level AR baseline (same size, KV cache) trained and measured on identical data.
- Synchronized benchmark at batch sizes 1–256 with p50/p95 and peak memory.
- Conditional generation from a token prefix (Sudoku puzzle cells); same-size AR baseline with prefix prefill.
- Sudoku task: synthetic training data, evaluation on the published NCM test boards, an OpenAI-compatible LLM baseline.
- `scripts/guarded_run.sh`: memory watchdog for long runs.
- Codec and generator full-state checkpoints, graceful pause and `--resume`; CPU tests match uninterrupted training exactly, including compiled generator steps. Mac experiments can run under launchd independently of a terminal. See `RECOVERY.md`.
- E17 was killed with its terminal session at 15:05. Its step-30,000 weights survive; a separate weights-only continuation completed step 32,000 and solves 71.25 / 31.75 / 3.70% of easy/medium/hard boards. Its optimizer and random state reset; see `EXPERIMENTS.md`. The independent remote E18 process survived and remains in progress.

## Measured (see EXPERIMENTS.md, E1)

| | NCM | AR baseline |
|---|---|---|
| Valid sentences, T=1.0 | 89.8% | 98.6% |
| Latency p50, batch 1 | 4.5 ms | ~20–30 ms |
| Throughput, batch 256 | 6,769 seq/s | 5,137 seq/s |
| Train time | 52 s codec + 266 s generator | 134 s |

All numbers are for 5M-parameter toy models on a synthetic language, on a desktop with other GPU services running.

**Sudoku (E4–E16).** Best NCM (E15: stronger conditioning corruption, unique puzzles, 8.2M training sequences): 96.2 / 76.6 / 31.9% solved on the upstream easy/medium/hard sets (paper: 99.5 / 96.5 / 86.1% with a 4× larger model and 51M sequences). At E11's matched 2.05M-sequence budget, the same-size AR baseline reaches 20.5 / 4.9 / 0.5%, against NCM's 55.2 / 18.1 / 1.6%. Trains on Apple Silicon, NVIDIA (RTX 3080) and NVIDIA A40 GPUs.

## Not done

- The upstream Sudoku result is not reproduced. Width-512 runs E10/E12 did not close the gap; a paper-scale training budget remains pending.
- No compatibility with the released NCM checkpoints.
- No `mx.compile`, quantisation or KV caching of the prefix in the NCM path.
- No power or energy measurements (no instrumentation without sudo).
- GitHub remote is private; a change to public visibility is pending.
