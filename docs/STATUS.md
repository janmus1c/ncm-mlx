---
title: Status
status: current
updated: 2026-10-30
---

# Status

## Works today (implemented and tested)

- Multiscale residual VQ codec in MLX: 6 levels, EMA codebooks, dead-code restart, fine-level dropout. 100% held-out sentence reconstruction on TinyWorld, 100% codebook utilisation.
- Next-concept generator with block-diagonal attention: one forward per level. Per-level inference proven equal to the training forward (exact on CPU).
- Level-by-level sampling with seeded determinism; per-level decodes show the refinement.
- Two-stage training from one command, best-validation checkpoint, weights saved and reloaded from disk.
- Character-level AR baseline (same size, KV cache) trained and measured on identical data.
- Synchronized benchmark at batch sizes 1–256 with p50/p95 and peak MLX memory; configurable sampling temperature and recorded timing settings. See [benchmark methodology](benchmarks.md).
- Conditional generation from a token prefix (Sudoku puzzle cells); same-size AR baseline with prefix prefill.
- Sudoku task: synthetic training data, evaluation on the published NCM test boards, an OpenAI-compatible LLM baseline.
- Released upstream checkpoints (TinyStories, Sudoku easy/medium/hard) run natively in MLX through `python -m ncm_mlx official`. Conversion is torch-free, per-level logits match upstream PyTorch in exact float64, and greedy outputs are identical. The released Sudoku models solve 99.40 / 96.30 / 86.20% on Metal (n = 2,000 each). See [released checkpoints](official-checkpoints.md).
- The released-checkpoint sampler caches the prefix by default and supports opt-in compiled level steps. Reference, cached and compiled paths retain separate correctness and timing evidence; the independently trained TinyWorld/Sudoku sampler is a separate implementation. See [prefix caching](official-checkpoints.md#prefix-kv-cache-and-compiled-step).
- `scripts/guarded_run.sh`: memory watchdog for long runs.
- Codec and generator full-state checkpoints, graceful pause and `--resume`; CPU tests match uninterrupted training exactly, including compiled generator steps. Mac experiments can run under launchd independently of a terminal. See `RECOVERY.md`.
- E17 was killed with its terminal session at 15:05. Its step-30,000 weights survive; a separate weights-only continuation completed step 32,000 and solves 71.25 / 31.75 / 3.70% of easy/medium/hard boards. Its optimizer and random state reset; see `EXPERIMENTS.md`. The independent remote E18 process survived and completed 128,000 steps without a recovery reset; it solves 84.25 / 49.10 / 10.55%, below E15. Its legacy checkpoint contains inference weights rather than full resumable training state.

## Measured (see EXPERIMENTS.md, E1)

| | MLX implementation | MLX AR baseline |
|---|---|---|
| Valid sentences, T=1.0 | 89.8% | 98.6% |
| Latency p50, batch 1 | 4.5 ms | ~20–30 ms |
| Throughput, batch 256 | 6,769 seq/s | 5,137 seq/s |
| Train time | 52 s codec + 266 s generator | 134 s |

All numbers are for 5M-parameter toy models on a synthetic language, on a desktop with other GPU services running.

**Sudoku (historical E4–E16).** Previous best MLX implementation (E15: stronger conditioning corruption, unique puzzles, 8.2M training sequences): 96.2 / 76.6 / 31.9% solved on the upstream easy/medium/hard sets (Original NCM, Kumbong et al.: 99.5 / 96.5 / 86.1% with a 4× larger generator and 51M presentations). At E11's matched 2.05M-sequence budget, the same-size AR baseline reaches 20.5 / 4.9 / 0.5%, against NCM's 55.2 / 18.1 / 1.6%. Trains on Apple Silicon, NVIDIA (RTX 3080) and NVIDIA A40 GPUs.

E19-funded is now the best completed local Sudoku run: **98.10 / 83.40 / 40.30%** on easy/medium/hard boards, n = 2,000 each, greedy generation. It completed 128,000 updates at batch 128 (16.384M presentations), twice E15/E18's exposure, so this is not an equal-budget advantage. The shorter E19-budget pilot completed 23,895 updates and scores 90.20/59.75/16.55%. E20 completed the same scientific recipe with seed 1 and scores **97.25 / 79.60 / 33.25%**, n = 2,000 each, at the same exposure. Both seeds finish without a recorded reset or truncation. E19-funded remains the best observed seed; the hard-set difference is 7.05 points. E20 retained aggregate reports but no per-board predictions or complete remote evaluation/source manifests; its benchmark contention log is empty. These are qualified historical measurements, not independently rescored results. See the [two-seed assessment](research/e20-seed-report.md). The original million-step E19 recipe remains unstarted. See the [completed report](research/e19-funded-report.md) and [experiment history](EXPERIMENTS.md).

## Research assessment

The E19/E20 phase is a partial implementation and small-model success: both seeds solve over 97% of easy and 79% of medium boards. Hard solving remains 33.25–40.30%, far below Original NCM’s 86.10%; neither reference reproduction nor reliable hard-Sudoku solving is achieved. The [assessment](research/e20-seed-report.md) separates observed failures, candidate explanations and tests needed to distinguish them. It motivates controlled text as the next research direction, with Sudoku retained for diagnostics; the subsequent pilot is described below.

**Controlled-text research, 2026-10-30:** E21–E23 complete a seed-0 TinyStories corruption ablation in a separate, unmerged training workflow. The [preliminary assessment](research/e21-e23-assessment.md) separates strong reconstruction from weak free-running generation and records the AR quality gap. Generator-seed repeats are in progress. TinyStories is synthetic prose; public clean-checkout reproduction and naturally occurring text validation remain pending. These experimental results do not add a supported training command to the released package.

## Not done

- The upstream Sudoku result is not reproduced. Width-512 runs E10/E12 did not close the gap; a paper-scale training budget remains pending.
- Released checkpoints other than TinyStories and Sudoku are not converted; the official port is inference-only and has no classifier-free guidance.
- Generator training supports `mx.compile` when `train.compile` is true. In the independent TinyWorld/Sudoku training path, codec/AR training and the samplers remain uncompiled, and the NCM sampler has no prefix cache. This differs from the released-checkpoint sampler described above. Quantisation for compressed inference is not implemented.
- No power or energy measurements (no instrumentation without sudo).

## Interpretation and reproduction

The historical E15 artifact records 96.25 / 76.65 / 31.90% solved; one-decimal historical summaries above retain their original rounding. All external Sudoku sets contain 2,000 boards. E15 has no matched-budget AR result, and its comparison with the original paper differs in generator size, data exposure and task setup. The common Sudoku codec contains 13,915,200 saved values including EMA state, in addition to the 6,440,128 generator values; “same size” in earlier comparisons refers to generator width/depth, not total system size.

Start with [architecture](architecture.md), [training](training.md), [benchmarks](benchmarks.md) and the [upstream map](research/upstream-map.md). Current results do not establish open-domain text quality, production suitability, cross-backend bitwise identity or compatibility with released checkpoints beyond TinyStories and Sudoku. Repository visibility and release availability are recorded in GitHub metadata rather than this scientific status page.
