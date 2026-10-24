---
title: NCM quality and inference benchmarks
description: Reproduce synchronized MLX generation timings and interpret TinyWorld and Sudoku quality, model sizes, matched budgets, hardware differences and uncertainty.
status: current
updated: 2026-10-24
---

# NCM quality and inference benchmarks

Quality, latency and training cost answer different questions. This page defines the measurements; [EXPERIMENTS.md](EXPERIMENTS.md) preserves the experiment history, and [`paper/data/results.json`](../paper/data/results.json) contains a public extract of measured results. Raw run logs and weights are not committed. Results describe experimental small models, not a general-purpose language model or a reproduction of upstream's full text benchmarks.

## Quality: TinyWorld and Sudoku

TinyWorld is a finite synthetic grammar. Its generated-sentence metrics check grammatical shape, number agreement, novelty relative to the training set, and diversity. These checks do not measure open-domain text quality. The historical 89.8% NCM versus 98.6% AR validity result uses temperature 1.0 and models of similar generator size.

Sudoku generation is conditioned on the puzzle cells. Greedy decoding (`temperature=0.0`) gives one candidate per board, with no search, reranking or solver repair. Each external difficulty set has 2,000 boards; the generated validation split has 1,000. A solved board is a valid Sudoku grid that preserves the givens. Cell accuracy against the supplied solution, blank-cell accuracy and the fraction preserving all givens are separate metrics. A valid alternative completion can matter on non-unique validation puzzles; the external boards are intended to have unique solutions.

| Experiment | Easy solved | Medium solved | Hard solved | Generator presentations |
|---|---:|---:|---:|---:|
| E13 | 70.85% | 31.25% | 4.05% | 2,048,000 |
| E15, best verified | 96.25% | 76.65% | 31.90% | 8,192,000 |
| E17, recovered | 71.25% | 31.75% | 3.70% | 2,048,000 intended; weights-only recovery |
| E18 | 84.25% | 49.10% | 10.55% | 8,192,000 |
| E19-budget pilot | 90.20% | 59.75% | 16.55% | 3,058,560 |

These percentages retain the precision of the evaluation artifacts; historical summaries sometimes use one decimal place. E17 restored step-30,000 weights and performed 2,000 further steps with fresh optimizer moments and random state. Its uncheckpointed work before interruption is unknown. It does not establish an improvement over E13 or a controlled effect of puzzle construction. See [recovery qualifications](RECOVERY.md#legacy-weights-only-recovery-e17).

For an indication of finite-board uncertainty, 95% Wilson intervals for E17 are 69.23–73.19%, 29.75–33.82% and 2.96–4.62%. They assume boards are independent samples from a population and do not account for training-seed variation, related puzzle structure, hyperparameter selection or distribution shift. The benchmark sets are fixed, so the observed scores themselves are exact for those boards. Comparing models on the same boards requires paired outcomes, not an independent-binomial significance claim. Most runs use one training seed; repeated sampling seeds are not independent training repetitions.

Codec reconstruction reads true held-out solutions through the encoder and decoder. E13/E15/E17 reconstruct all 1,000 validation solutions exactly using the full hierarchy. Generated solved-board rates above measure the separate generator task. Teacher-forced code accuracy and validation cross entropy also use true coarser codes and must not substitute for free-running quality.

## What is matched

The E11 NCM/AR comparison matches generator width/depth, dataset and 32,000 × 64 generator presentations. It reports NCM 55.2 / 18.1 / 1.6% versus AR 20.5 / 4.9 / 0.5% in the historical rounded summary. The AR baseline has a KV cache and puzzle-prefix prefill. It is a practical reference model, not an extensively tuned baseline.

Matching generator presentations does not match FLOPs, training time, optimizer path, or total system size: NCM also trains and retains a codec. E15 has no equal-budget AR quality result. Comparing E15 with the older AR run or with the original paper is an unmatched comparison. Upstream reports 99.5 / 96.5 / 86.1% with a 25.46M generator and 51.2M generator presentations, separate difficulty-specific checkpoints, and a different board tokenization. Those published numbers are reference results, not measurements made by this project. [Original paper, Tables 3, 5 and 6](https://arxiv.org/html/2610.14162v1).

## Parameter accounting

`eval` currently reports `params` by summing every public array in each MLX module's parameter tree. For the E6 codec reused by E13/E15/E17, that includes EMA accumulators, so it is not a count of gradient-trained weights alone.

| Quantity | Scalar values |
|---|---:|
| Generator, gradient-trained | 6,440,128 |
| Codec, gradient-trained | 13,816,704 |
| Codec codebooks, learned by EMA | 49,152 |
| Codec EMA sums and counts, training accumulators | 49,344 |
| Codec total reported by `params.codec` | 13,915,200 |
| Codec plus generator reported total | 20,355,328 |

The learned first-level down/up convolutions contribute 10,617,344 gradient-trained codec values. Earlier descriptions of this codec as about 3.3M omitted these layers. A claim that the local model is about one quarter of upstream's size refers only to the 6.44M versus 25.46M generators. Sampling needs the quantizer's codebooks/refinement/upsampling and decoder, but does not execute the codec encoder; the current loader retains the full codec. Saved-array counts, active inference components and measured memory are different quantities.

## Synchronized latency methodology

[`bench_sampler`](../ncm_mlx/eval/benchmark.py) times a complete generation call through conversion of its output to a NumPy host array. This waits for device work and includes output transfer; it never reports only MLX lazy graph construction. Generation also explicitly evaluates its output. Model loading, corpus preparation and training are outside the timed region. Prefix-array construction and prefix computation are inside it for conditional runs.

For each batch size the benchmark clears MLX's allocator cache, records a first call, performs three warmups, resets peak-memory accounting and times the requested iterations. It reports p50, p95, mean, sample standard deviation, and sequences/second derived from `batch / p50_seconds`. Peak memory is MLX's allocator peak, not total process memory or whole-device usage. First-call latency is not a fresh-process startup measurement. Three warmups and a small iteration count do not guarantee a stable steady state under changing GPU load.

Batch-one p50 is latency for one fixed-length output. Batch-256 p50 is latency for the whole batch; dividing it by 256 does not describe interactive request latency. Throughput at one batch size can be worse than at another because memory pressure and kernels change. The NCM forward count excludes the final decoder call and quantizer work; fewer forwards do not guarantee faster wall time.

Historical benchmark calls use temperature 1.0, including the E17 latency measurements below. Sudoku quality uses greedy temperature 0.0. Thus those timings do not directly time the decoding setting of the solved-board table. New runs can select and record temperature explicitly.

```bash
mkdir -p runs
python -m ncm_mlx bench --config configs/YOUR-RUN.json \
  --temperature 0.0 --batch-sizes 1 16 64 256 --iters 30 \
  --out runs/YOUR-RUN-bench.json
```

## Compilation and hardware

`train.compile: true` wraps the generator training step with `mx.compile`, including optimizer and MLX random state. It does not compile the sampling loop. Codec training, AR training and both samplers are currently uncompiled. Backend kernel optimizations still apply; “uncompiled” here refers specifically to the absence of an explicit `mx.compile` wrapper. Compiled-training speed measurements cannot be presented as compiled-inference speedups.

Verified execution includes Apple M5 Pro/Metal and NVIDIA RTX 3080 and A40/CUDA using MLX 0.32.3. CPU runs support exactness and recovery tests; they are not published GPU latency measurements. The recorded configurations use fp32 weights; exploratory bfloat16 tests are not the reported training recipe. Metal fp32 matrix products have reduced numerical precision, so exact algorithmic equivalence tests use CPU and device tests use tolerances. Cross-backend bitwise equivalence has not been established.

| E17 recovery, Apple M5 Pro | Batch 1 | Batch 16 | Batch 64 | Batch 256 |
|---|---:|---:|---:|---:|
| p50, ms | 14.059 | 78.835 | 648.257 | 1,952.542 |
| p95, ms | 14.499 | 93.883 | 754.231 | 2,346.734 |

This run used 20 timed iterations. GPU utilization was 80% immediately before benchmarking while other services remained running. A pre-run sample does not measure the load throughout the benchmark. Treat these as shared-desktop observations, not isolated hardware limits. Do not compare them directly with an uncontended CUDA run or infer a hardware speed ratio. Before any new benchmark, record device/software versions, configuration, checkpoint, sampling settings, compilation, batch sizes, iteration count, and contention without interrupting unrelated workloads.

## Training cost and future comparisons

Generator metadata times its training loop after code encoding. Periodic validation and checkpoint saving occur inside that loop; data generation, encoding and the final evaluation pipeline are separate costs. E17's 2,000 recovery steps took 1,247.53 seconds; the full recovery pipeline took 1,956 seconds. Neither is the full original E17 training cost. Always name the timing boundary, including repeated work after recovery.

Higher-value future comparisons are multiple training seeds, paired per-board outcomes, a budget-matched E15 AR baseline, matched compiled samplers and uncontended repeated timing sessions. No expensive new training is required to interpret the existing release accurately.
