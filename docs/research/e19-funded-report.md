---
title: E19 funded experiment report
status: completed
updated: 2026-10-25
---

# E19 funded experiment report

E19-funded completed all 128,000 generator updates and improves the best local Sudoku scores to **98.10 / 83.40 / 40.30%** on easy/medium/hard boards. Each set contains 2,000 boards, evaluated once with greedy generation. This is a single-seed result at twice E15's generator exposure, not an equal-budget or upstream-reproduction claim.

## Quality and comparisons

| Run | Generator presentations | Easy solved | Medium solved | Hard solved |
|---|---:|---:|---:|---:|
| E18, stronger corruption on base data | 8,192,000 | 84.25% | 49.10% | 10.55% |
| E19-budget pilot | 3,058,560 | 90.20% | 59.75% | 16.55% |
| E15, previous local best | 8,192,000 | 96.25% | 76.65% | 31.90% |
| E19-funded | 16,384,000 | 98.10% | 83.40% | 40.30% |
| Original NCM, published reference | 51,200,000 | 99.50% | 96.50% | 86.10% |

E19-funded gains **1.85 / 6.75 / 8.40 percentage points** over E15 and remains **1.40 / 13.10 / 45.80 points** below the original paper. The original uses a larger generator, different representation and task setup; its figures are published references from [Kumbong et al., Table 3 and Appendix C](https://arxiv.org/html/2610.14162v1), not measurements made here. E18 remains a negative result against E14 and E15. Differences in corpus, batch, schedule, exposure and hardware prevent these rows from isolating a causal effect.

| Funded metric | Validation (n=1,000) | Easy (n=2,000) | Medium (n=2,000) | Hard (n=2,000) |
|---|---:|---:|---:|---:|
| Solved | 70.90% | 98.10% | 83.40% | 40.30% |
| Blank-cell accuracy | 83.41% | 99.69% | 97.12% | 86.99% |
| Givens preserved | 99.80% | 100.00% | 100.00% | 99.60% |

Full-depth codec validation reconstruction is 100%, a separate measurement. Teacher-forced code loss is not a generated solve rate. Wilson 95% intervals for easy/medium/hard solve rates are 97.40–98.61 / 81.71–84.97 / 38.17–42.47%. These intervals assume independent boards, exclude training-seed variability and are not a paired significance test against earlier runs. No per-board paired outcome archive is available for such a test.

## Recipe, completion and provenance

`configs/sudoku-e19-funded.json`: frozen E6 codec; eight-layer width-256 generator; 200,000 base puzzles plus 8M cached unique puzzles, seeds 201–208; conditioning corruption 0.50→0.10; training seed 0, batch 128, warmup 1,000; AdamW, cosine schedule, learning rate 0.0006; full-state checkpoint every 1,000 updates. No pilot weights, optimizer or random state were carried into this fresh run.

Source commit: `80dd445e5d17f54f53687a88459e011de4f9006f`. NVIDIA A40, MLX 0.32.3 CUDA, Python 3.12.3. Training completed 128,000 updates without time-limit truncation; best-validation checkpoint 127,000, weighted CE 0.133244. Training-loop time 54,419.81 s; complete remote pipeline 55,700.91 s including preprocessing, evaluation, benchmarking and result transfer. Rental startup and the final cleanup wait are outside that pipeline timing boundary.

Best generator SHA-256: `843a8929c43ae8a976e7a71402d18d9a5ce7172ebfc3b3870cfe74fc3c8293b9`. The final full-state snapshot records 128,000/128,000 and preserves optimizer/random state. All 17 downloaded report/artifact files were independently rehashed against the completion manifest without mismatch; provider deletion was verified.

The first funded allocation failed during preflight before any training step and was deleted. Three regression cases reproduced SIGPIPE from early-closing version-check pipelines under pipefail; the corrected readers consume all output. The replacement passed all 78 tests in 105.59 s. Provider image-startup timeouts recovered before SSH readiness. These setup incidents are not additional training runs or optimizer-reset continuations.

## Synchronized generation benchmark

Temperature 1.0, uncompiled sampler, 3 warmups, 20 timed iterations, 12 generator forwards per sample batch, completion through a NumPy host copy. Quality above uses temperature 0, so this does not time the greedy quality setting.

| Batch | p50 ms | p95 ms | Sequences/s | MLX peak MB |
|---|---:|---:|---:|---:|
| 1 | 26.477 | 26.690 | 37.8 | 130.2 |
| 16 | 47.905 | 48.308 | 334.0 | 552.4 |
| 64 | 174.279 | 174.552 | 367.2 | 1,911.2 |
| 256 | 583.670 | 584.715 | 438.6 | 2,981.0 |

Before preflight, GPU utilisation and allocated memory were 0, with no compute applications. Health observations tracked the owned training process during the run; an exact immediate pre-benchmark load sample was not retained. Do not certify these as uncontended hardware limits or derive a hardware speed ratio against the pilot's different A40 host or the RTX 3080 runs.

## Next validation

E20 repeats the funded scientific recipe with training seed 1 and the same data split, frozen codec, batch and target updates. Its operational time cap is shorter; any early truncation must be reported. This will provide a second seed, not by itself a robust estimate of broad training variability. The original million-step E19 recipe remains unstarted.

The [sanitized result extract](completed-sudoku-results.json) contains complete quality reports, parameter accounting, runtime metadata and benchmark rows for E18 and both completed E19 variants. [Experiment history](../EXPERIMENTS.md) and [benchmark methodology](../benchmarks.md) retain the wider comparisons. The combined working-paper update follows these verified results; older releases remain immutable.
