---
title: Two-seed Sudoku assessment and the next research phase
description: Completed E20 results, the limits of the smaller-generator finding, the remaining hard-Sudoku gap and evidence requirements for text research.
status: completed
updated: 2026-10-27
---

# Two-seed Sudoku assessment and the next research phase

The current Sudoku study is a **partial success**. The MLX implementation trains a 6.44M-parameter next-concept generator that solves most easy and medium boards under two training seeds. Hard-board solve rates remain 33.25–40.30%, substantially below Original NCM's published 86.10%. This supports continuing the implementation and small-model research; it does not establish reproduction of the original result, a reliable general Sudoku solver or practical language capability.

The next main research direction is text generation, beginning with a controlled corpus, separate codec and generation evaluation, and an autoregressive baseline. Sudoku remains a bounded diagnostic track. Text support is planned, not implemented or validated by these results. This update analyzes saved outputs; it includes no new training, model evaluation or accelerator benchmark.

## What the model actually does

NCM means **Next-Concept Modeling**, the method introduced by [Kumbong et al.](https://arxiv.org/abs/2610.14162). It is a generation method, rather than a synonym for Sudoku or an alternative definition of an LLM. Conventional autoregressive text generation predicts the next token conditioned on previous tokens. NCM predicts all latent codes within one refinement level in parallel, then conditions the next level on the accumulated reconstruction.

A progressive image is a useful analogy: an initial coarse representation receives more detail in successive rounds. Here, however, the representation is learned discrete codes, not pixels or explicit logical deductions. Intermediate decodes need not improve monotonically or obey Sudoku constraints.

The codec learns to encode complete training solutions into a multiscale residual vector-quantization hierarchy and reconstruct them. A separate generator learns to predict that hierarchy, conditioned on puzzle clues. At inference the solution encoder is not given the answer: the generator produces codes and the codec decodes them. Sudoku uses 12 generator forwards followed by a final decode. The reported greedy policy has no solver repair, search, reranking or constraint mask that guarantees validity. The checker scores the completed output after generation.

## Completed results

“MLX implementation” identifies models trained in this project. “Original NCM (Kumbong et al.)” identifies a quoted result from their paper, not a checkpoint evaluated here. Every local external difficulty reports 2,000 boards; validation reports 1,000. Solved means a valid complete grid preserving every input clue. Easy, medium and hard designate 40, 35 and 30 clues in these benchmark sets, not a universal rating of human solving difficulty.

| Model or recipe | Configured training seed | Generator parameters | Generator presentations | Easy solved | Medium solved | Hard solved |
|---|---:|---:|---:|---:|---:|---:|
| MLX implementation, E15 | 0 | 6.44M | 8.192M | 96.25% | 76.65% | 31.90% |
| MLX implementation, E19 pilot | 0 | 6.44M | 3.059M | 90.20% | 59.75% | 16.55% |
| MLX implementation, E19-funded | 0 | 6.44M | 16.384M | 98.10% | 83.40% | 40.30% |
| MLX implementation, E20 | 1 | 6.44M | 16.384M | 97.25% | 79.60% | 33.25% |
| Original NCM (quoted reference) | Not compared | 25.46M | 51.200M | 99.50% | 96.50% | 86.10% |

The pilot's 3.059M is training exposure, not model size. Original NCM uses separate difficulty-specific checkpoints, a different representation and a different training setup. The reference figures come from its Table 3 and Appendix C. The comparison does not isolate capacity, data efficiency or total compute efficiency.

| Two-seed description | Easy | Medium | Hard |
|---|---:|---:|---:|
| E19-funded solved boards | 1,962 / 2,000 | 1,668 / 2,000 | 806 / 2,000 |
| E20 solved boards | 1,945 / 2,000 | 1,592 / 2,000 | 665 / 2,000 |
| E20 minus E19-funded | −0.85 pp | −3.80 pp | −7.05 pp |
| Descriptive mean of two seed scores | 97.675% | 81.500% | 36.775% |
| E20 Wilson 95% interval | 96.44–97.88% | 77.78–81.31% | 31.22–35.35% |

The intervals assume independent boards sampled from a population. They exclude training-seed variation, related puzzle structure, distribution shift and repeated benchmark-guided recipe selection. Scores are exact descriptions of the reported fixed-board aggregates. The two seeds reuse the same boards; pooling them into 4,000 independent boards would be inappropriate. No paired significance or solved-board overlap can be recovered without individual outcomes. Two seeds are insufficient for a reliable estimate of the distribution of training outcomes.

## What changed in E20

E20 repeats the funded scientific recipe with configured training seed 1 instead of 0. Both use the frozen E6 codec, identical shared input-manifest hashes, 200,000 base puzzles plus 8M cached unique puzzles, batch 128, 128,000 updates, learning rate 0.0006, 1,000-step warmup and corruption 0.50→0.10. The seed controls initialization, minibatch order, dropout and corruption randomness; it does not isolate one of them. The generator implementation internally initializes random streams with the configured seed plus one.

Run name and output directory differ. The operational generator cap changes from 1,440 to 960 minutes, but neither run reaches its cap. Both finish the full 16.384M presentations without optimizer reset or recorded truncation, on separate A40 allocations. There is no training/evaluation implementation or dependency-lock change between their declared source revisions. Remote source bytes were not independently inventoried, so this is a statement about recorded provenance and local Git objects.

| Training measurement | E19-funded | E20 |
|---|---:|---:|
| Completed updates | 128,000 | 128,000 |
| Selected best-validation step | 127,000 | 128,000 |
| Best weighted code CE | 0.133244 | 0.142558 |
| Generator training-loop seconds | 54,419.81 | 54,122.43 |
| Recorded pipeline seconds | 55,700.91 | 55,543.62 |
| Validation solved | 70.90% | 71.00% |
| Hard blank-cell accuracy | 86.99% | 84.92% |
| Hard boards preserving every clue | 99.60% | 99.30% |

Training-loop time includes periodic validation and checkpointing, but excludes prior data encoding. Pipeline time includes preprocessing, training, evaluation, benchmarking and transfer; rental startup and final cleanup lie outside it. Similar validation solve rates do not cancel the external hard-set difference. Code cross-entropy, codec reconstruction and generated solve rates measure different things.

## Why hard performance is lower

Two questions require different answers: why E20 is below E19-funded, and why both are below Original NCM on hard boards.

**Between the two seeds:** the recorded runs follow different stochastic training trajectories despite matching scientific settings. E20's higher best code loss and lower hard blank-cell accuracy accompany its lower solve rate. This is consistent with sensitivity to training randomness. It does not identify a defective component, prove a general advantage for seed 0, or strictly isolate randomness from unmeasured differences between physical GPU allocations. A shorter run, interrupted recovery or changed corruption recipe is not supported as an explanation.

**Across difficulties:** hard boards supply fewer clues and require filling 51 blanks rather than 41 on easy boards. A full-board metric fails when any required constraint fails. Small changes in cell accuracy can therefore accompany larger changes in solved-board rates. Cell errors can be correlated; raising average cell accuracy to a power is an illustration under an independence assumption, not a measured predictor or a model of these results. The aggregates do not show which deductions, cells or concept levels caused a failure.

A useful diagnosis is nevertheless possible from the saved metrics. E20's hard set contains 665 solved boards, 670 valid grids and 1,986 boards preserving all clues. Therefore 1,330 outputs are invalid grids and five additional outputs are valid grids that violate clues. Fourteen outputs violate clues overall, including nine invalid grids. Merely preserving clues cannot repair the dominant invalid-grid failure. These counts follow from the scoring definitions and aggregates; they do not identify individual boards or mechanisms.

**Against Original NCM:** generator capacity, training exposure, corpus construction, sequence representation and difficulty-specific training all differ. They are plausible contributors, not established causes. Historical width-512 runs E10/E12 did not close the gap at their tested budgets. That does not prove larger models cannot help; it shows that widening alone is not an evidence-based guarantee. Full-depth codec reconstruction on validation is 100%, but that does not establish useful coarse codes under generator errors or perfect reconstruction on every external board.

| Hypothesis to challenge | Current evidence | Discriminating next test |
|---|---|---|
| Training randomness materially affects hard quality | Two seeds differ by 7.05 points | Additional preregistered seeds and independent RNG streams, with paired predictions |
| Training data do not transfer well to hard boards | Validation and fixed-clue external sets behave differently | Controlled corpus/stratum comparisons at matched exposure; new untouched test boards |
| Early code errors are difficult to repair | Method conditions later levels on generated coarser codes | Save per-level outputs; intervene with true codes on a fixed diagnostic subset |
| More capacity or exposure closes the reference gap | Neither variable is isolated in the reference comparison | Change one variable at a time with fixed data, seed protocol and total-cost accounting |
| Selection by code loss misses generated quality | Code loss, validation solving and hard solving do not move together | Prespecify checkpoint selection using validation only; evaluate selected checkpoints once |

## What success means for this project

We are satisfied that the small-generator recipe merits further study: both seeds solve over 97% of easy and 79% of medium boards, and both exceed E15's reported rates at twice its exposure. The result shows that this parameter scale can learn substantial puzzle-conditioned structure under these settings. It does not locate a minimum sufficient capacity, demonstrate human-style reasoning or establish general combinatorial competence.

We are not satisfied with hard-board reliability, reference reproduction, seed characterization or end-to-end efficiency evidence. E20 fails 1,335 of 2,000 hard boards. A research demonstration can tolerate such failures when reported; an application promising reliable solutions cannot. The best-seed hard gap to Original NCM is 45.80 points; E20's is 52.85 points.

The 3.95× size reduction refers only to 6,440,128 versus 25.46M generator parameters. The shared codec separately contains 13,816,704 gradient-trained values, 49,152 EMA-trained codebook values and 49,344 EMA accumulator values. Their total plus the generator is 20,355,328 saved values, not all trainable parameters. Whole-system memory, latency and training costs need their own measurements. These results do not justify a fourfold deployment-memory or cost advantage.

## Evidence quality and reproducibility

All 15 retained E20 completion files were rehashed against their manifest. Inference weights match the selected best-model tensors in the full-state checkpoint; the saved optimizer and training step are 128,000. The reported generator has 6,440,128 values. Best-generator SHA-256: `b6ddadb7278e7f1c5542eecef676840f39cb829e802d189b65f2dadd645c2b3d`. Declared training source: `4ad3646189197fe1ca76e45308848793fefe9a3f`. The executed config is [committed](../../configs/sudoku-e20-seed1.json); runtime is A40, MLX 0.32.3 CUDA, Python 3.12.3.

These integrity checks verify retained bytes and internal consistency. They do not constitute an independent experimental reproduction. No per-board predictions, checkpoint-bound ordered scoring-input manifest or remote source digest inventory was saved. Current local evaluation files match the published hashes, but the remote transfer verification excludes those files. Remote evaluation identity, complete duplicate-free coverage and exact executed source therefore cannot be independently established from the retained outputs. This report publishes the historical aggregates with those limits rather than relabeling them independently verified scores.

E20's benchmark uses temperature 1, three warmups, 20 timed iterations, an uncompiled sampler and a host copy for completion. Batch 1/16/64/256 p50 is 29.453 / 48.506 / 174.326 / 581.791 ms; p95 is 30.323 / 49.274 / 175.352 / 583.397 ms. It does not time the greedy quality policy. Its GPU-load log is empty because a Metal-only probe was used on CUDA; contemporaneous benchmark contention is unknown. Do not infer a hardware speed ratio or attribute timing differences to seed choice.

The [sanitized extract](completed-sudoku-results.json) preserves quality and timing data. The [derived assessment](e20-seed-assessment.json) can be rebuilt without MLX or private logs:

```bash
python paper/analyze_replication.py
```

## Next research phase

1. **Improve evidence collection before new experiments.** Bind checkpoint, source, dataset revision, ordered input identities and scoring settings in an evaluation manifest. Retain per-board predictions, failures and per-level diagnostics. Record backend-appropriate GPU load. Rerunning saved checkpoints would create new evaluation evidence; it cannot reconstruct missing historical telemetry.
2. **Establish a reference.** Investigate independent conversion of released Original NCM weights, beginning with vocabulary, normalization, resampling and intermediate-output parity. Conversion parity is distinct from reproducing training. Keep the small generator as a controlled baseline before committing to larger runs.
3. **Begin controlled text research.** Pin a permitted corpus and tokenizer contract, preserve document-level split separation and test padding, Unicode and conditioning boundaries. TinyStories is a possible synthetic starting corpus; success there would not establish natural-document performance.
4. **Measure quality and actual cost.** Report codec reconstruction, free-running text samples and failures, a matched-data autoregressive baseline, synchronized Mac latency and memory. Generator-size matching alone does not match total cost. Text continuation precedes any claim about summarization, translation or instruction following.
5. **Advance only on evidence.** Test fixed level budgets before adaptive early exit, then caching and compilation as isolated changes. Sudoku has an exact validity checker; text stopping requires separate validation. Publish negative outcomes alongside improvements.

The immediate objective is a reproducible text experiment and a defensible quality/cost decision. A production assistant, successful benchmark reproduction or useful project integration remains contingent on future evidence. The [canonical roadmap](../NEXT.md) records dependencies and acceptance criteria without promising dates or future solve rates.
