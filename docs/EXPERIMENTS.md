---
title: Experiments
status: current
---

# Experiments

Every measured run, newest first. Unless a run specifies another device, hardware is Apple M5 Pro, 64 GB, macOS 26.5, MLX 0.32.3, Python 3.12.14. Raw JSON is written to `runs/` (gitignored); the numbers below are copied from it.

**Model labels.** All E-numbered NCM experiments below use this repository’s MLX implementation. “Original NCM (Kumbong et al.)” and “Original AR baseline” denote cited results from the original study; these were not measured here. “Published easy/medium/hard” identifies their shared test boards, not the model being scored.

**GPU contention.** Mac experiments generally ran alongside other local GPU services; per-run notes record exceptions and other hardware. `ioreg` reported 15% GPU utilisation from other processes just before the v0 benchmark started. Treat those latency numbers as "typical busy desktop" rather than an idle machine.

## Official checkpoints — released HazyResearch weights in MLX (2026-10-29)

Inference only; no training. These are the released upstream checkpoints (revision `d342ce8`, code `c702742`), converted by `python -m ncm_mlx official convert`; they are not models trained by this repository. Details and commands are in [released checkpoints](official-checkpoints.md); the parity reports and solve-rate records are in [`research/official-checkpoint-parity.json`](research/official-checkpoint-parity.json).

**Parity (CPU, `scripts/official_parity.py`, PyTorch 2.15.0).**

- TinyStories (4 texts, 8 greedy samples):
  - BPE ids are identical.
  - Per-level logits differ by at most 0.10 in fp32 (scale up to 853) and 3.2e-10 in exact fp64. Upstream fp32 itself differs by 0.071 from fp64.
  - Greedy codes and text are identical for 8/8 samples.
  - Codec codes agree 97.4–100% per level; each item's first divergence is a near-tie.
- Sudoku-easy (64 boards):
  - Per-level logits differ by at most 7.2e-5 in fp32 and 1.2e-13 in fp64.
  - Codes are 100% identical, and greedy outputs are identical for 64/64 boards.

**Sudoku solve rates**, greedy, published test sets, n = 2,000 each, MLX on Metal: **easy 99.40% (1,988), medium 96.30% (1,926), hard 86.20% (1,724)**. A first-200 easy sample gives 99.0% (198/200) on both Metal and CPU. On the full easy set, upstream PyTorch on this machine's CPU also solves 1,988/2,000 (188.75 s), and MLX's outputs equal upstream's on all 2,000 boards on both Metal and CPU. Original NCM (Kumbong et al.) reports 99.5 / 96.5 / 86.1%. Medium and hard were not run with upstream PyTorch here. Metal evaluation wall time is 21–24 s per 2,000 boards at batch 250, with ~19% GPU use by other applications before the medium/hard runs.

**Latency** (`official bench` / `scripts/official_torch_reference.py bench`). Timings are synchronized through a host copy, uncompiled, with no prefix cache. MLX uses 3 warmups and 20 iterations; PyTorch uses 2 warmups and 10 iterations on 6 threads. TinyStories samples at T=1 with the level-0 unigram; Sudoku is greedy.

| p50 / p95 ms | MLX Metal | Upstream PyTorch CPU |
|---|---:|---:|
| TinyStories b=1 | 39.8 / 40.8 | 280.0 / 289.9 |
| TinyStories b=16 | 519.3 / 742.3 | 3,883 / 4,076 |
| Sudoku-easy b=1 | 20.5 / 22.7 | 164.6 / 201.8 |
| Sudoku-easy b=16 | 228.6 / 277.0 | 1,663 / 1,746 |

- First-call MLX latency is 160 / 278 ms for TinyStories and 102 / 261 ms for Sudoku (batch 1 / 16).
- MLX peak memory is 635 / 2,269 MB for TinyStories and 269 / 2,012 MB for Sudoku.
- Upstream fails on MPS: PyTorch 2.15 does not implement non-divisible `adaptive_avg_pool1d` there, so there is no MPS row.
- **Contention:** `ioreg` sampled 0–33% GPU utilisation from desktop applications (window server, browser) before and between runs, with single readings of 69–98% at run boundaries. These are busy-desktop numbers, not idle-machine numbers.

At b=16 the Sudoku level forward is compute-bound: about 15 ms for the 89-position level, because the 91-token prefix is recomputed at every level.

## E20 — completed seed replication

`configs/sudoku-e20-seed1.json` repeats E19-funded’s scientific recipe with configured training seed 1, unchanged data split and frozen E6 codec, 200k base plus 8M cached unique puzzles, corruption 0.50→0.10 and batch 128. Both complete 128,000 updates (16,384,000 presentations). This is a fresh initialization, not a continuation. Its 960-minute cap is shorter than E19-funded’s 1,440-minute cap but did not trigger; there is no recorded optimizer reset or truncation.

**Completed, audited 2026-10-27.** Reported greedy solve rates are **97.25 / 79.60 / 33.25%**, n = 2,000 per published difficulty; validation is 71.00%, n = 1,000. E19-funded scores 98.10 / 83.40 / 40.30%, so E20 is lower by 0.85 / 3.80 / 7.05 percentage points. Both use twice E15/E18’s exposure. Best-validation and final full-state steps are 128,000; best weighted code CE 0.142558. Training-loop time 54,122.43 s; recorded pipeline time 55,543.62 s, excluding rental startup and final cleanup. Source declared by the controller: `4ad3646189197fe1ca76e45308848793fefe9a3f`; executed config SHA-256 `664a8295fb98ac9cbdb6aa5eab7e7a5f6a1f199a573bc12d481f927bea558e4b`. A40, MLX 0.32.3 CUDA, Python 3.12.3. All 15 retained completion files rehash correctly; inference weights equal the saved best-model tensors. Generator SHA-256: `b6ddadb7278e7f1c5542eecef676840f39cb829e802d189b65f2dadd645c2b3d`.

**Contention and audit limits.** Preflight records an idle allocation (0% GPU, 0 MiB) and 78 passing tests in 103.41 s; these are not benchmark-time observations. The benchmark GPU-load log is empty because a Metal-only probe ran on CUDA; contemporaneous contention is unknown. Synchronized temperature-1 p50 at batch 1/16/64/256 is 29.453 / 48.506 / 174.326 / 581.791 ms, with three warmups and 20 timed iterations, completion through a host copy and an uncompiled sampler. These timings do not time greedy quality or establish a cross-host speed ratio. No per-board predictions, ordered checkpoint-bound scoring-input manifest or remote source digest inventory was retained. Local board hashes match published files, but actual remote evaluation identity and duplicate-free coverage cannot be independently audited. Scores are recorded aggregates, not independently rescored results.

The [two-seed assessment](research/e20-seed-report.md) gives full metrics, failure accounting, uncertainty, interpretation and next tests; the [sanitized extract](research/completed-sudoku-results.json) and [derived statistics](research/e20-seed-assessment.json) preserve the public evidence. This phase is a partial small-model success, with hard-Sudoku reproduction and natural-text validation unresolved.

## E19-funded — completed larger-corpus experiment

Completed 128,000 updates, 16,384,000 presentations, seed 0, best-validation step 127,000 and weighted CE 0.133244. Training 54,419.81 s; whole remote pipeline 55,700.91 s. A40/CUDA, MLX 0.32.3, Python 3.12.3; no optimizer reset or time-limit truncation. All 17 downloaded files match their completion hashes; full-state checkpoint and provider deletion verified.

Greedy generation solves **98.10 / 83.40 / 40.30%**, n = 2,000 per upstream difficulty. E15's corresponding scores are 96.25 / 76.65 / 31.90 at half the generator exposure. Full-depth codec validation is 100%, separately. The initial preflight failure and corrected fresh attempt remain recorded in the [full report](research/e19-funded-report.md), with sample counts, uncertainty, recipe, timing boundaries, synchronized benchmarks and checkpoint hash. The [sanitized result extract](research/completed-sudoku-results.json) includes E18, the short pilot and this run. The original million-step E19 recipe remains unstarted.

## E19-budget — completed larger-corpus pilot

**Completion, 2026-10-24.** The A40 run completed all 23,895 updates without interruption; its best-validation checkpoint is also step 23,895, weighted CE 0.2000. Training-loop time was **10,225.84 s**. From controller readiness to final artifact copy and pod deletion was **11,445.52 s**, including setup, preprocessing, training, evaluation, benchmarking and transfers; it excludes the initial rental/readiness wait. All 16 downloaded artifact/log hashes match the completion manifest. Full training state records step 23,895 of 23,895 with no time-limit truncation, and provider deletion was independently verified.

**Recipe and actual target.** `configs/sudoku-e19-budget.json`: frozen E6 codec, width-256 eight-layer generator, corruption 0.50 → 0.10, 200,000 base puzzles plus 8M cached unique puzzles (seeds 201–208), batch 128, seed 0, warmup 1,000, full-state checkpoints every 1,000 updates. The executed CLI overrides the config's 1,000,000-step ceiling with **23,895 updates**, at most **3,058,560 sequence exposures**. Generator time cap is 600 minutes in the config; the outer pipeline cap is six hours including corpus processing, training, evaluation and benchmarking. Neither ceiling is a measured duration. This pilot has less exposure than E15 and cannot establish a longer-budget improvement or an equal-budget comparison.

**Preflight evidence.** MLX 0.32.3, Python 3.12.3, CUDA; 0% GPU utilisation, 0 MiB allocated and no compute applications observed before the checks. All 75 tests passed in 112.60 s, including exact CPU invariants and recovery tests on this host. A synthetic width-256/batch-128 training-step probe measured 542 ms/step and 236 sequences/s, with a 15.6 GB MLX-reported peak. The probe uses 3 warmups and 10 timed steps; it sizes the schedule to 90% of a four-hour allowance. It excludes dataset encoding and is not a measured end-to-end training or generation throughput.

**Recovery and reporting.** The source snapshot is commit `0c9ce98a63f029105b7ec85f044f8885e143960b`. Inputs have recorded SHA-256 hashes, remote files were copied back every ten minutes and at exit, and the pipeline ran independently of a terminal. No optimizer-reset continuation or cross-device exact recovery is claimed. Raw outputs use `runs/sudoku-e19-budget-*` and `artifacts/sudoku-e19-budget/`.

Best-generator SHA-256: `34ce3edeec7383fc8984af50ed2bb3ea2c45889215608cf509a195639021b638`.

**Held-out generation.** Temperature 0, one pass over 1,000 validation boards and 2,000 upstream boards per difficulty. Full-depth codec reconstruction remains 100% on validation; it is separate from these generator scores.

| Metric | Validation | Upstream easy | Upstream medium | Upstream hard |
|---|---:|---:|---:|---:|
| Solved | 62.40% | 90.20% | 59.75% | 16.55% |
| Blank-cell accuracy | 82.10% | 98.47% | 92.78% | 78.59% |
| Givens preserved | 98.70% | 99.90% | 99.05% | 96.40% |

This pilot exceeds E18's solve rates with fewer sequence exposures and trails E15. Corpus, batch size, schedule and accelerator differ across those comparisons; one seed does not isolate a causal effect or establish an equal-budget advantage.

**Synchronized timing.** MLX 0.32.3, Python 3.12.3, CUDA/A40, temperature 1.0, uncompiled sampler, 3 warmups and 20 timed iterations through a NumPy host copy. Batch 1 / 16 / 64 / 256 p50 was **37.415 / 51.420 / 174.178 / 582.918 ms**, with 12 generator forwards per batch. These timings use a different temperature from greedy quality evaluation. The rented device was idle before preflight; health checks observed the owned training workload at 100% utilisation, but no exact immediate pre-benchmark contention sample was retained. Do not treat these as certified idle-machine limits or infer a hardware speed ratio against earlier devices.

## E18 — stronger corruption on the base dataset

**Recipe.** `configs/sudoku-e18.json`: E14's 200,000 base training puzzles and 128,000 updates × batch 64 = 8.192M sequence exposures, with corruption changed from 0.30 → 0.05 to 0.50 → 0.10. The 6,440,128-value generator uses the frozen E6 codec (13,915,200 saved values including EMA state). No extra unique-puzzle corpus is added. NVIDIA RTX 3080 12 GB, MLX 0.32.3, Python 3.12.15, compiled generator training and uncompiled sampling.

**Completion and recovery.** The uninterrupted generator completed step 128,000; its best-validation checkpoint is also step 128,000, weighted CE 0.1750. Training-loop time was 23,366.69 s. The launcher recorded 11:38:47 CEST on 2026-10-24; the completed artifacts were copied and hash-verified at 18:12:07 CEST, 23,600 s later including watcher polling and transfer. This run used the older weights-only checkpoint implementation; no optimizer reset occurred during the run, but its saved inference weights are not full resumable training state. The remote job survived the earlier local terminal closure.

**Evaluation.** Greedy generation, temperature 0, one pass over 1,000 local validation boards and all 2,000 upstream boards per difficulty. These are single-run point estimates. Full-depth codec reconstruction on validation remains 100%; it is a separate measurement from solving generated boards.

| | Val | Upstream easy | Upstream medium | Upstream hard |
|---|---|---|---|---|
| Solved, E18 | 55.80% | 84.25% | 49.10% | 10.55% |
| Blank-cell accuracy, E18 | 81.22% | 97.50% | 90.62% | 76.70% |
| Givens preserved, E18 | 99.20% | 99.80% | 99.65% | 98.55% |
| Solved, E14 | 55.40% | 87.40% | 53.40% | 11.20% |
| Solved, E15 | 69.50% | 96.25% | 76.65% | 31.90% |

**Reading.** Stronger corruption alone did not improve transfer over E14 at this budget. E15, using the same stronger corruption and sequence budget plus 4M unique training puzzles, remains ahead by 12.00 / 27.55 / 21.35 percentage points on easy/medium/hard. This supports further dataset-coverage experiments; a single seed and these comparisons do not establish a general causal effect or a matched-budget advantage over the original paper.

**Timing and contention.** Synchronized generation p50 at batches 1 / 16 / 64 / 256 was 15.952 / 44.537 / 153.800 / 524.874 ms, with 20 timed iterations, 3 warmups and 12 generator forwards per sample batch. The executed CLI used the benchmark function's default temperature 1.0, verified from its original code; the older result JSON does not record that setting. Timing ends with a host copy. The coding-model service stayed stopped until training, evaluation and benchmarking were inactive; 100% GPU utilisation was observed while the benchmark was active; this does not reconstruct a pre-run idle reading. An exact pre-run contention measurement was not retained, so these are not certified idle-machine measurements. Raw artifacts: `runs/sudoku-e18-{train.log,eval.json,bench.json}` and `artifacts/sudoku-e18/`.

## E17 — interrupted Mac run and weights-only recovery

**Recipe.** E13 budget and corruption (32,000 × 64 = 2.048M intended sequences), with the unique subset of random-removal puzzles added to the base set. Original config: `configs/sudoku-e17.json`; recovery: `configs/sudoku-e17-recovery.json`, separate output directory, validation/checkpoint interval 100 instead of 2,000 steps. Apple M5 Pro, MLX 0.32.3, 20 GiB allocation cap and 24,000 MB watchdog.

**Interruption, 2026-10-24.** The original run started at 10:19:23 CEST. Closing its terminal killed the parent session and training tree at 15:05:17 (17,154 s elapsed wall time, including preprocessing). Its last durable checkpoint was step 30,000 at 14:49:25: loss 0.3265, weighted validation CE 0.2217. Uncheckpointed progress after that is unknown. The watchdog logged no limit breach; the final sampled process-tree footprint was 9,456 MB at 15:03:40, below 24,000 MB. All 143 tensors / 6,440,128 parameters in the saved weights are finite. Source SHA-256 and incident details are recorded in `docs/RECOVERY.md` and `runs/sudoku-e17-recovery-incident.json`.

**Recovery.** Started at 15:22:48 CEST under launchd. It loads the preserved step-30,000 weights and performs global steps 30,001–32,000, retaining learning-rate and shuffled-batch positions. AdamW moments and MLX random state reset because the legacy run did not save them. This is a weights-only continuation, not an uninterrupted matched-budget run. Original artifacts and logs were backed up before launch. The recovery performed 2,000 additional steps in 1,247.53 s of training-loop time; preprocessing, training, evaluation and benchmarking together took 1,956 s, finishing at 15:55:24. The best-validation model is step 31,900 (weighted CE 0.2204).

**GPU contention.** Device utilisation was 0% immediately before recovery launch; other local services stayed running. The original run's prelaunch contention was not recorded and cannot be reconstructed. Device utilisation was 80% immediately before the synchronized benchmark, so latency is a shared-desktop measurement.

**Recovery checks.** Before launch, 49 tests passed in 2.17 s, including exact CPU equality between uninterrupted and paused/resumed codec and generator training (compiled and uncompiled generator paths). Two additional cloud-runner tests passed using mocked SSH/rsync, including transfer after failure; no pod was rented or crashed for testing. The operational procedure is in `docs/RECOVERY.md`.

| | Val | Upstream easy | Upstream medium | Upstream hard |
|---|---|---|---|---|
| Solved, recovered E17 | 42.9% | 71.25% | 31.75% | 3.70% |
| Solved, E13 | 43.8% | 70.9% | 31.2% | 4.0% |
| Solved, E16 | 37.2% | 57.6% | 21.8% | 2.2% |
| Blank-cell accuracy, recovered E17 | 79.07% | 95.53% | 86.60% | 70.73% |

**Reading.** Recovered E17 is close to E13 and above the greedy-unique E16. This is consistent with puzzle construction affecting transfer, but the optimizer/random-state reset prevents a controlled causal conclusion. Codec reconstruction remains 100%; solved-board quality is the separate measurement above. Synchronized latency p50 at batches 1 / 16 / 64 / 256 is 14.059 / 78.835 / 648.257 / 1,952.542 ms (20 timed iterations). Raw results: `runs/sudoku-e17-recovery-{eval,bench}.json`.

## E15 — Sudoku, all levers combined (`configs/sudoku-e15.json`)

**Change from E14.** Corruption 0.50 → 0.10 (E14: 0.30 → 0.05) and 4M greedy-unique puzzles added to the 200k base set (seeds 201–204). Same budget (8.2M sequences, batch 64 × 128,000), RTX 3080, 23,374 s.

| | Val | Published easy | Published medium | Published hard |
|---|---|---|---|---|
| **Solved, E15** | **69.5%** | **96.2%** | **76.6%** | **31.9%** |
| Solved, E14 | 55.4% | 87.4% | 53.4% | 11.2% |

**Reading.** Within 3.3 points of Original NCM (Kumbong et al.) on easy (99.5%) with a generator about a quarter of its size and one sixth of its generator presentations; medium and hard still trail (Original NCM: 96.5 / 86.1%). It contradicts the reading of E16: at this scale, mixing greedy-unique puzzles with the base set does not hurt. E18 (E14 budget, stronger corruption, base data only) finished below E14; stronger corruption alone did not reproduce E15's improvement at this budget. See the E18 comparison above.

## E16 — Sudoku, fresh unique-solution puzzles instead of repetition (`configs/sudoku-e16.json`)

**Change from E13.** Training data: the 200k base puzzles plus 2M unique-solution puzzles generated by greedy removal (cells removed in random order, skipping any removal that breaks uniqueness), each seen about once, instead of ten passes over the 200k base puzzles. Same budget (2.05M sequences), recipe and corruption (0.50 → 0.10). Apple M5 Pro, 16,217 s.

| | Val | Published easy | Published medium | Published hard |
|---|---|---|---|---|
| Solved, E16 | 37.2% | 57.6% | 21.8% | 2.2% |
| Solved, E13 | 43.8% | 70.9% | 31.2% | 4.0% |

**Reading.** Worse than E13 by 13 points on easy. Not because of uniqueness: the published test boards are 100% unique (500 checked per difficulty), while only 24.8% of our base training puzzles are. The likely cause is construction: greedy unique removal changes which clues remain, while the test boards look like random removal that happens to be unique. E17 tests this by keeping the unique subset of random-removal puzzles.

## E12 — Sudoku, upstream's width with stronger corruption on a rented A40 (`configs/sudoku-e12.json`)

**Change from E13.** Generator 8 × 512 (25M), dropout 0.2, lr 3e-4, batch 64 × 50,776 steps = 3.25M sequences (steps sized from the measured 118 sequences/s for an 8.5 h budget). Corruption 0.50 → 0.10. NVIDIA A40 48 GB, 21,491 s of training. Spare CPU produced 11M unique-solution and 4M regular puzzles.

| | Val | Published easy | Published medium | Published hard |
|---|---|---|---|---|
| Solved, E12 | 48.5% | 74.6% | 35.2% | 6.0% |
| Blank-cell accuracy, E12 | 79.1% | 95.8% | 86.6% | 71.4% |

**Reading.** A few points above E13 (256 wide, 2.05M sequences: 70.9 / 31.2 / 4.0%) with 1.6× the data and twice the width; far below E14 (8.2M sequences). Consistent with E10: data dominates at these budgets. Latency on the A40 (p50, batch 1 / 16 / 64 / 256): 29.1 / 90.5 / 306 / 1,079 ms.

## E10 — Sudoku, upstream's generator width on the Mac (`configs/sudoku-e10.json`)

**Change from E8.** Generator 8 × 512 (25M parameters, upstream's width), dropout 0.2, lr 3e-4, batch 64 × 24,000 steps = 1.5M sequences; corruption 0.30 → 0.05. Apple M5 Pro, 36,906 s (10.3 h), MLX memory capped at 20 GB.

| | Val | Published easy | Published medium | Published hard |
|---|---|---|---|---|
| Solved, E10 | 36.6% | 55.9% | 18.4% | 2.0% |
| Blank-cell accuracy, E10 | 76.4% | 92.7% | 81.6% | 66.0% |

**Reading.** Twice the width (4× the parameters) on 1.5M sequences matches the 256-wide E11 on 2.05M (55.2 / 18.1 / 1.6%). At these budgets width is not the binding constraint; data and corruption are. Latency on the Mac (p50, batch 1 / 16 / 64 / 256): 19.5 / 143 / 785 / 2,363 ms.

## E14 — Sudoku, E11 with four times the data again (`configs/sudoku-e14.json`)

**Change from E11.** 128,000 steps × batch 64 = 8.2M training sequences (E11: 2.05M), over the same 200,000 puzzles (about 41 passes). Same codec, generator and corruption (0.30 → 0.05). RTX 3080, 23,341 s.

| | Val | Published easy | Published medium | Published hard |
|---|---|---|---|---|
| **Solved, E14** | 55.4% | 87.4% | 53.4% | 11.2% |
| Blank-cell accuracy, E14 | 80.6% | 98.0% | 91.7% | 77.1% |
| Givens kept, E14 | 99.3% | 100.0% | 99.9% | 99.1% |

**Reading.** Quadrupling the data again took easy from 55.2% to 87.4%, medium from 18.1% to 53.4% and hard from 1.6% to 11.2%. Validation CE 0.131 at the last step and still falling. Data remains the dominant lever.

## E13 — Sudoku, E11 with stronger corruption (`configs/sudoku-e13.json`)

**Change from E11.** Corruption 0.50 → 0.10 (E11: 0.30 → 0.05). Same data budget (2.05M), RTX 3080, 5,868 s.

| | Val | Published easy | Published medium | Published hard |
|---|---|---|---|---|
| **Solved, E13** | 43.8% | 70.9% | 31.2% | 4.0% |
| Blank-cell accuracy, E13 | 78.8% | 95.5% | 86.4% | 71.2% |

**Reading.** At equal data, stronger corruption gives +15.7 points on easy, +13.1 on medium, +2.4 on hard. Both levers (data, corruption strength) helped independently; E15 combines them.

## E11 — Sudoku, E8 recipe with four times the training data, on an RTX 3080 (`configs/sudoku-e11.json`)

**Change from E8.** Training length only: batch 64 × 32,000 steps = 2.05M training sequences (E8: 128 × 4,000 = 0.51M), warmup 200. Same frozen E6 codec, generator shape (8 × 256, 6.4M) and corruption schedule (0.30 → 0.05). First run on NVIDIA: RTX 3080 12 GB, MLX 0.32.3 CUDA backend, 5,849 s, peak 9.2 GB.

| | Val | Published easy | Published medium | Published hard |
|---|---|---|---|---|
| **Solved, E11** | **31.4%** | **55.2%** | **18.1%** | **1.6%** |
| Solved, E8 | 5.9% | 6.2% | 0.35% | 0.0% |
| Solved, AR (E4, 0.64M sequences) | 10.4% | 13.8% | 3.0% | 0.4% |
| Blank-cell accuracy, E11 | 76.8% | 93.1% | 82.5% | 66.8% |
| Givens kept, E11 | 99.8% | 100.0% | 99.7% | 99.3% |

**Reading.** Four times more data turned NCM from far behind to far ahead of the same-size AR model at every difficulty.

**Matched-budget AR baseline** (same config, batch 64 × 32,000 steps = 2.05M sequences, RTX 3080, 2,535 s): solved 17.9% val, 20.5% easy, 4.9% medium, 0.5% hard; blank cells 76.0% on easy. More data helps AR too (13.8% → 20.5% on easy), but at equal budget NCM solves 2.7× more easy boards, 3.7× more medium and 3× more hard. Teacher-forced accuracy rose at every level, most at level 0 (53% vs 23%). Validation CE was still falling at the last step. The gap to Original NCM (Kumbong et al.: 99.5 / 96.5 / 86.1%) remains large on medium and hard; its generator is 4× larger and uses 25× more generator presentations.

**Latency** on the RTX 3080 (p50, batch 1 / 16 / 64 / 256): 16.0 / 44.5 / 154 / 523 ms.

## E9 — Sudoku, corruption from the generator's own samples (`configs/sudoku-e9.json`)

**Change from E8.** Replacement codes come from the generator's own predictions (one extra forward without gradients per step; each level sampled given the true coarser levels) instead of uniformly from the codebook. Same probabilities, codec, shape and steps. 7,565 s (1.56× E8).

| | Val | Published easy | Published medium | Published hard |
|---|---|---|---|---|
| Solved, E9 | 4.5% | 3.0% | 0.3% | 0.0% |
| Solved, E8 | 5.9% | 6.2% | 0.35% | 0.0% |
| Blank-cell accuracy, E9 | 63.1% | 74.7% | 61.5% | 48.5% |
| Givens kept, E9 | 93.8% | 95.7% | 90.6% | 85.4% |

**Reading.** Model-sampled corruption matches uniform corruption on blank cells and solves fewer easy boards (3.0% vs 6.2%; with 2,000 boards the standard error is about 0.5 points). Its validation CE is lower (0.242 vs 0.297), as its corrupted inputs are closer to the truth. At this scale it does not beat uniform corruption and costs 1.56× the training time, so uniform corruption stays the default. The paper's suggestion is not supported here; a one-step sample of each level given true coarser levels may be too close to the truth to teach recovery.

## E8 — Sudoku, conditioning corruption (`configs/sudoku-e8.json`)

**Change from E6.** Generator only: conditioning corruption (upstream Section 3.4), replacement probability falling linearly from 0.30 at level 0 to 0.05 at level 10, replacements uniform from the level's codebook, labels unchanged. Same frozen E6 codec, same generator shape and steps; training step compiled and attention segmented (identical results, faster). 4,847 s.

| | Val | Published easy | Published medium | Published hard |
|---|---|---|---|---|
| **Solved, E8** | **5.9%** | **6.2%** | **0.35%** | 0.0% |
| Solved, E6 | 0.0% | 0.0% | 0.0% | 0.0% |
| Blank-cell accuracy, E8 | 63.2% | 74.7% | 61.1% | 48.2% |
| Blank-cell accuracy, E6 | 47.0% | 53.9% | 44.8% | 36.8% |
| Blank-cell accuracy, AR (E4) | 59.5% | 71.0% | 58.4% | 47.4% |
| Givens kept, E8 | 92.9% | 94.1% | 90.3% | 84.3% |

**Reading.** The first NCM configuration here to solve boards. Teacher-forced validation CE is *worse* than E6 (0.297 vs 0.238), because the model now spends capacity on recovering from wrong coarse codes, while greedy generation is much better: blank-cell accuracy rises by about 20 points on every set and exceeds the same-size AR baseline on all three. Solved boards remain below AR on easy (6.2% vs 13.8%) and medium (0.35% vs 3.0%). Train–test mismatch, not codec quality, was the largest single gap.

**Latency** (p50, batch 1 / 16 / 64 / 256): 15.1 / 58 / 618 / 2,154 ms, as E6 within noise.

## E6 — Sudoku, learned level-0 sampler (`configs/sudoku-e6.json`)

**Change from E5.** Codec only: upstream's learned level-0 down/upsampling (`learned_l0`: one strided convolution from 81 positions to 1 and one transposed convolution back, initialised to area pooling and repetition), pre-quantisation GroupNorm(8), codec dropout 0.3. The generator is E5's, unchanged (8 × 256, 4,000 steps), so the difference isolates the codec.

**Stability.** At E5's codec learning rate (1e-3) the learned sampler diverged within 100 steps, both as a four-layer cascade (81 = 3⁴, upstream's `learned_l1_cascade` plan) and as a single layer: the VQ loss reached 1e12. The likely mechanism is the straight-through gradient moving the level-0 input while the EMA codebook trails it. At 2e-4 it trained stably. Upstream has not released its training configuration, so its learning rate is unknown.

**Codec** (3,000 steps, 531 s): exact reconstruction kept (100% cells with all levels). Level 0 is now used: code perplexity 15.9 (E5: 1.0). Cell accuracy decoding from the first k levels: 16, 16, 16, 16, 16, 20, 26, 34, 45, 57, 72, 100% (E5: 11, 11, 13, 14, 15, 18, 23, 32, 45, 56, 72, 100%).

**Generator** (5,241 s, GPU not shared, peak 16.6 GB): validation CE 0.238 at step 4,000 (E5: 0.250), 0.481 at step 1,000 (E5: 0.548). The level-0 code is close to unpredictable from the puzzle (CE 2.29 nats of a possible 2.77); level 1 became deterministic instead.

| | Val | Published easy | Published medium | Published hard |
|---|---|---|---|---|
| Solved | 0.0% | 0.0% | 0.0% | 0.0% |
| Blank-cell accuracy, E6 | 47.0% | 53.9% | 44.8% | 36.8% |
| Blank-cell accuracy, E5 | 45.2% | 51.8% | 43.4% | 36.0% |
| Givens kept, E6 | 91.6% | 92.7% | 89.6% | 87.1% |

**Latency** (12 levels, T = 1.0, p50, 20 iterations, GPU not shared): 13.6 / 54 / 531 / 2,012 ms at batch 1 / 16 / 64 / 256, peak 3.5 GB at 256. Against E4's AR baseline (57 / 90 / 365 / 1,425 ms, measured in a different run), the 12-level NCM is faster at batch ≤16 and slower from batch 64. With 314 codes plus an 81-token prefix per sequence, the per-level forwards stop being cheap at large batch.

**Reading.** Each upstream component added so far moves every secondary metric the right way, by small amounts. None moves solved boards off zero. The two remaining large gaps are the generator width (upstream 8 × 512) and the training budget.

## E5 — Sudoku, upstream-style 12-level hierarchy (`configs/sudoku-ref12.json`)

**Change from E4.** Codec only follows upstream's released Sudoku configuration more closely: 12 levels `1,2,4,6,9,13,18,26,37,50,67,81` × 16 codes, lookup-only encoder (`enc_layers: 0`), one decoder block, width 256, EMA decay 0.99, fine-level dropout 0.5. Generator unchanged in shape (8 × 256, 6.44M params), 4,000 steps. Codec 3.30M params, 345 s; generator 7,784 s, peak 16.6 GB. The Qwen baseline below shared the GPU for most of the generator run. Greedy decoding.

**Solved boards: 0.0% on val and on all three upstream sets** (2,000 boards each).

| | Val | Published easy | Published medium | Published hard |
|---|---|---|---|---|
| Blank-cell accuracy, E5 | 45.2% | 51.8% | 43.4% | 36.0% |
| Blank-cell accuracy, E4 | — | 29% | — | — |
| Blank-cell accuracy, AR (E4) | — | 71% | — | — |
| Givens kept | 89.8% | 89.5% | 86.9% | 84.7% |

**What changed.** The hierarchy is now gradual, as intended: decoding from the first k levels gives cell accuracy 11, 11, 13, 14, 15, 18, 23, 32, 45, 56, 72, 100% for k = 1…12. Level 0 is still dead (perplexity 1.0, one code used) and level 4 is underused (perplexity 4.6). Teacher-forced, the generator gets 59–95% of the codes right at levels 1–10 (CE about 1.0 nats per code at levels 1, 2 and 5), and validation loss was still falling at step 4,000 (0.254 → 0.250 over the last 500 steps). With about 300 codes per board, an exact solve needs nearly every one of them right, so per-code accuracy in this range gives no solved boards even though cell accuracy almost doubled over E4.

**Still different from upstream** (`config/eval/sudoku-*.yaml` and `models/quant.py` in HazyResearch/ncm):

1. Level 0 uses a learned, cascaded strided-conv down/upsampler (`learned_l1_sampling`, `learned_l1_cascade`) instead of area pooling. A mean over all 81 cells is nearly constant, which is the likely reason level 0 is dead here.
2. Pre-quantisation GroupNorm(8) on the encoder output.
3. Generator 8 × 512 (about 4× the parameters; paper: 25M) vs 8 × 256 here.
4. Dropout 0.2 (generator) and 0.3 (codec) vs 0.1.
5. Codec sequence of 89 tokens (grid plus separators) vs 81.
6. One model per difficulty upstream vs one model for every difficulty here; upstream's training budget is not stated in the released configs.

This is a configuration replication with known gaps, not a tuned attempt. Closing gaps 1–3 is the next experiment (`docs/NEXT.md`).

## Reference — Qwen3.8 27B (4-bit, local oMLX) on the upstream boards

`python -m ncm_mlx.baselines.openai_llm`, 50 boards per difficulty, temperature 0, thinking disabled, 400 completion tokens. A reference point, not a matched comparison: the model is about 4,000× larger and was not trained on the task.

| | Easy | Medium | Hard |
|---|---|---|---|
| Solved | 0 / 50 | 0 / 50 | 0 / 50 |
| Blank-cell accuracy | 15.3% | 11.4% | 11.0% |
| Givens kept | 10% | 6% | 4% |
| No parseable grid | 24 | 22 | 25 |
| Latency p50 | 14.2 s | 17.2 s | 55.6 s |

With reasoning enabled (up to 6,000 completion tokens, 10 easy boards) the run had not finished after about 95 minutes on the shared GPU and was stopped, so there is no result for it. Without reasoning, its blank-cell accuracy is at chance (about 11%), and about half the replies are not a 9 × 9 grid. Latency was measured while E5 trained on the same GPU.

## E4 — Sudoku, conditional, 6-level hierarchy (`configs/sudoku.json`)

**Setup.** 200,000 synthetic training puzzles (randomized-backtracking grids, 28–45 givens, solution uniqueness not enforced), 1,000 val. One model for every difficulty. Codec 1.16M params: contextual encoder (2 blocks), levels `1,3,9,27,54,81` × 32 codes; 3,000 steps, 130 s. Generator 6.41M params (8 × 256, 8 heads), conditioned on the 81 puzzle cells as a token prefix; 5,000 steps, 3,444 s, peak 10 GB. AR baseline 6.33M params, same width and depth, same prefix (bidirectional), 5,000 steps, 1,657 s. Greedy decoding (T=0) for both, as in the paper's greedy rows. Test sets: own val and the published NCM boards (2,000 each; 40/35/30 givens).

**Solved boards.**

| Model | Val (28–45 givens) | Published easy (40) | Published medium (35) | Published hard (30) |
|---|---|---|---|---|
| MLX implementation (6 levels) | 0.0% | 0.0% | 0.0% | 0.0% |
| MLX AR baseline, greedy (6.3M) | 10.4% | **13.8%** | 3.0% | 0.4% |
| Original AR baseline, greedy (Kumbong et al., 25M) | — | 14.6% | 5.1% | 1.0% |
| Original NCM (Kumbong et al., 25M, 12 levels) | — | 99.5% | 96.5% | 86.1% |

The MLX AR baseline is numerically close to the Original AR baseline (13.8 vs 14.6 / 13.9% on easy). This is a limited sanity check on boards, scoring and conditioning, not a reproduction of its model or training recipe. Blank-cell accuracy: NCM 29% on easy (chance is ~11%), AR 71%.

**Diagnosis.** The codec puts no information in the coarse levels: decoding from levels 0–2 gives 11% cell accuracy (chance), and utilisation at those levels is 19–28% of the codebook. Nearly all the information sits in level 4 (54 codes, whose pooling windows straddle cell boundaries). The generator predicts level 4 with only 36% accuracy (teacher-forced), and level 5 is then almost deterministic (99.8%). So this hierarchy turns Sudoku into one big, hard parallel guess, which is the opposite of coarse-to-fine. Upstream's released Sudoku configuration differs exactly here: 12 gradual levels (`1,2,4,6,9,13,18,26,37,50,67,89`), 16 codes per level and a lookup-only encoder.

**Latency** (T=1.0, batch 1/16/64/256, p50): NCM 6.6 / 27 / 152 / 722 ms; AR 57 / 90 / 365 / 1,425 ms. NCM is 2–8× faster but solves nothing, so the speed is meaningless for this configuration.

## E3 — block-causal attention (`configs/toy-grammar-12k-causal.json`)

**Change from E1.** Generator only, `attn_pattern: block_causal`: level k also attends to the inputs of levels < k. Same frozen E1 codec, same data and steps. Train 328 s (E1: 266 s).

| | E1 block-diagonal | E3 block-causal |
|---|---|---|
| Valid, T=1.0 | 0.898 | **0.922** |
| Valid, T=0.7 | **0.958** | 0.953 |
| Valid & novel, T=1.0 | 0.163 | 0.173 |
| Forced levels 0–3, valid | 0.984 | 0.978 |
| p50 batch 1 / 16 / 256 | **4.5 / 5.8 / 37.8 ms** | 6.9 / 9.5 / 70.6 ms |
| Peak memory batch 256 | **1.7 GB** | 2.0 GB |

Teacher-forced val CE per level is unchanged to two decimals. The T=1.0 gain (+2.4 points, n=1,024, standard error ≈ 0.9) is small and vanishes at T=0.7. Level k now runs over every coarser level as well, so latency grows with the total code count instead of the level length. **Kept as an option; the default stays block-diagonal.**

## E2 — where NCM loses validity (`examples/level_ablation.py`, E1 checkpoint)

**Forced prefix.** For 1,000 val sentences, use the codec's true codes for levels 0..k and sample the remaining levels at T=1.0 (2 seeds).

| True levels given | none | 0 | 0–1 | 0–2 | 0–3 | 0–4 | all |
|---|---|---|---|---|---|---|---|
| Valid | 0.907 | 0.897 | 0.920 | 0.946 | 0.984 | 0.991 | 1.000 |
| Reproduces the val sentence | 0.041 | 0.052 | 0.232 | 0.666 | 0.951 | 0.991 | 1.000 |

The ~10% of invalid samples is spread across the hierarchy: about 4.5 points come from sampling levels 0–2 (1–4 codes per level), about 5.5 from levels 3–5 (8–32 codes). Sampling the last level alone, with everything coarser correct, still breaks 1% of sentences. Parallel sampling of neighbouring fine codes is a real error source, but not the only one.

**Per-level temperature** (unconditional, 2 × 512):

| Schedule | valid | valid & novel | unique |
|---|---|---|---|
| T=1.0 everywhere | 0.898 | 0.163 | 0.967 |
| T=0.7 everywhere | **0.958** | 0.148 | 0.926 |
| T=1.0 levels 0–1, 0.5 levels ≥2 | 0.938 | 0.148 | 0.944 |
| T=1.0 levels 0–1, greedy levels ≥2 | 0.946 | 0.126 | 0.811 |
| T=1.0 levels 0–2, greedy levels ≥3 | 0.937 | 0.153 | 0.932 |

Cooling only the fine levels does not beat uniform T=0.7, and greedy fine levels cut diversity (unique 0.81): fine levels carry some of the sentence's identity, not only spelling. The hypothesis "diversity from coarse levels, accuracy from fine levels" is **not supported** at this size.

## E1 — toy-grammar v1, 12k train (`configs/toy-grammar-12k.json`, current default)

**Change from E0.** 12,000 train / 1,000 val / 2,360 unseen sentences; generator 3,000 steps with best-validation checkpoint selection (best at step 1,500). AR baseline retrained on the same split for 3,000 steps. GPU: 16% from other processes before the run.

**Codec.** Val sentence-exact reconstruction from all levels **1.000** (char 1.000); from levels 0–2 0.366, 0–3 0.760. Utilisation 100% at every level. Train time 52 s.

**Generator (teacher-forced, val, best step).** CE per level `4.81, 2.04, 0.58, 0.23, 0.15, 0.20`, accuracy `0.02, 0.30, 0.73, 0.90, 0.95, 0.94`. More data roughly halved CE at levels 2–5 compared with E0. Train time 266 s.

**Samples (2 × 512 per temperature).**

| | valid | agreement ∣ well-formed | valid & not in train | unique |
|---|---|---|---|---|
| MLX implementation T=1.0 | 0.898 | 0.998 | 0.163 | 0.967 |
| MLX implementation T=0.7 | 0.958 | 0.999 | 0.148 | 0.926 |
| AR T=1.0 | **0.986** | 1.000 | **0.235** | 0.964 |
| AR T=0.7 | 0.998 | 1.000 | 0.224 | 0.954 |

Unseen sentences are 16.4% of the language, so a model sampling uniformly from the whole language would score `valid & not in train` ≈ 0.164 at validity 1.0. Both models are near that rate; AR is slightly above it (its argmax sentence is unseen).

AR val CE is 0.303 nats/char ≈ ln(15,360)/32 = 0.301, the entropy of a uniform distribution over the language at 32 positions: **the AR baseline has essentially learned this language exactly.** NCM has not. Its validity drops compared with E0 because it memorises less (E0 samples were mostly training copies, which are valid by construction).

**Failure modes (NCM, T=1.0, 22 invalid of 256).** Mostly local spelling slips with correct structure: `cy has one ret fog.`, `cy has two big coxs.`, `eve sees five wed cups.`. A few are positional breakdowns: `fin   s threeeedd   has__daassn`. Number agreement, the long-range constraint, is almost never violated (99.8% given well-formed). Reading: the coarse levels capture the global plan; errors come from sampling neighbouring fine-level codes independently.

**Latency.** Unchanged from E0 within noise (same architecture): NCM p50 4.51 ms at batch 1, 5.76 ms at 16, 37.8 ms at 256; AR 30.5 ms (first iterations noisy, p95 37.7), 22.9 ms, 49.8 ms.

**Conclusion.** On this task, at this size, coarse-to-fine gives **~4–5× lower latency at batch ≤ 16 and ~1.3× at 256**, for **~9 points lower validity at T=1.0 (~4 at T=0.7)** than an equal-size AR model.

## E0 — toy-grammar v0 (`configs/toy-grammar.json`)

**Setup.** TinyWorld split seed 0: 4,000 train / 500 val / 10,860 unseen sentences. Codec 1.30M params (dim 128, 2+2 blocks, 6 levels `1,2,4,8,16,32` × 128 codes), 3,000 steps, 52 s. Generator 4.97M params (6 × 256, 4 heads), 4,000 steps, 369 s. Batch 128, AdamW, cosine schedule.

**Codec (held-out val, decode from levels 0..k).**

| Levels used | 0 | 0–1 | 0–2 | 0–3 | 0–4 | 0–5 |
|---|---|---|---|---|---|---|
| Char accuracy | 0.752 | 0.874 | 0.957 | 0.984 | 0.995 | **1.000** |
| Sentence exact | 0.000 | 0.050 | 0.392 | 0.704 | 0.882 | **0.994** |

Codebook utilisation 97% at level 0, 100% at every other level; no collapse.

**Generator (teacher-forced, val).** CE per level `4.83, 2.10, 1.26, 0.52, 0.48, 0.62`; accuracy `0.01, 0.30, 0.68, 0.90, 0.92, 0.91`. Level 0 CE ≈ ln(code perplexity 108) is expected: level 0 is unconditional, so its best prediction is the code marginal. **Overfitting:** val CE was lowest at step 1,000 (levels 2–5: `0.85, 0.35, 0.35, 0.40`) and rose afterwards while train loss kept falling. The checkpoint used is the final (step 4,000) one; best-val selection was added after this run.

**Samples (2 × 512 per temperature).**

| | valid | well-formed | agreement ∣ well-formed | valid & not in train | unique |
|---|---|---|---|---|---|
| MLX implementation T=1.0 | 0.946 | 0.952 | 0.994 | **0.252** | 0.915 |
| MLX implementation T=0.7 | 0.952 | 0.961 | 0.991 | 0.179 | 0.871 |
| MLX implementation T=0 | 1.000 | 1.000 | 1.000 | 0.000 | 0.001 (one sentence) |
| AR baseline T=1.0 | **0.990** | 0.990 | 1.000 | 0.076 | 0.905 |
| AR baseline T=0.7 | 0.999 | 0.999 | 1.000 | 0.029 | 0.868 |

AR baseline: 4.75M params, same width/depth, 4,000 steps, 159 s, val CE 0.42 nats/char. It is more often valid, and NCM produces about 3× more valid sentences outside its training set. Neither result is tuned; both models overfit.

**Generation latency (T=1.0, 30 timed iterations after 3 warm-ups, timed through a host copy of the tokens).**

| Batch | MLX implementation p50 / p95 ms | MLX AR baseline p50 / p95 ms | MLX implementation seq/s | MLX AR baseline seq/s | MLX implementation / MLX AR peak MB |
|---|---|---|---|---|---|
| 1 | 4.56 / 5.07 | 19.75 / 24.73 | 219 | 51 | 35 / 19 |
| 4 | 4.75 / 4.89 | 20.41 / 21.20 | 842 | 196 | 59 / 25 |
| 8 | 5.10 / 5.19 | 21.01 / 24.22 | 1,570 | 381 | 89 / 33 |
| 16 | 5.66 / 5.83 | 22.26 / 24.59 | 2,825 | 719 | 144 / 46 |
| 64 | 10.65 / 10.79 | 26.66 / 31.83 | 6,011 | 2,401 | 491 / 129 |
| 256 | 35.55 / 36.12 | 49.79 / 52.57 | 7,200 | 5,142 | 1,704 / 460 |

NCM: 6 generator forwards + 1 decoder forward per batch. AR: 32 single-token forwards with a KV cache. First-call latency (graph setup, kernel compilation) at batch 1: NCM 580 ms, AR 36 ms. Neither uses `mx.compile`. At small batches NCM is about 4× faster, mostly because it makes 6 dispatch round trips instead of 32. The gap closes as batch size grows and both become compute bound. NCM peak memory is 3–4× higher because every level attends over its whole length and the codec decoder runs at full length.

## E-precision — Metal fp32 matmul precision (2026-10-22)

`a @ b` for `(4,4,S,16) @ (4,4,16,S)` standard-normal inputs, compared with float64 NumPy: GPU max abs error 1.4e-2 (S=17), 1.6e-2 (S=64), 2.4e-2 (S=256); CPU 2–4e-6. The generator's per-level forward vs. the full teacher-forced forward differs by up to 5.7e-4 on GPU and 0 on CPU (2-layer test model). Tests check exact equivalence on CPU and allow 5e-3 on GPU.
