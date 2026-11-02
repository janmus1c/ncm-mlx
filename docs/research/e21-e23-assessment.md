---
title: E21–E23 controlled-text assessment
type: research-assessment
status: complete
aliases: [TinyStories corruption ablation, E21 E22 E23, E24 seed-1 repeats]
updated: 2026-11-02
evidence: e21-e23-evidence.json
---

# E21–E23 controlled-text assessment

E21 establishes independently trained, unconditional TinyStories generation in MLX. E22 and E23 test how much corrupted conditioning the generator needs during training. All three seed-0 runs, their aligned evaluations and their benchmarks are complete, as are the three generator-seed-1 repeats (E24, E22-seed1 and E23-seed1). The main result is a useful failure: reducing corruption improves prediction when earlier levels are supplied correctly, while making independently generated text worse. E21 is the strongest of these three under the fixed external text scorer; the common autoregressive control remains substantially better under that scorer.

The seed-1 repeats train fresh generators and a fresh AR control with training seed 1, on the same frozen seed-0 codec. E24 repeats E21's recipe; E22-seed1 and E23-seed1 repeat E22 and E23. They test generator reliability conditional on one codec, not full two-stage training variability. The ranking E21 < E22 < E23 in free-running scorer PPL holds in both generator seeds (seed 0: 47.897 / 54.923 / 97.776; seed 1: 45.568 / 55.114 / 93.833), and the AR control stays ahead of every NCM generator (14.676 for seed 0, 14.311 for seed 1). Seed-0 values were audited from retained sample and scorer records; seed-1 values are read from the completed aligned reports, whose recorded result audits pass (see below). No training, inference or benchmark workload was started for this assessment.

## Summary

- **E21 asks:** Can a much smaller, independently trained NCM learn short story openings, using the released TinyStories architecture family? It learns recognizable story language, but its own generations contain broken grammar, repeated words, changing characters and unfinished thoughts. Its codec is much better at reconstructing supplied text than its generator is at inventing text.
- **E22 changes:** Reduce the random damage applied to earlier concept levels during training, keeping the codec, model size, training exposure and optimizer schedule fixed. Predicting the next level from correct earlier levels gets easier; actual generated text scores worse.
- **E23 changes:** Remove that training damage entirely. Correct-context prediction gets better again, while actual generated text deteriorates sharply. A model can do well when given correct context and fail when it must live with its own mistakes.
- **Strongest experiment:** E21 has the best free-running external-scorer result among the three completed runs. E23 has the lowest clean code-prediction loss, which does not make it the best text generator.
- **Progress toward language:** Yes, this moves independent training beyond a hand-written grammar and Sudoku into prose. The corpus is still synthetic GPT-4 stories. It does not establish useful general language generation, natural-document generalization, prompt following or chat.
- **Seed-1 repeats:** E24, E22-seed1 and E23-seed1 repeat the ordering. Lower clean code CE (E24 2.24909, E22-seed1 1.66106, E23-seed1 1.38638) again accompanies worse free-running PPL (45.568, 55.114, 93.833). The gap to the AR control remains large.
- **Scope:** The result supports a reproducibility and failure-analysis account of the training implementation and the corruption ablation. It does not support a claim of competitive natural-language generation.

## Evidence and local state

The runs were produced by the text-training workflow that is now part of this repository (see [text training](../text-training.md)); the experiments used a frozen source snapshot whose file hashes are in the evidence extract, from development commit `d323f4f7803d4e3607dd21dd9d28ae41c1cce2ae`. That commit is not part of the public history. Of its 44 digested training files, 41 are byte-identical in this repository; the other three (`ncm_mlx/official/cli.py`, `model.py` and `sample.py`) also contain the later prefix-cache, compilation and benchmark changes to the released-checkpoint port. The 44 frozen training files, exactly as run, are attached to the paper release as `ncm-mlx-text-training-source-d323f4f.tar.gz`, with a SHA-256 checksum file ([download](https://github.com/janmus1c/ncm-mlx/releases/download/paper-v0.12/ncm-mlx-text-training-source-d323f4f.tar.gz)); each file's digest matches the evidence extract. The evaluation and benchmark driver is not included. The retained evidence describes local Apple Silicon runs under a local sequential controller; it does not describe remote GPU experiments.

The [status page](../STATUS.md) states that public clean-checkout reproduction and naturally occurring text validation remain pending, and that remains the capability boundary. The workflow uses additions to `ncm_mlx/official/`; its name denotes architectural compatibility with the released checkpoint port, not official authorship of the independently trained weights. It differs from the existing TinyWorld/Sudoku training path.

The [sanitized evidence extract](e21-e23-evidence.json) retains normalized configs, source hashes, data and scorer revisions, checkpoint identities, aggregate diagnostics, per-seed scores, benchmark raw times and the first eight NCM samples from sampling seed 42 for the seed-0 runs, plus the aggregate seed-1 results under `seed1_repeats`. Logical `runs/` paths identify original retained files; those ignored files are not supplied by a clean public checkout. Training snapshots, full sample JSONL, per-sample scorer records and the controller remain preserved locally.

Independent CPU-only checks in this assessment verified:

- All 44 frozen training-source files and 38 optimized inference-source files against their recorded SHA-256 hashes.
- Every E21–E23 config, exported checkpoint, tokenizer and stored sample-file hash; all three data-manifest identities and the ordered 4,096 test-window hash.
- Unique, complete sample identities; 1,280 samples per model; 128 in-range BPE token IDs per raw output; every decoded-text hash against its scorer record.
- Every reported per-seed and mean generative perplexity, recomputed from retained individual NLL sums and scored-token counts.

- For the six aligned reports (E21, E22, E23, E24, E22-seed1, E23-seed1), the recorded result audit passes and its claimed report hash equals the computed one. It covers 5,120 scored samples for E21 and 2,560 for each of the other five. The independent per-sample reaggregation described above covers the seed-0 runs; the seed-1 means and standard deviations are taken from the reports and their passing audits.

These checks establish consistency of retained evidence, not an independent training replication or a fresh model-inference run. Large raw-corpus and scorer-weight hashes are retained declared identities and were not rehashed here. The original reports' result auditors also record passing checks. Historic CPU preflight records cover training, recovery, attention equivalence and sampler tests; this assessment did not rerun those tests.

## Shared architecture, data and budgets

| Component | E21–E23 configuration |
|---|---|
| Codec | Token and positional lookup encoder, GroupNorm with 8 groups, learned strided level-0 cascade, area pooling/linear interpolation elsewhere, per-level residual transforms, one transformer decoder block; width 256, 4 heads |
| Hierarchy | Lengths `[1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]`; 2,048 codes at each of 12 levels |
| Codec size | 15,508,096 saved values, including 6,291,456 EMA codebook values; 9,216,640 gradient-trained values |
| Generator | 6 transformer blocks, width 384, 6 heads, QK normalization, RoPE with base 100,000, tied distance heads, block-diagonal level attention; 11,832,587 recorded values |
| Generation | Level 0 sampled from the stored empirical unigram; exactly 11 generator forwards predict the remaining levels; decode to 128 output tokens |
| Conditioning | A single null EOS prefix; previous concept levels provide internal conditioning. No supplied text prompt, continuation or other conditional text task is evaluated |
| AR control | Same 6 × 384 transformer block stack and 6 heads, tied 8,192-token embedding; 13,794,048 parameters; 128 cached next-token forwards |
| Corpus | `roneneldan/TinyStories` V2 GPT-4 train/valid text, revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` |
| Tokenization | Released 8,192-token BPE, EOS 0, PAD 1; verbatim story with leading newline, first 128 tokens of story + EOS, then right padding |
| Data split | 2,717,495 training stories; valid file split by SHA-1 parity of stripped story into 13,960 validation and 13,670 test stories; document overlap with training is checked during preprocessing |
| Environment | Apple M5 Pro, 64 GB unified memory, macOS 26.5, Python 3.12.14, MLX 0.32.3, `mlx-lm` 0.32.2, NumPy 2.5.4; Metal float32 with reduced-precision matmul |
| Configured training seeds | Codec, E21 generator, common AR, E22 generator and E23 generator: seed 0. E24, E22-seed1 and E23-seed1 generators and their AR control: seed 1, on the same frozen seed-0 codec; E22-seed1 and E23-seed1 reuse the seed-1 AR from E24 |
| Codec training | E21 only: 14,000 updates × 256 = 3,584,000 presentations; reused unchanged by E22/E23 |
| Generator training | Each: 15,000 updates × 64 = 960,000 presentations; 122,880,000 output-token positions |
| AR training | Once: 7,500 updates × 128 = 960,000 presentations; reused as the common E22/E23 control |

The pinned [dataset card](https://huggingface.co/datasets/roneneldan/TinyStories/blob/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md) identifies the V2 file as GPT-4-generated and lists CDLA-Sharing-1.0. This is a controlled synthetic-prose study. It does not test naturally occurring documents. Windowing drops later portions of long stories; only about 5% of training windows contain their story's EOS. Roughly 0.38% of training positions are padding. Losses include padding.

All stages use AdamW, peak learning rate `5e-4`, cosine decay to `5e-5`, betas `(0.9, 0.95)` and gradient clipping at 1. Codec warmup is 1,000 updates with weight decay 0.01; generator/AR warmup is 500 with weight decay 0.1. Training uses compiled steps. Codec dropout is 0.3, nested fine-level dropout probability is 0.5, commitment weight is 0.25 and EMA decay is 0.95, with dead-code restart below count 0.01. Generator/AR dropout is 0.1. Generator loss uses `alpha=0`, weighting code positions equally. Exported final weights are evaluated; these are not best-validation selections. The implementation derives generator initialization and minibatch seeds as configured seed + 1, and AR seeds as configured seed + 2; their residual-projection initialization uses a further offset of 7. Thus configured seed 0 does not mean every internal stream is seeded with zero.

| Stage | Actual training-loop seconds | Hours | Time-limited? |
|---|---:|---:|---|
| Shared E21 codec | 5,517.25 | 1.53 | No |
| E21 generator | 13,898.17 | 3.86 | No |
| E22 generator | 14,435.59 | 4.01 | No |
| E23 generator | 20,734.07 | 5.76 | No |
| Shared seed-0 AR | 2,062.03 | 0.57 | No |
| E24 generator (seed 1) | 18,842.73 | 5.23 | No |
| E22-seed1 generator | 22,419.19 | 6.23 | No |
| E23-seed1 generator | 18,706.68 | 5.20 | No |

Training-loop totals exclude setup, evaluation and publication work. The first E21 codec + generator costs 19,415.42 seconds; later ablations reuse the codec. Training logs record a generator peak of about 13.39 GiB. The follow-up driver caps MLX memory at 26 GiB and cache at 2 GiB. These are local elapsed times, not dollar, energy or FLOP measurements. E23's longer wall time cannot be attributed to corruption alone: desktop load and runtime conditions vary.

The comparison matches generator/AR window exposure and block stack, **not total compute or total model size**. The 11 predicted levels sum to 640 code positions; together with the EOS prefix, NCM's teacher-forced input contains 641 positions per window versus 128 AR positions. The teacher-forced input is therefore 641 positions, not 640. Sequence length alone is not a FLOP estimate, and codec training is additional. The frozen codec also adds model storage and inference work. Released TinyStories is a separate, much larger reference: 36,434,816 codec values and 88,605,707 generator values. The pilot is not a matched reproduction of that model or its training budget.

## Evaluation contract

Reconstruction and teacher-forced losses use the same ordered first 4,096 held-out test windows, with hash in the evidence extract. Generations use sampling seeds `[42, 0, 1, 2, 3]`, 256 samples per seed, batch 128, temperature 1, top-k 0 and top-p 1. These five sampling seeds do not provide five independently trained models. Sub-batches derive their actual random seeds from the reported seed and offset; batch size is part of the reproduction contract.

Generative perplexity is an external probability score of generated text under [GPT-2 large at pinned revision `32b71b1`](https://huggingface.co/openai-community/gpt2-large/tree/32b71b12589c2f8d625668d2335a01cac3249519). It is not NCM's own token perplexity. Cleanup stops at the first EOS and removes PAD tokens before decoding. GPT-2 tokenization adds no BOS; every token after the first is scored. Right-padded scorer batches mask padded targets. NLL sums divide by scored-token counts within each sampling seed, then seed perplexities are averaged. Empty or one-token strings are unscorable and counted separately; none occurred. The scorer cap is 1,024 GPT-2 tokens. Retained records have at most 155 scored tokens, so that cap did not truncate this evaluation.

This score is useful for a fixed comparison, but does not directly establish coherence, grammatical correctness, truthfulness or usefulness. No LLM-judge score is used. Qualitative observations below come from an informal, unblinded reading, not a human quality study. The AR's own held-out token CE is distinct from external generative perplexity; NCM code CE is distinct from both. Temperature/top-k/top-p sensitivity has not been evaluated in these aligned reports.

### Sensitivity of the saved scores

A subsequent CPU-only audit checks 7,680 saved text identities against their scorer-record hashes and reproduces every reported mean to its published three-decimal precision. It then reaggregates the same NLLs; no samples are generated or rescored. Pooled token PPL exponentiates total NLL divided by total scored tokens. Equal-sample geometric PPL exponentiates the mean per-output token NLL, giving each output equal weight. Median sample PPL describes the central output. These are alternative summaries of one scorer, rather than independent quality measurements.

| Model | Reported mean seed PPL | Pooled token PPL | Equal-sample geometric PPL | Median sample PPL |
|---|---:|---:|---:|---:|
| E21 | 47.897 | 47.891 | 47.862 | 46.196 |
| E22 | 54.923 | 54.906 | 54.868 | 52.635 |
| E23 | 97.776 | 97.757 | 97.680 | 95.498 |
| Common AR | 14.676 | 14.671 | 14.708 | 13.612 |

The ranking survives each aggregation. Mean scored lengths are similar: 127.1, 127.1 and 127.5 GPT-2 tokens for E21/E22/E23. Within the 96–127-token group, pooled PPL is 46.089 / 53.804 / 94.413; within 128–159 tokens, it is 50.807 / 56.304 / 100.457. Each group contains at least 494 outputs per NCM model, and the AR remains ahead within both groups. These post-hoc length groups do not control every generation difference or establish human-rated quality. They show that these particular aggregation and length checks do not reverse the first-seed result. Exact values and input hashes are retained under `additional_cpu_audit` in the [evidence extract](e21-e23-evidence.json).

## E21: establish a text-training pilot

**Objective and hypothesis.** Test whether an independently implemented training recipe for a smaller member of the released architecture family can learn controlled prose and generate it in 11 concept forwards. Corrupt earlier conditioning codes at rates falling from 0.75 to 0.25 across levels, while keeping targets clean, to reduce dependence on perfect teacher-forced context.

**What changed.** Earlier independent training focused on TinyWorld and Sudoku. E21 introduces BPE story windows, a new codec trained on those windows, the compatible text generator, a token AR control and a pinned external scorer. It uses the inference-port architecture, rather than assuming the earlier task-specific codec is adequate for prose. Differences from the inferred upstream training recipe remain explicit assumptions; upstream checkpoint parity does not prove training-recipe equivalence.

**Result.** Non-padding reconstruction accuracy is 99.816% and exact-window reconstruction is 84.937%. Clean teacher-forced code CE is 2.24875 nats per code. Unconditional generation has external generative perplexity **47.897**, compared with **14.676** for the common AR and **16.259** for the released reference. Held-out prose scores **8.529** under the same scorer. These references differ in budget and/or model size.

**Conclusion.** The full text-training and evaluation path works, and E21 is the strongest completed ablation on this scorer. Reconstruction is excellent; free-running language quality remains weak.

## E22: reduce conditioning corruption

**Objective and hypothesis.** Test whether E21 damages its conditioning too aggressively for this small model and budget. Lower corruption to 0.30 → 0.05, keeping the frozen E21 codec, unigram, tokenizer, architecture, seed, optimizer and window exposure fixed. Train a fresh generator; do not count reuse of the AR checkpoint as a new AR experiment.

**Result.** Clean code CE improves to **1.66515**, while external generative perplexity worsens to **54.923**. Codec results remain identical because the codec is shared.

**Conclusion.** Lower clean-context loss does not transfer to better free-running text in this seed-0 comparison. Reduced corruption is not an improvement under the fixed generation protocol. The direction is consistent across the five sampling-seed scores, and the seed-1 repeat (below) points the same way.

## E23: remove conditioning corruption

**Objective and hypothesis.** Establish the zero-corruption endpoint at the same generator exposure. Train a fresh seed-0 generator with both corruption endpoints zero, sharing all other scientific settings with E22.

**Result.** Clean code CE improves further to **1.38995**, while external generative perplexity deteriorates to **97.776**. The common codec and AR control are unchanged.

**Conclusion.** This is the clearest negative result. Training on perfect earlier codes produces the best clean-context loss of the three and the worst independently generated text score. It is consistent with sensitivity to self-generated context errors; it does not isolate every mechanism behind the failure.

## E24 and the seed-1 repeats

**Objective and hypothesis.** Check whether the E21 < E22 < E23 ordering is a property of the corruption schedule or of one generator seed. E24 repeats E21's generator recipe, and E22-seed1 and E23-seed1 repeat E22 and E23, each with training seed 1 and the same frozen seed-0 codec, BPE and level-0 unigram. A fresh seed-1 AR control (E24's) serves all three. All three generators complete 15,000 of 15,000 updates at batch 64 and none is time-limited. A disk-headroom guard stopped the E24 controller once, after generator training had finished; the queue resumed at AR training without retraining the generator or changing a recipe. The same evaluation contract applies: 4,096 held-out windows, 256 outputs × sampling seeds 42, 0, 1, 2, 3, temperature 1, top-k 0, top-p 1.

| Run | Corruption | Training seed | Clean code CE, nats/code | External generative PPL | SD across sampling seeds |
|---|---|---:|---:|---:|---:|
| E21 | 0.75 → 0.25 | 0 | 2.24875 | 47.897 | 0.797 |
| E24 | 0.75 → 0.25 | 1 | 2.24909 | 45.568 | 0.909 |
| E22 | 0.30 → 0.05 | 0 | 1.66515 | 54.923 | 1.548 |
| E22-seed1 | 0.30 → 0.05 | 1 | 1.66106 | 55.114 | 1.568 |
| E23 | 0 → 0 | 0 | 1.38995 | 97.776 | 2.107 |
| E23-seed1 | 0 → 0 | 1 | 1.38638 | 93.833 | 1.424 |
| AR control, seed 0 | n/a | 0 | AR token CE 1.31454 | 14.676 | 0.450 |
| AR control, seed 1 | n/a | 1 | AR token CE 1.31806 | 14.311 | 0.237 |

Per-sampling-seed values (42, 0, 1, 2, 3): E24 45.737, 44.409, 45.102, 46.870, 45.722; E22-seed1 54.969, 55.192, 56.967, 55.768, 52.675; E23-seed1 94.068, 91.984, 93.083, 95.816, 94.213; AR seed 1 14.246, 14.254, 13.977, 14.534, 14.547. No sample was unscorable in any run, and codec reconstruction is the shared 99.816% non-pad token accuracy.

**Conclusion.** The ordering of free-running scorer quality is the same in both generator seeds, as is the opposite ordering of clean code CE. Between generator seeds the means differ by 2.329 for E21/E24, 0.191 for E22 and 3.943 for E23, all much smaller than the gaps of 7.0 to 42.9 between adjacent corruption settings within a seed, but not negligible beside the sampling-seed SDs. With two generator seeds on one codec, these differences describe run-to-run variation, not a variance estimate or significance test. The AR controls differ slightly between seeds (14.676 and 14.311), so seed-1 gaps are measured against the seed-1 control. The seed-1 repeats do not test codec-seed variation.

## Comparison

| Result | E21 | E22 | E23 | Common AR | Released reference | Held-out data |
|---|---:|---:|---:|---:|---:|---:|
| Corruption endpoints | 0.75 → 0.25 | 0.30 → 0.05 | 0 → 0 | n/a | Not trained here | n/a |
| Clean code CE, nats/code | 2.24875 | 1.66515 | 1.38995 | n/a | 1.37032 | n/a |
| Non-pad reconstruction accuracy | 99.816% | Shared | Shared | n/a | 99.981% | n/a |
| Exact reconstruction windows | 84.937% | Shared | Shared | n/a | 92.969% | n/a |
| External generative PPL, lower is better | 47.897 | 54.923 | 97.776 | 14.676 | 16.259 | 8.529 |
| SD across sampling seeds | 0.797 | 1.548 | 2.107 | 0.450 | 0.315 | 0.067 |
| Mean output BPE tokens after cleanup | 127.36 | 127.20 | 127.14 | 127.61 | 128.00 | 127.32 |
| Outputs with EOS in window | 7.11% | 8.05% | 7.50% | 3.12% | 0.31% | 5.94% |
| Distinct 3-grams | 0.4951 | 0.5374 | 0.6239 | 0.4910 | 0.4595 | 0.4850 |
| Within-output repeated 4-grams | 0.51% | 0.40% | 0.19% | 0.84% | 0.58% | 1.01% |
| Exact copied training opening, 32 tokens | 0.00% | 0.00% | 0.00% | 1.25% | 0.70% | 4.61% |

All rows use 1,280 outputs per model for generation diagnostics; reconstruction uses 4,096 supplied windows. The reused AR is one control, not three independent runs. The sampling-seed SD is not a confidence interval over training runs. Lower repetition and higher distinct-n in E23 coexist with substantially worse text scores: diverse grammatical noise can improve these diagnostics. Zero exact opening matches does not establish absence of memorization.

The preprocessing record reports no exact held-out document overlap with training. Near duplicates remain: about 4.7% of test windows share their first 32 tokens with a training window, and 4.61% of the 1,280 sampled held-out data windows match that diagnostic. Only opening overlap is measured here; substring copying, semantic near-duplicates and full training-membership analysis remain untested. The raw corpus split was not rebuilt in this assessment.

## Retained examples and qualitative inspection

These are the **first** NCM output for sampling seed 42, index 0, in each aligned report, shown as opening excerpts rather than selected for quality. Their complete text, raw token IDs and seven subsequent outputs are retained in the evidence extract. All 1,280 outputs per model remain in the original JSONL files.

**E21:**

> Once there was a little boy called Jack. Joe was was three years old and discover. One One day, he wanted to go and look for exploring.

**E22:**

> He and his mum were walking by the beach beach. T Tise was very happy because he wanted to learn how to try it. He wanted to show his plane plane and take off his boat.

**E23:**

> Jack and his mum were walking by the beach at. T T gears. Tom loved to harbor because learn learned he wanted to travel beyond and was and excited, green trees and trees.

Inspection of the first eight retained outputs from each run identifies recognizable story vocabulary and short plausible phrases mixed with word duplication, incomplete clauses, name/pronoun changes, implausible actions and unstable narrative continuity. The first E21 output switches among Jack, Joe and John. E23's first output shifts among beach, trees, a park, a lizard, a squirrel and ducks with broken transitions. These are qualitative observations from a fixed small sample, not human ratings, a blinded comparison or an estimated grammatical-error rate. Many outputs, including the data reference, end mid-story because the protocol caps length at 128 BPE tokens.

## Latency, memory and interpretation

| Runner | Batch 1 p50/p95 ms | Batch 16 p50/p95 ms | Batch 64 p50/p95 ms |
|---|---:|---:|---:|
| E21 eager NCM | 12.791 / 13.005 | 45.432 / 45.888 | 223.878 / 227.090 |
| E21 common AR, KV cache | 81.841 / 83.411 | 142.712 / 144.585 | 357.411 / 365.647 |
| E22 eager NCM | 13.270 / 14.057 | 51.638 / 59.402 | 275.516 / 331.925 |
| E23 eager NCM | 12.903 / 13.142 | 50.851 / 96.635 | 332.046 / 445.853 |

Three warmups and 20 iterations time completion through a decoded-token host copy. E21 eager NCM peaks at approximately 184.6 / 1,114.5 / 2,256.6 MiB for batches 1/16/64; the AR batch-1 peak is about 62.6 MiB. These are MLX allocation peaks, not full process RSS. First calls, raw times, each run's AR timings and GPU-load observations are in the extract. First-call costs are per mode after model loading; earlier modes may already have warmed device kernels.

Desktop services were active. E21's benchmark pre-row utilization reads zero, but these discrete observations cannot exclude load during timing. E22/E23 include nonzero boundary observations and larger latency tails; do not interpret their slower batch timings as an architectural regression from changing training corruption. Evaluations record before/after utilization of 0/99% (E21), 21/99% (E22), and 0/94% (E23); after-run utilization includes the experiment itself and is not a measure of foreign contention. Periodic load logs remain retained. No competing NCM task was scheduled concurrently by the controller.

The 11-forward generator is faster in these recorded eager comparisons, while scoring much worse than the AR on generated text. This is not a matched-quality speedup. Cached and cached+compiled NCM modes have separate timings. E21 and E23 cold GPU outputs differ from eager at batches 16 and 64: changed-token fractions are about 0.049–0.317% for E21 and 0.342–0.403% for E23. E22's tested cold outputs match. This is one seed per tested batch and mode, not a distribution-level parity study. Optimized timing cannot inherit eager quality results without a separately justified equivalence or evaluation claim.

## What is established and what remains open

**Established:** independently trained short-prose reconstruction and unconditional generation run end to end; retained evidence separates code prediction from actual generated text; this corruption ablation, in two generator seeds on one codec, exhibits opposite movements in clean-context code loss and free-running scorer quality. The large reconstruction/generation gap is measurable rather than inferred from loss alone.

**Unproven:** useful natural-document language quality, conditional continuation, long-form coherence, robust grammar, full memorization safety, temperature sensitivity, causal attribution beyond two generator seeds, independent codec-seed replication, and matched-quality or matched-total-compute superiority to AR. The pilot also does not test the ceiling of the architecture under the released model's far larger budget. No energy measurement, rental cost or independent hardware replication is available.

**Reproducibility:** final exports, full-state training snapshots, configs, source digests, ordered scoring-input hash, all sample outputs and per-sample scorer values exist locally. The audit can recover the published aggregates from those values. The training and evaluation workflow is now in this repository, but the aligned evaluation and benchmark job driver that produced these reports is not part of it, so a clean-checkout reproduction has not been verified. The optimized sampler is pinned by frozen source hashes rather than a selected public release commit. A reproducible release needs a reviewed public evaluation driver or an equivalent deterministic reproduction path, a complete source snapshot, sample and scorer artifacts, and data-rights notes. The sanitized extract alone does not provide that.

## Next research steps

1. Keep seed-0 and seed-1 results side by side; do not replace either with a selected winner.
2. Use the retained outputs to perform a small fixed, blinded human comparison with the AR and reference, recording instructions, sample identities and failures. Expand novelty checks beyond exact openings. These address quality and memorization questions the scorer cannot answer.
3. Before further compute, choose a targeted test: a small sampling-policy sweep can separate poor sampling calibration from poor learned generation; a teacher-forced versus self-generated per-level error analysis can probe the corruption hypothesis. Both need a documented budget and a free accelerator. Neither requires another large training campaign. A second codec seed is the first test of codec-conditional variation.
4. Verify the public text-training and evaluation recipe from a clean checkout. Assess data and weight distribution terms before any new checkpoint release. Keep actual model generation separate from illustrative refinement explanations in any demo.

**Scope of claims.** The controlled-text workflow and the corruption ablation extend the documented findings with a reproducibility and failure analysis. The evidence does not support claims of strong natural-language generation or superiority to AR.
