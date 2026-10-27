---
title: Inference research experiment protocol
status: proposed
updated: 2026-10-27
evidence: [literature-review, source-inspection]
---

# Inference research experiment protocol

This protocol turns the [prefix-caching and speculative-drafting review](prefix-caching-and-speculative-drafting.md) into bounded, reviewable experiments. These are proposals, not executed runs. No compute has been allocated by this document. Assign an unused experiment ID in [EXPERIMENTS](../EXPERIMENTS.md) when a study actually runs; the labels below identify protocols rather than completed experiments.

The order is: prove the inference contract, measure existing checkpoints, establish the text baseline, then decide whether drafter training has a viable cost/acceptance target. Read [NEXT](../NEXT.md), [benchmark methodology](../benchmarks.md) and [RECOVERY](../RECOVERY.md) before running work.

## PC-1 — generation-local prefix cache

**Question:** Can the repeated prefix computation be removed without changing the existing model's inference contract?

**Intervention:** retain normalized/rotated prefix K and matching V for each transformer layer during level zero; reuse those entries for later levels. Keep the full hierarchy, checkpoint, token inputs and sampling settings fixed. Initially support `block_diagonal` only, or add explicit independent tests before enabling another attention pattern. Do not combine this patch with compilation, dtype changes, different levels or early exit.

**Required checks:**

- Existing CPU teacher-forced/per-level exactness and one-forward-per-level tests pass without changed assertions or tolerances.
- Cached and uncached CPU logits agree exactly at every level, not merely after final argmax. Exercise both learned unconditional prefixes and token-conditioned prefixes, learned level-zero resampling, and more than one batch size.
- Deterministic and sampled generation retain their declared code/token and random-key contract. Cache construction does not consume new random draws.
- A different input prefix, repeated generation, changed model, batch permutation or changed device/dtype cannot read stale state. Cache lifetime is one generation in the initial design.
- Padding behavior is explicit. Sudoku zero is a real blank-cell input, not masked padding. Proposed variable-length text masks need their own tests.
- The implementation actually avoids prefix Q/projection/feed-forward work after its first level; merely storing K/V while recomputing all prefix hidden states does not answer the experiment.

**Gate:** exact CPU checks pass, real computation reuse is observable and no forward is hidden in an uncounted prefill. If changing matrix shapes breaks exact CPU agreement, investigate the execution path rather than weakening the invariant. Training behavior and checkpoint format remain unchanged.

## PC-2 — cache quality, cost and memory

**Question:** Does the equivalent path improve the workload of interest on a verified backend?

Use the same completed checkpoint and complete board identities for both paths. A weights/config prerequisite must be stated if a checkpoint is not publicly distributed; do not present a private-artifact-dependent command as a self-contained reproduction. Time full generation, including cache construction, all levels, quantizer/resampling work, decoder and a host-visible result. Each request starts with a fresh generation-local cache.

Record a first-call measurement separately, then use at least three warmups and 30 measured repetitions per supported batch size. Start with batches 1, 16 and 64 when memory permits; record skipped sizes and the reason. Interleave or counterbalance method order and retain every timing. Record backend-appropriate GPU load before measurement and during the window where available. Do not start a competing workload or stop another service to create an artificial result.

Run final paired generation evaluation over all 2,000 boards per available Sudoku difficulty, with the same greedy policy. Retain input/checkpoint/source identities and per-board outputs, including failures. Compare exact codes/tokens, clue preservation and solved status. Accelerator logit tolerances alone do not establish unchanged argmax choices or quality.

Report first-call, full-request p50/p95, total elapsed time, logical cache payload and actual peak memory. Include host allocation/copy and synchronization behavior where it materially changes the result. Warm kernels are permissible; a cache silently prefilled outside the timed request is a different experiment. Any cross-request cache timing must be reported separately with hit rate and invalidation policy.

**Gate:** publish the measured trade-off. A recommendation to enable caching needs preserved output/quality evidence and a repeatable latency or memory benefit on the stated workload. If output choices differ or speed does not improve, retain the result and keep the optimization experimental. Analytic position savings alone do not pass this gate.

## SD-0 — verifier contract before a learned assistant

**Question:** Does a greedy verifier reproduce an independently validated target decoder?

Choose one supported target architecture, exact checkpoint revision, tokenizer/template revision, quantization and positive output budget. Compare against that target framework's ordinary greedy generation, not only a second path through the new wrapper. Establish how the final prompt token/logits and cache offsets are handled before adding proposals.

Use deterministic fixtures covering:

| Fixture | Required behavior |
|---|---|
| No proposal | Same target output and strict length/EOS rules as ordinary decoding |
| Oracle proposal | All eligible matching tokens accepted; oracle construction is a correctness diagnostic, not a speed result |
| Wrong first token | Immediate target correction; rejected suffix absent from committed cache |
| Mismatch at each later position | Longest valid prefix only, with correctly rewound cache and position offsets |
| All tokens match | Bonus only within remaining budget; next round has a consistent last-token convention |
| EOS within a block | Output terminates at EOS; later proposed tokens are not emitted |
| Remaining budget smaller than draft | No overrun; reported token counts match saved outputs |
| Token-boundary or special-token change | Valid token IDs or explicit proposal rejection; no silent prefix rewrite |
| Unsupported or non-trimmable cache | Clear rejection or a separately validated fallback |

Validate changes in batch, cancellation and reentrant calls. A trace should identify committed tokens, cached processed tokens, draft origin, acceptance length and rewind amount after every round. Keep sampled decoding disabled in this first study. Any target numerical mismatch must be investigated before claiming greedy identity.

**Gate:** complete target-token identity and correct state transitions on the declared contract. Do not infer sampling-distribution preservation from this result.

## SD-1 — training-free cost baseline

**Question:** When can proposal verification beat ordinary target generation without paying to train an assistant?

After the verifier gate, compare ordinary decoding, no-proposal verification and prompt-lookup proposals. Sweep draft caps such as 2, 4, 8 and 16 as planned settings, not supported optima. Use a held-out workload stratified into copying/extraction, constrained editing and novel continuation. Keep source documents disjoint from later training and tuning data. Mark any quick smoke set as a smoke test; a headline result needs a prespecified larger workload and repeated timing.

Measure full-request latency and decode-only latency separately, including prefill in the former. Retain cold/warm conditions, output length, proposal time, verification time, state-management time, drafted/accepted/committed counts, empty-proposal frequency and the empirical acceptance-length survival curve. Report pooled throughput as total committed output tokens divided by total measured time; also report per-request latency distributions. Do not substitute the mean of token-rate ratios for a declared aggregate without labeling it.

Use paired inputs and identical output policy. A paired latency comparison with resampling uncertainty may describe the measured workload; preserve repeated-case dependence and do not generalize that interval to all hardware. Report every task stratum, including regressions. There is no automatic benefit on abstractive output merely because the input contains reusable text.

**Gate:** a documented verifier overhead and break-even accepted-prefix length exist. If verification alone is too expensive, reduce runtime overhead or stop before training a drafter. A lookup success is a baseline for an application, not evidence for a learned NCM assistant.

## SD-2 — bounded history-conditioned NCM draft training

This phase depends on the controlled-text gates in [NEXT](../NEXT.md). It cannot begin from Sudoku solve rates alone.

Before training, freeze the target and define a shared target-token interface or a separately verified tokenizer bridge. Create a licensed, revision-pinned document-level split and disjoint evaluation strata. Record teacher generation costs and any stored target logits/features. Build a codec reconstruction baseline and a small autoregressive assistant baseline with explicit data and training costs.

Train the initial NCM drafter to propose the next block from committed history. Start with token-history conditioning and greedy verification. Compare teacher-forced training with an explicitly defined rollout/on-policy data variant; record which model produced each history. Codec reconstruction, latent validation accuracy, standalone text quality and target-prefix acceptance remain separate measurements.

Vary one factor at a time: draft block size, concept-level budget, alignment data or target-feature conditioning. Do not add all of them in one run and attribute the outcome to one change. Benchmark every candidate against the same ordinary target, lookup baseline and conventional assistant. Include discarded drafts and all drafting/codec work in latency.

**Go/no-go:** allocate a larger run only after a small held-out pilot demonstrates plausible end-to-end benefit under the measured cost model. Set a hard run-duration/spend limit and checkpoint/evaluation plan before starting. A failure to break even is a valid result, not a reason to keep increasing model size without a new hypothesis.

## SD-3 — optional stochastic verification

Treat this as a separate design and validation effort. Specify normalized target/proposal distributions after all sampling transforms, support handling, correction, RNG streams, EOS and length rules. For a latent drafter, identify whether token proposals are deterministic, factorized conditional on a sampled hierarchy or sequential; do not use latent-code likelihood as token probability.

Begin with exact finite distributions, including zero support and deterministic proposals. Then test the actual sampler on enumerable conditional fixtures, using both analytic checks and repeated sampling with declared statistical tolerances. Check cache/numerical behavior separately on the accelerator. Matching a few sampled strings is not a distribution test, and matching a distribution does not promise fixed-seed string identity. Approximate/lenient acceptance needs a separately labeled policy and measured quality trade-off.

## Minimum retained evidence

Every measured run needs the source commit and executed-code digest; full config; checkpoint, tokenizer/template and ordered input hashes; data rights/split; backend/library/device/precision settings; and GPU contention record. Save per-input outputs plus per-round draft/accept/commit/rewind records, raw timing samples, failures and recovery state. Token traces can contain source documents: review their redistribution rights and private content before publishing them.

A public summary must identify the tested policy and its prerequisites, sample/repetition counts, quality and numerical checks, p50/p95 and aggregate throughput, peak memory, training/teacher costs where applicable, and remaining reproduction limits. Preserve immutable published paper versions; a research protocol is not a new empirical paper or a completed run.

## What must be decided before the next training allocation

1. Whether prefix-cache correctness work can proceed independently of the paid training window.
2. Which controlled-text corpus, vocabulary and conditioning contract form the first evaluated baseline.
3. Whether the intended product is standalone NCM generation or assistance for a particular AR target; their training objectives differ.
4. The minimum useful measured latency/acceptance target, a bounded pilot budget and the evidence that ends an unsuccessful run.

No calendar date or future quality score is implied. These decisions keep implementation feasibility, measured performance and training expenditure separate.
