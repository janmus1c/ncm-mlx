---
title: Parallel generation and candidate use cases
description: Where coarse-to-fine NCM fits among parallel generation methods, what this repository measures and which proposed uses remain untested.
status: exploratory
updated: 2026-10-25
---

# Parallel generation and candidate use cases

Next-Concept Modeling generates a hierarchy of discrete latent codes, predicting all positions of each level in parallel. This page separates that mechanism from other ways to accelerate generation and identifies experiments that could test useful applications. The [related-work comparison](../related-work.md) covers specific implementations; [capabilities](../capabilities.md) records what NCM-MLX actually supports.

## Different routes to parallel work

| Approach | Sequential dependency | Parallel work | Relation to NCM |
|---|---|---|---|
| Autoregressive decoding | Previously generated tokens | Separate batch elements; prompt processing | The local character-level baseline |
| Speculative decoding | Verified token chunks | A target model verifies a draft span | Using NCM as a drafter is a hypothesis, not an implemented feature |
| Discrete diffusion | Denoising/refinement steps | Many sequence positions per step; details depend on the sampler | The original NCM paper's principal comparison family |
| Visual autoregressive modeling (VAR) | Image resolution scales | Code positions within one scale | A precursor to NCM's next-scale formulation |
| Next-Concept Modeling | Coarse-to-fine latent levels | Code positions within each level | Six levels in TinyWorld; twelve in the later local Sudoku configs |

See [VAR, Tian et al.](https://arxiv.org/abs/2404.02905), [masked diffusion, Sahoo et al.](https://arxiv.org/abs/2406.07524), and [NCM, Kumbong et al.](https://arxiv.org/abs/2610.14162) for the methods. A forward count alone does not determine latency: processed positions, prefix work, decoder cost, model size, kernel execution and host dispatch all matter. The local [benchmark guide](../benchmarks.md) specifies the timing boundary.

## Evidence established so far

Original NCM (Kumbong et al.) reports text and Sudoku results, including 99.5 / 96.5 / 86.1% solved on easy/medium/hard Sudoku. Those are upstream reference results. Its Sudoku generator has about 25.46M parameters and sees 51.2M training presentations; its task representation and difficulty-specific checkpoints differ from this repository. These differences prevent an equal-budget reproduction claim.

In the MLX implementation, the early E4–E6 runs solved no external boards. Conditioning corruption, training exposure and unique training puzzles improved later runs: E15 reaches 96.25 / 76.65 / 31.90% after 8.192M generator presentations. E19-funded later reaches 98.10 / 83.40 / 40.30% at 16.384M presentations, twice E15’s exposure. Its easy score is within 1.40 points of Original NCM, while medium/hard remain substantially behind. The MLX implementation’s 6.44M generator is 3.95× smaller than Original NCM’s generator and also uses a separate codec with 13.915M saved values, including EMA state. E20 repeats the funded scientific recipe with seed 1 at equal exposure and reports 97.25 / 79.60 / 33.25%; its 7.05-point hard-set deficit to seed 0 limits best-seed generalization. The [two-seed assessment](e20-seed-report.md) records the aggregate-only evidence and missing remote provenance. E17's recovered 71.25 / 31.75 / 3.70% remains qualified by reset optimizer and random state. Negative results and intermediate steps remain in [EXPERIMENTS](../EXPERIMENTS.md).

The local general-model reference also solved no boards in its small no-reasoning evaluation, while its reasoning-enabled trial was stopped under heavy contention. That experiment does not establish a general limit on chat models, long reasoning, search or solver-assisted systems.

On TinyWorld, NCM improves synchronized latency while the character-level AR baseline generates more valid sentences. These are small synthetic models. They establish neither a speedup over large language models nor useful real-text quality. Training compilation exists for the generator; the two inference samplers remain uncompiled, leaving dispatch overhead as an unresolved comparison factor.

## Applications to test, rather than advertised features

A promising shape is a short, fixed-size output with dependencies across positions and a reliable external validator. Sudoku supports investigating this shape; it does not validate every proposed application.

| Candidate application | Test required | Main alternative or risk |
|---|---|---|
| Constraint grids, schedules or assignments | Valid complete outputs per second against a budget-matched baseline and an independent checker | A conventional constraint solver may be more reliable and efficient |
| Structured synthetic records or test fixtures | Schema validity, coverage, diversity and downstream usefulness on held-out cases | A template engine may already solve the task |
| Multi-field structured prediction | Exact agreement across fields and quality on unseen inputs | Constrained AR decoding or a discriminative model may be sufficient |
| Short summaries, translations or UI text | Real-text quality and latency at matched system size and training cost | TinyWorld spelling quality does not establish open-vocabulary capability |
| A speculative drafter | Accepted draft tokens per target-model verification, with total wall time | Poor acceptance can erase any drafting advantage |

No local service, scheduling tool, JSON-constrained sampler or speculative-decoding integration is implemented. These uses require new datasets, conditioning designs and evaluation before becoming public capabilities.

## Research questions and next steps

1. Do coarse levels carry useful semantic information on real text, or primarily surface structure?
2. How much of the small-model latency gap remains with both samplers compiled and contention controlled?
3. How do quality and total training cost compare across several training seeds and matched budgets?
4. Can the released upstream checkpoints be imported and evaluated without changing their task semantics?
5. Can a single conditional model generalize across difficulty or task distributions?

[Current status](../STATUS.md) and [NEXT](../NEXT.md) define the active research sequence. The release does not initiate new training or turn the exploratory applications above into a delivery commitment.
