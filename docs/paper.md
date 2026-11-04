---
title: Working paper and downloadable editions
description: Read the NCM-MLX working paper, its experimental evidence, version history and reproducible source in six formats.
---

# Working paper

**Coarse-to-Fine Text Generation on a Laptop: A Native MLX Study of Next-Concept Modeling**<br>
Jan Music ([ORCID](https://orcid.org/0009-0001-5376-7913)) · Working paper v0.13 · 2026-11-04 · [DOI: 10.5281/zenodo.23953213](https://doi.org/10.5281/zenodo.23953213)

## Abstract

Next-Concept Modeling (NCM) generates a sequence through successive levels of discrete latent codes, predicting each level in parallel. The method is by Kumbong et al.; we implement it independently in MLX and study synthetic language, Sudoku and short synthetic stories on Apple Silicon and NVIDIA GPUs. On a finite synthetic grammar, a 5M-parameter generator achieves lower small-batch latency than a similarly sized autoregressive generator, with lower sentence validity (89.8% versus 98.6%). Two Sudoku training seeds use the same 6.44M-parameter generator recipe and 16.384M presentations each. Retained greedy evaluation aggregates report easy/medium/hard solve rates of 98.10/83.40/40.30% for seed 0 and 97.25/79.60/33.25% for seed 1, with 2,000 boards per difficulty; two seeds do not establish population training variability or its cause. Original NCM reports 99.50/96.50/86.10% with a 25.46M generator, 51.2M presentations and a different task setup, so the comparison establishes neither task parity nor matched-cost efficiency. Missing per-board predictions and remote provenance manifests limit independent audit of the Sudoku runs.

This version adds two bounded results. First, an inference-only MLX port of the released TinyStories and Sudoku checkpoints agrees with upstream PyTorch on CPU in the tested computations: the largest compared fp32 TinyStories logit difference is 0.100 at logit magnitudes up to 853, and 3.2e-10 in a composed fp64 diagnostic. The converted Sudoku checkpoints solve 99.40/96.30/86.20% of 2,000 boards per difficulty. This concerns upstream-trained weights, not their training. Second, we train small TinyStories models (an 11.8M-value generator on a frozen 15.5M-value codec) and ablate conditioning corruption with two generator seeds that share one codec. The codec reconstructs held-out windows accurately (99.816% of non-padding tokens). Clean teacher-forced code cross-entropy and free-running quality rank the three corruption schedules in opposite order in both seeds. Under a fixed external scorer (GPT-2 large), generative perplexity for strong, reduced and no corruption is 47.897/54.923/97.776 for generator seed 0 and 45.568/55.114/93.833 for seed 1. An MLX autoregressive baseline with the same block stack scores 14.676 and 14.311, the released TinyStories checkpoint 16.259 and held-out text 8.529. Total training compute is not matched, the corpus is synthetic, generation is unconditional and no conditional task is tested. We characterize the work as a partial implementation, a small-model Sudoku success and a documented failure of free-running text quality at pilot scale.

Version 0.13 adds a scaling study in the same TinyStories setup, with one frozen codec and one external scorer. With the generator raised from 11.8M to 49.6M and 87.4M parameters at the original 960,000 windows, the ratio of NCM to AR generative perplexity falls from 3.33 to 2.43 and 2.38; with four times the data (and bf16 training) it is 2.21 and 2.13 for the two larger sizes and 3.05 and 2.89 for two seeds of the 11.8M generator. Most of the change comes from the first size step and from the extra data; the step from 49.6M to 87.4M is smaller than the difference between two training seeds of the 11.8M generator. The best NCM scores 24.404 against 11.484 for its AR baseline. Optional bf16 mixed-precision training matches the fp32 quality of the 11.8M recipe closely (clean code cross-entropy 2.24705 against 2.24708–2.24909; generative perplexity 49.091 against 45.568–48.964 for three fp32 runs, 0.13 above the highest). Several follow-ups give no gain: a matched-constant corruption control (47.052), stronger corruption (55.329 and 53.297 for two seeds), equal level loss (47.543 and 45.740) and a sampling-policy study that attributes roughly 5–10% of the gap (in log terms) to sampling. A small FineWeb-Edu run is reported only as a pipeline check on web text, not as a result. The scaling rows are single-seed except one, mix fp32 and bf16, compare unequal compute and use synthetic prose, so they describe a trend under stated conditions and are not a scaling law. An inference-only port of the released XSum generator matches upstream PyTorch in CPU parity checks with greedy decoding and scores ROUGE-1/2/L of 34.93/12.71/27.94 on the 11,334-article test set, against 35.0/12.8/28.0 quoted from Kumbong et al.

[Read the working paper (PDF)](https://janmus1c.github.io/ncm-mlx/paper/ncm-mlx.pdf){ .primary-link }

## Download an edition

| Edition | Intended use |
|---|---|
| [PDF](../paper/dist/ncm-mlx.pdf) | Read or print the typeset paper |
| [HTML](../paper/dist/ncm-mlx.html) | Download a self-contained browser edition with mathematical notation |
| [EPUB](../paper/dist/ncm-mlx.epub) | Read on an e-reader |
| [Word](../paper/dist/ncm-mlx.docx) | Review or annotate in Word |
| [LaTeX](../paper/dist/ncm-mlx.tex) | Starting point for a submission, rather than a publisher-ready template |
| [JATS XML](../paper/dist/ncm-mlx.jats.xml) | Structured scholarly archiving |

Version 0.13 adds a TinyStories scaling study: with one frozen codec, the ratio of NCM to AR generative perplexity falls from 3.33 (11.8M generator) to 2.38 (87.4M) at the original data and to 2.13 with four times the data, a trend and not a scaling law (single seeds, mixed precision, unmatched compute, synthetic text). It also reports bf16 parity at 11.8M, follow-ups that gave no gain, and the released XSum generator in the port. Version 0.12 added inference parity for the released checkpoints and a [TinyStories corruption ablation](research/e21-e23-assessment.md) with two generator seeds: less conditioning corruption lowers clean teacher-forced loss and worsens free-running text, and the best run scores 45.568 GPT-2-large generative perplexity against 14.311 for the AR baseline. Version 0.11 added completed E20 and a [two-seed assessment](research/e20-seed-report.md): preserved easy/medium performance, a 7.05-point hard-set spread, aggregate-only audit limits and a controlled-text research direction. It characterizes the Sudoku phase as a partial success, without reference parity or a demonstrated text capability. The [paper release](https://github.com/janmus1c/ncm-mlx/releases/tag/paper-v0.11) preserves all six editions. The [earlier v0.10 DOI archive](https://doi.org/10.5281/zenodo.23640777) remains unchanged and does not identify this revision. This is an unreviewed working paper. The [v0.11 DOI archive](https://doi.org/10.5281/zenodo.23686842) preserves the checked PDF.

The [Markdown manuscript](../paper/manuscript.md), [references](../paper/references.bib) and [figure data](../paper/data/results.json) are committed with the editions. The [build guide](../paper/README.md) explains how to rebuild them without private run logs. The manuscript's version history records substantive changes; it is separate from the software release version.

For the method, cite [Kumbong et al., *Next-Concept Modeling*](https://arxiv.org/abs/2610.14162). For this implementation, use the repository's [citation metadata](../CITATION.cff). The [benchmark guide](benchmarks.md) separates local results, upstream reference numbers, matched exposure and total system cost.

## Cite this working paper

```bibtex
@misc{music2026ncmmlx,
  author = {Jan Music},
  title = {Coarse-to-Fine Text Generation on a Laptop: A Native MLX Study of Next-Concept Modeling},
  year = {2026},
  note = {Unreviewed working paper, version 0.13},
  doi = {10.5281/zenodo.23953213},
  url = {https://doi.org/10.5281/zenodo.23953213}
}
```
