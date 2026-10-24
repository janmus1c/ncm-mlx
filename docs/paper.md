---
title: Working paper and downloadable editions
description: Read the NCM-MLX working paper, its experimental evidence, version history and reproducible source in six formats.
---

# Working paper

**Coarse-to-Fine Text Generation on a Laptop: A Native MLX Study of Next-Concept Modeling**<br>
Jan Music · Working paper v0.8 · 2026-10-24

## Abstract

Next-concept modeling (NCM) writes a sequence the way a painter works: a rough sketch first, then finer and finer detail, with every piece of a level produced at once. We independently implement NCM from its paper and released configurations in MLX, Apple's array framework, and study it on an Apple M5 Pro laptop and single NVIDIA GPUs. On a synthetic 15,360-sentence language, a 5M-parameter NCM generator is faster than a character-level autoregressive (AR) model of similar generator size at small batches but less accurate (89.8% vs 98.6% valid sentences). On the published Sudoku boards, NCM with conditioning corruption and unique-solution puzzles solves 96.25% of easy, 76.65% of medium and 31.90% of hard boards after 8.192M generator presentations. The original reports 99.5%, 96.5% and 86.1% with a roughly 4× larger generator and 51.2M presentations; this is an unmatched comparison. A separate 2.048M-presentation comparison favors NCM over the AR baseline, but excludes NCM's additional codec training cost. We trace the effects of the level schedule, learned first level, conditioning corruption and training exposure. We also measure reduced precision in fp32 matrix products on Apple GPUs and a 1.35× training improvement from segmented block-diagonal attention.

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

Version 0.8 adds parameter accounting, training-budget qualifications and a simplified visual identity. This is an unreviewed working paper.

The [Markdown manuscript](../paper/manuscript.md), [references](../paper/references.bib) and [figure data](../paper/data/results.json) are committed with the editions. The [build guide](../paper/README.md) explains how to rebuild them without private run logs. The manuscript's version history records substantive changes; it is separate from the software release version.

For the method, cite [Kumbong et al., *Next-Concept Modeling*](https://arxiv.org/abs/2610.14162). For this implementation, use the repository's [citation metadata](../CITATION.cff). The [benchmark guide](benchmarks.md) separates local results, upstream reference numbers, matched exposure and total system cost.

## Cite this working paper

```bibtex
@misc{music2026ncmmlx,
  author = {Jan Music},
  title = {Coarse-to-Fine Text Generation on a Laptop: A Native MLX Study of Next-Concept Modeling},
  year = {2026},
  note = {Working paper, version 0.8},
  url = {https://janmus1c.github.io/ncm-mlx/paper/}
}
```
