---
title: Working paper and downloadable editions
description: Read the NCM-MLX working paper, its experimental evidence, version history and reproducible source in six formats.
---

# Working paper

**Coarse-to-Fine Text Generation on a Laptop: A Native MLX Study of Next-Concept Modeling**<br>
Jan Music ([ORCID](https://orcid.org/0009-0001-5376-7913)) · Working paper v0.11 · 2026-10-27 · [DOI: 10.5281/zenodo.23686842](https://doi.org/10.5281/zenodo.23686842)

## Abstract

Next-Concept Modeling (NCM) generates a sequence through successive levels of discrete latent codes, predicting each level in parallel. We independently implement the method in MLX and study synthetic language and Sudoku on Apple Silicon and NVIDIA GPUs. On a finite synthetic grammar, a 5M-parameter generator achieves lower small-batch latency than a similarly sized autoregressive generator, with lower sentence validity (89.8% versus 98.6%). Two Sudoku training seeds use the same 6.44M-parameter generator recipe and 16.384M presentations each. Retained greedy evaluation aggregates report easy/medium/hard solve rates of 98.10/83.40/40.30% for seed 0 and 97.25/79.60/33.25% for seed 1, with 2,000 boards per difficulty. The hard-set difference is 7.05 percentage points; two seeds do not establish population training variability or its cause. Original NCM reports 99.50/96.50/86.10% with a 25.46M generator, 51.2M presentations and a different task setup. Our generator is 3.95 times smaller, excluding its additional codec; the comparison establishes neither task parity nor matched-cost efficiency. E20 completes its schedule without a recorded reset or truncation, but missing per-board predictions and remote provenance manifests limit independent audit. We characterize this phase as a partial implementation and small-model success, retain the substantial hard-Sudoku deficit, and define controlled text evaluation as the next research direction rather than a demonstrated capability.

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

Version 0.11 adds completed E20 and a [two-seed assessment](research/e20-seed-report.md): preserved easy/medium performance, a 7.05-point hard-set spread, aggregate-only audit limits and a controlled-text research direction. It characterizes the Sudoku phase as a partial success, without reference parity or a demonstrated text capability. The [paper release](https://github.com/janmus1c/ncm-mlx/releases/tag/paper-v0.11) preserves all six editions. The [earlier v0.10 DOI archive](https://doi.org/10.5281/zenodo.23640777) remains unchanged and does not identify this revision. This is an unreviewed working paper. The [v0.11 DOI archive](https://doi.org/10.5281/zenodo.23686842) preserves the checked PDF.

The [Markdown manuscript](../paper/manuscript.md), [references](../paper/references.bib) and [figure data](../paper/data/results.json) are committed with the editions. The [build guide](../paper/README.md) explains how to rebuild them without private run logs. The manuscript's version history records substantive changes; it is separate from the software release version.

For the method, cite [Kumbong et al., *Next-Concept Modeling*](https://arxiv.org/abs/2610.14162). For this implementation, use the repository's [citation metadata](../CITATION.cff). The [benchmark guide](benchmarks.md) separates local results, upstream reference numbers, matched exposure and total system cost.

## Cite this working paper

```bibtex
@misc{music2026ncmmlx,
  author = {Jan Music},
  title = {Coarse-to-Fine Text Generation on a Laptop: A Native MLX Study of Next-Concept Modeling},
  year = {2026},
  note = {Unreviewed working paper, version 0.11},
  doi = {10.5281/zenodo.23686842},
  url = {https://doi.org/10.5281/zenodo.23686842}
}
```
