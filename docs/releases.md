---
title: Releases and citation
description: Versioned NCM-MLX research software releases, supported workflows, limitations and citation guidance.
---

# Releases and citation

## Version 0.1.0 — initial research release

This release provides an independent MLX implementation of Next-Concept Modeling for synthetic character-level text and puzzle-conditioned Sudoku. It includes a multiscale residual VQ codec, next-concept transformer, from-scratch training, coarse-to-fine sampling, autoregressive baseline, evaluation and synchronized benchmarks.

- **Platforms:** verified Apple Silicon Metal and Linux NVIDIA CUDA execution; portable tests run without MLX. See [platform setup](platforms.md).
- **Recovery:** atomic full-state codec/generator snapshots, graceful pause and resume, preceding-snapshot fallback and tests of exact short CPU continuation.
- **Results:** TinyWorld quality/latency comparisons; completed Sudoku experiments through E17. E15 remains best at 96.2 / 76.6 / 31.9% solved. E17's 71.25 / 31.75 / 3.70% is qualified as weights-only recovery. E18 is ongoing.
- **Limits:** experimental software; no general text benchmark, chat behavior, upstream checkpoint compatibility or cross-device exact-resume guarantee. Inference samplers are uncompiled. Some comparisons use different training budgets or contended hardware.
- **Distribution:** source and working-paper editions; no model checkpoints or downloaded datasets. Install from the tagged source using the [platform guide](platforms.md), then follow [reproduction instructions](reproducibility.md).

The release tag fixes the software revision. [GitHub releases](https://github.com/janmus1c/ncm-mlx/releases) contain the immutable tag target and release notes. Later research results require a new version; an existing released tag must not move.

## Attribution and archival readiness

The underlying method is [Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation](https://arxiv.org/abs/2610.14162), with [reference code](https://github.com/HazyResearch/ncm). NCM-MLX is an independent implementation, not an upstream release. Software metadata is in [CITATION.cff](https://github.com/janmus1c/ncm-mlx/blob/main/CITATION.cff); the source license is MIT.

A Zenodo DOI would provide an archived software citation. It requires an authorized Zenodo account and repository integration; no DOI is claimed here. [GitHub documents that integration](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content). No Hugging Face checkpoint upload is included: model cards and dataset provenance need review before separate artifact releases. PyPI publication is deferred while the repository installation and explicit CUDA setup remain the supported developer workflow.

## Maintaining useful documentation

For a new public capability or research milestone, update its authoritative documentation, capabilities and platform support where behavior changed, benchmark summary where measurements changed, and release notes. Preserve the experiment log, exact configuration, sampling settings, sample counts and uncertainty. Rebuild paper editions when their source changes. Change the README only when its overview or headline evidence changes. Keep topics and project identity stable unless the project's actual scope changes.

Documentation maintenance and discovery checks are described in [discoverability](discoverability.md).
