---
title: Releases and citation
description: Versioned NCM-MLX research software releases, supported workflows, limitations and citation guidance.
---

# Releases and citation

Follow new software and paper versions through the [research updates feed](feed.xml) or [GitHub releases feed](https://github.com/janmus1c/ncm-mlx/releases.atom).

## Working paper v0.10 — completed larger-corpus results

The [archived paper v0.10](https://doi.org/10.5281/zenodo.23640777) reports E18 and both completed E19 runs. The MLX implementation solves **98.10 / 83.40 / 40.30%** on easy/medium/hard boards, n = 2,000 each. Easy accuracy is within 1.40 points of **Original NCM (Kumbong et al.)**, with a 6.44M versus 25.46M generator. The medium/hard gaps, additional codec and unmatched training recipes remain explicit; the result is not overall parity or an equal-budget efficiency claim.

The [paper release](https://github.com/janmus1c/ncm-mlx/releases/tag/paper-v0.10) preserves all six editions, source data and figures at a new immutable tag. The [completed report](research/e19-funded-report.md) gives checkpoint provenance, timing boundaries, uncertainty and setup failures. [Roadmap scenarios](research/planning-projections.md) are separate planning assumptions, not future quality claims. E20 is an ongoing seed replication with no completed result yet. Earlier paper versions and software v0.1.0 remain unchanged. The paper is a preprint, without peer review or arXiv acceptance.

## Version 0.1.0 — initial research release

This release provides an independent MLX implementation of Next-Concept Modeling for synthetic character-level text and puzzle-conditioned Sudoku. It includes a multiscale residual VQ codec, next-concept transformer, from-scratch training, coarse-to-fine sampling, autoregressive baseline, evaluation and synchronized benchmarks.

- **Platforms:** verified Apple Silicon Metal and Linux NVIDIA CUDA execution; portable tests run without MLX. See [platform setup](platforms.md).
- **Recovery:** atomic full-state codec/generator snapshots, graceful pause and resume, preceding-snapshot fallback and tests of exact short CPU continuation.
- **Results:** TinyWorld quality/latency comparisons; completed Sudoku experiments through E17. At that release, E15 was best at 96.2 / 76.6 / 31.9% solved. E17's 71.25 / 31.75 / 3.70% is qualified as weights-only recovery. E18 was ongoing when v0.1.0 was released; its later results are in the [experiment log](EXPERIMENTS.md).
- **Limits:** experimental software; no general text benchmark, chat behavior, upstream checkpoint compatibility or cross-device exact-resume guarantee. Inference samplers are uncompiled. Some comparisons use different training budgets or contended hardware.
- **Distribution:** source and working-paper editions; no model checkpoints or downloaded datasets. Install from the tagged source using the [platform guide](platforms.md), then follow [reproduction instructions](reproducibility.md).

The release tag fixes the software revision. [GitHub releases](https://github.com/janmus1c/ncm-mlx/releases) contain the immutable tag target and release notes. Later research results require a new version; an existing released tag must not move.

## Attribution and archival readiness

The underlying method is [Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation](https://arxiv.org/abs/2610.14162), with [reference code](https://github.com/HazyResearch/ncm). NCM-MLX is an independent implementation, not an upstream release. Software metadata is in [CITATION.cff](https://github.com/janmus1c/ncm-mlx/blob/main/CITATION.cff); the source license is MIT.

Software version 0.1.0 is archived on Zenodo with DOI [10.5281/zenodo.23620792](https://doi.org/10.5281/zenodo.23620792). Its source ZIP matches the released commit `f2516a90e3d3ffac493ded861bfb7350cf611031`; the anonymous archive download was checked byte for byte. This is a manually reviewed deposit; automatic GitHub-to-Zenodo integration is not enabled.

The earlier [working paper v0.9](https://doi.org/10.5281/zenodo.23620914) has its own preprint DOI and [paper release](https://github.com/janmus1c/ncm-mlx/releases/tag/paper-v0.9). It corrects reproduction instructions and experimental settings and adds an AI-assistance disclosure. The paper is not peer reviewed, and this DOI is not an arXiv identifier. The original software tag and its v0.8 paper assets remain unchanged.

No Hugging Face checkpoint upload is included: model cards and dataset provenance need review before separate artifact releases. PyPI publication is deferred while the repository installation and explicit CUDA setup remain the supported developer workflow.

## Maintaining useful documentation

For a new public capability or research milestone, update its authoritative documentation, capabilities and platform support where behavior changed, benchmark summary where measurements changed, and release notes. Preserve the experiment log, exact configuration, sampling settings, sample counts and uncertainty. Rebuild paper editions when their source changes. Change the README only when its overview or headline evidence changes. Keep topics and project identity stable unless the project's actual scope changes.

Documentation maintenance and discovery checks are described in [discoverability](discoverability.md).
