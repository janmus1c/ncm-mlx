---
title: Releases and citation
description: Versioned NCM-MLX research software releases, supported workflows, limitations and citation guidance.
---

# Releases and citation

Follow new software and paper versions through the [research updates feed](feed.xml) or [GitHub releases feed](https://github.com/janmus1c/ncm-mlx/releases.atom).

## Version 0.1.1 — packaging and citation metadata

Software 0.1.1 is the first release published on [PyPI](https://pypi.org/project/ncm-mlx/) (`pip install ncm-mlx` on Apple Silicon; `pip install "ncm-mlx[cuda]"` on Linux NVIDIA). It adds PyPI metadata and trusted publishing, the author ORCID in citation metadata and a refreshed README. Model code, training behavior and recorded results are unchanged from 0.1.0. The project's own trained Sudoku and TinyWorld weights are published separately on Hugging Face ([sudoku](https://huggingface.co/janmusic/ncm-mlx-sudoku), [tinyworld](https://huggingface.co/janmusic/ncm-mlx-tinyworld)); they are not included in the package. Software 0.1.1 is archived on Zenodo with DOI [10.5281/zenodo.23780571](https://doi.org/10.5281/zenodo.23780571); the concept DOI [10.5281/zenodo.23620791](https://doi.org/10.5281/zenodo.23620791) always resolves to the latest software version.

## Working paper v0.11 — two-seed assessment and next research phase

[Archived working paper v0.11](https://doi.org/10.5281/zenodo.23686842) adds E20’s completed seed-1 run: **97.25 / 79.60 / 33.25%**, n = 2,000 per difficulty. E19-funded’s seed-0 result remains **98.10 / 83.40 / 40.30%**, at the same 16.384M presentations and 6.44M generator size. Both seeds finish their full schedules without a recorded reset or truncation.

The [assessment](research/e20-seed-report.md) treats this as a partial small-model success, retains the hard-board deficit and reports missing per-board predictions, remote provenance manifests and benchmark contention. It distinguishes measurements from hypotheses and defines tests needed to challenge the interpretation. Controlled text generation is the next research direction, with explicit data, quality and cost gates. No new training or model evaluation is included. The revision is an unreviewed working paper; it does not imply arXiv acceptance or peer review. The [versioned GitHub release](https://github.com/janmus1c/ncm-mlx/releases/tag/paper-v0.11) preserves all six editions; the public PDF download matches the checked artifact byte for byte. Earlier releases and their DOI records remain immutable.

## Working paper v0.10 — completed larger-corpus results

The [archived paper v0.10](https://doi.org/10.5281/zenodo.23640777) reports E18 and both completed E19 runs. The MLX implementation solves **98.10 / 83.40 / 40.30%** on easy/medium/hard boards, n = 2,000 each. Easy accuracy is within 1.40 points of **Original NCM (Kumbong et al.)**, with a 6.44M versus 25.46M generator. The medium/hard gaps, additional codec and unmatched training recipes remain explicit; the result is not overall parity or an equal-budget efficiency claim.

The [paper release](https://github.com/janmus1c/ncm-mlx/releases/tag/paper-v0.10) preserves all six editions, source data and figures at a new immutable tag. The [completed report](research/e19-funded-report.md) gives checkpoint provenance, timing boundaries, uncertainty and setup failures. [Roadmap scenarios](research/planning-projections.md) are separate planning assumptions, not future quality claims. E20 was pending at v0.10; its completed result and qualifications are reported in v0.11 above. Earlier paper versions and software v0.1.0 remain unchanged. The paper is a preprint, without peer review or arXiv acceptance.

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

No Hugging Face checkpoint upload is included: model cards and dataset provenance need review before separate artifact releases. PyPI publication is described in [Installing from PyPI](#installing-from-pypi); the repository installation remains the reproducible research workflow.

## Installing from PyPI

Available after the first PyPI publish; before then `pip install ncm-mlx` finds nothing. Once published, `pip install ncm-mlx` installs on Apple Silicon, where MLX 0.32.3 is a dependency. On Linux NVIDIA use `pip install "ncm-mlx[cuda]"` and set up CUDA headers as in [platform setup](platforms.md). Without MLX the package imports and the platform-independent configuration and data code run, but the CLI needs an accelerator. The wheel contains only the `ncm_mlx` package, with no configs, checkpoints or datasets; run it with `python -m ncm_mlx`. Use a repository checkout to reproduce experiments, because the recipes live in `configs/`.

Software releases are published by the `publish-pypi` workflow when a GitHub release with a `v*` tag matching the package version is published, using PyPI Trusted Publishing from the `pypi` environment. `paper-v*` releases are not uploaded. A PyPI version cannot be replaced; a correction needs a new version.

## Maintaining useful documentation

For a new public capability or research milestone, update its authoritative documentation, capabilities and platform support where behavior changed, benchmark summary where measurements changed, and release notes. Preserve the experiment log, exact configuration, sampling settings, sample counts and uncertainty. Rebuild paper editions when their source changes. Change the README only when its overview or headline evidence changes. Keep topics and project identity stable unless the project's actual scope changes.

Documentation maintenance and discovery checks are described in [discoverability](discoverability.md).
