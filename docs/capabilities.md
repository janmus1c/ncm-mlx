---
title: NCM-MLX capabilities
description: Supported training, generation, evaluation and accelerator workflows, with explicit research limitations.
status: current
updated: 2026-10-24
---

# NCM-MLX capabilities

NCM-MLX independently implements Next-Concept Modeling in MLX for small, reproducible experiments in non-autoregressive generation. A multiscale residual vector-quantized codec learns discrete latent levels; a transformer predicts each level's positions in parallel. It is research software for TinyWorld synthetic text and puzzle-conditioned Sudoku. The architecture originates with [Kumbong et al.](https://arxiv.org/abs/2610.14162), not this implementation.

## Supported workflows

| Capability | Status and evidence |
|---|---|
| Training from scratch | Codec then generator; `ncm_mlx/training/`. The TinyWorld quickstart includes both stages. Some Sudoku experiments reuse a separately trained codec. |
| Coarse-to-fine generation | One generator forward per level, followed by codec decoding; stochastic temperature/top-k or greedy decoding. Inference/training-forward equivalence is tested. |
| Synthetic text | Fixed-length character-level TinyWorld grammar; reconstruction, grammatical validity, agreement and training-set novelty metrics. |
| Conditional Sudoku | Fixed 81-cell puzzle-token prefix; checks row/column/box validity and preservation of givens. Synthetic training data and the upstream test-board loader are separate. |
| Autoregressive comparison | Character/cell AR model with KV cache, conditioning support and the same evaluator. Only explicitly matched experiments establish equal training budgets. |
| Benchmarks | Synchronized host-copy timing, warmups, p50/p95 latency and batched throughput. See [methodology](benchmarks.md). |
| Apple Silicon | Metal execution, measured on M5 Pro; exactness tests use MLX's CPU device. |
| NVIDIA CUDA | Linux MLX 0.32.3 CUDA execution, measured on RTX 3080 and A40. See [platform setup](platforms.md); other configurations require verification. |
| Training compilation | Generator training optionally uses `mx.compile` through `train.compile`. NCM and AR samplers are not compiled. |
| Pause/resume | Atomic full-state codec/generator checkpoints, previous-snapshot fallback and graceful interruption. Short CPU tests verify resumed versus uninterrupted training. Cross-backend bitwise identity is not established. See [recovery](RECOVERY.md). |

## Unsupported or unverified

- General instruction following, open-domain chat and production inference serving.
- Real-text language-model quality, translation and summarization. These are upstream tasks, not validated workflows here.
- Importing released upstream PyTorch checkpoints or using their BPE tokenizers.
- Classifier-free guidance, top-p sampling, codebook-tied prediction heads and NCM prefix KV caching.
- Quantized inference, distributed training and AMD accelerator execution.
- General claims of superior quality, energy use or wall-clock speed across hardware or model scales.

TinyWorld and Sudoku use different codec and generator settings. Later Sudoku configurations include learned level-0 resampling, GroupNorm and conditioning corruption; the TinyWorld defaults do not. See the [upstream map](research/upstream-map.md) for configuration-specific differences.

## Reproduce a supported workflow

From a clean checkout on Apple Silicon with `mise` and `uv` installed:

```bash
mise trust
mise install
mise run setup
mise run smoke
mise run test
mise run train
mise run demo
mise run eval
mise run bench
python -m ncm_mlx baseline --bench
```

Linux NVIDIA setup, including runtime CUDA headers, is in [platforms](platforms.md). Portable tests require no accelerator: `uv sync --locked && uv run pytest -q`; accelerator tests skip when MLX is absent. The CLI itself requires Metal or CUDA.

Do not use the quickstart duration as a Sudoku training estimate. Reproduction of a Sudoku result requires its recorded config, codec, data construction, training budget and test boards. Published aggregate evidence is in [experiments](EXPERIMENTS.md) and `paper/data/results.json`; local weights and full run logs are ignored rather than bundled with source releases.

## Which document is authoritative?

- [STATUS](STATUS.md): current measured progress and open limitations.
- [EXPERIMENTS](EXPERIMENTS.md): individual results, budgets and interpretation, including negative outcomes.
- [Benchmarks](benchmarks.md): how to interpret quality and timing comparisons.
- [RECOVERY](RECOVERY.md): full-state resume and the qualified E17 weights-only recovery.
- [AGENTS](../AGENTS.md): commands, source map and invariants for repository changes.

Update this page when a supported workflow or platform changes. New experiment scores belong first in the experiment log; they do not require rewriting the capability list.
