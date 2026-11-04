---
title: NCM-MLX capabilities
description: Supported training, generation, evaluation and accelerator workflows, with explicit research limitations.
status: current
updated: 2026-11-02
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
| Released checkpoints | Inference port of upstream TinyStories, Sudoku and XSum (plus the WMT14 codec only, no generator): pinned download, torch-free conversion, BPE and GPT-2 tokenizers, per-level generation, Sudoku scoring and XSum summarization with ROUGE. CPU parity with upstream PyTorch is tested against a committed fixture; exact float64 agreement is checked by `scripts/official_parity.py`. See [official checkpoints](official-checkpoints.md). |
| Pause/resume | Atomic full-state codec/generator checkpoints, previous-snapshot fallback and graceful interruption. Short CPU tests verify resumed versus uninterrupted training. Cross-backend bitwise identity is not established. See [recovery](RECOVERY.md). |

## Unsupported or unverified

- General instruction following, open-domain chat and production inference serving.
- Real-text language-model quality and translation, and summarization with models trained here. The released XSum generator runs through the [official port](official-checkpoints.md) with ROUGE close to the paper's, but its summaries often contain wrong facts and broken words.
- Released upstream checkpoints other than TinyStories, Sudoku, XSum and the WMT14 codec; fine-tuning or training any released checkpoint.
- Prompt-conditioned text quality. Prefix-conditioned TinyStories training exists (see [text training](text-training.md#prompt-continuation-prefix-conditioning)), but no such model has been trained or evaluated.
- Classifier-free guidance and NCM prefix KV caching in the independent training implementation's sampler. These, top-p sampling and codebook-tied heads exist only in the official-checkpoint port (guidance for XSum).
- Quantized inference, distributed training and AMD accelerator execution.
- General claims of superior quality, energy use or wall-clock speed across hardware or model scales.

TinyWorld and Sudoku use different codec and generator settings. Later Sudoku configurations include learned level-0 resampling, GroupNorm and conditioning corruption; the TinyWorld defaults do not. See the [upstream map](research/upstream-map.md) for configuration-specific differences.

## Scope

**Experimental research code.** Small models trained on a synthetic language, Sudoku and synthetic children's stories, using Apple Silicon and NVIDIA GPUs. It is not a chat model, follows no instructions, and its from-scratch models use their own format; the released NCM checkpoints run through the separate [official port](official-checkpoints.md). It is not affiliated with Stanford or HazyResearch. Results are labelled with hardware, training budget and measurement conditions.

An independent [MLX](https://github.com/ml-explore/mlx) implementation of [Next-Concept Modeling (NCM)](https://arxiv.org/abs/2610.14162): **parallel, coarse-to-fine text generation** with multiscale residual vector quantization. The non-autoregressive generator predicts a whole level of discrete latent codes per forward pass, rather than one text token at a time. This repository supports training from scratch, sampling, evaluation and synchronized comparisons with a character-level autoregressive baseline.

MLX makes these experiments runnable on consumer Apple Silicon GPUs; Linux NVIDIA CUDA execution is also verified. The current tasks are TinyWorld synthetic text, puzzle-conditioned Sudoku and, experimentally, unconditional TinyStories text. General-purpose language modeling and instruction following are not demonstrated. The architecture belongs to Kumbong et al. (HazyResearch, 2026); this project independently investigates its implementation and small-model behavior.

## Limitations

- Models trained here use synthetic data (TinyWorld, Sudoku, TinyStories), apart from a small FineWeb-Edu pilot that is a pipeline check; quality on naturally occurring text is unknown.
- Conditioning is a fixed-length token prefix: Sudoku boards, and from 0.3.0 the first tokens of a TinyStories story ([prompt continuation](text-training.md#prompt-continuation-prefix-conditioning)). No prompt-conditioned model has been trained yet, and nothing here summarises, translates or follows instructions.
- Sudoku: 33.25–40.30% of hard boards solved across E20 and E19-funded, against 86.1% for Original NCM (Kumbong et al.), with different model size and training budget.
- Weights trained here use their own format. The released TinyStories and Sudoku checkpoints run through the inference-only [official port](official-checkpoints.md). XSum summarization runs through the same port; the WMT14 codec converts and runs without its generator; LM1B and OpenWebText are not converted yet.
- Generator training can use `mx.compile`; the independent implementation's inference sampler is not compiled, so its latency gap to AR may change once it is. The official-checkpoint port has an opt-in compiled step.
- On Metal, fp32 matmuls run at reduced precision (~1e-3 relative error, measured); exactness tests therefore run on the CPU device.
- No energy or power measurements.

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

## For agents

```yaml
repo: ncm-mlx
read_first: [AGENTS.md, docs/STATUS.md, docs/NEXT.md]
platform: "MLX Metal on Apple Silicon; MLX CUDA on verified Linux NVIDIA systems"
setup: "mise trust && mise install && mise run setup"
verify: "mise run smoke && mise run test"
train: "mise run train"            # writes artifacts/toy-grammar-12k/
entrypoint: "python -m ncm_mlx {smoke,train,demo,eval,bench,baseline}"
default_config: configs/toy-grammar-12k.json
invariants:
  - per-level inference equals the teacher-forced forward (exact on mx.cpu)
  - one generator forward per level
  - benchmarks time through a host copy; never enqueue-only
  - codec accuracy and generator quality are reported separately
record_results_in: docs/EXPERIMENTS.md
```

| Path | Contents |
|---|---|
| `ncm_mlx/latent/` | `resample.py` level resampling matrices · `quantizer.py` multiscale residual VQ · `codec.py` encoder/decoder |
| `ncm_mlx/models/` | `blocks.py` transformer block · `generator.py` next-concept generator |
| `ncm_mlx/generation/sampler.py` | level-by-level sampling |
| `ncm_mlx/training/` | two-stage training, checkpoints |
| `ncm_mlx/eval/` | metrics, benchmark |
| `ncm_mlx/baselines/ar.py` | autoregressive baseline |
| `ncm_mlx/text/` | tokenizer, TinyWorld grammar and checker |
| `docs/` | status, decisions, experiments, ADRs, research |

[AGENTS](https://github.com/janmus1c/ncm-mlx/blob/main/AGENTS.md) and [llms.txt](https://github.com/janmus1c/ncm-mlx/blob/main/llms.txt) provide concise entry points for automated tools.

## Which document is authoritative?

- [STATUS](STATUS.md): current measured progress and open limitations.
- [EXPERIMENTS](EXPERIMENTS.md): individual results, budgets and interpretation, including negative outcomes.
- [Benchmarks](benchmarks.md): how to interpret quality and timing comparisons.
- [RECOVERY](RECOVERY.md): full-state resume and the qualified E17 weights-only recovery.
- [AGENTS](../AGENTS.md): commands, source map and invariants for repository changes.

Update this page when a supported workflow or platform changes. New experiment scores belong first in the experiment log; they do not require rewriting the capability list.
