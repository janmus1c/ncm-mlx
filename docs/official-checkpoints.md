---
title: Released NCM checkpoints in MLX
description: Convert and run the official HazyResearch TinyStories and Sudoku checkpoints natively in MLX, with parity evidence against upstream PyTorch.
status: current
updated: 2026-10-29
---

# Released NCM checkpoints in MLX

`ncm_mlx/official/` runs the released checkpoints of the original Next-Concept Modeling work in MLX on Apple Silicon. It covers TinyStories (unconditional BPE text) and the three Sudoku checkpoints (easy, medium, hard). It is an inference-only reimplementation, written from the upstream architecture and released configurations. It is separate from this repository's own training models and contains no upstream source code.

**Credit and license.** The checkpoints, configurations and reference PyTorch implementation are by HazyResearch: [checkpoints](https://huggingface.co/hazyresearch/ncm-checkpoints), [code](https://github.com/HazyResearch/ncm) and the paper [Kumbong et al., arXiv:2610.14162](https://arxiv.org/abs/2610.14162). They are MIT licensed, Copyright (c) 2025 Hermann Kumbong. Weights are downloaded from Hugging Face on demand and never redistributed here.

## Commands

```bash
python -m ncm_mlx official fetch --task tinystories       # pinned revision, SHA-256 checked, into artifacts/official/
python -m ncm_mlx official convert --task tinystories     # .pt -> MLX safetensors + spec.json + bpe.json
python -m ncm_mlx official generate --task tinystories --n 2 --levels   # per-level refinement, Metal GPU

python -m ncm_mlx official fetch --task sudoku-easy
python -m ncm_mlx official convert --task sudoku-easy
bash scripts/fetch_upstream_sudoku.sh                     # published test boards
python -m ncm_mlx official sudoku --task sudoku-easy --n 200   # greedy; --n 0 for all 2,000
python -m ncm_mlx official bench --task tinystories --batch-sizes 1 16
```

Tasks are `tinystories`, `sudoku-easy`, `sudoku-medium` and `sudoku-hard`. Add `--cpu` to run on `mx.cpu`. `generate` takes `--temperature`, `--top-k`, `--top-p`, `--seed` and `--level0 CODE`; the defaults follow the released config (temperature 1.0 for TinyStories, greedy for Sudoku).

Conversion does not need PyTorch. `.pt` files are read by `torch_reader`, which accepts only the four globals a tensor checkpoint uses and refuses all others. The BPE pickles are plain dicts and lists of bytes, read by an unpickler that refuses every global, and are converted once to JSON. Each upstream tensor name must match exactly one explicit rule: it either becomes an MLX parameter with a stated layout change or is dropped. Dropped names are the EMA training state and two derived buffers. The level ids and block-diagonal mask are first checked against the values the MLX model assumes. An unmatched name, missing parameter or shape mismatch stops the conversion.

## What the port implements

Sudoku and TinyStories share one architecture:

- **Codec:** token plus position lookup, then GroupNorm(8) without affine parameters over channels and positions.
- **Level-0 resampling:** a cascade of learned strided convolutions each way.
- **Other levels:** area pooling down and linear interpolation up.
- **Phi:** a per-level blend, `(1-r)h + r Conv1d_k3(h)`.
- **Decoder:** one block.
- **Generator:** a prefix of raw-token embeddings (`raw_token_prefix`), a level embedding, RoPE that restarts per level, QK RMSNorm, LayerNorm eps 1e-6 and exact GELU. It runs one forward per level, in which the prefix attends only to itself.

The two task families differ in a few places:

- **Level 0:** TinyStories samples it from the stored unigram; Sudoku uses learned queries.
- **Heads:** TinyStories uses codebook-tied distance heads, `(2z·E - |E|²)/mean|E|² · exp(log_inv_T)`; Sudoku uses untied linear heads.
- **Sampling:** temperature, top-k and nucleus sampling follow upstream semantics; greedy is argmax.

## Parity with upstream PyTorch

`scripts/official_parity.py` runs upstream PyTorch 2.15 on CPU and the MLX port on `mx.cpu` in one process, with identical inputs. It routes teacher-forced codes through each implementation's own generation loop. Reports are in [`research/official-checkpoint-parity.json`](research/official-checkpoint-parity.json). Upstream code revision `c702742`, checkpoint revision `d342ce8`.

| Check | TinyStories | Sudoku-easy |
|---|---|---|
| BPE pre-tokenization, ids, round trip (13 cases) | identical | n/a |
| Codec embedding, max abs diff | 2.9e-6 | 1.4e-6 |
| Codec codes from real inputs | 97.4–100% per level (near-tie cascades, below) | 100% (64 boards) |
| Per-level logits, fp32, max abs diff | 0.10 at logit scale up to 853 (1.2e-4 relative) | **7.2e-5** at scale 29 |
| Same, both sides in exact fp64 | **3.2e-10** | **1.2e-13** |
| Greedy generation, codes and decoded text | identical, 8/8 samples | identical, 64/64 boards |
| Greedy, all 2,000 easy boards | | identical to upstream on 2,000/2,000, on Metal and CPU |

**Why the TinyStories fp32 difference exceeds 1e-4.** The fp64 row shows that the computation is the same function. In fp32, upstream PyTorch is itself 0.071 away from exact fp64, and MLX is 0.126 away. The model carries activations of magnitude about 435 in its residual stream, and the tied head subtracts large `|E|²` terms, so float32 rounding of about 3e-6 relative reaches about 1e-4 relative in the logits. Argmax agrees at every compared position.

**How the fp64 check works.** For that run only, numerics are made exact on both sides:

- Upstream's RMSNorm, tied-head norm buffers and RoPE tables, which it computes in float32 even for float64 inputs, are recomputed in float64.
- MLX's CPU `conv1d`, fused norms, fused attention, `exp` and `erf` round float64 inputs to about float32 accuracy, so the parity script swaps in composed exact operations. `exp` and `erf` are evaluated on the host.

The production float32 path is unchanged.

**Codec code disagreements are near-ties.** For each item, the first differing code is a near-tie: distance gaps of 4.5e-3 and 2.4e-4 at distances near 10. The changed residual then shifts later levels. `tests/test_official.py` checks this property: codes must be identical before the first divergence, and the two candidates must be within 1e-3 relative distance.

## Results

**Sudoku**, greedy, MLX on Metal, published test sets:

| Checkpoint | MLX, first 200 boards | MLX, all 2,000 | Upstream PyTorch CPU, all 2,000 (this machine) | Paper (Kumbong et al.) |
|---|---:|---:|---:|---:|
| easy | 99.0% (198/200) | **99.40%** (1,988) | 99.40% (1,988), the same boards | 99.5% |
| medium | | **96.30%** (1,926) | not run | 96.5% |
| hard | | **86.20%** (1,724) | not run | 86.1% |

The 200-board row is a sample. The 0.1-point gaps to the paper are not attributable to the port: on the easy set, upstream PyTorch on this machine produces exactly the same 2,000 outputs. The paper's figures come from its own CUDA evaluation.

**TinyStories.** `generate --levels` prints the decode of the cumulative reconstruction after each level, followed by the final text. Generative perplexity under GPT-2 large, the upstream metric, is not computed here.

## Benchmark

Apple M5 Pro, 64 GB, macOS 26.5, MLX 0.32.3, PyTorch 2.15.0 (6 CPU threads by default). Latency is synchronized through a host copy of the decoded tokens and includes all level forwards and the codec decode. MLX: 3 warmups and 20 iterations. PyTorch: 2 warmups and 10 iterations. Samplers are uncompiled and have no prefix KV cache. TinyStories samples at temperature 1.0 and Sudoku is greedy.

| p50 / p95 ms | TinyStories b=1 | TinyStories b=16 | Sudoku b=1 | Sudoku b=16 |
|---|---:|---:|---:|---:|
| MLX, Metal GPU | 39.8 / 40.8 | 519 / 742 | 20.5 / 22.7 | 229 / 277 |
| Upstream PyTorch, CPU | 280 / 290 | 3,883 / 4,076 | 165 / 202 | 1,663 / 1,746 |

MLX on Metal is 7–8× faster than upstream on this machine's CPU. Upstream does not run on MPS: PyTorch 2.15 rejects area interpolation with non-divisible sizes on MPS. The GPU was shared with desktop applications at 0–33% utilisation around these runs, with brief higher readings, so treat the numbers as busy-desktop timings. See [EXPERIMENTS](EXPERIMENTS.md). At batch 16 a Sudoku level forward is compute-bound (2,880 tokens through the 25M-parameter generator), mostly because the 91-token prefix is recomputed at every level. Prefix caching is a separate roadmap item.

## Limitations

- Inference only: no fine-tuning, no EMA codebook updates and no training loss.
- Not converted: LM1B, OpenWebText, WMT14 and XSum. WMT14 and XSum also need classifier-free guidance and a pretrained source tokenizer, which are not implemented. Upstream's experimental options outside the released configs, such as a context encoder, `predict_level0` and block-causal attention, are rejected by the pinned specs rather than supported.
- The pinned spec restates each `config.yaml`. The converter refuses a config whose SHA-256 differs, and a test compares the spec field by field with the downloaded YAML.
- The BPE pre-tokenizer reimplements upstream's `regex` pattern with `unicodedata`. It is verified on the parity cases, including Unicode, but rare whitespace classes such as `\x1c`–`\x1f` may differ.
- Metal fp32 matmul is reduced precision (see E-precision), so GPU outputs can differ from CPU at near-ties. On the 2,000 easy boards, Metal and CPU outputs were identical.
- Upstream's TinyStories generative-perplexity protocol is not reproduced, and sampled (non-greedy) outputs use MLX's RNG, so they do not match upstream sample for sample.

## Reproduce the parity check

```bash
# upstream checkout outside the repository, e.g. a download of HazyResearch/ncm at c702742
uv run --with torch --with omegaconf --with hydra-core --with regex \
    python scripts/official_parity.py --upstream /path/to/ncm --task tinystories
uv run --with torch --with omegaconf --with hydra-core --with regex \
    python scripts/official_torch_reference.py sudoku --upstream /path/to/ncm --task sudoku-easy --n 0
uv run pytest -q tests/test_official.py    # skips parity checks when converted weights are absent
```

PyTorch is used only in these ephemeral runs. It is not a dependency of `ncm_mlx`.
