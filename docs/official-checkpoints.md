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
python -m ncm_mlx official bench --task sudoku-easy      # eager vs cached vs cached+compiled, batch 1 4 16 64
```

Tasks are `tinystories`, `sudoku-easy`, `sudoku-medium` and `sudoku-hard`. Add `--cpu` to run on `mx.cpu`. `generate` and `sudoku` take `--no-prefix-cache` (reference path that recomputes the prefix in every level) and `--compile` (`mx.compile` of the per-level step). The prefix cache is on by default; compilation is opt-in. `generate` takes `--temperature`, `--top-k`, `--top-p`, `--seed` and `--level0 CODE`; the defaults follow the released config (temperature 1.0 for TinyStories, greedy for Sudoku).

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

Apple M5 Pro, 64 GB, macOS 26.5, MLX 0.32.3, PyTorch 2.15.0 (6 CPU threads by default). Latency is synchronized through a host copy of the decoded tokens and includes all level forwards and the codec decode. MLX: 3 warmups and 20 iterations. PyTorch: 2 warmups and 10 iterations. These baseline rows are the reference path (`--no-prefix-cache`, uncompiled), measured before the prefix cache existed. TinyStories samples at temperature 1.0 and Sudoku is greedy.

| p50 / p95 ms | TinyStories b=1 | TinyStories b=16 | Sudoku b=1 | Sudoku b=16 |
|---|---:|---:|---:|---:|
| MLX, Metal GPU | 39.8 / 40.8 | 519 / 742 | 20.5 / 22.7 | 229 / 277 |
| Upstream PyTorch, CPU | 280 / 290 | 3,883 / 4,076 | 165 / 202 | 1,663 / 1,746 |

MLX on Metal is 7–8× faster than upstream on this machine's CPU. Upstream does not run on MPS: PyTorch 2.15 rejects area interpolation with non-divisible sizes on MPS. The GPU was shared with desktop applications at 0–33% utilisation around these runs, with brief higher readings, so treat the numbers as busy-desktop timings. See [EXPERIMENTS](EXPERIMENTS.md). At batch 16 a Sudoku level forward is compute-bound (2,880 tokens through the 25M-parameter generator), mostly because the 91-token prefix is recomputed at every level. The next section measures the prefix cache and compilation against this baseline.

## Prefix K/V cache and compiled step

Under the released attention contract the prefix attends only to itself and its RoPE positions restart at 0, so its hidden states do not depend on the level. `Generator.build_prefix_cache` runs the prefix tokens through every block once per `generate` call and keeps each layer's rotated K and V. Each level forward (`level_logits_cached`) then computes only its own rows and attends over `[cached prefix K/V, level K/V]` with the prefix padding mask. The cache is a local object of one call: it is never stored on the model, so repeated calls, other prefixes and permuted batches cannot read stale state. Cache construction draws no random numbers. The last block's prefix attention output and feed-forward are skipped because no level reads them. `--no-prefix-cache` keeps the original path, which recomputes the prefix inside every level forward.

`--compile` wraps three pieces in `mx.compile`: the level forward plus head (one trace per level and batch size), the "add level contribution and pool for the next level" step, and the final decode with argmax. Sampling stays eager, so random keys are unchanged. Wrappers are kept on the loaded model and reused, so the first call at each batch size pays tracing cost (reported as `first_call_ms`); they close over the loaded weights and must be dropped if weights are replaced.

**Invariant, adapted.** The test `test_generation_uses_one_generator_forward_per_level` now runs for both paths. Reference path: `level_logits` is called once per level, as before. Cached path: `level_logits_cached` is called once per level, `build_prefix_cache` once per generation (it runs only the prefix tokens and is not a level forward), and `level_logits` is not called. The teacher-forced training equivalence tests are unchanged.

**Exactness evidence** (`tests/test_official_prefix_cache.py`, `mx.cpu`):

- Random tiny configs, with no weights needed: with and without level-0 queries, tied and untied heads, with and without prefix padding; batch 1, 3, 8. Per-level logits of the cached path equal the reference within 2e-5 of the logit scale (observed differences come only from different matmul shapes), sampled codes and decoded tokens are identical, forced (teacher-forced) logits also agree, and cached prefix K/V equal the prefix rows of the full forward to 1e-6.
- Repeated calls, an interleaved different prefix and a permuted batch give identical outputs; cached sampling at temperature 0.7 with top-k matches the reference codes for four seeds; compiled equals eager-cached across batch-size changes.
- Real checkpoints, weight-gated: greedy TinyStories (8 samples) and Sudoku easy, medium and hard (first 64 published boards each) give identical codes and tokens in reference, cached and compiled modes, with per-level logits within 2e-5 relative.
- On Metal, greedy Sudoku-hard on the first 500 boards solves 426 with identical cell accuracy (98.34%) in all three modes. Metal matmul is reduced precision, so this is an outcome check rather than a bitwise one.

### Measured latency

Apple M5 Pro, MLX 0.32.3, Metal, synchronized through a host copy, 3 warmups and 20 iterations, `official bench` as above. Sudoku-easy is greedy and TinyStories samples at T=1.0. Rows are p50 / p95 ms.

**The GPU was not idle.** `ioreg` sampled `Device Utilization %` every 2 s during the runs: median 65%, 96% of samples at or above 50%, peak 100%, from desktop applications and local inference services. No training run was active (checked with the process list). Absolute numbers are therefore noisy; read the ratios between modes, which were measured back to back. A second Sudoku run for batch 16 and 64 (75% and 37% at its start and end) reproduced the first.

| Sudoku-easy | eager | cached | cached + compiled |
|---|---:|---:|---:|
| b=1 | 20.5 / 21.9 | 23.4 / 26.2 | 17.9 / 18.7 |
| b=4 | 36.4 / 37.7 | 28.9 / 31.5 | 23.2 / 24.1 |
| b=16 | 207.8 / 334.5 | 65.4 / 70.9 | 52.0 / 55.5 |
| b=64 | 1,238 / 1,718 | 648 / 837 | 458 / 510 |
| b=16, rerun | 199.7 / 237.3 | 63.8 / 68.1 | 57.1 / 64.9 |
| b=64, rerun | 1,179 / 1,627 | 392.5 / 416.0 | 435 / 488 |

| TinyStories | eager | cached | cached + compiled |
|---|---:|---:|---:|
| b=1 | 40.3 / 46.0 | 48.3 / 50.3 | 38.4 / 39.1 |
| b=4 | 62.1 / 63.6 | 72.9 / 83.7 | 58.7 / 60.1 |
| b=16 | 334.9 / 660.9 | 389.1 / 405.8 | 316.7 / 468.6 |
| b=64 | 1,636 / 1,749 | 1,706 / 1,894 | 1,572 / 1,734 |

Compile and warmup cost, separated from steady state: the first call at each batch size (including the trace) was 114 to 474 ms for Sudoku and 76 to 2,608 ms for TinyStories compiled, against 43 to 1,461 ms (Sudoku) and 79 to 1,170 ms (TinyStories) eager; the steady-state rows above exclude it.

Reading, with the contention caveat. For Sudoku (91-token prefix) the cache gives about 3.1x at b=16 and 1.9 to 3.0x at b=64; compilation adds roughly 10 to 20% at b=4 to 16 and is inconclusive at b=64 (458 vs 435 against 392 in the reruns). At b=1 the cached path was slower than eager (23.4 vs 20.5 ms): the extra prefix launches are not paid back by one sequence. For TinyStories the prefix is one token, so the cache saves almost nothing and its extra concatenations cost a little (all differences at or below the run-to-run noise at b>=16); only compilation helps there, by about 5%. The gains are for conditional, long-prefix tasks. Peak-memory and first-call numbers are in the raw JSON, which is not tracked.

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
