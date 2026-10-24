---
title: Upstream NCM map
status: current
upstream: https://github.com/HazyResearch/ncm (shallow clone read 2026-10-22, default branch)
paper: https://arxiv.org/html/2610.14162v1 (full text and released Sudoku config checked 2026-10-24)
updated: 2026-10-24
---

# Upstream NCM map

What the reference implementation does, where, and what `ncm-mlx` takes from it. The initial source inspection was on 2026-10-22; the [released Sudoku configuration](https://github.com/HazyResearch/ncm/blob/main/config/eval/sudoku-easy.yaml), quantizer and generator were rechecked on 2026-10-24. No upstream code is copied into this repository; every module here is a fresh implementation of the concepts below. Local behavior is configuration-specific: the TinyWorld defaults differ substantially from the later Sudoku codec.

## Pipeline

| Step | Reference pipeline |
|---|---|
| Encode | BPE-8192 text or 12-token Sudoku vocabulary → token/position lookup → GroupNorm |
| Quantize | Multiscale residual VQ produces a hierarchy of codes with EMA codebooks |
| Generate | A block-diagonal next-concept transformer predicts each level in parallel |
| Decode | Sampled codes → cumulative reconstruction → one-block decoder → tokens |

The [architecture guide](../architecture.md) illustrates the shared two-stage structure and distinguishes local codec settings from the released configurations.

Stage 1 trains encoder, quantizer and decoder together as a VQ-VAE. Stage 2 freezes it and trains the generator on its codes.

## Components and source locations

| Component | Upstream location | Behaviour |
|---|---|---|
| Multiscale residual VQ | `models/quant.py` `MultiscaleResidualVectorQuantizer.forward` | For each level `k`: area-downsample the residual to `l_k`, nearest code by squared distance, linear-upsample to `L`, apply `Phi_k`, add to `z_hat`, subtract from residual. |
| Codebook learning | same, `self.training` branch | EMA of assignment counts and summed inputs, Laplace smoothing with `epsilon`; codebooks have `requires_grad=False`. Optional dead-code restart (`restart_threshold`, default 0). |
| `Phi` (quant_resi) | `models/quant.py` `Phi`, `_build_quant_resi` | `(1-r)·h + r·Conv1d_k3(h)`, `r = 0.5`, one per level (`share_mode=0`) selected by `level/(K-1)`. |
| VQ loss | `forward` | per level: `β·mse(sg(z_hat), z) + mse(z_hat, sg(z))`, averaged over levels; final `z_hat = z + sg(z_hat - z)` (straight-through). |
| Fine-level dropout | `forward`, `aux.fine_drop` | With prob 0.5, the decoder receives cumulative `z_hat` truncated at a random level ≥ `min_keep`. |
| Learned level-0 resampling | `_make_cascaded_down_l1/up_l1` | Released configs replace area/linear at level 0 with strided convs initialised to average/repeat, and use a straight-through estimator at level 0 only. |
| Encoder | `models/vae.py` `Encoder` | Released configs: token + position embedding only (`enc_layers=0`), optional compression conv. |
| Decoder | `models/vae.py` `Decoder` | One `Block` (`dec_layers=1`), LayerNorm, linear to vocab; no positional embedding by default. |
| Generator input | `quant.py` `idx_to_ncm_input` | For level `k+1`: cumulative `z_hat` after levels `0..k`, area-pooled to `l_{k+1}`. |
| Generator | `models/next_concept.py` `NextConceptTransformer` | Prefix (`prefix_len` tokens) + all target levels in one sequence; `wte` linear from codec dim; level embedding `wle`; RoPE with positions restarting per level; QK RMSNorm; one head per level, optionally tied to the codebook as a distance head. |
| Attention mask | same, `attn_pattern` | `block_diagonal` (default): prefix sees prefix; level `k` sees prefix and level `k`. `block_causal` lets level `k` also see coarser levels. |
| Level 0 | same, `generate` | Three options: `query_level0` learned queries (Sudoku), `predict_level0` from prefix positions, or a stored level-0 unigram (`generation.level0_unigram`, TinyStories/OWT/LM1B). |
| Generation | same, `generate` | One forward of the stack per level over `prefix + l_k` positions; optional prefix KV cache; temperature, top-k, top-p; optional CFG for conditional tasks. |

Released checkpoint shapes (from `config/eval/*.yaml`): TinyStories uses BPE-8192, `L=128`, 12 levels `1, 8, 16, …, 128` with 4096 codes each, codec `embed_dim=384`, generator 12 × 768. Sudoku uses 12 levels `1, 2, 4, …, 89` with 16 codes each and an 8 × 512 generator.

## What ncm-mlx adopts, modifies or replaces

| Idea | Status here | Notes |
|---|---|---|
| Two-stage codec → generator | **Adopted** | `ncm_mlx/latent`, `ncm_mlx/models`, `ncm_mlx/training`. |
| Multiscale residual VQ with area down / linear up | **Adopted** | Resampling is precomputed as matrices (`latent/resample.py`); tests pin PyTorch's `area` and `linear, align_corners=False` semantics. |
| EMA codebooks, smoothing, commitment + fit loss | **Adopted** | EMA is applied after the optimiser step from statistics returned by the forward, because MLX transforms are functional. Upstream updates codebook `k` before looking up level `k` in the same pass; here all levels use the pre-update codebooks. |
| `Phi` blend conv, ratio 0.5, non-shared | **Adopted** | |
| Fine-level dropout | **Adopted** | prob 0.5, `min_keep=1`. |
| Dead-code restart | **Modified** | Enabled by default (`restart_threshold=0.05`); upstream default is off. The toy corpus is small enough that collapse is a real risk. |
| Learned strided level-0 resampling + STE | **Implemented for Sudoku from E6** | `codec.learned_l0` uses one full-sequence convolution each way. TinyWorld retains fixed area/linear resampling. A multi-layer cascade for length 81 diverged in an earlier test. |
| Encoder | **Configurable** | TinyWorld uses 2 bidirectional transformer blocks; Sudoku E5 and later use the lookup-only encoder. |
| Pre-quant norm | **Configurable** | TinyWorld uses per-position LayerNorm without affine; Sudoku E6 and later use GroupNorm(8) without affine. |
| Decoder | **Modified / configurable** | TinyWorld uses 2 blocks; Sudoku E5 and later use 1. Both retain a learned positional embedding. |
| Pre-tokenizer | **Replaced** | Fixed-length characters for TinyWorld, 81 digit cells for Sudoku. Upstream text uses BPE; upstream Sudoku includes separators in its 89-token target. |
| Block-diagonal generator, per-level RoPE, level embedding, QK-norm | **Adopted** | Equivalence of per-level inference and full teacher-forced training is tested. |
| Level 0 | **Adopted (`query_level0`)** | |
| Untied per-level heads | **Adopted** | Codebook-tied distance heads not implemented yet. |
| Token-prefix conditioning | **Implemented** | Sudoku embeds 81 puzzle cells as the prefix; TinyWorld uses a learned unconditional prefix. |
| Conditioning corruption | **Implemented from E8** | Per-level linear rates, uniform replacement; optional model-sampled replacement tested in E9. Labels remain true codes. |
| CFG, top-p, NCM prefix KV cache | **Not implemented** | Sampling supports temperature, top-k and per-level temperatures. The separate AR baseline has KV caching and prefix prefill. |

## Checkpoint compatibility

No importer is implemented. Released checkpoints are PyTorch `.pt` files. Text checkpoints use BPE and may use codebook-tied heads, unlike the character-level TinyWorld path. Sudoku now implements lookup encoding, GroupNorm and learned level-0 resampling, but target/prefix lengths, token vocabulary, decoder positional embedding and some update semantics still differ. Similar component names do not establish tensor compatibility. Importing weights requires a separate configuration that reproduces the released model exactly, followed by logit and evaluation checks; that work has not started.

## Limitations and open hypotheses

- The paper's speedups against diffusion models concern its own hardware, tasks and sampling protocol. Local AR comparisons are separate measurements; see [benchmark methodology](../benchmarks.md).
- With a per-character final level, the codec can reconstruct perfectly from the finest level alone; whether the coarse levels carry useful global structure is an empirical question answered by the per-level decode metrics, not assumed.
- Parallel sampling within a level samples positions independently given the level input. Agreement errors in generated text are the expected failure mode and are measured.
- Unproven here: open-domain text quality on this training budget and the effect of codebook-tied heads. Learned first-level resampling is implemented and its ablations are recorded, but it alone did not produce solved Sudoku boards.

## License and citation

Upstream is MIT licensed, © 2025 Hermann Kumbong. This repository contains no upstream source; it cites the work:

> Hermann Kumbong, Ishan S. Khare, Iddah Mlauzi, Andrew Woen, Charles Ding, E. Kelly Buchanan, Simran Arora, Azalia Mirhoseini, Christopher Ré. *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation.* arXiv:2610.14162, 2026.

## Training recipe (paper, Appendix C, D.3, F)

From [the paper's Tables 5 and 6](https://arxiv.org/html/2610.14162v1), checked 2026-10-24. The released repository provides model/evaluation code and configs, but no training loop in the inspected tree. Reproduction therefore also depends on recipe details stated in the paper.

| Sudoku | Tokenizer (Table 5) | Generator (Table 6) | This repo, E6 |
|---|---|---|---|
| Parameters | 14.90M | 25.46M | 13.915M codec arrays including EMA state; 6.440M generator |
| Updates | 10,000 | 50,000 | 3,000 codec, 4,000 generator |
| Global batch | 1,024 | 1,024 | 128 |
| Peak learning rate (floor) | 5e-4 (5e-5) | 3e-4 (3e-5) | 2e-4 codec, 6e-4 generator |
| Warmup | 500 | 1,000 | 100 |
| AdamW β₁, β₂ / weight decay | 0.9, 0.99 / 0.01 | β₂ 0.99 / 0.01 | 0.9, 0.999 / 0.01 |
| Other | EMA decay 0.99, target length 89, 12 levels × 16 codes, width 256, 1 decoder block | 8 × 512, 8 heads, prefix 91 | |

The generator budget is 50,000 × 1,024 ≈ 51M training sequences; this repo's E6 used 0.5M, about 1%.

The local codec has 13,816,704 gradient-trained values, 49,152 EMA-trained codebook values and 49,344 EMA accumulators. The earlier 3.3M description omitted the learned level-0 down/up kernels. Comparisons describing the local model as one quarter of upstream's size concern the generators, not the full codec-plus-generator systems. See [parameter accounting](../benchmarks.md#parameter-accounting). The released Sudoku evaluation YAML specifies generator dropout 0.2, while paper Table 6 lists 0.1; this difference should be resolved before a strict checkpoint/recipe reproduction.

- **Conditioning corruption** (Section 3.4, Appendix F.3): during generator training, a fraction of the *conditioning* codes (the coarser levels the generator reads) is replaced by other codes from the same level's codebook. Labels stay the ground-truth codes. The replacement probability decreases linearly across levels. Moderate corruption "greatly improved generation quality"; too much hurt. The exact probabilities are not stated in the text read.
- **Minimum Bayes risk decoding** (Section 4.5, Appendix F.4): generate K candidates and keep the one most similar on average to the others. K = 12 for Sudoku in the appendix ablation. Table 3 lists 12 NFE for NCM, which matches one candidate with 12 levels, so the headline Sudoku numbers appear to be single-sample.
- **Number of levels** (F.1): K ∈ {4, 8, 16, 32} tested on text; generative perplexity improves with more levels, with diminishing returns beyond 16.
- Sudoku training data follows S-FLM; its size is not given in the paper.
- **Loss weighting** (Appendix F.2.3): $w_k \propto l_k^{-\alpha}$. $\alpha = 0$ weights every code equally (this repo's choice: CE weighted by level length); $\alpha = 1$ weights every level equally. On TinyStories, raising $\alpha$ from 0 to 1.5 moves the coarse levels' share of the loss from 19% to 63%, with results stable for $\alpha \le 1$.
- **Corruption, more detail** (F.3): codes are replaced uniformly from the same level's codebook at a per-level rate that decreases linearly; the authors note that "corruption based on the model's actual errors may better match inference conditions". That variant is not in the paper.

## Related implementations

See [related work](../related-work.md) for the dated comparison of official code, independent implementations and related architectures. This project makes no claim to be the first MLX port.
