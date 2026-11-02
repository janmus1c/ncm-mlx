---
title: Training the TinyStories architecture
description: A two-stage NCM training recipe for real text in MLX, in the released TinyStories architecture family, with an autoregressive baseline of the same block stack and a fixed generative-perplexity scorer.
status: experimental
updated: 2026-11-02
---

# Training the TinyStories architecture

Upstream released NCM inference code and checkpoints but no training code ([HazyResearch/ncm](https://github.com/HazyResearch/ncm) issues #2 and #10). This page documents this repository's training recipe for the released TinyStories architecture family. The codec, generator and sampler are the inference modules of `ncm_mlx/official/`, so a trained checkpoint loads through the same `official generate` and `official bench` path as the released weights. The objectives are inferred from the paper ([Kumbong et al., arXiv:2610.14162](https://arxiv.org/abs/2610.14162), Sections 3.3–3.4, Appendices C and F) and from the upstream quantizer and model code (MIT, Copyright (c) 2025 Hermann Kumbong). No upstream code is copied. Measured runs are in [EXPERIMENTS](EXPERIMENTS.md) (E21–E24) and the [controlled-text assessment](research/e21-e23-assessment.md).

## Commands

```bash
python -m ncm_mlx official fetch --task tinystories && python -m ncm_mlx official convert --task tinystories  # BPE + reference
python -m ncm_mlx official prepare-data    --config configs/tinystories-e21.json   # pinned download, tokenize (~1 min)
python -m ncm_mlx official train-codec     --config configs/tinystories-e21.json   # stage 1
python -m ncm_mlx official train-generator --config configs/tinystories-e21.json   # stage 2, frozen codec
python -m ncm_mlx official train-ar        --config configs/tinystories-e21.json   # AR baseline
uv run --group text-eval python -m ncm_mlx official eval-text --config configs/tinystories-e21.json
python -m ncm_mlx official generate --task tinystories --weights artifacts/tinystories-e21/export --n 4 --levels
```

`scripts/run_text_pipeline.sh CONFIG` runs all five steps, skips finished stages and resumes an interrupted one. Every training command takes `--resume`, which restores the full state: weights, EMA codebooks, AdamW moments, schedule position, MLX random state and the shuffled-batch position. A graceful pause (`SIGUSR1`, `SIGINT` or `SIGTERM`) saves and exits 75, as described in [RECOVERY](RECOVERY.md). `configs/tinystories-smoke.json` is a 10-minute end-to-end check. `scripts/text_train_probe.py` measures training throughput for candidate shapes.

Outputs go to `<out_dir>/export/`: `codec.safetensors`, `generator.safetensors`, `inference.safetensors` (level-0 unigram), `bpe.json`, `ar.safetensors` and `spec.json` with the full config and training history. Full-state snapshots stay in `<out_dir>/<stage>-training.safetensors`.

## Data contract

| Item | Choice | Basis |
|---|---|---|
| Corpus | `roneneldan/TinyStories` at `f54c09f`, `TinyStoriesV2-GPT4-{train,valid}.txt`, SHA-256 checked | The upstream tokenized stream (`hazyresearch/ncm-datasets`) begins with the first V2 story; inferred, not documented |
| Tokenizer | Released 8,192-token BPE, EOS 0, PAD 1 | Released checkpoint |
| Story | Text between `<|endoftext|>` markers, verbatim, with a leading newline | Upstream rows start with `\n` after EOS |
| Window | First 128 tokens of `encode(story) + [EOS]`, right-padded with PAD | Released samples start at a story opening and usually hit the 128 limit; later windows of long stories are unused |
| Prefix | One EOS token, embedded by the codec's token table | Released config (`raw_token_prefix`, prefix 1) and upstream evaluation |
| Split | Train file → train; valid file → validation / test by SHA-1 parity of the stripped story; held-out stories also present in train are removed | Document-level separation |

On the pinned files: 2,717,495 train, 13,960 validation and 13,670 test stories; no held-out story text occurs in train. 0.38% of train positions are padding and 5.0% of windows contain their story's EOS. Near-duplicate openings exist: 4.7% of test windows share their first 32 tokens with some training window.

## Stage 1: codec

Architecture: the released codec (token + position lookup, GroupNorm(8) without affine, learned strided level-0 cascade, area pooling and linear interpolation elsewhere, per-level `Phi`, one decoder block). Width, heads and codebook sizes come from the config. The released size is width 384 with 4,096 codes per level, 36.43M values, as in paper Table 5. E21 uses width 256 with 2,048 codes per level.

Per batch:

1. `z = GroupNorm(dropout(tok + pos))`.
2. Residual VQ on `stop_gradient(z)`: nearest code by squared distance at every level. Level 0 passes gradients into its learned cascade through a straight-through estimator; the upstream code applies it at level 0 only.
3. `vq = mean_k [β·mse(sg(ẑ_k), z) + mse(ẑ_k, sg(z))]` over all levels, with β = 0.25.
4. Fine-level (nested) dropout: with probability 0.5 the decoder reads ẑ truncated after a uniformly drawn level in `[min_keep − 1, K − 2]`.
5. The decoder reads `z + sg(ẑ − z)`, the straight-through path to the encoder. Token cross-entropy is taken over all 128 positions, padding included.
6. AdamW step with gradient clipping, then EMA codebook update.

Codebooks are not gradient-trained. They follow an EMA (decay 0.95) of assigned inputs with Laplace-smoothed counts (ε = 1e-5), as upstream. The level-0 unigram is the empirical distribution of level-0 codes over the first 200,000 training windows, stored for sampling.

## Stage 2: generator

Architecture: the released generator (in-projection from codec width, learned level embedding, RoPE restarting per segment, QK RMSNorm, LayerNorm eps 1e-6, exact GELU, codebook-tied distance heads with a learned log inverse temperature, prefix-pad masking). Level 0 is not predicted.

Per batch:

1. Targets are the frozen codec's greedy codes of the clean window.
2. Conditioning corruption: codes of levels `0..K−2` are replaced by uniform random codes of the same level with probability falling linearly from 0.75 to 0.25. Labels stay clean.
3. One teacher-forced forward over `[prefix, level 1, …, level K−1]`. Level `k` reads the cumulative reconstruction of the (corrupted) levels `< k`, area-pooled to `l_k`. Attention is block-diagonal: each level sees the prefix and itself, and the prefix sees itself. It is computed segment by segment, which is the per-level inference computation and avoids forming 640 × 640 scores.
4. Loss `Σ_k l_k^(1−α) CE_k / Σ_k l_k^(1−α)` (paper eq. 10) with α = 0, so every code position has equal weight.
5. Residual dropout 0.1; residual projections initialised N(0, 0.02/√(2·n_layer)) as upstream.

`tests/test_text_training.py` checks on CPU that these training logits equal the logits of the official sampler's level-by-level forwards (max error < 1e-5). It also checks that sampling runs exactly one generator forward per level.

## AR baseline

`ncm_mlx/official/ar.py`: the generator's `Block` (QK norm, RoPE base 10⁵), same width, depth and dropout, tied 8,192-token embedding. The EOS prefix is its first input. Next-token cross-entropy is taken over the same 128 positions, padding included. Sampling uses a KV cache (128 forwards) and the NCM sampler's temperature/top-k/top-p semantics. The two models share the block stack and the number of training windows, and nothing else is matched. The NCM's teacher-forced input has 641 positions per window (640 code positions plus the EOS prefix) against the AR's 128; sequence length alone is not a FLOP estimate. The NCM codec's training is additional.

## Prompt continuation (prefix conditioning)

`data.prefix_len` (P, default 0) switches the task from unconditional stories to continuing a prompt. With P = 0 nothing above changes: the key is left out of the saved configuration, so existing configurations, config hashes, checkpoints and `--resume` behave as before, and a short CPU run of codec, generator and AR training gives bit-identical state to the code before this option was added. `configs/tinystories-prefix-p32.json` is E21 with P = 32 (same codec, corruption 0.75 → 0.25, 15,000 × 64 generator steps, AR 7,500 × 128); `configs/tinystories-prefix-smoke.json` is a small P = 16 end-to-end check on 20,000 stories.

**Data.** A row is the first P + 128 tokens of `encode(story) + [EOS]`, right-padded with PAD. Tokens 0..P−1 are the prefix (including the leading newline token), tokens P..P+127 are the target the codec encodes. Stories with an EOS inside the first P tokens (fewer than P real tokens) are dropped from every split; a story of exactly P tokens is kept and its target is EOS followed by padding. The train/validation/test assignment is made per story before this filter, so held-out stories stay disjoint from training stories. The cache lives in `v2-bpe8192-L128-P<P>/`. Targets are therefore mid-story windows, not story openings.

**Generator.** The prefix is the P prompt tokens, embedded by the codec's token table, with the released semantics: pad positions are masked, RoPE restarts at 0, and the prefix attends only to itself. Level 0 is predicted from the prompt with the generator's learned level-0 queries (`query_level0`, as in the upstream prefix tasks) instead of being drawn from the stored unigram, since otherwise the global code of the continuation would ignore the prompt. Only target codes are labels, with the same level weights, corruption and dropout as before. A CPU test checks that the training logits equal the official sampler's per-level logits for prompts with padding, with and without the prefix cache. The generator sees 32 more positions per sequence (672 instead of 640).

**AR baseline.** The prompt tokens are prepended to the target, there is no BOS token, and the loss covers only the 128 target positions, the same positions the NCM codes describe. Block stack, width and depth are unchanged; the prompt is read once into the KV cache before sampling.

**Codec.** It is trained on the target windows only. E21's frozen codec can be reused; it was trained on story openings, so check `codec_test_target` in the evaluation report before trusting a comparison:

```bash
mkdir -p artifacts/tinystories-prefix-p32/export
cp artifacts/tinystories-e21/export/{codec.safetensors,inference.safetensors,bpe.json} artifacts/tinystories-prefix-p32/export/
scripts/run_text_pipeline.sh configs/tinystories-prefix-p32.json   # skips the codec stage, trains generator and AR, evaluates
```

Without the copy the pipeline trains a new codec on the continuation windows.

**Evaluation.** `official eval-text --models data ncm ar` detects P > 0 and continues the first 256 held-out prompts (the same prompts for every model and seed). The released checkpoint is skipped because its prefix has one token. Reported separately:

- Teacher-forced held-out loss of the continuation, with the true prompt and with another story's prompt (`*_shuffled_prompt`). The gap shows how much each model reads its prompt.
- The sample diagnostics from above, on continuations only.
- Generative perplexity under the same GPT-2 large scorer. The scorer reads prompt plus continuation; only tokens after the prompt count. The boundary is the number of GPT-2 tokens the prompt encodes on its own, so a token that straddles it counts as continuation. Gold continuations are scored the same way.
- A prompt-faithfulness **diagnostic**, labelled as such in the report: share of continuations that mention a character name from the prompt (capitalised words that do not start a sentence), content-word overlap, and the share of continuation 4-grams copied verbatim from the prompt, each next to a control that pairs every continuation with the next story's prompt. These are surface heuristics. They do not measure coherence, and verbatim copying is a failure mode, not a success.

Latency is not measured in this mode. A single prompt can be continued with `python -m ncm_mlx official continue-text --weights artifacts/tinystories-prefix-p32/export --prompt "..."`; the model reads the first P tokens of the prompt, and shorter prompts are padded, which training never showed it.

No prefix-conditioned model has been trained or evaluated yet; the tests use tiny CPU models and check shapes, masking, equivalence and resume, not quality.

## Assumptions where the paper and code are silent

| Topic | Assumption here | Upstream evidence |
|---|---|---|
| Data variant and windowing | V2 GPT-4 stories, one story-anchored window each | Inferred from the released token stream and samples |
| Padding in the loss | Included for codec, generator (codes of padded windows) and AR | Codec reconstructs all `L` positions; nothing indicates masking |
| VQ loss levels | Averaged over all levels, not the dropout cutoff | Upstream code; paper eq. 9 writes the cutoff |
| Fine-drop draw | Per sequence | Upstream draws once per batch |
| EMA timing | After the optimizer step, from pre-update assignments | Upstream updates level `k` before its lookup in the same pass |
| Dead codes | Restart from a random batch input when the EMA count falls below 0.01 | Upstream default 0 (off). Without restarts, unused codes divide by the smoothing constant and grow without bound; the released level-0 codebook norms are about 4·10¹⁰ |
| Corruption schedule | Linear 0.75 → 0.25 over conditioning levels, mean 0.50 | Paper: "shifted linear decreasing", selected mean rate 0.50; endpoints not given |
| Loss weighting | α = 0 | Paper F.2.3: α ≤ 1 gives the same generative perplexity |
| Dropout placement | Residual branches only. No attention-probability dropout (MLX fused attention); the codec's 0.3 also applies to the input embedding | Upstream uses attention and residual dropout |
| Weight decay | All gradient-trained parameters | Not stated |
| LR schedule | Linear warmup, cosine decay to the stated floor | Paper gives peak, floor and warmup only |
| Precision | float32 (Metal reduced-precision matmul) | Paper: BF16 mixed precision, `torch.compile` |
| Best checkpoint | Final weights exported; validation is monitored only | Not stated |

## Evaluation

`official eval-text` writes one JSON report and the samples (`runs/<name>-eval/`). It reports these numbers separately:

- Held-out codec reconstruction on test windows: token accuracy (all and non-pad), exact windows, per-level cumulative accuracy and code usage.
- Generator teacher-forced **code** cross-entropy per level, in nats per code. This is not a text perplexity.
- AR next-token cross-entropy on the same windows.
- Free-running samples, 5 seeds × 256 at temperature 1, with the stored level-0 unigram for NCM. The diagnostics are length, EOS rate, distinct-1/2/3, within-sample repeated 4-grams, per-sample unigram entropy (upstream `H`), duplicate samples and the share of samples copying a training window's first 32 tokens. The first eight samples of the first seed are stored verbatim, failures included.
- Generative perplexity under [`openai-community/gpt2-large@32b71b1`](https://huggingface.co/openai-community/gpt2-large) run locally in MLX (`mlx-lm` 0.32.2). It is modelled on the upstream TinyStories protocol (no BOS token, every GPT-2 token after the first is scored, corpus-level perplexity per seed, then the mean over seeds) but is an independent implementation; agreement with upstream's reported scores is not established. Held-out data windows are scored the same way as a reference.
- Synchronized latency at batch 1 and 16 (host-copy completion, 3 warmups, 20 iterations) with GPU-load readings.

Under this scorer the released TinyStories checkpoint scores 16.259 (E21 evaluation). This is a reference value for this protocol, not a reproduction of the paper's reported number.

## Limitations

- One codec, three corruption schedules and two generator seeds. The other assumptions above are untested by ablation; differences from the released model's quality cannot be attributed to any single choice.
- The E21 budget is a small pilot, much shorter than the original authors' training. Low quality at this budget does not measure the recipe's ceiling.
- TinyStories is a synthetic corpus of simple stories. Results here do not establish quality on naturally occurring text, long-form coherence or instruction following.
- Trained weights and token caches are not distributed (see [THIRD_PARTY](https://github.com/janmus1c/ncm-mlx/blob/main/THIRD_PARTY.md) for the CDLA-Sharing-1.0 data terms).
