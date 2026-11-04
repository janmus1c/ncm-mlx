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

Latency is not measured in this mode. A single prompt can be continued with `python -m ncm_mlx official continue-text --weights artifacts/tinystories-prefix-p32/export --prompt "..."`; the model reads the first P tokens of the prompt, and shorter prompts are padded, which training never showed it. Options: `--n` continuations (default 2), `--temperature`, `--top-k`, `--top-p`, `--seed`, `--cpu`. Output is one block per continuation, `prompt[|]continuation`, and is repeatable for a fixed seed. `tests/test_text_prefix.py::test_continue_text_cli_continues_a_prompt_on_cpu` runs exactly this command on a tiny CPU model and checks the format and repeatability; it does not check quality.

No prefix-conditioned model has been trained or evaluated yet; the tests use tiny CPU models and check shapes, masking, equivalence and resume, not quality.

Example, for a model trained with `data.prefix_len > 0`:

```bash
python -m ncm_mlx official continue-text --weights artifacts/tinystories-prefix-p32/export \
    --prompt "Once upon a time, a little fox found a red kite." --n 2 --seed 0
```

It prints `prompt[|]continuation` per sample. A CPU test runs this command on a tiny model; no prompt-conditioned model has been trained at a useful scale yet.

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
| Precision | float32 by default (Metal reduced-precision matmul); optional bfloat16 mixed precision for the generator and AR stages, see below | Paper: BF16 mixed precision, `torch.compile` |
| Best checkpoint | Final weights exported; validation is monitored only | Not stated |

## Optional Muon optimizer, neighbour corruption and the official-recipe config

Three more options, all off by default. With the defaults the config dictionary has none of the new keys, so config hashes, checkpoints and `--resume` are unchanged, and tiny three-stage CPU runs give byte-identical checkpoints before and after (also across the MLX 0.32.3 to 0.32.4 bump; the tests compare the default path with a copy of the old corruption function and train with the options spelled out).

- `train.<stage>.optimizer = "muon"` (default `"adamw"`) puts the 2-D weight matrices inside the transformer blocks (and the generator's input projection) on `mlx.optimizers.Muon` and everything else (embeddings, tied heads, codebooks, norms, biases, convolutions) on AdamW. `muon_lr` (default 0.02, on the same warmup and cosine shape as `lr`), `muon_momentum` (0.95), `muon_weight_decay` (0.0) and `muon_ns_steps` (5) apply to the Muon part; the logged `lr` is the Muon rate. The fused `qkv` matrix is orthogonalised as one matrix.
- `generator.corruption_mode = "neighbor"` (default `"uniform"`) replaces a corrupted conditioning code with one of its `corruption_neighbors` (default 8) nearest other codes by distance between codebook vectors, instead of a uniform draw. `generator.corruption_rate_jitter = j` (default 0) multiplies each sequence's per-level rates by a factor drawn uniformly from `[1 - j, 1 + j]`, so the mean schedule is kept. Labels stay the true codes. Neither has been trained at a useful scale; the experiment configs are `configs/tinystories-muon-12m.json` and `configs/tinystories-neighbor-12m.json` (bfloat16, otherwise E21 with the frozen E21 codec copied in).
- `configs/tinystories-official-recipe.json` trains a new codec and generator with the details that the released TinyStories config and the paper fix: 4,096 codes per level, codec width 384 with 6 heads, block-diagonal attention, corruption mean 0.50 falling linearly, generator learning rate 3e-4, at E21's size and budget. The released `config.yaml` holds inference settings only, so the corruption endpoints, batch size and step count are not public beyond the paper's numbers.

## Optional bfloat16 mixed precision

`"train": {"precision": "bfloat16"}` (default `"float32"`) switches the generator and AR stages to bfloat16 compute. The codec stage always trains in float32: its nearest-code assignments and EMA codebook statistics are the part most sensitive to rounding, and it is the smallest stage.

What changes, per training step:

- The model keeps float32 master weights. Each step casts them to bfloat16 inside the differentiated function, runs the forward and backward in bfloat16, and receives float32 gradients for the master weights. Gradient clipping, AdamW (moments and update) and the weight update are float32.
- LayerNorm, the QK RMSNorm and the codebook-tied heads stay float32 (norm statistics are computed on a float32 copy of the activations). RoPE is applied in float32 and the result is cast back. The softmax inside fused attention is float32 in MLX by its own documentation. Logits and cross-entropy are float32 (the AR's tied output matmul is bfloat16; the generator's tied heads are float32).
- The frozen codec is not touched: it stays float32, so the codes that serve as generator targets are exactly the float32 codec's codes (checked in the tests).
- Checkpoints and exports hold the float32 master weights and float32 optimizer state, so `--resume` works as before, and a bfloat16 run and a float32 run cannot resume each other's checkpoints (the config differs). Validation during training and the final reports run in float32 on the master weights, which is how the exported model is used.
- With the default, the config dictionary (and so the checkpoint metadata and `spec.json`) has no `precision` key and the computation is the pre-existing one. On CPU, tiny three-stage runs gave byte-identical checkpoints before and after this change, compiled and uncompiled. On a GPU the run-to-run results already differ at the bit level, so that comparison does not apply there.

On a tiny CPU model the bfloat16 loss and gradient norm were within 0.01% and 0.15% of float32 at identical weights. This says the implementation is consistent, not that training dynamics are unchanged. At a useful scale, one comparison exists: the E21 recipe trained in bfloat16 on an RTX 3080 matched the fp32 runs closely in quality, and its compiled generator step was about 1.5 times faster with 36% less peak memory at batch 32 ([E29](EXPERIMENTS.md#e29-bf16-mixed-precision-quality-parity-at-the-e21-recipe-2026-11-03)). That is one size, one seed and one device; the speed-up depends on the device and model size. `scripts/text_step_timing.py` times the real compiled training step at a chosen precision and prints step time, tokens per second and peak memory as JSON:

```bash
python scripts/text_step_timing.py configs/tinystories-e24-seed1.json --stage generator --precision float32
python scripts/text_step_timing.py configs/tinystories-e24-seed1.json --stage generator --precision bfloat16
```

It uses random token windows and a randomly initialised frozen codec, so it needs no data. Run each precision on an otherwise idle device and compare like with like. Training logs report peak memory as before.

## Sampling study

`scripts/sampling_study.py` is the evaluation driver behind the sampling-policy study (E27 in the [experiment log](EXPERIMENTS.md)). It trains nothing: it draws samples from a trained NCM and its matched AR baseline under different policies (global and per-level temperature, top-k, top-p, a sharpened level-0 unigram for the NCM), scores them with the pinned GPT-2-large scorer and writes one JSON result and one sample file per run, with the same diversity diagnostics as `official eval-text`.

```bash
python scripts/sampling_study.py scripts/sampling_study_runs.json \
    --checkpoint e21=artifacts/tinystories-e21/export --checkpoint e24-seed1=artifacts/tinystories-e24-seed1/export \
    --data artifacts/tinystories/v2-bpe8192-L128 --out artifacts/sampling-study --selftest
```

`scripts/sampling_study_runs.json` lists the 98 runs of E27 (seeds, number of samples and sampling parameters). Each name after `--checkpoint` is the `ckpt` field the runs refer to. The first run is a gate: it stops the study if the default-setting NCM perplexity falls outside a known range, so a setup that does not reproduce is caught early. Finished runs are skipped, so an interrupted study resumes. Memory is capped by `NCM_MLX_MEMORY_LIMIT_GB`. The script was cleaned up from the private one that produced the published tables and has not been re-run on a clean checkout.

## Evaluation

`official eval-text` writes one JSON report and the samples (`runs/<name>-eval/`). It reports these numbers separately:

- Held-out codec reconstruction on test windows: token accuracy (all and non-pad), exact windows, per-level cumulative accuracy and code usage.
- Generator teacher-forced **code** cross-entropy per level, in nats per code. This is not a text perplexity.
- AR next-token cross-entropy on the same windows.
- Free-running samples, 5 seeds × 256 at temperature 1, with the stored level-0 unigram for NCM. The diagnostics are length, EOS rate, distinct-1/2/3, within-sample repeated 4-grams, per-sample unigram entropy (upstream `H`), duplicate samples and the share of samples copying a training window's first 32 tokens. The first eight samples of the first seed are stored verbatim, failures included.
- Generative perplexity under [`openai-community/gpt2-large@32b71b1`](https://huggingface.co/openai-community/gpt2-large) run locally in MLX (`mlx-lm` 0.32.2). It is modelled on the upstream TinyStories protocol (no BOS token, every GPT-2 token after the first is scored, corpus-level perplexity per seed, then the mean over seeds) but is an independent implementation; agreement with upstream's reported scores is not established. Held-out data windows are scored the same way as a reference.
- Synchronized latency at batch 1 and 16 (host-copy completion, 3 warmups, 20 iterations) with GPU-load readings.

Under this scorer the released TinyStories checkpoint scores 16.259 (E21 evaluation). This is a reference value for this protocol, not a reproduction of the paper's reported number.

## Results so far

**Text models trained here from scratch (2026-11-02).** Small TinyStories models reconstruct held-out text accurately (99.8% of tokens) but generate stories that GPT-2 large scores far worse than an autoregressive baseline: generative perplexity 45.6–97.8 against 14.3–14.7, across three corruption schedules and two generator seeds on one codec. Less conditioning corruption lowers teacher-forced loss and worsens generated text. Text training is experimental; see the [controlled-text assessment](research/e21-e23-assessment.md).

**Size and data (2026-11-04).** With the same frozen codec, a larger generator and four times the data narrow the gap without closing it: the ratio of NCM to AR generative perplexity falls from 3.33 (12M generator) to 2.38 (87M) at the original data and to 2.13 (87M, 24.40 against 11.48) at four times the data in bf16. These are single-seed runs except one, compute is not matched, and the 50M-to-87M step is within seed variation; see E30 to E32 in [EXPERIMENTS](EXPERIMENTS.md#e32-50m-and-87m-generators-at-four-times-the-data-bf16-2026-11-04).

The [corruption-ablation figure](../paper/figures/corruption-ablation.svg) from the working paper plots both measures for both generator seeds.

## Limitations

- One codec, three corruption schedules and two generator seeds. The other assumptions above are untested by ablation; differences from the released model's quality cannot be attributed to any single choice.
- The E21 budget is a small pilot, much shorter than the original authors' training. Low quality at this budget does not measure the recipe's ceiling.
- TinyStories is a synthetic corpus of simple stories. Results here do not establish quality on naturally occurring text, long-form coherence or instruction following.
- Trained weights and token caches are not distributed (see [THIRD_PARTY](https://github.com/janmus1c/ncm-mlx/blob/main/THIRD_PARTY.md) for the CDLA-Sharing-1.0 data terms).
