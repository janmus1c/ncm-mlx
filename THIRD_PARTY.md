# Third-party provenance and licenses

This repository's original implementation and documentation are licensed under [MIT](LICENSE). Dependencies and external datasets retain their own licenses. The software license does not grant rights to arbitrary training data, third-party models or generated artifacts from other projects.

## NCM method and reference implementation

The architecture originates in [Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation](https://arxiv.org/abs/2610.14162), Kumbong et al. (2026). The [HazyResearch reference implementation](https://github.com/HazyResearch/ncm) has an [MIT license](https://github.com/HazyResearch/ncm/blob/main/LICENSE), copyright 2025 Hermann Kumbong. NCM-MLX independently implements the architecture; no upstream source code is copied. The [upstream map](docs/research/upstream-map.md) describes conceptual correspondences and differences. Preserve the upstream notice with any future copied code.

## Red-fuchs/ncm-modular

Two features of this release follow [Red-fuchs/ncm-modular](https://github.com/Red-fuchs/ncm-modular) (MIT, Copyright (c) 2026 the ncm-modular authors; reviewed at `eae154b`). No code from it is copied or vendored; the code here is written against this repository's own modules. Keep the ncm-modular MIT notice with any future copied code.

- **XSum and WMT14 codecs.** ncm-modular established that these checkpoints need GroupNorm-normalized level-0 resampling and reported the WMT14 level-9 observation. It served as a reference for cross-checks during development. The source tokenizers it names (`openai-community/gpt2`, `Qwen/Qwen3-0.6B`) are not downloaded or redistributed. `tests/fixtures/official_codec_configs.json` restates the `tokenizer.vqvae` sections of two upstream `config.yaml` files as data (HazyResearch, MIT, Copyright (c) 2025 Hermann Kumbong), and `tests/fixtures/official_codec_parity.json` holds token ids and codes computed with upstream PyTorch.
- **Prefix-conditioned text training.** The prompt-conditioned training in `ncm_mlx/official/` (`data.prefix_len`) follows the design of ncm-modular's prefix-conditioned NCM loop (MASSIVE pilot, `RESULTS.md` section 3): a frozen codec over the target, a pad-masked prefix of text tokens that attends only to itself, level 0 predicted from the prefix, and block-diagonal per-level attention.

## Datasets

| Data | Origin and use | License and distribution |
|---|---|---|
| TinyWorld | Synthetic grammar defined in `ncm_mlx/text/toygrammar.py` | Generated locally by this repository's MIT implementation; no external corpus |
| Sudoku training puzzles | Generated locally by `ncm_mlx/text/sudoku.py`; generation and uniqueness rules depend on config | Generated locally; no third-party training boards bundled |
| Published Sudoku evaluation boards | [`hazyresearch/ncm-datasets`](https://huggingface.co/datasets/hazyresearch/ncm-datasets/tree/6580ba41a4b0a976dde876fbe84d25327617b0e6/sudoku), six puzzle/solution files | Dataset card declares MIT; downloaded separately, never bundled in this repository or wheel |
| TinyStories (V2, GPT-4 stories) | [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories/tree/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64), `TinyStoriesV2-GPT4-train.txt` and `TinyStoriesV2-GPT4-valid.txt`, SHA-256 pinned in `ncm_mlx/official/tinystories.py`; text training data for `official train-*` | Dataset card declares [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/); synthetic stories generated with GPT-3.5/GPT-4 (Eldan and Li, arXiv:2305.07759). Downloaded on demand into ignored `artifacts/`, never bundled; the token caches and models trained on it are not distributed |

The dataset's [versioned card](https://huggingface.co/datasets/hazyresearch/ncm-datasets/blob/6580ba41a4b0a976dde876fbe84d25327617b0e6/README.md) and public metadata were checked on 2026-10-24. Its MIT declaration covers the dataset repository as published; this release downloads only Sudoku files and does not make licensing claims about other included text corpora. The repository does not provide a separate dataset license notice beyond that card. Keep the source attribution and check applicable terms before redistributing datasets. The downloader pins this revision and verifies the six hashes in `scripts/sudoku.sha256`.

The TinyStories card at the pinned revision was checked on 2026-10-29: its front matter declares `license: cdla-sharing-1.0` and describes the stories as synthetic GPT-3.5/GPT-4 generations. CDLA-Sharing-1.0 places conditions on publishing the data or enhanced data, not on publishing results computed from it; check its terms, and the terms of the generating models' providers, before redistributing the text, token caches or trained weights.

## Model weights

Text evaluation (`official eval-text`) scores samples with [`openai-community/gpt2-large`](https://huggingface.co/openai-community/gpt2-large/tree/32b71b12589c2f8d625668d2335a01cac3249519) (model card: MIT), downloaded at a pinned revision into ignored `artifacts/scorers/` and run locally through `mlx-lm` (MIT); it is an optional `text-eval` dependency group and is never redistributed.

No NCM-MLX weights, upstream checkpoints, tokenizer assets or training datasets are distributed in this source release. `artifacts/`, `runs/` and weight extensions are ignored. Public JSON summaries in `paper/data/` report measurements and provenance, rather than weights. `python -m ncm_mlx official fetch` downloads released HazyResearch checkpoints (MIT, copyright 2025 Hermann Kumbong) at a pinned revision with SHA-256 checks into ignored `artifacts/official/`; converted MLX weights stay there and are not redistributed. The committed parity fixture contains only small reference values computed from them. Any future weight release needs its own model card covering training data, license, configuration, evaluation, intended use and limitations.

## Runtime and build dependencies

MLX is [MIT licensed](https://github.com/ml-explore/mlx/blob/v0.32.4/LICENSE). NumPy uses a [BSD license](https://github.com/numpy/numpy/blob/main/LICENSE.txt). Optional NVIDIA CUDA libraries have their own terms and are installed from their published packages, rather than redistributed here. `pyproject.toml` declares dependencies; `uv.lock` records resolved packages and hashes. Paper and documentation tools are optional dependency groups. Installing this repository does not replace or relicense any dependency.
