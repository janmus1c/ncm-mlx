---
title: Related work and NCM implementations
description: Primary-source comparison of Next-Concept Modeling implementations and related parallel generative methods.
status: current
updated: 2026-10-24
---

# Related work and NCM implementations

The original Next-Concept Modeling architecture is due to [Kumbong et al., *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*](https://arxiv.org/abs/2610.14162). NCM-MLX is an independent engineering implementation and experimental investigation. It does not claim invention of the method, affiliation with its authors or priority among MLX ports.

NCM first trains a multiscale residual vector-quantized autoencoder. A separate transformer then generates the latent hierarchy from coarse to fine, predicting positions within each level in parallel. This retains a sequential dependency across levels while avoiding left-to-right token generation. The original paper evaluates text tasks and Sudoku. Those results are distinct from this repository's smaller TinyWorld and Sudoku experiments; paper speedups against diffusion baselines do not establish speedups against arbitrary autoregressive models.

## NCM software comparison

Checked on 2026-10-24 using public repository documentation and source listings. “Reported” means the other project's own evidence, not an experiment rerun by NCM-MLX. This is a scope comparison, not a benchmark leaderboard.

| Project and inspected revision | Architecture/framework | Training and inference | Checkpoints | Hardware and evaluation |
|---|---|---|---|---|
| [HazyResearch/ncm](https://github.com/HazyResearch/ncm/tree/c70274273f489abc73832b2b98584a8483b6b203) (`c702742`) | Official PyTorch NCM, multiscale residual VQ codec and coarse-to-fine transformer | Model, generation and evaluation source; no training loop in the inspected release | Official released checkpoints and task configs linked from its README | CUDA-oriented evaluation; text benchmarks and Sudoku. This is the reference implementation, not an independent reproduction. |
| [Bakhaow/ncm-forge](https://github.com/Bakhaow/ncm-forge/tree/62b6af4ae2759c8c8a1d327097e5d0c822a7ad1e) (`62b6af4`) | PyTorch; reuses upstream modules with added training pipeline and fill-in-the-middle variant | Codec and generator training, generation helpers and latency tools | Claims upstream-evaluator-compatible checkpoint serialization; no trained model shipped at inspection. Compatibility was not independently tested here | Reports RTX 4070 measurements. Its documented latency measurements use random weights; its hierarchy probe uses an official Sudoku checkpoint. These are not trained-model quality results. |
| NCM-MLX (this repository) | Independent MLX implementation; character/cell tokenization and task-specific codec settings | Both training stages, level-by-level generation, AR baseline, full-state recovery | MLX safetensors; released TinyStories and Sudoku checkpoints convert through a separate inference port with tested parity | Measured Metal M5 Pro and CUDA RTX 3080/A40 runs. TinyWorld grammar metrics and shared upstream Sudoku test boards. See [experiments](EXPERIMENTS.md), [benchmarks](benchmarks.md) and [upstream differences](research/upstream-map.md). |

The implementations make different choices about tokenization, shapes, level-0 prediction and training support. A shared method name does not make their weights interchangeable. In particular, NCM-MLX's 81-cell Sudoku representation and its TinyWorld character vocabulary differ from the released upstream task formats. Training results at different model sizes or sequence budgets are not controlled comparisons.

## Other projects using the NCM name or hierarchy

| Project | Scope established by its documentation | Relationship to NCM-MLX |
|---|---|---|
| [kavai-no/ncm-forge](https://github.com/kavai-no/ncm-forge/tree/06bac11989834ed501ff63d7aae1995cc0d22583) (`06bac11`) | Qwen speculative-decoding drafter, PyTorch/Transformers notebooks and a web demonstration; documented T4 workflow. Proposals are checked against greedy Qwen. It explicitly has no codebook | A distinct approach despite the similar name. It does not implement the multiscale VQ hierarchy or establish NCM checkpoint compatibility. Acceptance rate controls any speed benefit. |
| [yoonalexander/Concept-Firewall](https://github.com/yoonalexander/Concept-Firewall/tree/98833cd97a466beec3e1ff0a7d6f36b32b288d99) (`98833cd`) | Frozen official NCM Sudoku hierarchy analysis using PyTorch/CUDA; upstream code pinned as a submodule. Reports checkpoint baseline reproduction and a bounded probe study closed with a negative finding | Evaluation/interpretability tooling, rather than a new generative implementation. Its official-checkpoint study does not validate NCM-MLX's independently trained weights. Reported downstream probe training differs from training the NCM codec or generator. |

GitHub repository search for `"next-concept" mlx`, together with web searches for Next-Concept Modeling MLX implementations, did not identify another independently documented MLX port during this review. Search coverage and indexing are incomplete; this observation is not evidence for a “first implementation” claim. Other forks or unindexed work may exist.

## Methodological context

These papers explain relevant design families. They are not implemented benchmark baselines here unless an experiment explicitly says so.

| Work | Connection and distinction |
|---|---|
| [van den Oord et al., *Neural Discrete Representation Learning* (2017)](https://arxiv.org/abs/1711.00937) | Introduces VQ-VAE discrete representation learning. Discrete codes alone do not imply NCM's multiscale hierarchy or parallel level generator. |
| [Tian et al., *Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction* (2024)](https://arxiv.org/abs/2404.02905) | Generates images across resolution scales with parallel prediction within a scale. It provides a useful architectural comparison for coarse-to-fine latent generation; its image results are not text-generation evidence. |
| [Gu et al., *Non-Autoregressive Neural Machine Translation* (2017/2018)](https://arxiv.org/abs/1711.02281) | Earlier parallel conditional token generation using latent fertilities and distillation. This establishes broader non-autoregressive context; it is not NCM's two-stage residual VQ method. |
| [Sahoo et al., *Simple and Effective Masked Diffusion Language Models* (2024)](https://arxiv.org/abs/2406.07524) | Masked discrete diffusion iteratively unmasks token sequences. NCM instead predicts levels of a learned latent hierarchy and decodes the accumulated representation. Both support parallel work, but their steps, sequence lengths and quality costs differ. |

Use synchronized wall-clock timing, matched budgets and task-specific quality measures when comparing these families. Counting generator forwards alone cannot establish equal computation or useful output quality. See [benchmark methodology](benchmarks.md).

## Provenance and maintenance

NCM-MLX copies no upstream NCM source. Its MIT license covers this implementation; upstream code, checkpoints and datasets retain their own terms. See [CITATION.cff](../CITATION.cff), the [source map](research/upstream-map.md) and [reproducibility guide](reproducibility.md). Descriptions of other repositories are limited to their inspected revisions and should be rechecked before adopting their code or artifacts.

Update this comparison when a material implementation feature or compatibility claim changes. Keep measured NCM-MLX scores in the experiment log rather than duplicating a changing leaderboard here.

## Inference optimization literature

The [prefix-caching and speculative-drafting review](research/prefix-caching-and-speculative-drafting.md) examines primary literature on K/V reuse, speculative verification, drafter alignment and parallel proposals, with a versioned source ledger. Its [experiment protocol](research/inference-experiment-protocol.md) separates equivalent cache execution from changes to generation policy and future text-model training. These are research proposals rather than new measured capabilities.
