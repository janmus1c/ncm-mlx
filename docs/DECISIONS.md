---
title: Decisions
status: current
---

# Decisions

Newest first. Long-form records live in `docs/adr/`.

| Date | Decision | Why |
|---|---|---|
| 2026-10-22 | Exactness tests run on `mx.cpu`; GPU tests use ~1e-3 tolerance | Measured on the M5 Pro: Metal fp32 matmul differs from float64 by ~1.4e-2 absolute on unit-variance 16-dim dot products (CPU: ~2e-6). A per-level vs full-sequence forward therefore differs by ~5e-4 on GPU purely from precision; on CPU it is exact. |
| 2026-10-22 | Character-level AR baseline with KV cache, same width/depth as generator | Speed claims need a baseline measured on the same data and hardware; the paper's numbers are vs. diffusion LMs on GPUs. |
| 2026-10-22 | Synthetic TinyWorld grammar as the first corpus | 15,360 sentences, automatically checkable (form + number agreement), held-out sentences measure generalisation vs. memorisation, no licensing or privacy questions. |
| 2026-10-22 | Character tokenizer, fixed length 32, levels `1,2,4,8,16,32`, 128 codes per level | Reconstruction accuracy is directly interpretable; levels divide the length so area pooling is exact. |
| 2026-10-22 | Contextual encoder (2 blocks) instead of upstream's lookup-only encoder | With a lookup encoder, coarse codes can only average local embeddings; a contextual encoder lets level 0 carry sentence-level content. Listed as a modification in `research/upstream-map.md`. |
| 2026-10-22 | EMA codebook update applied after the optimiser step, from statistics returned by the loss function | MLX `value_and_grad` is functional; in-place buffer mutation inside the traced loss is not an option. |
| 2026-10-22 | Dead-code restart on by default (`restart_threshold=0.05`) | Small dataset, 128 codes per level; collapse risk is higher than upstream's. |
| 2026-10-22 | Resampling as precomputed dense matrices | One batched matmul per level, no custom kernels; semantics pinned by tests to PyTorch's `area` and `linear` interpolation. |
| 2026-10-22 | Option A (native independent implementation) for v0 | See `adr/0001-native-independent-implementation.md`. |
| 2026-10-22 | MIT license | Dependencies are MIT (MLX) and BSD (NumPy); upstream is MIT and no upstream code is copied. |
| 2026-10-22 | Python 3.12.14 via mise, `mlx==0.32.3` pinned via uv | Same versions model-lab already uses on this machine: no new global installs. |
| 2026-10-23 | Learned level-0 sampler is one conv each way, codec lr 2e-4 | A four-layer cascade and a single layer both diverged at lr 1e-3; upstream's own Sudoku codec (L = 89, prime) also uses a single layer. |
| 2026-10-23 | Keep the name `ncm-mlx` | It says what the project is (an MLX implementation of NCM) and is what people would search for; a new name would only rename paths and docs. |
