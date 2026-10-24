---
title: "ADR 0001: native independent MLX implementation for v0"
status: accepted
date: 2026-10-22
---

# ADR 0001: native independent MLX implementation for v0

## Context

The goal for v0 is a coarse-to-fine text generator that genuinely runs on MLX/Metal on an M5 Pro today, with honest measurements. Three routes were considered.

| Option | What it means | For | Against |
|---|---|---|---|
| **A. Native independent implementation** | Implement the NCM concepts (multiscale residual VQ, level-parallel generator) from scratch in MLX, train small models locally | Runs today; every line is understood and tested; free to pick MLX-friendly operators | Not checkpoint compatible; results are toy scale |
| B. Faithful MLX checkpoint port | Reimplement the exact released configuration and convert `.pt` weights | Real pretrained quality; direct comparison with the paper | Needs BPE tokenizer, learned resampling, tied heads, GroupNorm parity and tensor-by-tensor tests before any output; multi-day and blocks the demo |
| C. PyTorch MPS reference | Run upstream code on `mps` | Least work if it runs | Not MLX; upstream assumes CUDA (flex attention, distributed); would not answer the question of a native runtime |

## Decision

Option A for v0. Option B becomes a separate `reference-compatible` variant in milestone M4, gated on M1–M3, never mixed with the experimental variant. Option C is not pursued.

## Consequences

- No upstream code is copied; attribution is by citation.
- All quality numbers are for models trained here on a synthetic corpus and must be labelled as toy results.
- The codec deliberately differs from upstream where small scale needs it (contextual encoder, dead-code restart); differences are listed in `docs/research/upstream-map.md`.
