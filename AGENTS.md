# AGENTS.md — ncm-mlx

Independent MLX research implementation of coarse-to-fine next-concept text generation (inspired by HazyResearch NCM, arXiv:2610.14162). Apple Silicon only for anything that touches MLX.

Read this file, then `docs/STATUS.md` (what is true now) and `docs/NEXT.md` (what to do next).

## Commands

| Task | Command |
|---|---|
| Install | `mise install && mise run setup` |
| Metal + shape check | `mise run smoke` |
| Tests | `mise run test` |
| Train quickstart model | `mise run train` (~5.5 min on M5 Pro) |
| Coarse-to-fine samples | `mise run demo` |
| Quality metrics | `mise run eval` |
| Latency benchmark | `mise run bench` |
| AR baseline | `python -m ncm_mlx baseline --bench` |

## Map

| Path | Role |
|---|---|
| `ncm_mlx/text/` | Character tokenizer; TinyWorld synthetic grammar and its checker |
| `ncm_mlx/latent/` | Resampling matrices, multiscale residual VQ, codec (encoder/decoder) |
| `ncm_mlx/models/` | Shared transformer block; next-concept generator (block-diagonal) |
| `ncm_mlx/generation/` | Level-by-level sampler |
| `ncm_mlx/training/` | Two-stage training, checkpoint I/O |
| `ncm_mlx/eval/` | Codec / sample metrics, synchronized benchmark |
| `ncm_mlx/baselines/` | Character-level autoregressive baseline |
| `configs/` | Run configs (JSON, validated on load) |
| `docs/` | `STATUS`, `DECISIONS`, `EXPERIMENTS`, `NEXT`, `adr/`, `research/upstream-map.md` |
| `artifacts/`, `runs/` | Weights and logs — gitignored |

## Invariants — do not break

- Per-level inference must equal the teacher-forced training forward. `test_level_forward_equals_teacher_forced_forward_exactly_on_cpu` guards this; never loosen it.
- Generation performs exactly one generator forward per level (`test_generation_uses_one_forward_per_level`).
- Benchmarks time through a host copy of the result. Never report enqueue-only time.
- On Metal, fp32 matmul is reduced precision (~1e-3 relative). Exactness tests run on `mx.cpu`; GPU tests use a tolerance.
- Codec accuracy and generator quality are separate numbers. Do not report one as the other.
- No upstream NCM code is copied. If that ever changes, keep upstream's MIT notice next to the copied code.

## Working rules

- Do not stop, restart or reconfigure other local services or GPU jobs on the host machine. Check GPU load (`ioreg -r -d 1 -c IOAccelerator | grep 'Device Utilization'`) before benchmarking and record it.
- Record every measured run in `docs/EXPERIMENTS.md` with config, steps, wall time and GPU contention.
- Commits: one imperative subject line, no body, no AI attribution.
- Public pushes need explicit human approval; run a secret scan and read the staged diff first.
