# AGENTS.md — ncm-mlx

Independent MLX research implementation of Next-Concept Modeling: multiscale residual vector quantization followed by parallel coarse-to-fine generation. The original method is [Kumbong et al., arXiv:2610.14162](https://arxiv.org/abs/2610.14162); this repository does not invent it or copy upstream code.

Training and inference run on Apple Silicon with Metal and on Linux with MLX's NVIDIA CUDA backend. Recorded CUDA experiments use an RTX 3080 and A40; this does not establish support for every NVIDIA GPU. Portable data/configuration tests run without MLX. The CLI requires an accelerator; CPU execution is used for exactness tests. This is experimental TinyWorld and Sudoku software, without instruction following, production chat serving, or real-text validation. The released TinyStories and Sudoku checkpoints run through a separate inference-only port ([official checkpoints](docs/official-checkpoints.md)); other released checkpoints are unsupported. See [capabilities](docs/capabilities.md) for scope and [platforms](docs/platforms.md) for setup.

Read this file, then `docs/STATUS.md` (what is true now) and `docs/NEXT.md` (what to do next).

## Commands

| Task | Command |
|---|---|
| Install on Apple Silicon | `mise trust && mise install && mise run setup` |
| Install on Linux NVIDIA | `uv sync --locked --python 3.12 --group cuda` (CUDA headers: `docs/platforms.md`) |
| Accelerator + shape check | `uv run python -m ncm_mlx smoke` |
| Tests | `uv run pytest -q` (MLX tests skip when MLX is absent) |
| Lint | `uv run ruff check .` |
| Train quickstart model | `mise run train` (~5.5 min on M5 Pro) |
| Coarse-to-fine samples | `mise run demo` |
| Quality metrics | `mise run eval` |
| Latency benchmark | `mise run bench` |
| AR baseline | `python -m ncm_mlx baseline --bench` |
| Released checkpoints | `python -m ncm_mlx official fetch|convert|generate|sudoku|bench --task TASK` (see `docs/official-checkpoints.md`) |
| Resume full training state | `python -m ncm_mlx train --config CONFIG --resume` (read `docs/RECOVERY.md`) |

The `mise` tasks use the TinyWorld quickstart. Pass `--config configs/NAME.json` to CLI commands for other recipes. `demo` starts training when its checkpoint is absent; check artifacts before running it. Several Sudoku configs reuse a codec checkpoint, so they are not self-contained fresh-checkout commands. See the training guide and experiment log for prerequisites. Dependencies are pinned in `uv.lock`; accelerator MLX is 0.32.3.

## Map

| Path | Role |
|---|---|
| `ncm_mlx/text/` | Character tokenizer; TinyWorld grammar; Sudoku synthesis, checker and upstream-board loading |
| `ncm_mlx/latent/` | Resampling matrices, multiscale residual VQ, codec (encoder/decoder) |
| `ncm_mlx/models/` | Shared transformer block; next-concept generator (block-diagonal) |
| `ncm_mlx/generation/` | Level-by-level sampler |
| `ncm_mlx/training/` | Two-stage training, checkpoint I/O |
| `ncm_mlx/eval/` | Codec / sample metrics, synchronized benchmark |
| `ncm_mlx/baselines/` | Character-level autoregressive baseline |
| `ncm_mlx/official/` | Inference port of released upstream checkpoints: torch-free reader, converter, BPE, model, sampler, CLI |
| `configs/` | Run configs (JSON, validated on load) |
| `docs/` | `STATUS`, `DECISIONS`, `EXPERIMENTS`, `NEXT`, `adr/`, `research/upstream-map.md` |
| `artifacts/`, `runs/` | Weights and logs — gitignored |

## Invariants — do not break

- Per-level inference must equal the teacher-forced training forward. `test_level_forward_equals_teacher_forced_forward_exactly_on_cpu` guards this; never loosen it.
- Generation performs exactly one generator forward per level (`test_generation_uses_one_forward_per_level`).
- Keep configuration differences explicit: TinyWorld uses a contextual encoder and LayerNorm; later Sudoku configs enable a lookup-only encoder, GroupNorm, learned level-0 resampling, token-prefix conditioning and conditioning corruption. Check the config before describing the architecture.
- Benchmarks time through a host copy of the result. Never report enqueue-only time.
- On Metal, fp32 matmul is reduced precision (~1e-3 relative). Exactness tests run on `mx.cpu`; GPU tests use a tolerance.
- Codec accuracy and generator quality are separate numbers. Do not report one as the other.
- `ncm_mlx/official/` must keep CPU parity with upstream PyTorch (`tests/test_official.py`, `scripts/official_parity.py`); converter rules fail on any unmapped or missing tensor.
- No upstream NCM code is copied. If that ever changes, keep upstream's MIT notice next to the copied code.

## Working rules

- Do not stop, restart or reconfigure other local services or GPU jobs on the host machine. Check GPU load before benchmarking and record it: `ioreg -r -d 1 -c IOAccelerator | grep 'Device Utilization'` on Metal; `nvidia-smi` on CUDA. Do not start a competing GPU workload during an active experiment.
- Record every measured run in `docs/EXPERIMENTS.md` with config, steps, wall time and GPU contention. Preserve negative results, evaluation sample counts, unmatched-budget qualifications and recovery limitations. Best-validation checkpoints do not replace held-out generation evaluation.
- Read `docs/RECOVERY.md` before interruption or recovery. Full-state resume and weights-only continuation are different experiments; never describe reset optimizer/random state as exact recovery. Do not edit an executing shell script in place.
- Commits: one imperative subject line, no body, no AI attribution.
- Public pushes need explicit human approval; run a secret scan and read the staged diff first.
- Keep private paths, credentials and infrastructure identifiers out of tracked files and generated documents. Artifacts and local-only operational notes stay ignored. Update capabilities and benchmark summaries when supported behavior changes, rather than after every run.
