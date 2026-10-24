---
title: Metal and CUDA setup
description: Install NCM-MLX on Apple Silicon or verified NVIDIA CUDA hardware and understand portable test coverage.
---

# Metal and CUDA setup

NCM-MLX uses MLX 0.32.3 on Apple Silicon through Metal and on NVIDIA GPUs through MLX's Linux CUDA backend. Training and evaluation have run on an Apple M5 Pro, NVIDIA RTX 3080 and NVIDIA A40. This is evidence for those configurations, rather than a claim that every supported MLX device has been tested. The experiment log records hardware with each result.

Python 3.11 or later is declared; the reproducible development environment pins Python 3.12.14. Ordinary Linux CI installs no MLX and tests the portable configuration, tokenizer, data, scoring, process-control and remote-runner components. GPU execution and CPU exactness tests require MLX and are skipped when it is absent. Windows and Intel Mac GPU execution are unsupported.

## Apple Silicon and Metal

Use native ARM64 Python on macOS 14 or later. A Python process running through Rosetta is unsuitable. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and clone the repository, then:

```bash
uv sync --locked --python 3.12
uv run python -m ncm_mlx smoke
uv run pytest -q
```

The platform marker installs MLX automatically on macOS ARM64. Alternatively, `mise install && mise run setup` selects the pinned Python version. `mise run smoke` and `mise run test` use that environment.

## NVIDIA CUDA on Linux

The CUDA 12 backend needs an NVIDIA GPU with compute capability at least 7.5, driver at least 550.54.14 and a Linux distribution with glibc at least 2.35. The [MLX installation guide](https://ml-explore.github.io/mlx/build/html/install.html) documents upstream requirements. This repository pins CUDA runtime and CCCL headers to the 12.9 series to match the locked NVRTC runtime compiler.

```bash
uv sync --locked --python 3.12 --group cuda
scripts/check_gpu.sh
export CUDA_HOME="$PWD/.venv/cuda-home"
uv run python -m ncm_mlx smoke
```

`scripts/check_gpu.sh` checks the GPU, driver and glibc, installs the locked CUDA environment, assembles matching headers under `.venv/cuda-home`, runs the tests, and times short Sudoku training steps. Run it only on an idle host where this work is authorized; it is an accelerator workload, rather than a read-only health probe. `scripts/run_experiment.sh` discovers this header directory automatically. Direct CLI invocations need the `CUDA_HOME` export in each shell.

The `cuda` dependency group is opt-in because the NVIDIA libraries are large. `uv sync --locked` on Linux alone installs portable dependencies, without a GPU backend. `pip install .` has the same limitation on Linux; use the checked-out, locked CUDA workflow above. CUDA support does not imply binary compatibility with upstream PyTorch checkpoints or identical floating-point results across backends.

## Portable development without a GPU

```bash
uv sync --locked --python 3.12
uv run ruff check .
uv run pytest -q
uv run python scripts/validate_repository.py
uv build
```

On Linux without the CUDA group these commands exercise the portable suite and packaging. Training, sampling, evaluation and benchmarking require an installed MLX GPU backend; the CLI checks that a GPU is selected. Exactness tests explicitly use MLX's CPU device, while GPU comparisons allow floating-point tolerance.

See [reproducibility](reproducibility.md), [benchmarks](benchmarks.md) and [remote execution](CLOUD.md) for workloads and measurement conditions.
