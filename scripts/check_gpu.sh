#!/usr/bin/env bash
# Validate a Linux NVIDIA host for ncm-mlx, then time one training step. Run it on the host,
# from a checkout of the repository: scripts/check_gpu.sh [WIDTHxBATCH...]
#
# Checks MLX's CUDA requirements (compute capability >= 7.5, driver >= 550, glibc >= 2.35),
# installs the `cuda` dependency group with uv, runs the test suite on the GPU and times a
# Sudoku training step. Exits non-zero at the first failed requirement.
set -euo pipefail
cd "$(dirname "$0")/.."
fail() { echo "FAIL: $*" >&2; exit 1; }
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }

command -v nvidia-smi >/dev/null || fail "nvidia-smi not found (no NVIDIA driver)"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,compute_cap --format=csv
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
glibc=$(ldd --version | head -1 | grep -oE '[0-9]+\.[0-9]+$')
ver_ge "$driver" 550 || fail "driver $driver < 550"
ver_ge "$cap" 7.5 || fail "compute capability $cap < 7.5"
ver_ge "$glibc" 2.35 || fail "glibc $glibc < 2.35"
echo "OK: driver $driver, compute capability $cap, glibc $glibc"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --python 3.12 --group dev --group cuda
# MLX's run-time kernel compiler needs CUDA headers matching its NVRTC (12.9), not the system's.
site=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")/nvidia
rm -rf .venv/cuda-home && mkdir -p .venv/cuda-home/include
for pkg in cuda_runtime cuda_cccl cuda_nvrtc; do cp -rs "$(realpath "$site/$pkg/include")/." .venv/cuda-home/include/; done
export CUDA_HOME=$PWD/.venv/cuda-home
.venv/bin/python -c "import mlx.core as mx; print('MLX', mx.__version__, mx.default_device()); a = mx.random.normal((2048, 2048)); mx.eval(a @ a); print('matmul ok')"
.venv/bin/python -m pytest -q 2>&1 | tail -3
.venv/bin/python scripts/step_timing.py "$@"
