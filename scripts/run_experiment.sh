#!/usr/bin/env bash
# Train, evaluate and benchmark one config under the memory watchdog.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${1:-}" != "--worker" ]; then
  cfg=$1
  max_mb=${2:-24000}
  shift
  [ $# -gt 0 ] && shift
  exec scripts/guarded_run.sh -m "$max_mb" -i 120 -l "runs/$(basename "$cfg" .json)-mem.log" -- \
    bash scripts/run_experiment.sh --worker "$cfg" "$@"
fi
shift
cfg=$1
shift
name=$(basename "$cfg" .json)
export PYTHONUNBUFFERED=1
if [ "$(uname)" = Darwin ]; then
  export NCM_MLX_MEMORY_LIMIT_GB=${NCM_MLX_MEMORY_LIMIT_GB:-20}
fi
[ -d .venv/cuda-home ] && export CUDA_HOME=$PWD/.venv/cuda-home
py=.venv/bin/python
train_args=(--config "$cfg" "$@")
if [ -n "${NCM_MLX_GENERATOR_STEPS:-}" ]; then
  train_args+=(--generator-steps "$NCM_MLX_GENERATOR_STEPS")
fi
child=
pause_child() {
  [ -z "$child" ] || kill -USR1 "$child" 2>/dev/null || true
}
trap pause_child USR1 TERM INT
run_stage() {
  "$@" &
  child=$!
  local status=0
  while :; do
    wait "$child" && status=0 || status=$?
    kill -0 "$child" 2>/dev/null || break
  done
  child=
  return "$status"
}
run_stage "$py" -m ncm_mlx train "${train_args[@]}" >> "runs/$name-train.log" 2>&1
run_stage "$py" -m ncm_mlx eval --config "$cfg" --temperatures 0 --seeds 1 \
  --out "runs/$name-eval.json" >> "runs/$name-eval.log" 2>&1
ioreg -r -d 1 -c IOAccelerator 2>/dev/null | grep 'Device Utilization' >> "runs/$name-bench-gpu.log" || true
run_stage "$py" -m ncm_mlx bench --config "$cfg" --batch-sizes 1 16 64 256 --iters 20 \
  --out "runs/$name-bench.json" >> "runs/$name-bench.log" 2>&1
