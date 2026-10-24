#!/usr/bin/env bash
set -euo pipefail
host=${1:?host}; cfg=${2:?config}; shift 2
max_hours=12; train_minutes=0; resume=0
while [ $# -gt 0 ]; do
  case $1 in
    --max-hours) max_hours=$2; shift 2 ;;
    --train-minutes) train_minutes=$2; shift 2 ;;
    --resume) resume=1; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
cd "$(dirname "$0")/.."
trap 'echo "remote_run: failed at line $LINENO (exit $?)" >&2' ERR
step() { echo "$(date '+%T') remote_run: $*"; }
name=$(basename "$cfg" .json)
port=22
case $host in *:*) port=${host##*:}; host=${host%:*} ;; esac
ssh() { command ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=60 -p "$port" "$@"; }
rsync() { command rsync -e "ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -p $port" "$@"; }
finish() {
  status=$?
  rsync -az "$host:ncm-mlx/runs/$name-*" runs/ || true
  rsync -az "$host:ncm-mlx/artifacts/$name" artifacts/ || true
  exit "$status"
}
busy_status=0
ssh "$host" "pgrep -f '[p]ython.*-m ncm_mlx (train|eval|bench)' >/dev/null" || busy_status=$?
if [ "$busy_status" -eq 0 ]; then
  echo "an NCM job is already running on $host; refusing to replace its files" >&2
  exit 1
fi
[ "$busy_status" -eq 1 ] || { echo "cannot inspect $host (ssh exit $busy_status)" >&2; exit 1; }
trap finish EXIT

codec=$(python3 -c "import json; print(json.load(open('$cfg')).get('codec_checkpoint') or '')")
extra=(artifacts/cache)
[ -d artifacts/upstream-sudoku ] && extra+=(artifacts/upstream-sudoku)
[ -n "$codec" ] && extra+=("$codec" "${codec%.safetensors}.json")
resume_args=""
steps=""
if [ "$resume" -eq 1 ]; then
  [ -d "artifacts/$name" ] || { echo "missing local recovery directory artifacts/$name" >&2; exit 1; }
  steps=$(python3 scripts/checkpoint_steps.py "artifacts/$name")
  extra+=("artifacts/$name")
  resume_args=--resume
fi
step "prepare $host"; ssh "$host" "mkdir -p ncm-mlx && { command -v rsync >/dev/null || { apt-get update -qq && apt-get install -y -qq rsync >/dev/null; }; }"
step "copy code"; rsync -az --delete --exclude .venv --exclude runs --exclude artifacts --exclude .git --exclude .private --exclude scripts/cloud --exclude .idea --exclude "*.local.md" --exclude ".env*" ./ "$host:ncm-mlx/"
step "copy data: ${extra[*]}"; rsync -azR "${extra[@]}" "$host:ncm-mlx/"
width=$(python3 -c "import json; print(json.load(open('$cfg'))['generator']['n_embd'])")
batch=$(python3 -c "import json; print(json.load(open('$cfg'))['train']['batch_size'])")
ssh "$host" "cd ncm-mlx && mkdir -p runs && scripts/check_gpu.sh ${width}x$batch > runs/$name-gpu-check.log 2>&1 && tail -4 runs/$name-gpu-check.log"
if [ "$train_minutes" != 0 ] && [ "$resume" -eq 0 ]; then
  rate=$(ssh "$host" "grep -oE 'width $width batch [0-9]+: [0-9]+ ms/step, [0-9]+ sequences/s' ncm-mlx/runs/$name-gpu-check.log" | grep -oE '[0-9]+ sequences/s' | grep -oE '[0-9]+' | sort -n | tail -1)
  steps=$(( ${rate:?no throughput measured for width $width} * train_minutes * 60 * 9 / 10 / batch ))
  echo "throughput $rate sequences/s at width $width: $steps steps of $batch for $train_minutes minutes"
fi
ssh "$host" "cd ncm-mlx && NCM_MLX_GENERATOR_STEPS=$steps timeout ${max_hours}h scripts/run_experiment.sh $cfg 24000 $resume_args"
echo "results in runs/$name-* and artifacts/$name/"
