#!/usr/bin/env bash
# Run a long GPU job with a memory watchdog on its own process tree (macOS).
#
#   scripts/guarded_run.sh [-m MAX_MB] [-i SECONDS] [-l LOG] -- command [args...]
#
# Every interval the watchdog sums `footprint` (physical footprint, which
# includes Metal/unified-memory buffers) over the command and all its
# descendants, appends one line to LOG, and kills the tree if the total
# exceeds MAX_MB. It watches only this job, so other applications' memory use
# cannot trigger it. Exit status is the command's, or 137 if the watchdog
# stopped it.
set -uo pipefail

max_mb=24000
interval=30
log=""
while getopts "m:i:l:h" opt; do
  case "$opt" in
    m) max_mb=$OPTARG ;;
    i) interval=$OPTARG ;;
    l) log=$OPTARG ;;
    *) sed -n '2,11p' "$0"; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { sed -n '2,11p' "$0"; exit 2; }
log=${log:-/dev/stderr}

descendants() {
  local pid=$1
  echo "$pid"
  for child in $(pgrep -P "$pid"); do descendants "$child"; done
}

footprint_mb() {
  footprint -p "$1" 2>/dev/null | awk '/phys_footprint:/ {
    v = $2; if ($3 == "GB") v *= 1024; else if ($3 == "KB") v /= 1024; print int(v); exit }'
}

"$@" &
job=$!
stopped=0
pause_job() {
  kill -USR1 "$job" 2>/dev/null || true
}
trap pause_job USR1 TERM INT

while kill -0 "$job" 2>/dev/null; do
  sleep "$interval"
  kill -0 "$job" 2>/dev/null || break
  total=0
  for pid in $(descendants "$job"); do
    mb=$(footprint_mb "$pid")
    total=$((total + ${mb:-0}))
  done
  echo "$(date +%Y-%m-%dT%H:%M:%S) footprint_mb=$total limit_mb=$max_mb" >> "$log"
  if [ "$total" -gt "$max_mb" ]; then
    echo "$(date +%Y-%m-%dT%H:%M:%S) over limit, stopping job $job" >> "$log"
    pids=$(descendants "$job")
    for pid in $pids; do kill -KILL "$pid" 2>/dev/null || true; done
    stopped=1
    break
  fi
done

wait "$job"
status=$?
[ "$stopped" -eq 1 ] && exit 137
exit "$status"
