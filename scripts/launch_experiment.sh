#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cfg=$1
shift
name=$(basename "$cfg" .json)
label=com.ncm-mlx.$name
domain=gui/$(id -u)
mkdir -p runs
if launchctl print "$domain/$label" >/dev/null 2>&1; then
  echo "job already loaded: $domain/$label; inspect it before launching again" >&2
  exit 1
fi
if pgrep -f '[p]ython.*-m ncm_mlx (train|eval|bench)' >/dev/null; then
  echo "another local NCM job is running" >&2
  exit 1
fi
python3 - "$PWD" "$cfg" "$label" "$@" <<'PY'
import os
import plistlib
import sys
from pathlib import Path

root, cfg, label, *args = sys.argv[1:]
name = Path(cfg).stem
job = {
    "Label": label,
    "ProgramArguments": ["/bin/bash", f"{root}/scripts/run_experiment.sh", cfg, "24000", *args],
    "WorkingDirectory": root,
    "EnvironmentVariables": {"PATH": os.environ["PATH"], "NCM_MLX_MEMORY_LIMIT_GB": "20"},
    "RunAtLoad": True,
    "KeepAlive": False,
    "ExitTimeOut": 300,
    "StandardOutPath": f"{root}/runs/{name}-launcher.log",
    "StandardErrorPath": f"{root}/runs/{name}-launcher.log",
}
Path(f"{root}/runs/{name}.plist").write_bytes(plistlib.dumps(job))
PY
launchctl bootstrap "$domain" "$PWD/runs/$name.plist"
echo "launched $domain/$label"
