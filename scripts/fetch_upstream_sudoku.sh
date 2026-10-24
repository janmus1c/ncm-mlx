#!/usr/bin/env bash
# Download the NCM Sudoku test boards (MIT, hazyresearch/ncm-datasets) into artifacts/.
set -euo pipefail
dest="${1:-artifacts/upstream-sudoku}"
revision=6580ba41a4b0a976dde876fbe84d25327617b0e6
base="https://huggingface.co/datasets/hazyresearch/ncm-datasets/resolve/$revision/sudoku"
python3 - "$dest" "$base" "$(dirname "$0")/sudoku.sha256" <<'PY'
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

dest, base, manifest = sys.argv[1:]
for line in Path(manifest).read_text().splitlines():
    expected, relative = line.split()
    data = urlopen(f"{base}/{relative}", timeout=60).read()
    if hashlib.sha256(data).hexdigest() != expected:
        raise SystemExit(f"SHA-256 mismatch: {relative}")
    target = Path(dest) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as f:
        temporary = f.name
        f.write(data)
    os.replace(temporary, target)
PY
echo "saved to $dest"
