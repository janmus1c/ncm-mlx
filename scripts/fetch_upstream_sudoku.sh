#!/usr/bin/env bash
# Download the NCM Sudoku test boards (MIT, hazyresearch/ncm-datasets) into artifacts/.
set -euo pipefail
dest="${1:-artifacts/upstream-sudoku}"
base="https://huggingface.co/datasets/hazyresearch/ncm-datasets/resolve/main/sudoku"
for level in easy medium hard; do
  mkdir -p "$dest/$level"
  for file in puzzles solutions; do
    curl -fsSL -o "$dest/$level/$file.pickle" "$base/$level/$file.pickle"
  done
done
echo "saved to $dest"
