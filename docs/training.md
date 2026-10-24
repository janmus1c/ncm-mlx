---
title: Train NCM from scratch with MLX
description: Reproduce the two-stage TinyWorld and Sudoku training workflows, reuse a compatible codec, evaluate generated samples, and recover interrupted training.
status: current
updated: 2026-10-24
---

# Train NCM from scratch with MLX

Training has two stages: learn a reconstructing codec, then freeze it and train a next-concept generator on its discrete codes. Supported tasks are the synthetic TinyWorld grammar and conditional Sudoku. This repository does not train general instruction-following language models or reproduce the original paper's real-text experiments.

Use the [platform setup](platforms.md) before these commands. Ordinary training and evaluation require a supported GPU. Check for existing GPU jobs before launching; do not run a benchmark alongside training or stop unrelated services to obtain a cleaner number.

## Reproduce the TinyWorld quickstart

```bash
mise trust
mise install
mise run setup
mise run smoke
python -m ncm_mlx train --config configs/toy-grammar-12k.json
python -m ncm_mlx demo --config configs/toy-grammar-12k.json --n 4
mkdir -p runs
python -m ncm_mlx eval --config configs/toy-grammar-12k.json \
  --temperatures 1.0 0.7 0.0 --out runs/tinyworld-eval.json
```

The quickstart uses 12,000 training sentences, 1,000 validation sentences, six latent levels, and 3,000 updates per stage. A previous measured run took about 5.5 minutes on an Apple M5 Pro; runtime depends on hardware, software version and contention. The demonstration command trains automatically if no generator checkpoint exists, so use `eval` to inspect existing weights without triggering training.

## Reproduce the Sudoku codec before a generator experiment

`configs/sudoku-e6.json` describes the shared Sudoku codec used by later experiments, but points `codec_checkpoint` at its already-trained file. A fresh checkout has no model weights. Make a training configuration with that field cleared and a separate output directory:

```bash
python - <<'PY'
import json
from pathlib import Path

cfg = json.loads(Path("configs/sudoku-e6.json").read_text())
cfg["name"] = "sudoku-codec-reproduction"
cfg["out_dir"] = "artifacts/sudoku-codec-reproduction"
cfg["codec_checkpoint"] = None
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/sudoku-codec-reproduction.json").write_text(
    json.dumps(cfg, indent=2) + "\n"
)
PY
python -m ncm_mlx train --config artifacts/sudoku-codec-reproduction.json \
  --generator-steps 1
```

This trains the 3,000-step codec and a one-step placeholder generator. For a later experiment, copy its JSON configuration, set `codec_checkpoint` to `artifacts/sudoku-codec-reproduction/codec.safetensors`, and give the reproduction a new `out_dir`. Preserve the codec's architecture and tokenizer settings. Regenerating a codec does not guarantee bitwise equality with the historical frozen E6 codec across backends.

E13 uses 32,000 generator updates × batch 64 = 2,048,000 sequence presentations. E15 uses 128,000 × 64 = 8,192,000 and adds four million greedily constructed unique-solution puzzles. Presentations count repeated examples; they are not counts of distinct puzzles. These are substantial research runs, not quickstart requirements. Configurations E10/E12 additionally use wider generators. No published model weights are required to run the synthetic-data workflow.

The base Sudoku puzzles come from independently generated solution grids with random clue removal. They may admit multiple solutions. Optional extras either use greedy uniqueness-preserving removal or retain only the unique subset of random-removal puzzles. These constructions have different distributions; record which one a run uses.

## Evaluate on the published test boards

```bash
scripts/fetch_upstream_sudoku.sh
python -m ncm_mlx eval --config configs/YOUR-RUN.json \
  --temperatures 0.0 --out runs/YOUR-RUN-eval.json
python -m ncm_mlx baseline --config configs/YOUR-RUN.json \
  --temperatures 0.0 --out runs/YOUR-RUN-ar.json
```

Without downloaded test boards, evaluation reports only the generated validation split. The upstream loader removes row separators and beginning-of-sequence tokens to preserve the 81 board cells. It refuses pickle globals. Downloaded test boards and all weights remain outside version control. Check the [dataset provenance](../THIRD_PARTY.md) before redistributing external artifacts.

`baseline` trains if no AR weights exist. By default it uses the generator's update count, batch size, width and depth, but it does not pay the codec training cost. Use the same corpus and data budget for a controlled quality comparison, and report total system parameters and training cost separately. See [benchmarks](benchmarks.md) for evaluation definitions and limitations.

## Configuration and outputs

JSON configurations are validated by [`Config`](../ncm_mlx/config.py). Important controls are codec shape, generator shape and attention pattern, corruption schedule, batch size, update counts, seed, checkpoint interval and generator training compilation. The learning rate warms up and then follows cosine decay to 5% of its peak; AdamW uses weight decay. `train.compile` applies only to generator training.

Training saves the resolved configuration, tokenizer metadata, inference weights and full training state under `out_dir`. Generator dataset encoding happens before its timed training loop. Training-loop time includes work inside the loop such as validation and periodic saving; end-to-end pipeline time also includes data preparation, evaluation and benchmarking. Record both when available.

The generator selects its inference checkpoint by minimum weighted teacher-forced validation cross entropy. That criterion is distinct from solved-board rate or generated-sentence validity. The full training snapshot follows the latest training step and may differ from the best inference model.

## Interruptions and reproducibility

Use a fresh output directory for an independent run. `--resume` restores full training state; it is not a way to change the total schedule or configuration. Legacy `--generator-checkpoint` recovery resets optimizer and random state and must be labelled as weights-only continuation. The [recovery guide](RECOVERY.md) gives pause, restart and replacement-machine procedures, including session-independent Mac execution and limits of ephemeral storage.

Record the code revision, resolved config, dataset construction and seeds, MLX/Python versions, device, complete update budget, best step, checkpoint hashes, timing boundaries and GPU contention in [EXPERIMENTS.md](EXPERIMENTS.md). Preserve unsuccessful runs and negative findings. Short CPU recovery tests establish exact continuation in their tested environment; they do not establish cross-device or cross-backend bitwise identity.
