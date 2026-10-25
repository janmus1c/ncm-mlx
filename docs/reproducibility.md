---
title: Reproduce training and evaluation
description: Reproduce NCM-MLX experiments from configs, pinned dependencies and versioned Sudoku evaluation boards.
---

# Reproduce training and evaluation

Results refer to a config, training budget, selected checkpoint, evaluation procedure and hardware. A matching model shape alone does not reproduce a result. Read [platform setup](platforms.md) before using the commands below and [benchmark methodology](benchmarks.md) before comparing latency or quality.

## TinyWorld quickstart

```bash
uv sync --locked --python 3.12
uv run python -m ncm_mlx smoke
uv run python -m ncm_mlx train --config configs/toy-grammar-12k.json
uv run python -m ncm_mlx demo --config configs/toy-grammar-12k.json
uv run python -m ncm_mlx eval --config configs/toy-grammar-12k.json
uv run python -m ncm_mlx bench --config configs/toy-grammar-12k.json
uv run python -m ncm_mlx baseline --config configs/toy-grammar-12k.json --bench
```

TinyWorld is generated from the grammar in `ncm_mlx/text/toygrammar.py`; no external text dataset is downloaded. The config selects deterministic train and validation splits. Training writes weights and metadata to its `out_dir` under `artifacts/`. The autoregressive baseline trains separately. Repeat runs may differ across hardware and library versions; seeds are not a cross-backend reproducibility guarantee.

## Sudoku evaluation data

```bash
scripts/fetch_upstream_sudoku.sh
```

The downloader fetches only the six puzzle/solution files from `hazyresearch/ncm-datasets` at revision `6580ba41a4b0a976dde876fbe84d25327617b0e6`. It verifies SHA-256 hashes from `scripts/sudoku.sha256` before replacing each destination. There are 2,000 boards per difficulty. `ncm_mlx/text/upstream_sudoku.py` reads the tokenized lists with a restricted unpickler that refuses Python globals. These boards are evaluation inputs, not the locally generated training set. See [third-party provenance](../THIRD_PARTY.md).

Sudoku configs can reuse a trained codec via `codec_checkpoint`; inspect that field before launching a generator experiment. Weight files are not included in the release. Reproduce the referenced codec training first, then use the generator config and recorded step budget from [EXPERIMENTS.md](EXPERIMENTS.md). E15 uses 8.192M generator presentations. The newer E19-funded result requires `configs/sudoku-e19-funded.json`, the frozen E6 codec, eight million additional cached unique puzzles and 128,000 generator updates at batch 128 (16.384M presentations). Its best-validation checkpoint is step 127,000. Neither result is expected from the short quickstart. The [completion report](research/e19-funded-report.md) records checkpoint identity, backend, timings and uncertainty.

```bash
uv run python -m ncm_mlx train --help
uv run python -m ncm_mlx eval --help
uv run python -m ncm_mlx bench --help
```

For a completed Sudoku checkpoint, evaluate with `--temperatures 0 --seeds 1` and retain the complete JSON report. Keep validation and the three published test difficulties separate. Report codec reconstruction independently from the generator's solved-board rate.

## Record a run

Preserve the source revision, complete config, dependency lock, backend and GPU model, initial GPU utilization, seeds, training steps, actual sequence budget, wall time and checkpoint-selection rule. Store raw JSON reports locally under `runs/` and weights under `artifacts/`; these directories are ignored by Git. Add the measured result and conditions to `docs/EXPERIMENTS.md`, and update the committed, sanitized `paper/data/results.json` when publishing a paper milestone. Running `uv run --group paper python paper/make_figures.py` collects local reports and rebuilds figures; `--plot` uses the committed data without collecting reports.

`scripts/run_experiment.sh CONFIG` runs training, evaluation and synchronized benchmarks under the memory watchdog. Mac runs default to a 20 GiB MLX memory limit and a 24,000 MB process-tree watchdog; choose a workload that fits rather than interpreting the watchdog as an allocation guarantee. Check for other accelerator workloads before launching anything.

## Interruptions and continued training

Use full-state checkpoints and the [pause/resume procedure](RECOVERY.md). The recovery tests compare short interrupted and uninterrupted trajectories on the same CPU backend, including compiled generator steps. They do not establish exactness across CPU, Metal and CUDA or across different MLX versions. Keep an external copy of checkpoints if machine or disk loss matters.

E17 is a documented exception: its legacy checkpoint contained weights only. Its last 2,000 steps reset optimizer and random state, so it is not an exact continuation of the uninterrupted trajectory. Preserve that qualification in any comparison.
