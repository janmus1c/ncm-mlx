---
title: Training interruption and recovery
status: current
updated: 2026-10-24
---

# Training interruption and recovery

Codec and generator training save full state at each `train.log_every` boundary, at the final step, and after a requested pause. Use the same code, configuration, datasets, codec and MLX version when resuming. A weights file alone cannot reproduce an interrupted optimizer trajectory.

## Checkpoint contents

Each stage writes `<stage>-training.safetensors` and retains the preceding readable file as `<stage>-training.previous.safetensors`. A temporary file is written and flushed before atomic replacement; readers see a complete old or new snapshot. An unreadable latest snapshot falls back to the previous one, with a log message.

Full snapshots contain:

- Model parameters, including the codec's VQ codebooks and EMA accumulators.
- AdamW moments, optimizer step and learning-rate schedule position.
- MLX random key, shuffled-data RNG and position within its current epoch.
- Codec dropout's Python RNG and the VQ update key.
- Generator best-validation weights, best step and loss history.
- Configuration, target step count and elapsed training-loop time across restarts.

`generator.safetensors` remains the best-validation inference model. Full generator state holds the latest training step, which can differ from the best step. Copy both snapshots and the inference files when moving an experiment.

Checkpoint interval is measured in steps. `log_every: 100` bounds a hard crash's unsaved training work to at most 99 completed steps after the latest snapshot. A graceful pause also checkpoints between regular boundaries. Data preparation before the first training step is recomputed after an interruption.

## Start a Mac experiment independently of a terminal

```bash
scripts/launch_experiment.sh configs/sudoku-e17-recovery.json \
  --generator-checkpoint artifacts/sudoku-e17/generator.safetensors \
  --generator-start-step 30000
```

For a new experiment, omit the two generator arguments. The launcher refuses another local NCM training/evaluation/benchmark process. It uses launchd, a 20 GiB MLX allocation cap and a 24,000 MB process-tree watchdog. Logs and the launchd plist are under `runs/<config-stem>-*` and `runs/<config-stem>.plist`.

Closing the terminal or agent session does not own this process tree. The job is not configured to rerent GPUs, retry forever, restart after logout or resume automatically after a reboot. Inspect the last exit and checkpoint before restarting it.

```bash
launchctl print gui/$(id -u)/com.ncm-mlx.sudoku-e17-recovery
tail -n 8 runs/sudoku-e17-recovery-train.log
```

Do not edit executing shell scripts in place. Wait for the job to finish, or write a replacement file and rename it atomically; an existing shell then continues reading its original inode.

## Pause and resume

Send `SIGUSR1` to the training Python process, after checking its full command:

```bash
pgrep -fl '[p]ython.*-m ncm_mlx train'
kill -USR1 TRAINING_PID
```

Replace `TRAINING_PID` with the inspected numeric PID. The trainer completes its current step, evaluates if needed, saves full state, prints `paused ... full training state saved`, and exits 75. The experiment wrapper then skips evaluation and benchmarking. `SIGINT` and `SIGTERM` also request a checkpoint while the training loop is active. The wrappers forward these requests; `launchctl kill SIGUSR1 gui/$(id -u)/com.ncm-mlx.sudoku-e17-recovery` can take up to the watchdog's two-minute interval to reach Python.

Confirm the pause message and that the process exited before starting another job. Remove the finished launchd entry, then relaunch with `--resume`:

```bash
launchctl bootout gui/$(id -u)/com.ncm-mlx.sudoku-e17-recovery
scripts/launch_experiment.sh configs/sudoku-e17-recovery.json --resume
```

Direct foreground use on a machine where terminal lifetime is managed:

```bash
NCM_MLX_MEMORY_LIMIT_GB=20 \
  scripts/run_experiment.sh configs/sudoku-e17-recovery.json 24000 --resume
```

Resume requires an unchanged configuration and total step budget. The saved learning-rate offset is restored; warmup does not start over. A completed training snapshot can be resumed to regenerate final metadata and proceed to evaluation. Logs append across restarts. Preserve the original `NCM_MLX_GENERATOR_STEPS` override if one was used; `scripts/checkpoint_steps.py artifacts/NAME` reads its value without importing MLX.

`SIGKILL`, power loss and a destroyed container cannot execute a final save. The memory watchdog deliberately uses a hard kill when the process tree exceeds its limit. Recover from the last complete snapshot and report how much work was replayed; uncheckpointed progress is unknown unless another log records it.

## Recover on another machine or replacement pod

First check that the original job is stopped. Preserve the config, all stage snapshots, frozen codec and its metadata, caches required by the data configuration, and logs. Record SHA-256 hashes before copying. Keep the same relative paths in the replacement checkout.

```bash
scripts/remote_run.sh HOST:PORT configs/NAME.json --resume --max-hours 12
```

The remote runner refuses an existing NCM process before replacing code. With `--resume`, it uploads `artifacts/NAME`, retains the saved generator step budget instead of sizing a new schedule, and downloads artifacts/logs on both success and failure. This convention requires the config's `out_dir` to be `artifacts/<config-stem>`. Use a session-independent orchestrator on the Mac, and a bounded runtime on the remote machine.

Store snapshots outside an ephemeral container at a documented interval and once more at shutdown when the machine remains reachable. If it disappears permanently, only the last successful external copy is recoverable: potential loss is the copy interval plus the training checkpoint interval. A checkpoint stored only inside a deleted container does not survive it. The remote runner copies state at exit; periodic external backup must be configured by the operator. No remote-machine crash or cross-backend recovery has been tested yet.

## Legacy weights-only recovery: E17

On 2026-10-24, closing the terminal terminated its background launcher and E17 child tree. The launcher's task record says `killed` at 15:05:17 CEST; macOS records the trainer disconnecting at the same time and the launcher exiting at 15:05:19. A live snapshot had recorded the trainer at 15:04:26. The watchdog never logged `over limit`; the nearby memory-warning event belonged to XProtect.

The last durable model is step 30,000 of 32,000, saved at 14:49:25. Validation CE was 0.2217. All 143 tensors / 6,440,128 float32 parameters were present and finite. Its SHA-256 is `75b5fd8e816f26f7b48065c3d547a535e8bd6becab0df9f8195f7a749e32905d`. Original logs and weights were copied into timestamped `runs/sudoku-e17-interrupted-20261024-150517/` and `artifacts/sudoku-e17-interrupted-20261024-150517/` directories before recovery.

The old trainer saved no AdamW state or random key, so exact continuation is impossible from that file. `configs/sudoku-e17-recovery.json` writes to a separate directory, loads the saved weights and runs global steps 30,001–32,000. It retains the original cosine schedule position and shuffled-batch position, starts fresh optimizer moments and MLX random state, validates the initial model, and keeps it if later validation worsens. Validation/checkpoint frequency changes from 2,000 to 100 steps. The continuation finished all 32,000 steps; its best model is step 31,900 and solves 71.25 / 31.75 / 3.70% of easy/medium/hard boards. The 2,000 continuation steps took 1,247.53 s; the full recovery pipeline finished at 15:55:24 CEST. Future interruptions of runs using this code can use full-state `--resume`.

This is a weights-only continuation, not an uninterrupted E17 run. The number of steps performed after the last old checkpoint is unknown and must not be counted as durable progress. Any recovered quality result must be labelled with the optimizer reset, replayed work and separate recovery time. E18's remote process was independently verified still running after the local tab closed; the new checkpoint code has not been installed into that active run.

## Verification and reporting

`tests/test_recovery.py` checks exact saved-state equality on CPU between uninterrupted and paused/resumed codec and generator training, including compiled generator steps, dropout, corruption, EMA, optimizer moments and batch order. It also checks weights-only schedule offsets and fallback from an unreadable latest file. These are short deterministic tests on MLX 0.32.3; they do not establish bitwise equality across Metal/CUDA backends or different hardware.

For every interrupted experiment, record the cause and evidence, last durable step, recovery mode, source checkpoint hash, repeated or lost steps if known, training-loop time for each segment, end-to-end wall time, hardware and GPU contention. Keep scientific outcomes separate from operational recovery tests. A pause/resume test is not a Sudoku quality result.
