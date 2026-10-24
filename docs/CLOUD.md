# Remote NVIDIA training

MLX runs on NVIDIA GPUs under Linux through its CUDA backend (`uv sync --group cuda`). Configure your own SSH host and keep credentials outside this repository.

```bash
scripts/remote_run.sh HOST:PORT configs/NAME.json --max-hours 12
scripts/remote_run.sh HOST:PORT configs/NAME.json --resume --max-hours 12
```

The remote runner refuses an existing NCM process before copying code. It transfers cached data and any reused codec, validates the NVIDIA driver and CUDA requirements with `scripts/check_gpu.sh`, runs the experiment with a bounded runtime, and copies artifacts and logs back on success or failure.

For a replacement machine, `--resume` uploads the full training snapshots in `artifacts/NAME` and retains the original step budget. The config's `out_dir` must follow `artifacts/<config-stem>`. Run the orchestrator independently of a terminal and keep periodic external backups for machine loss; see [RECOVERY.md](RECOVERY.md).

Provisioning, host configuration and machine lifecycle management belong to the operator. This repository does not provision or delete infrastructure.
