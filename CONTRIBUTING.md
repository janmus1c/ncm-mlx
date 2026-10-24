# Contributing

NCM-MLX is experimental research software. Contributions that improve correctness, reproducibility, platform coverage and useful evaluation are welcome. Read [AGENTS.md](AGENTS.md), [current status](docs/STATUS.md) and [next work](docs/NEXT.md) before changing the implementation.

## Development

Use the [platform setup](docs/platforms.md), then run:

```bash
uv sync --locked
uv run ruff check .
uv run pytest -q
uv run python scripts/validate_repository.py
uv build
```

Portable Linux CI does not require MLX. Changes to models, training, recovery or sampling also need the MLX tests on a supported Metal or CUDA setup. Include the hardware and test command with a proposed change. Do not weaken exact CPU forward-equivalence tests or the one-forward-per-level sampling invariant. Benchmark measurements must synchronize through a host copy.

## Research changes

Record measured experiments in `docs/EXPERIMENTS.md` with config, step and sequence budget, wall time, seeds, hardware and GPU contention. Preserve unfavorable results and distinguish hypotheses from evidence. Keep codec reconstruction, validation loss and held-out generation quality separate. Compare training budgets and evaluation sets explicitly; fewer model calls do not establish a wall-clock speedup.

For a meaningful capability or research milestone, update the relevant current documentation, benchmark summary and release notes. Update the README, `docs/capabilities.md`, `AGENTS.md` and supported-platform page when their claims change. Routine experiments need only the experiment log. Keep citation identifiers and existing release tags stable. Repository search topics need updating only when the project's scope changes.

## Proposed changes and attribution

Use the [issue chooser](https://github.com/janmus1c/ncm-mlx/issues/new/choose) for bug reports, training/evaluation/recovery questions or feature proposals. Include a minimal reproduction and relevant environment details; existing evidence is enough, and opening a report does not require an expensive training run. Remove credentials, host identities, private paths, cloud account details and costs from all shared output. Report vulnerabilities through the process in [SECURITY.md](SECURITY.md).

The [pull request template](.github/pull_request_template.md) covers fixes, features and documentation changes. Describe resulting behavior, relevant verification and any untested limits; qualify research comparisons and document third-party provenance when applicable.

Use focused pull requests that explain the problem, resulting behavior and validation. Commit subjects should use the imperative mood, with no body or generated attribution. Do not include credentials, private filesystem paths, host identities, provisioning details or raw local artifacts. Keep datasets and weights outside Git; document provenance and redistribution rights before proposing their publication.

The original NCM method belongs to its authors. This is an independent implementation; avoid novelty or upstream-compatibility claims without evidence. Do not copy upstream code without retaining its required copyright and MIT notice. See [THIRD_PARTY.md](THIRD_PARTY.md) and [CITATION.cff](CITATION.cff).
