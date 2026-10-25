"""Build planning scenarios from completed evidence: uv run --group paper python scripts/project_roadmap.py."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/research/projections"


def main() -> None:
    evidence = json.loads((ROOT / "docs/research/completed-sudoku-results.json").read_text())
    funded = evidence["experiments"]["sudoku-e19-funded"]
    baseline_hours = funded["training"]["train_seconds"] / 3600
    baseline_exposure = funded["sequence_presentations"]
    scenarios = []
    for exposure in (8_192_000, 16_384_000, 32_768_000, 51_200_000):
        hours = baseline_hours * exposure / baseline_exposure
        scenarios.append({"generator_presentations": exposure, "linear_training_hours": hours,
                          "sensitivity_hours": [hours * 0.8, hours * 1.2]})
    result = {
        "kind": "planning_scenarios_not_quality_forecasts",
        "baseline_run": "sudoku-e19-funded",
        "baseline_generator_parameters": funded["parameters"]["generator"],
        "baseline_training_hours": baseline_hours,
        "baseline_presentations": baseline_exposure,
        "assumption": "Same generator, batch, accelerator and observed mean time per presentation",
        "sensitivity": "Arbitrary +/-20% runtime sensitivity, not a confidence or prediction interval",
        "excludes": ["rental startup", "data generation and transfer", "encoding", "evaluation",
                     "codec training", "failures", "new model shapes", "real-text workloads"],
        "training_scenarios": scenarios,
        "planned_level_budgets": [{"levels": n, "generator_forwards": n, "quality": None}
                                  for n in (3, 6, 9, 12)],
        "original_reference": {"label": "Original NCM (Kumbong et al.)", "generator_parameters": 25_460_000,
                               "generator_presentations": 51_200_000,
                               "source": "https://arxiv.org/html/2610.14162v1"},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "scenarios.json").write_text(json.dumps(result, indent=2) + "\n")
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "ncm-mlx-planning"})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), layout="constrained")
    x = [s["generator_presentations"] / 1e6 for s in scenarios]
    y = [s["linear_training_hours"] for s in scenarios]
    axes[0].fill_between(x, [t * 0.8 for t in y], [t * 1.2 for t in y], color="#dfe8e2",
                         label="Assumed +/-20% runtime sensitivity")
    axes[0].plot(x, y, color="#38634c", ls="--", marker="o", label="Linear planning scenario")
    axes[0].scatter([baseline_exposure / 1e6], [baseline_hours], color="#1c2b22", s=65,
                    zorder=5, label="Measured E19-funded anchor")
    axes[0].set(xlabel="Generator presentations (millions)", ylabel="Generator training hours",
                title="Same-model compute planning")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].bar(["3 levels", "6 levels", "9 levels", "12 levels"], [3, 6, 9, 12],
                color=["#b9cdbf"] * 3 + ["#38634c"])
    axes[1].set(ylabel="Generator forwards per batch", title="Planned fixed-level budgets", ylim=(0, 14))
    axes[1].text(1.5, 13, "Quality and latency are unmeasured", ha="center", fontsize=9)
    fig.suptitle("Roadmap planning: assumptions, not predicted solve rates", fontsize=13)
    for ext in ("svg", "png"):
        path = OUTPUT / f"compute-and-levels.{ext}"
        fig.savefig(path, dpi=180)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
