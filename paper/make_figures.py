"""Collect the paper's numbers from runs/ into data/results.json and draw the figures.

    .venv/bin/python paper/make_figures.py            # collect from runs/, then plot
    .venv/bin/python paper/make_figures.py --plot     # plot from the committed data/results.json only

runs/ is gitignored; data/results.json is committed so the figures can be
rebuilt from the repository alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
PAPER = Path(__file__).resolve().parent
DATA = PAPER / "data" / "results.json"
FIGS = PAPER / "figures"

SUDOKU_RUNS = {
    "E4": ("eval-sudoku.json", "train-sudoku.log", [1, 3, 9, 27, 54, 81]),
    "E5": ("eval-e5.json", "train-e5.log", [1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 81]),
    "E6": ("eval-e6.json", "train-e6.log", [1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 81]),
    "E8": ("sudoku-e8-eval.json", "sudoku-e8-train.log", [1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 81]),
    "E11": ("sudoku-e11-eval.json", "sudoku-e11-train.log", [1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 81]),
    "E13": ("sudoku-e13-eval.json", "sudoku-e13-train.log", [1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 81]),
    "E14": ("sudoku-e14-eval.json", "sudoku-e14-train.log", [1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 81]),
    "E15": ("sudoku-e15-eval.json", "sudoku-e15-train.log", [1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 81]),
    "E16": ("sudoku-e16-eval.json", "sudoku-e16-train.log", [1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 81]),
    "E17 recovery": ("sudoku-e17-recovery-eval.json", "sudoku-e17-recovery-train.log", [1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 81]),
}
SETS = ["upstream-easy", "upstream-medium", "upstream-hard"]
GEN_STEP = re.compile(r"gen step\s+(\d+)\s+loss ([\d.]+)\s+val ([\d.]+)")


def _load(name: str):
    path = RUNS / name
    return json.loads(path.read_text()) if path.exists() else None


def _bench(rows):
    return {str(r["batch"]): {"p50_ms": r["p50_ms"], "peak_memory_mb": r["peak_memory_mb"]} for r in rows}


def collect() -> dict:
    out: dict = {"sudoku": {}, "latency": {}}
    for name, (ev, log, levels) in SUDOKU_RUNS.items():
        d = _load(ev)
        if d is None:
            continue
        curve = []
        if (RUNS / log).exists():
            curve = [[int(s), float(v)] for s, _, v in GEN_STEP.findall((RUNS / log).read_text())]
        out["sudoku"][name] = {
            "levels": levels,
            "params": d["params"],
            "codec_acc_by_levels": d["codec_val"]["char_acc_by_levels_used"],
            "code_perplexity": d["codec_val"]["code_perplexity"],
            "teacher_forced_acc": d["generator_val_teacher_forced"]["acc_per_level"],
            "scores": {s: d["sudoku"]["0.0"][s] for s in SETS + ["val"] if s in d["sudoku"]["0.0"]},
            "val_curve": curve,
        }
        meta_path = ROOT / "artifacts" / ev.removesuffix("-eval.json") / "generator.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            out["sudoku"][name]["training"] = {
                k: meta[k] for k in ("steps", "best_step", "train_seconds") if k in meta
            }
            if continuation := meta.get("continuation"):
                recovery = {k: continuation[k] for k in (
                    "start_step", "optimizer_reset", "random_state_reset")}
                source = ROOT / continuation["checkpoint"]
                if source.exists():
                    with source.open("rb") as f:
                        recovery["source_checkpoint_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
                out["sudoku"][name]["recovery"] = recovery
    ar = _load("baseline-sudoku.json")
    if ar:
        out["sudoku"]["AR"] = {"params": ar["params"], "scores": ar["samples"]["0.0"]}
        out["latency"]["sudoku-ar"] = _bench(ar["bench"])
    for key, name in [("sudoku-ncm-6", "bench-sudoku.json"), ("sudoku-ncm-12", "bench-e6.json"),
                      ("tinyworld-ncm", "bench-ncm-v1.json")]:
        d = _load(name)
        if d:
            out["latency"][key] = _bench(d["rows"])
    ar1 = _load("baseline-ar-v1.json")
    if ar1:
        out["latency"]["tinyworld-ar"] = _bench(ar1["bench"])
    qwen = {}
    for level in ("easy", "medium", "hard"):
        d = _load(f"qwen27b-nothink-{level}.json")
        if d:
            qwen[f"upstream-{level}"] = {**d["scores"], "unparsed": d["unparsed"], "latency_s_p50": d["latency_s_p50"]}
    if qwen:
        out["sudoku"]["Qwen3.8-27B"] = {"params": 27_000_000_000, "scores": qwen}
    out["paper_table3"] = {
        "AR sample": [13.9, 5.1, 0.6], "AR greedy": [14.6, 5.1, 1.0], "MDLM": [92.0, 77.1, 30.2],
        "DUO": [96.3, 84.7, 58.4], "CANDI": [79.3, 45.9, 16.7], "FLM": [94.2, 82.7, 44.5],
        "S-FLM": [94.8, 85.2, 45.0], "FMLM+": [97.7, 92.8, 81.3], "NCM": [99.5, 96.5, 86.1],
    }
    return out


INK, MUTED, ACCENT, WARM = "#1d2420", "#7a857f", "#2f6b4f", "#b5552f"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": "#e3e6e1", "grid.linewidth": 0.6, "legend.frameon": False,
})


def _save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=200)
    plt.close(fig)


OVERVIEW_SAMPLE = [  # one E1 generation, per-level decodes as printed by `ncm_mlx demo`
    (1, "eue likes one oed egg."), (2, "eve likes one oig egg."), (4, "dee likes one red egg."),
    (8, "dee likes one ret egg."), (16, "dee likes one ret egg."), (32, "dee likes one ret egg."),
]


def fig_overview(_: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, len(OVERVIEW_SAMPLE) + 0.2)
    width = 4.6
    for row, (n, text) in enumerate(OVERVIEW_SAMPLE):
        y = len(OVERVIEW_SAMPLE) - 1 - row
        w = width / n
        fill = ACCENT if row >= 2 else "#cfe3d7"
        for i in range(n):
            ax.add_patch(plt.Rectangle((0.9 + i * w + 0.01, y + 0.12), w - 0.02, 0.62, color=fill, lw=0))
        ax.text(0.75, y + 0.43, f"level {row}", ha="right", va="center", fontsize=8, color=MUTED)
        ax.text(5.75, y + 0.43, f"{n:>2} code{'s' if n > 1 else ''}", ha="left", va="center", fontsize=8, color=MUTED)
        ax.text(6.75, y + 0.43, text, ha="left", va="center", fontsize=9, family="monospace", color=INK)
    ax.text(6.75, -0.45, "decoded text after this level", fontsize=7.5, color=MUTED)
    ax.text(0.9, -0.45, "codes of each level, predicted in parallel", fontsize=7.5, color=MUTED)
    _save(fig, "overview")


def fig_codec_refinement(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    styles = {"E4": (MUTED, "--"), "E5": (WARM, "-"), "E6": (ACCENT, "-")}
    for name, (color, ls) in styles.items():
        run = data["sudoku"].get(name)
        if not run:
            continue
        total = sum(run["levels"])
        used, x = 0, []
        for l in run["levels"]:
            used += l
            x.append(used / total)
        ax.plot(x, [100 * a for a in run["codec_acc_by_levels"]], ls, color=color, marker="o", ms=3,
                label=f"{name}: {len(run['levels'])} levels")
    ax.axhline(100 / 9, color=MUTED, lw=0.8, ls=":")
    ax.text(0.98, 100 / 9 + 2, "chance (1/9)", color=MUTED, fontsize=7, ha="right")
    ax.set_xlabel("share of codes used (coarse to fine)")
    ax.set_ylabel("cells decoded correctly (%)")
    ax.set_ylim(0, 102)
    ax.legend(loc="upper left")
    _save(fig, "codec-refinement")


def fig_latency(data: dict) -> None:
    lat = data["latency"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=False)
    panels = [("TinyWorld (32 chars)", [("tinyworld-ncm", "NCM, 6 levels", ACCENT, "-"),
                                         ("tinyworld-ar", "AR + KV cache", WARM, "--")]),
              ("Sudoku (81 cells)", [("sudoku-ncm-6", "NCM, 6 levels", MUTED, "-"),
                                      ("sudoku-ncm-12", "NCM, 12 levels", ACCENT, "-"),
                                      ("sudoku-ar", "AR + KV cache", WARM, "--")])]
    for ax, (title, series) in zip(axes, panels):
        for key, label, color, ls in series:
            if key not in lat:
                continue
            b = sorted(lat[key], key=int)
            ax.plot([int(x) for x in b], [lat[key][x]["p50_ms"] for x in b], ls, color=color, marker="o", ms=3,
                    label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("batch size")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("p50 latency per batch (ms)")
    _save(fig, "latency")


def fig_val_curves(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for name, color in [("E5", WARM), ("E6", ACCENT), ("E8", INK)]:
        run = data["sudoku"].get(name)
        if run and run["val_curve"]:
            s, v = zip(*run["val_curve"])
            ax.plot(s, v, color=color, marker="o", ms=3, label=name)
    ax.set_xlabel("generator step (batch 128)")
    ax.set_ylabel("validation CE per code (nats)")
    ax.legend()
    _save(fig, "sudoku-val-curves")


def fig_sudoku_cells(data: dict) -> None:
    runs = [(n, data["sudoku"][n]) for n in ("E4", "E6", "E8", "E11", "E13", "E14", "AR", "Qwen3.8-27B") if n in data["sudoku"]]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    width = 0.8 / len(runs)
    colors = [MUTED, WARM, "#9cc5ad", ACCENT, "#6b7fa8", INK, "#8a6a12", "#c9a7a0"]
    for i, (name, run) in enumerate(runs):
        vals = [100 * run["scores"].get(s, {}).get("blank_cell_acc", 0) for s in SETS]
        ax.bar([j + i * width for j in range(3)], vals, width, color=colors[i % len(colors)], label=name)
    ax.axhline(100 / 9, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks([j + 0.4 - width / 2 for j in range(3)], ["easy (40 givens)", "medium (35)", "hard (30)"])
    ax.set_ylabel("blank cells correct (%)")
    ax.legend(ncol=3, fontsize=7, loc="upper right")
    _save(fig, "sudoku-blank-cells")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plot", action="store_true", help="skip collection and plot from data/results.json")
    args = p.parse_args()
    if not args.plot:
        DATA.parent.mkdir(parents=True, exist_ok=True)
        DATA.write_text(json.dumps(collect(), indent=1) + "\n")
    data = json.loads(DATA.read_text())
    for fig in (fig_overview, fig_codec_refinement, fig_latency, fig_val_curves, fig_sudoku_cells):
        fig(data)
    print(f"wrote {DATA.relative_to(ROOT)} and {len(list(FIGS.glob('*.svg')))} figures to {FIGS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
