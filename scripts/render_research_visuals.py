"""Render the Sudoku evidence and the project roadmap without running a model."""

from __future__ import annotations

import json
import re
import textwrap
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from render_diagrams import PALETTES, Drawing

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"


def save(drawing: Drawing, name: str, theme: str, mobile: bool) -> None:
    suffix = "-mobile" if mobile else ""
    drawing.save(ASSETS / f"{name}{suffix}-{theme}.svg")


def grid(d: Drawing, x: int, y: int, size: int, filled: bool) -> None:
    for row in range(9):
        for col in range(9):
            given = (row * 9 + col) % 4 == 0
            d.box(x + col * size, y + row * size, size - 2, size - 2, filled or given)


def sudoku(theme: str, mobile: bool) -> None:
    levels = json.loads((ROOT / "configs/sudoku-e20-seed1.json").read_text())["codec"]["levels"]
    d = Drawing(420 if mobile else 1080, 1230 if mobile else 650, theme,
                "Sudoku: from clues to a checked grid.",
                "The 81-cell puzzle conditions twelve parallel code levels. Decode once, then check rows, "
                "columns, boxes and every input clue. Invalid candidates remain failures in the solve rate. "
                "Grid icons are schematic, not model predictions.")
    if mobile:
        d.text(28, 48, "From clues to a checked grid.", 24, weight=650)
        d.text(28, 83, "Sudoku · current full-depth policy", 18, muted=True)
        d.box(28, 112, 364, 192)
        grid(d, 48, 138, 16, False)
        d.text(218, 164, "01  Puzzle", 21, weight=600)
        d.text(218, 198, "81-cell prefix", 18, muted=True)
        d.text(218, 231, "0 = blank", 18, muted=True)
        d.text(218, 266, "Clues condition", 17, muted=True)
        d.text(218, 290, "all levels", 17, muted=True)
        d.arrow(210, 321, True)
        d.text(28, 393, "02  Generate twelve code levels", 22, weight=600)
        for index, count in enumerate(levels):
            y = 425 + index * 21
            d.text(28, y + 8, f"{index:02}", 15, muted=True)
            d.text(77, y + 8, str(count), 16)
            for position in range(count):
                d.box(118 + 3 * position, y, 2, 8, True)
        d.text(28, 713, "One forward per level; parallel within it.", 17, muted=True)
        d.arrow(210, 735, True)
        d.box(28, 788, 364, 117)
        d.text(48, 826, "03  Decode once", 23, weight=600)
        d.text(48, 860, "Produce a candidate 9 × 9 grid", 19, muted=True)
        d.text(48, 887, "Validity is not guaranteed.", 18, muted=True)
        d.arrow(210, 922, True)
        d.text(28, 994, "04  Check, then score", 24, weight=600)
        d.text(28, 1030, "Rows + columns + boxes are valid", 19, muted=True)
        d.text(28, 1061, "AND every input clue is preserved.", 19, muted=True)
        d.rule(28, 1091, 392, 1091)
        d.text(28, 1128, "Solved / all evaluated boards", 23, weight=600)
        d.text(28, 1160, "Failures stay in the denominator.", 18, muted=True)
        d.text(28, 1205, "Schematic icons · no solver repair or search", 16, muted=True)
    else:
        d.text(32, 52, "Sudoku: from clues to a checked grid.", 32, weight=650)
        d.text(32, 91, "Current full-depth inference · the grid is generated through latent codes, not cell by cell.",
               20, muted=True)
        d.box(32, 131, 232, 319)
        d.text(52, 171, "01  Puzzle prefix", 23, weight=600)
        grid(d, 64, 198, 18, False)
        d.text(52, 397, "81 cells · 0 = blank", 19, muted=True)
        d.text(52, 428, "Clues condition all levels", 17, muted=True)
        d.arrow(277, 285)
        d.text(332, 171, "02  Generate latent codes", 23, weight=600)
        for index, count in enumerate(levels):
            y = 194 + index * 19
            d.text(332, y + 8, f"{index:02}", 14, muted=True)
            d.text(370, y + 8, str(count), 15)
            for position in range(count):
                d.box(408 + 3 * position, y, 2, 8, True)
        d.text(332, 453, "12 forwards · parallel within each level", 18, muted=True)
        d.arrow(754, 285)
        d.box(804, 131, 244, 319)
        d.text(824, 171, "03  Decode once", 23, weight=600)
        grid(d, 848, 198, 18, True)
        d.text(824, 397, "One candidate grid", 19, muted=True)
        d.text(824, 428, "Validity not guaranteed", 17, muted=True)
        d.rule(32, 491, 1048, 491)
        d.text(32, 534, "04  Check independently", 24, weight=600)
        d.text(32, 570, "Valid rows, columns and boxes", 20, muted=True)
        d.text(32, 602, "+ every input clue preserved", 20, muted=True)
        d.arrow(465, 553)
        d.text(540, 534, "Solved / all evaluated boards", 26, weight=600)
        d.text(540, 570, "Failures remain in the denominator.", 20, muted=True)
        d.text(540, 608, "Schematic icons · no solver repair or search", 17, muted=True)
    save(d, "sudoku-generation", theme, mobile)


def seed_chart(theme: str, mobile: bool) -> None:
    data = json.loads((ROOT / "docs/research/e20-seed-assessment.json").read_text())
    bg, ink, muted, line, surface = PALETTES[theme]
    width, height = (420, 1130) if mobile else (1080, 580)
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "sans-serif"],
                         "svg.fonttype": "none", "svg.hashsalt": "ncm-mlx-seeds"})
    fig = plt.figure(figsize=(width / 72, height / 72), facecolor=bg)

    def text(x, y, label, size=18, bold=False, color=ink):
        fig.text(x / width, 1 - y / height, label, fontsize=size,
                 color=color, weight=600 if bold else 400, va="top")

    text(28 if mobile else 32, 27, "Two seeds. Three difficulties.", 25 if mobile else 32, True)
    if mobile:
        text(28, 73, "MLX implementation · 6.44M generator", 17, color=muted)
        text(28, 101, "16.384M presentations per seed", 17, color=muted)
    else:
        text(32, 76, "MLX implementation · 6.44M generator · 16.384M presentations per seed", 20, color=muted)
    for index, (name, result) in enumerate(data["difficulties"].items()):
        x = 28 if mobile else 32 + index * 352
        top = 157 + index * 260 if mobile else 147
        chart_width, chart_height = (364, 126) if mobile else (312, 126)
        text(x, top - 7, name.capitalize(), 23, True)
        ax = fig.add_axes([x / width, 1 - (top + 181) / height,
                           chart_width / width, chart_height / height], facecolor=bg)
        values = [result[f"seed_{seed}"]["solved_percent"] for seed in (0, 1)]
        ax.barh([1, 0], values, height=.36, color=[ink, surface], edgecolor=ink, linewidth=1.2)
        ax.patches[1].set_hatch("///")
        reference = result["quoted_original_ncm_percent"]
        ax.axvline(reference, color=muted, linewidth=1.5, linestyle=(0, (3, 3)))
        for row, value in zip((1, 0), values, strict=True):
            ax.text(3, row + .26, f"E{19 if row else 20} · seed {0 if row else 1}",
                    color=muted, fontsize=14, va="bottom")
            ax.text(value - 2, row, f"{value:.2f}%", color=bg if row else ink,
                    fontsize=16, weight=600, va="center", ha="right",
                    bbox={"facecolor": ink if row else surface, "edgecolor": "none", "pad": 1})
        ax.set(xlim=(0, 100), ylim=(-.5, 1.75), yticks=[], xticks=[0, 50, 100])
        ax.set_xticklabels(["0", "50", "100%"], fontsize=13, color=muted)
        ax.tick_params(axis="x", colors=muted, length=3)
        for edge in ("top", "right", "left"):
            ax.spines[edge].set_visible(False)
        ax.spines["bottom"].set_color(line)
        text(x, top + 219, f"Dashed: Original NCM {reference:.2f}%", 15, color=muted)
    if mobile:
        footnotes = ["2,000 boards per difficulty · greedy decoding",
                     "Same boards recur across the two seeds.",
                     "Original NCM: 25.46M generator;",
                     "51.2M presentations; different task recipe.",
                     "Historical aggregate reports; audit limits apply.",
                     "Two seeds do not establish population variance."]
        for index, label in enumerate(footnotes):
            text(28, 958 + index * 26, label, 16, color=muted)
    else:
        for index, label in enumerate([
            "2,000 boards per difficulty · greedy decoding · the same boards recur across seeds.",
            "Original NCM: 25.46M generator, 51.2M presentations and a different task recipe; not a matched comparison.",
            "Historical aggregate reports with audit limits. Two seeds do not establish population training variability.",
        ]):
            text(32, 448 + index * 36, label, 17, color=muted)
    suffix = "-mobile" if mobile else ""
    title = "Sudoku solve rates across two training seeds"
    description = (
        "Measured aggregate solve rates from e20-seed-assessment.json. "
        "E19 seed 0: 98.10, 83.40, 40.30 percent. E20 seed 1: 97.25, 79.60, 33.25 percent. "
        "Dashed Original NCM references are not a matched-budget comparison."
    )
    path = ASSETS / f"sudoku-seeds{suffix}-{theme}.svg"
    fig.savefig(path, metadata={"Date": None, "Title": title, "Description": description})
    svg = path.read_text()
    root = re.search(r"<svg\b[^>]*>", svg).group()
    accessible_root = root.replace('<svg ', '<svg role="img" aria-labelledby="title description" ', 1)
    accessible_root = accessible_root.replace(f'width="{width}pt"', f'width="{width}"')
    accessible_root = accessible_root.replace(f'height="{height}pt"', f'height="{height}"')
    svg = svg.replace(root, accessible_root + f'\n<title id="title">{escape(title)}</title>'
                      f'<desc id="description">{escape(description)}</desc>', 1)
    path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
    plt.close(fig)


def scaling_chart(theme: str, mobile: bool) -> None:
    data = json.loads((ROOT / "docs/research/tinystories-scaling.json").read_text())
    runs = data["runs"]
    width, height = (420, 980) if mobile else (1080, 750)
    title = "With a larger generator and more data the NCM/AR ratio falls, and stays above two."
    description = (
        "NCM over AR ratio of GPT-2-large generative perplexity on TinyStories by generator size. "
        "Ratios at 1x data and fp32: " + ", ".join(
            f"{runs[key]['ncm_over_ar']:.2f}" for key in ("ladder-12m", "ladder-50m", "ladder-87m"))
        + " for 12M, 50M and 87M generators. Ratios at 4x data and bf16: "
        + ", ".join(f"{runs[key]['ncm_over_ar']:.2f}" for key in (
            "ladder-12m-x4-bf16", "ladder-12m-x4-bf16-seed1", "ladder-50m-x4-bf16", "ladder-87m-x4-bf16"))
        + " (12M seeds 0 and 1, then 50M and 87M). One training seed per point except the 12M point at 4x data; "
        "the 50M to 87M step is within seed variation. Compute is not matched."
    )
    d = Drawing(width, height, theme, title, description)
    m = 28 if mobile else 32
    if mobile:
        d.text(m, 48, "NCM versus AR, by size", 26, weight=650)
        d.text(m, 80, "Ratio of GPT-2-large perplexities", 17, muted=True)
        d.text(m, 104, "TinyStories · lower is better", 17, muted=True)
        left, right, top, bottom = 72, 398, 222, 592
    else:
        d.text(m, 52, "NCM versus AR: the ratio falls with size and data, and stays above 2.", 28, weight=650)
        d.text(m, 90, "Ratio of GPT-2-large generative perplexity, NCM over the AR trained on the same windows · "
               "TinyStories · lower is better.", 19, muted=True)
        left, right, top, bottom = 110, 1040, 168, 508
    low, high = 0.9, 3.7

    def ypos(value: float) -> float:
        return bottom - (value - low) / (high - low) * (bottom - top)

    slots = [("12M", 0), ("50M", 1), ("87M", 2)]
    span = right - left
    slot_x = {name: left + span * (0.15 + index * 0.35) for name, index in slots}
    for value, label in ((1, "1 · parity"), (2, "2"), (3, "3")):
        y = ypos(value)
        dash = ' stroke-dasharray="6 5"' if value == 1 else ""
        d.parts.append(f'<path d="M{left} {y:.1f}H{right}" stroke="{d.line}"{dash}/>')
        d.text(left - 10, y + 6, label, 15, muted=True, anchor="end")
    for name, _ in slots:
        d.text(slot_x[name], bottom + 30, name if mobile else f"{name} generator", 16 if mobile else 18, muted=True, anchor="middle")

    def dot(x, y, kind, filled):
        fill = d.ink if filled else d.bg
        if kind == "circle":
            d.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{fill}" stroke="{d.ink}" stroke-width="2"/>')
        elif kind == "square":
            d.parts.append(f'<rect x="{x - 7:.1f}" y="{y - 7:.1f}" width="14" height="14" fill="{fill}" '
                           f'stroke="{d.ink}" stroke-width="2"/>')
        else:
            d.parts.append(f'<path d="M{x:.1f} {y - 9:.1f}L{x + 9:.1f} {y:.1f}L{x:.1f} {y + 9:.1f}L{x - 9:.1f} {y:.1f}Z" '
                           f'fill="{fill}" stroke="{d.muted}" stroke-width="2"/>')

    def path(points, dashed):
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        d.parts.append(f'<polyline points="{coords}" fill="none" stroke="{d.ink}" stroke-width="2"{dash}/>')

    ratio = {key: runs[key]["ncm_over_ar"] for key in runs if "ncm_over_ar" in runs[key]}
    one = [("12M", ratio["ladder-12m"]), ("50M", ratio["ladder-50m"]), ("87M", ratio["ladder-87m"])]
    four_12 = [ratio["ladder-12m-x4-bf16"], ratio["ladder-12m-x4-bf16-seed1"]]
    four = [("12M", sum(four_12) / 2), ("50M", ratio["ladder-50m-x4-bf16"]), ("87M", ratio["ladder-87m-x4-bf16"])]
    path([(slot_x[n], ypos(v)) for n, v in one], False)
    path([(slot_x[n], ypos(v)) for n, v in four], True)
    for n, v in one:
        dot(slot_x[n], ypos(v), "circle", False)
        left_side = n == "12M"
        d.text(slot_x[n] + (-12 if left_side else 12), ypos(v) - 12, f"{v:.2f}", 17, weight=600,
               anchor="end" if left_side else "start")
    mac = [data["mac_fp32_references"][k]["ncm_over_ar"] for k in ("e21", "e24")]
    for v in mac:
        d.parts.append(f'<circle cx="{slot_x["12M"]:.1f}" cy="{ypos(v):.1f}" r="3" fill="{d.muted}"/>')
    for v in four_12:
        dot(slot_x["12M"], ypos(v), "square", True)
    d.parts.append(f'<path d="M{slot_x["12M"]:.1f} {ypos(four_12[0]):.1f}V{ypos(four_12[1]):.1f}" '
                   f'stroke="{d.ink}" stroke-width="2"/>')
    d.text(slot_x["12M"] - 14, ypos(four_12[0]) + 6, f"{four_12[0]:.2f}", 17, weight=600, anchor="end")
    d.text(slot_x["12M"] - 14, ypos(four_12[1]) + 6, f"{four_12[1]:.2f}", 17, weight=600, anchor="end")
    for n, v in four[1:]:
        dot(slot_x[n], ypos(v), "square", True)
        d.text(slot_x[n] + 4, ypos(v) + 34, f"{v:.2f}", 17, weight=600, anchor="middle")
    ly = 124 if mobile else 118
    lx = m
    dot(lx + 9, ly + 24 if mobile else ly + 10, "circle", False)
    d.text(lx + 28, ly + (30 if mobile else 16), "1× data · 960k windows · fp32", 16)
    ly2 = ly + 30 if mobile else ly
    lx2 = m if mobile else 440
    dot(lx2 + 9, ly2 + 24 if mobile else ly2 + 10, "square", True)
    d.text(lx2 + 28, ly2 + (30 if mobile else 16), "4× data · 3.84M windows · bf16", 16)
    notes = [
        ("Dashed line joins the 4× points; the 12M point is the mean of two seeds. "
         "Small dots at 12M: the Mac fp32 repeats (E21, E24)."),
        "Frozen E21 codec · 5 sampling seeds × 256 outputs per point · one training seed per point except 12M at 4×.",
        ("The two 12M seeds at 4× differ more than the 50M and 87M points do. "
         "The released checkpoint (16.259) is not plotted: no AR baseline here matches its budget."),
        "Compute is not matched; GPU and precision differ between the two series.",
    ]
    ny = bottom + 66
    row = 0
    for note in notes:
        for line in textwrap.wrap(note, 50 if mobile else 100):
            d.text(m, ny + row * (24 if mobile else 25), line, 14 if mobile else 16, muted=True)
            row += 1
    save(d, "scaling-ladder", theme, mobile)


COMPLETED = [
    ("TinyWorld", ["Synthetic grammar", "Codec, generator, AR", "Latency and quality"]),
    ("Sudoku", ["E6 to E20, two seeds", "Easy 98.1%, medium 83.4%", "Hard boards still behind"]),
    ("Checkpoint port", ["Runs Kumbong et al.", "TinyStories, Sudoku, XSum", "Agrees with PyTorch, tested"]),
    ("TinyStories", ["E21 to E32, size and data", "NCM/AR ratio 3.3 to 2.1", "Still above 2 (single seeds)"]),
]
PLANNED = [
    ("Understand the remaining gap", ["Released recipe details, Muon,", "neighbour corruption, codec seeds"]),
    ("Prompt-conditioned training", ["Prefix training of our own models;", "released XSum generator ported"]),
    ("Mixed-precision throughput", ["bf16 done for the 12M recipe;", "larger sizes untested"]),
    ("Natural-text corpus", ["Several billion tokens"]),
    ("100M model, about 2B tokens", ["The go / no-go against AR"]),
    ("300M, then about 1B", ["Only if NCM is close to AR", "at matched compute"]),
]


def project_roadmap(theme: str, mobile: bool) -> None:
    d = Drawing(420 if mobile else 1080, 1620 if mobile else 735, theme,
                "From a laptop study to a text model: completed work and planned steps.",
                "Four completed milestones: TinyWorld, Sudoku, the released-checkpoint port (TinyStories, Sudoku, "
                "XSum) and TinyStories training up to E32, where the NCM/AR ratio falls from 3.3 to 2.1. Six planned "
                "steps in order: understand the remaining gap, prompt-conditioned training, mixed-precision "
                "throughput (done for the 12M recipe), a natural-text corpus, a 100M go/no-go run, then 300M and "
                "about 1B only if NCM is close to the baseline. Planned steps are plans, not results.")
    m = 28 if mobile else 32
    d.text(m, 49, "Roadmap" if mobile else "Roadmap: measured so far, planned next", 26 if mobile else 32, weight=650)
    goal = ["Goal: about 1B parameters, text not worse than", "a matched autoregressive baseline."]
    if mobile:
        d.text(m, 84, goal[0], 17, muted=True)
        d.text(m, 108, goal[1], 17, muted=True)
        y0 = 148
    else:
        d.text(m, 88, " ".join(goal), 19, muted=True)
        y0 = 128
    d.text(m, y0, "COMPLETED", 16, muted=True)
    if mobile:
        for i, (title, lines) in enumerate(COMPLETED):
            y = y0 + 20 + i * 150
            d.box(m, y, 364, 136)
            d.text(m + 20, y + 36, f"{i + 1:02}  {title}", 24, weight=600)
            for r, line in enumerate(lines):
                d.text(m + 20, y + 68 + r * 26, line, 17, muted=True)
        y1 = y0 + 20 + 4 * 150 + 8
    else:
        w = 246
        for i, (title, lines) in enumerate(COMPLETED):
            x = m + i * (w + 10)
            y = y0 + 20
            d.box(x, y, w, 146)
            d.text(x + 20, y + 36, title, 24, weight=600)
            for r, line in enumerate(lines):
                d.text(x + 20, y + 72 + r * 27, line, 16, muted=True)
        y1 = y0 + 20 + 146 + 30
    d.rule(m, y1, 1048 if not mobile else 392, y1)
    d.text(m, y1 + 40, "PLANNED, IN ORDER", 16, muted=True)
    y2 = y1 + 60
    for i, (title, lines) in enumerate(PLANNED):
        if mobile:
            x, y, w, h = m, y2 + i * 114, 364, 100
        else:
            x, y, w, h = m + (i % 3) * 344, y2 + (i // 3) * 126, 328, 112
        d.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="none" '
                       f'stroke="{d.line}" stroke-dasharray="5 5"/>')
        d.text(x + 20, y + 34, f"{i + 1:02}  {title}", 19 if not mobile else 20, weight=600)
        for r, line in enumerate(lines):
            d.text(x + 20, y + 62 + r * 24, line, 16 if not mobile else 15, muted=True)
    yf = (y2 + 6 * 114 + 12) if mobile else (y2 + 2 * 126 + 14)
    d.rule(m, yf, 392 if mobile else 1048, yf)
    d.text(m, yf + 38, "Planned steps are plans, not results.", 19, weight=600)
    if mobile:
        d.text(m, yf + 68, "No delivery dates.", 16, muted=True)
    else:
        d.text(1048, yf + 38, "No delivery dates.", 17, muted=True, anchor="end")
    save(d, "roadmap-2026-11", theme, mobile)


def main() -> None:
    for theme in PALETTES:
        for mobile in (False, True):
            sudoku(theme, mobile)
            seed_chart(theme, mobile)
            scaling_chart(theme, mobile)
            project_roadmap(theme, mobile)
    print("Rendered Sudoku workflow, two-seed evidence, scaling ladder and roadmap in both themes and layouts")


if __name__ == "__main__":
    main()
