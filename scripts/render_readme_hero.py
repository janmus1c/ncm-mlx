"""Render the README refinement animation and the GitHub social preview.

The trace is read from docs/research/official-tinystories-traces.json (real outputs of the official
TinyStories checkpoint running in the MLX port).
Outputs: docs/assets/tinystories-refinement-{light,dark}.svg, docs/assets/tinystories-refinement-mobile-{light,dark}.svg
and assets/social-preview.svg.
"""

from __future__ import annotations

import json
import re
import textwrap
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

THEMES = {
    "light": {
        "canvas": "#ffffff",
        "ink": "#20252b",
        "muted": "#59636e",
        "rule": "#d6dce2",
        "panel": "#f3f5f7",
    },
    "dark": {
        "canvas": "#13161a",
        "ink": "#f1f2f3",
        "muted": "#b4bac1",
        "rule": "#454c55",
        "panel": "#1c2025",
    },
}
FONT = "Inter, Helvetica Neue, Arial, sans-serif"
LOGO_PATH = (
    "M4 4h24v4H4zm0 10h10v4H4zm14 0h10v4H18zM4 24h4v4H4zm6.667 0h4v4h-4zm6.666 0h4v4h-4zM24 24h4v4h-4z"
)

TRACE_FILE = ROOT / "docs" / "research" / "official-tinystories-traces.json"
SAMPLE = 9
SHOWN_LEVELS = (0, 2, 4, 6, 8, 11)
SOCIAL_LEVELS = (0, 4, 8, 11)
TITLE = "Official NCM TinyStories model, refined level by level"
SUBTITLE = "REAL SAMPLE · HazyResearch checkpoint · MLX on Apple GPU · 12 levels"
CAPTION = (
    "Recorded with ncm-mlx 0.2.0 on an M5 Pro (seed 0, temperature 1.0, sample 9). "
    "Excerpt of the first sentence at six of twelve levels; level 11 of 12 briefly regresses before level 12 "
    "(see docs/research/official-tinystories-traces.json)."
)
TAGLINE = ("Next-Concept Modeling on the Apple GPU —", "runs the released NCM checkpoints in MLX")


def excerpt(text: str, limit: int) -> str:
    """Collapse whitespace and cut at a word boundary to at most `limit` characters."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    cut = flat[:limit]
    if flat[limit] != " ":
        cut = cut.rsplit(" ", 1)[0]
    cut = cut.rstrip()
    return cut if cut.endswith((".", "!", "?")) else cut + "…"


def load_trace() -> tuple[list[tuple[int, int, str]], dict]:
    data = json.loads(TRACE_FILE.read_text())
    sample = data["samples"][SAMPLE]
    sizes = data["level_sizes"]
    final = " ".join(sample["final"].split())
    sentence = re.match(r".*?[.!?](?=\s|$)", final).group(0)
    if sample["levels"][-1] != sample["final"]:
        raise SystemExit("level 11 must equal the final text")
    rows = [(k, sizes[k], excerpt(sample["levels"][k], len(sentence))) for k in SHOWN_LEVELS]
    if rows[-1][2] != sentence:
        raise SystemExit(f"final excerpt {rows[-1][2]!r} is not the first sentence {sentence!r}")
    if data["n"] != len(sizes) or SHOWN_LEVELS[-1] != len(sizes) - 1:
        raise SystemExit("unexpected level count")
    return rows, data


def wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width, break_long_words=False)


def level_dots(x: float, y: float, size: float, gap: float, level: int, levels: int, c: dict) -> str:
    """One square per level; squares up to and including ``level`` are filled."""
    return "".join(
        f'<rect x="{x + k * (size + gap):g}" y="{y:g}" width="{size:g}" height="{size:g}" rx="1.5" '
        f'fill="{c["ink"] if k <= level else c["rule"]}"/>'
        for k in range(levels)
    )


def words_markup(prev: str | None, text: str, ink: str, muted: str) -> str:
    cur = text.split(" ")
    old = prev.split(" ") if prev else [None] * len(cur)
    parts = []
    for i, word in enumerate(cur):
        changed = prev is None or i >= len(old) or word != old[i]
        fill, weight = (ink, 650) if changed else (muted, 400)
        parts.append(f'<tspan fill="{fill}" font-weight="{weight}">{escape(word)}</tspan>')
    return " ".join(parts)


def animation_css(rows: int, cycle: float, step: float) -> str:
    css = ["@media (prefers-reduced-motion: no-preference){"]
    for i in range(rows):
        t0 = 100 * (0.6 + i * step) / cycle
        t1 = t0 + 100 * 0.5 / cycle
        out0, out1 = 100 * (cycle - 1.2) / cycle, 100 * (cycle - 0.4) / cycle
        css.append(
            f"@keyframes r{i}{{0%,{t0:.2f}%{{opacity:0}}{t1:.2f}%,{out0:.2f}%{{opacity:1}}"
            f"{out1:.2f}%,100%{{opacity:0}}}}.r{i}{{animation:r{i} {cycle}s ease-out infinite}}"
        )
    css.append("}")
    return "".join(css)


def hero(theme: str, mobile: bool) -> str:
    c = THEMES[theme]
    trace, data = load_trace()
    levels = len(data["level_sizes"])
    w = 420 if mobile else 1080
    m = 28 if mobile else 32
    n = len(trace)
    desc = (
        "A recorded generation from the official HazyResearch TinyStories checkpoint running in MLX. "
        f"Level 1 of 12 is mostly punctuation; by level 9 the first sentence is nearly formed; level 12 reads: {trace[-1][2]}"
    )
    if mobile:
        row_h, top, fsz = 112, 240, 21
        h = top + n * row_h + 150
    else:
        row_h, top, fsz = 62, 140, 24
        h = top + n * row_h + 76
    out = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title description">'
            f'<title id="title">{escape(TITLE)}</title><desc id="description">{escape(desc)}</desc>'
        ),
        f"<style>{animation_css(n, 14, 2.0)}</style>",
        f'<rect width="{w}" height="{h}" rx="16" fill="{c["canvas"]}"/>',
        f'<g font-family="{FONT}" fill="{c["ink"]}">',
    ]
    if mobile:
        for j, line in enumerate(("Official NCM TinyStories", "model, refined level", "by level")):
            out.append(f'<text x="{m}" y="{50 + j * 30}" font-size="25" font-weight="650">{escape(line)}</text>')
        for j, line in enumerate(("REAL SAMPLE · HazyResearch", "checkpoint · MLX on Apple GPU", "· 12 levels")):
            out.append(f'<text x="{m}" y="{140 + j * 20}" font-size="15" fill="{c["muted"]}">{escape(line)}</text>')
    else:
        out.append(f'<text x="{m}" y="54" font-size="30" font-weight="650">{escape(TITLE)}</text>')
        out.append(f'<text x="{m}" y="84" font-size="18" fill="{c["muted"]}">{escape(SUBTITLE)}</text>')
    prev = None
    for i, (lvl, codes, text) in enumerate(trace):
        y = top + i * row_h
        label = f"LEVEL {lvl + 1} OF 12 · {codes} codes" if codes != 1 else f"LEVEL {lvl + 1} OF 12 · 1 code"
        out.append(f'<g class="r{i}">')
        out.append(
            f'<line x1="{m}" y1="{y - 26}" x2="{w - m}" y2="{y - 26}" stroke="{c["rule"]}" stroke-width="2" stroke-linecap="round" stroke-dasharray="0 6"/>'
        )
        out.append(f'<text x="{m}" y="{y - 2}" font-size="15" fill="{c["muted"]}">{label}</text>')
        if mobile:
            out.append(level_dots(m, y + 6, 10, 6, lvl, levels, c))
            lines = wrap(text, 27)
            pieces, pp = [], (prev or "").split(" ") if prev else None
            idx = 0
            for j, line in enumerate(lines):
                words = line.split(" ")
                ref = " ".join(pp[idx : idx + len(words)]) if pp else None
                pieces.append(
                    f'<text x="{m}" y="{y + 38 + j * 26}" font-size="{fsz}">'
                    f"{words_markup(ref, line, c['ink'], c['muted'])}</text>"
                )
                idx += len(words)
            out.extend(pieces)
        else:
            out.append(level_dots(m, y + 6, 10, 6, lvl, levels, c))
            out.append(
                f'<text x="{m + 270}" y="{y + 8}" font-size="{fsz}">{words_markup(prev, text, c["ink"], c["muted"])}</text>'
            )
        out.append("</g>")
        prev = text
    cy = top + n * row_h - 20
    out.append(f'<line x1="{m}" y1="{cy}" x2="{w - m}" y2="{cy}" stroke="{c["rule"]}" stroke-width="2" stroke-linecap="round" stroke-dasharray="0 6"/>')
    cap = wrap(CAPTION, 46 if mobile else 130)
    for j, line in enumerate(cap):
        out.append(
            f'<text x="{m}" y="{cy + 26 + j * (20 if mobile else 22)}" font-size="{14 if mobile else 15}" '
            f'fill="{c["muted"]}">{escape(line)}</text>'
        )
    out.append("</g></svg>")
    return "\n".join(out) + "\n"


def social() -> str:
    c = THEMES["light"]
    trace, data = load_trace()
    levels = len(data["level_sizes"])
    rows = [r for r in trace if r[0] in SOCIAL_LEVELS]
    width, height, margin = 1280, 640, 56
    px, py, pw, ph = margin, 262, width - 2 * margin, 322
    desc = (
        "Next-Concept Modeling on the Apple GPU: runs the released NCM checkpoints in MLX. "
        "A recorded sample of the official TinyStories model resolves from punctuation to a sentence over successive levels."
    )
    o = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            'role="img" aria-labelledby="title description">'
        ),
        '<title id="title">NCM-MLX</title>',
        f'<desc id="description">{escape(desc)}</desc>',
        f'<rect width="{width}" height="{height}" fill="{c["canvas"]}"/>',
        f'<g font-family="{FONT}" fill="{c["ink"]}">',
        f'<g transform="translate({margin - 8} {margin - 8}) scale(2)"><path fill="{c["ink"]}" d="{LOGO_PATH}"/></g>',
        f'<text x="{margin + 84}" y="{margin + 40}" font-size="56" font-weight="700">NCM-MLX</text>',
        f'<text x="{margin}" y="160" font-size="32" fill="{c["muted"]}">{escape(TAGLINE[0])}</text>',
        f'<text x="{margin}" y="202" font-size="32" fill="{c["muted"]}">{escape(TAGLINE[1])}</text>',
        f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="4" fill="{c["panel"]}"/>',
        (
            f'<text x="{px + 28}" y="{py + 38}" font-size="16" fill="{c["muted"]}">'
            "REAL SAMPLE · official TinyStories model · refined level by level</text>"
        ),
        (
            f'<text x="{px + pw - 28}" y="{py + 38}" font-size="16" fill="{c["muted"]}" text-anchor="end">'
            "github.com/janmus1c/ncm-mlx</text>"
        ),
    ]
    prev = None
    for i, (lvl, codes, text) in enumerate(rows):
        y = py + 100 + i * 62
        o.append(f'<text x="{px + 28}" y="{y}" font-size="16" fill="{c["muted"]}">LEVEL {lvl + 1}</text>')
        o.append(level_dots(px + 138, y - 12, 8, 4, lvl, levels, c))
        o.append(
            f'<text x="{px + 304}" y="{y + 4}" font-size="34">{words_markup(prev, text, c["ink"], c["muted"])}</text>'
        )
        prev = text
    o.append("</g></svg>")
    return "\n".join(o) + "\n"


def main() -> None:
    assets = ROOT / "docs" / "assets"
    for theme in THEMES:
        (assets / f"tinystories-refinement-{theme}.svg").write_text(hero(theme, False))
        (assets / f"tinystories-refinement-mobile-{theme}.svg").write_text(hero(theme, True))
    (ROOT / "assets" / "social-preview.svg").write_text(social())


if __name__ == "__main__":
    main()
