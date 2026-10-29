"""Render the README refinement animation and the GitHub social preview.

The trace is the real TinyWorld sample documented in README.md (`mise run demo`).
Outputs: docs/assets/refinement-{light,dark}.svg, docs/assets/refinement-mobile-{light,dark}.svg
and assets/social-preview.svg.
"""

from __future__ import annotations

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

# (level, code positions, text) from the recorded sample in README.md. Level 4 is not in the excerpt.
TRACE = [
    (0, 1, "eue likes one oed egg."),
    (1, 2, "eve likes one oig egg."),
    (2, 4, "dee likes one red egg."),
    (3, 8, "dee likes one ret egg."),
    (5, 32, "dee likes one ret egg."),
]
FINAL_NOTE = "final: invalid (spelling)"
TITLE = "One real sample, refined level by level"
DESC = (
    "A recorded TinyWorld generation. Level 0 predicts one code and gives a rough guess, "
    "eue likes one oed egg. Levels 1 to 3 settle the words, reaching dee likes one red egg. "
    "A fine level then breaks red into ret, and the final level 5 output, dee likes one ret egg., "
    "is invalid for spelling."
)
CAPTION = (
    "Real sample from the trained TinyWorld model (mise run demo), "
    "chosen because it shows both the refinement and the typical failure."
)


def words_markup(prev: str | None, text: str, ink: str, muted: str) -> str:
    cur = text.split(" ")
    old = prev.split(" ") if prev else [None] * len(cur)
    parts = []
    for i, word in enumerate(cur):
        changed = prev is None or word != old[i]
        fill, weight = (ink, 650) if changed else (muted, 400)
        parts.append(f'<tspan fill="{fill}" font-weight="{weight}">{escape(word)}</tspan>')
    return " ".join(parts)


def squares(x: float, y: float, n: int, size: int, gap: int, fill: str) -> str:
    return "".join(
        f'<rect x="{x + i * (size + gap):g}" y="{y:g}" width="{size}" height="{size}" rx="2" fill="{fill}"/>'
        for i in range(n)
    )


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
    w = 420 if mobile else 1080
    m = 28 if mobile else 32
    n = len(TRACE)
    if mobile:
        row_h, top, fsz, tile, gap = 128, 168, 25, 8, 3
        h = top + n * row_h + 112
    else:
        row_h, top, fsz, tile, gap = 74, 120, 30, 8, 4
        h = top + n * row_h + 76
    out = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title description">'
            f'<title id="title">{escape(TITLE)}</title><desc id="description">{escape(DESC)}</desc>'
        ),
        f"<style>{animation_css(n, 14, 2.0)}</style>",
        f'<rect width="{w}" height="{h}" rx="16" fill="{c["canvas"]}"/>',
        f'<g font-family="{FONT}" fill="{c["ink"]}">',
    ]
    if mobile:
        out.append(f'<text x="{m}" y="54" font-size="25" font-weight="650">One real sample,</text>')
        out.append(f'<text x="{m}" y="84" font-size="25" font-weight="650">refined level by level</text>')
        out.append(
            f'<text x="{m}" y="114" font-size="16" fill="{c["muted"]}">REAL SAMPLE · TinyWorld · 6 levels</text>'
        )
    else:
        out.append(f'<text x="{m}" y="54" font-size="30" font-weight="650">{escape(TITLE)}</text>')
        out.append(
            f'<text x="{m}" y="84" font-size="19" fill="{c["muted"]}">REAL SAMPLE · TinyWorld · 6 levels, 32 characters</text>'
        )
    prev = None
    for i, (lvl, codes, text) in enumerate(TRACE):
        y = top + i * row_h
        plural = "s" if codes != 1 else ""
        markup = words_markup(prev, text, c["ink"], c["muted"])
        out.append(f'<g class="r{i}">')
        out.append(
            f'<line x1="{m}" y1="{y - 24}" x2="{w - m}" y2="{y - 24}" stroke="{c["rule"]}" stroke-width="1"/>'
        )
        if mobile:
            out.append(
                f'<text x="{m}" y="{y}" font-size="16" fill="{c["muted"]}">LEVEL {lvl}  ·  {codes} code{plural}</text>'
            )
            out.append(squares(m, y + 14, codes, tile, gap, c["ink"]))
            out.append(f'<text x="{m}" y="{y + 66}" font-size="{fsz}">{markup}</text>')
            if i == n - 1:
                out.append(
                    f'<text x="{m}" y="{y + 92}" font-size="16" fill="{c["muted"]}">{FINAL_NOTE}</text>'
                )
        else:
            out.append(f'<text x="{m}" y="{y + 4}" font-size="16" fill="{c["muted"]}">LEVEL {lvl}</text>')
            out.append(
                f'<text x="{m + 90}" y="{y + 4}" font-size="16" fill="{c["muted"]}">{codes} code{plural}</text>'
            )
            out.append(squares(m, y + 20, codes, tile, gap, c["ink"]))
            out.append(f'<text x="{m + 440}" y="{y + 18}" font-size="{fsz}">{markup}</text>')
            if i == n - 1:
                out.append(
                    f'<text x="{w - m}" y="{y + 18}" font-size="19" fill="{c["muted"]}" text-anchor="end">{FINAL_NOTE}</text>'
                )
        out.append("</g>")
        prev = text
    cy = top + n * row_h - 4
    out.append(f'<line x1="{m}" y1="{cy}" x2="{w - m}" y2="{cy}" stroke="{c["rule"]}" stroke-width="1"/>')
    if mobile:
        lines = [
            "Real sample from the trained TinyWorld",
            "model (mise run demo), chosen because it",
            "shows the refinement and the typical failure.",
            "Level 4 is omitted from the excerpt.",
        ]
        for j, line in enumerate(lines):
            out.append(
                f'<text x="{m}" y="{cy + 28 + j * 22}" font-size="15" fill="{c["muted"]}">{line}</text>'
            )
    else:
        out.append(f'<text x="{m}" y="{cy + 30}" font-size="16" fill="{c["muted"]}">{escape(CAPTION)}</text>')
        out.append(
            f'<text x="{m}" y="{cy + 52}" font-size="16" fill="{c["muted"]}">'
            "Level 4 is omitted from the recorded excerpt. Bold words changed at that level.</text>"
        )
    out.append("</g></svg>")
    return "\n".join(out) + "\n"


def social() -> str:
    c = THEMES["light"]
    width, height, margin = 1280, 640, 56
    px, py, pw, ph = margin, 258, width - 2 * margin, 326
    o = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            'role="img" aria-labelledby="title description">'
        ),
        '<title id="title">NCM-MLX</title>',
        (
            '<desc id="description">Next-Concept Modeling in MLX: parallel within a level, coarse to fine across levels. '
            "A real TinyWorld sample resolves from a rough guess to a sentence over successive levels.</desc>"
        ),
        f'<rect width="{width}" height="{height}" fill="{c["canvas"]}"/>',
        f'<g font-family="{FONT}" fill="{c["ink"]}">',
        f'<g transform="translate({margin - 8} {margin - 8}) scale(2)"><path fill="{c["ink"]}" d="{LOGO_PATH}"/></g>',
        f'<text x="{margin + 84}" y="{margin + 40}" font-size="56" font-weight="700">NCM-MLX</text>',
        (
            f'<text x="{margin}" y="170" font-size="32" fill="{c["muted"]}">'
            "Next-Concept Modeling in MLX — parallel within a level,</text>"
        ),
        f'<text x="{margin}" y="212" font-size="32" fill="{c["muted"]}">coarse to fine across levels</text>',
        f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="4" fill="{c["panel"]}"/>',
        (
            f'<text x="{px + 28}" y="{py + 38}" font-size="16" fill="{c["muted"]}">'
            "REAL SAMPLE · TinyWorld · one forward pass per level</text>"
        ),
        (
            f'<text x="{px + pw - 28}" y="{py + 38}" font-size="16" fill="{c["muted"]}" text-anchor="end">'
            "github.com/janmus1c/ncm-mlx</text>"
        ),
    ]
    prev = None
    for i, (lvl, codes, text) in enumerate(TRACE[:4]):
        y = py + 100 + i * 62
        o.append(f'<text x="{px + 28}" y="{y}" font-size="16" fill="{c["muted"]}">LEVEL {lvl}</text>')
        o.append(squares(px + 118, y - 10, codes, 8, 4, c["ink"]))
        o.append(
            f'<text x="{px + 330}" y="{y + 4}" font-size="34">{words_markup(prev, text, c["ink"], c["muted"])}</text>'
        )
        prev = text
    o.append("</g></svg>")
    return "\n".join(o) + "\n"


def main() -> None:
    assets = ROOT / "docs" / "assets"
    for theme in THEMES:
        (assets / f"refinement-{theme}.svg").write_text(hero(theme, False))
        (assets / f"refinement-mobile-{theme}.svg").write_text(hero(theme, True))
    (ROOT / "assets" / "social-preview.svg").write_text(social())


if __name__ == "__main__":
    main()
