"""Render the documentation's native SVG diagrams in both reading themes."""

from __future__ import annotations

from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PALETTES = {
    "light": ("#ffffff", "#20252b", "#59636e", "#d6dce2", "#f3f5f7"),
    "dark": ("#13161a", "#f1f2f3", "#b4bac1", "#454c55", "#1c2025"),
}


class Drawing:
    def __init__(self, width: int, height: int, theme: str, title: str, description: str):
        self.bg, self.ink, self.muted, self.line, self.surface = PALETTES[theme]
        self.parts = [
            (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
             'role="img" aria-labelledby="title description">'),
            f'<title id="title">{escape(title)}</title><desc id="description">{escape(description)}</desc>',
            f'<rect width="{width}" height="{height}" rx="16" fill="{self.bg}"/>',
            f'<g font-family="Inter, Helvetica Neue, Arial, sans-serif" fill="{self.ink}">',
        ]

    def text(self, x, y, value, size=23, muted=False, weight=400, anchor="start"):
        color = self.muted if muted else self.ink
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color}" text-anchor="{anchor}">{escape(value)}</text>'
        )

    def rule(self, x1, y1, x2, y2):
        self.parts.append(f'<path d="M{x1} {y1}H{x2}" stroke="{self.line}"/>' if y1 == y2 else
                          f'<path d="M{x1} {y1}L{x2} {y2}" stroke="{self.line}"/>')

    def box(self, x, y, w, h, solid=False):
        color = self.ink if solid else self.surface
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{color}"/>')

    def arrow(self, x, y, vertical=False):
        path = f'M{x} {y}v30m-5 -6l5 6 5 -6' if vertical else f'M{x} {y}h30m-6 -5l6 5 -6 5'
        self.parts.append(f'<path d="{path}" fill="none" stroke="{self.muted}" stroke-width="2"/>')

    def tiles(self, x, y, count, size=12, columns=8):
        for i in range(count):
            self.box(x + (i % columns) * (size + 4), y + (i // columns) * (size + 4), size, size, True)

    def save(self, path):
        path.write_text("\n".join([*self.parts, "</g></svg>\n"]))


def architecture(theme: str, mobile: bool) -> None:
    title = "Two stages. One hierarchy."
    desc = ("Train a codec to encode, quantize and reconstruct text. Freeze the codec, then train a generator "
            "to predict the code hierarchy. During inference, generate codes and decode once.")
    d = Drawing(420 if mobile else 1080, 1050 if mobile else 570, theme, title, desc)
    d.text(32, 52, title, 28 if mobile else 34, weight=650)
    if mobile:
        d.text(32, 96, "01  Learn a discrete code hierarchy", 19, muted=True)
        d.box(32, 122, 356, 94)
        d.text(52, 160, "Text", 23, weight=600)
        d.text(52, 192, "gus finds two blue hens.", 20, muted=True)
        d.arrow(210, 230, vertical=True)
        d.text(52, 298, "Encoder → residual VQ", 24, weight=600)
        d.text(52, 332, "Coarse codes, then finer residuals", 19, muted=True)
        for j, n in enumerate((1, 2, 4, 8)):
            d.tiles(52, 355 + j * 23, n, 14)
        d.text(232, 404, "Discrete", 21, muted=True)
        d.text(232, 433, "hierarchy", 21, muted=True)
        d.arrow(210, 465, vertical=True)
        d.text(52, 532, "Decoder → rebuilt text", 24, weight=600)
        d.rule(32, 566, 388, 566)
        d.text(32, 610, "02  Freeze the codec. Learn to generate.", 18, muted=True)
        d.box(32, 638, 356, 102)
        d.text(52, 677, "Next-concept transformer", 23, weight=600)
        d.text(52, 710, "Predict a whole level in parallel", 19, muted=True)
        d.arrow(210, 755, vertical=True)
        d.text(52, 823, "Coarse → fine codes", 24, weight=600)
        d.text(52, 857, "Earlier levels condition the next", 19, muted=True)
        d.arrow(210, 882, vertical=True)
        d.text(52, 949, "Decode once → generated text", 22, weight=600)
        d.text(32, 1013, "Shared idea; task-specific codec settings.", 17, muted=True)
    else:
        d.text(32, 101, "01  Learn a discrete code hierarchy", 20, muted=True)
        d.box(32, 133, 242, 132)
        d.text(52, 180, "Text", 25, weight=600)
        d.text(52, 218, "gus finds two blue hens.", 19, muted=True)
        d.arrow(287, 199)
        d.text(340, 167, "Encoder → residual VQ", 26, weight=600)
        d.text(340, 204, "A hierarchy of discrete codes", 20, muted=True)
        for j, n in enumerate((1, 2, 4, 8)):
            d.tiles(340, 224 + j * 17, n, 10)
        d.text(510, 249, "coarse → fine", 19, muted=True)
        d.arrow(727, 199)
        d.box(780, 133, 268, 132)
        d.text(802, 180, "Decoder", 25, weight=600)
        d.text(802, 218, "Rebuild the input text", 20, muted=True)
        d.rule(32, 320, 1048, 320)
        d.text(32, 365, "02  Freeze the codec. Learn to generate its hierarchy.", 20, muted=True)
        d.box(32, 396, 332, 118)
        d.text(52, 437, "Next-concept transformer", 24, weight=600)
        d.text(52, 474, "Every position of a level at once", 19, muted=True)
        d.arrow(379, 455)
        d.text(432, 437, "Coarse → fine codes", 25, weight=600)
        d.text(432, 474, "Earlier levels condition the next", 19, muted=True)
        d.arrow(727, 455)
        d.box(780, 396, 268, 118)
        d.text(802, 437, "Decode once", 25, weight=600)
        d.text(802, 474, "Generated text", 20, muted=True)
        d.text(32, 551, "Shared architecture; TinyWorld and Sudoku use different codec settings.", 17, muted=True)
    suffix = "-mobile" if mobile else ""
    d.save(ROOT / f"docs/assets/architecture{suffix}-{theme}.svg")


def generation(theme: str, mobile: bool) -> None:
    desc = ("TinyWorld generates levels of 1, 2, 4, 8, 16 and 32 code positions. Each level uses one "
            "generator forward, and its positions are predicted in parallel. One final decoder produces text.")
    d = Drawing(420 if mobile else 1080, 790 if mobile else 350, theme,
                "Sequential between levels. Parallel within each level.", desc)
    if mobile:
        d.text(28, 48, "Six levels. One final decode.", 25, weight=650)
        for k, n in enumerate((1, 2, 4, 8, 16, 32)):
            y = 105 + 92 * k
            d.text(28, y + 16, f"{k:02}", 18, muted=True)
            d.text(76, y + 17, f"{n} code" + ("s" if n > 1 else ""), 21, weight=600)
            d.tiles(210, y, n, 11, 8)
            if k < 5:
                d.rule(76, y + 53, 388, y + 53)
        d.rule(28, 663, 392, 663)
        d.text(28, 707, "6 generator forwards + 1 decode", 23, weight=600)
        d.text(28, 744, "32 output characters · TinyWorld recipe", 17, muted=True)
        d.text(28, 772, "Each row is predicted in parallel.", 17, muted=True)
    else:
        d.text(32, 51, "Sequential between levels. Parallel within each level.", 30, weight=650)
        for k, n in enumerate((1, 2, 4, 8, 16, 32)):
            x = 32 + 174 * k
            d.text(x, 107, f"LEVEL {k}", 16, muted=True)
            d.text(x, 147, str(n), 32, weight=600)
            d.text(x + (24 if n < 10 else 44), 147, "code" + ("s" if n > 1 else ""), 19, muted=True)
            d.tiles(x, 175, n, 12, 8)
            if k < 5:
                d.arrow(x + 124, 102)
        d.rule(32, 266, 1048, 266)
        d.text(32, 310, "6 generator forwards + 1 decoder pass", 24, weight=600)
        d.text(1048, 310, "32 characters · TinyWorld", 20, muted=True, anchor="end")
    suffix = "-mobile" if mobile else ""
    d.save(ROOT / f"docs/assets/generation{suffix}-{theme}.svg")


def main() -> None:
    mark = (ROOT / "assets/logo.svg").read_text()
    (ROOT / "assets/logo-dark.svg").write_text(mark.replace("#20252b", "#f1f2f3"))
    (ROOT / "docs/assets/logo.svg").write_text(mark)
    (ROOT / "docs/assets/logo-dark.svg").write_text(mark.replace("#20252b", "#f1f2f3"))
    for theme in PALETTES:
        for mobile in (False, True):
            architecture(theme, mobile)
            generation(theme, mobile)
    print("Rendered theme-aware architecture and generation diagrams")


if __name__ == "__main__":
    main()
