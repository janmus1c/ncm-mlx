"""Render proposed inference flows; these figures contain no measured performance."""

from textwrap import wrap

from render_diagrams import PALETTES, ROOT, Drawing

FIGURES = {
    "prefix-cache-proposal": {
        "title": "Compute once. Reuse the prefix.",
        "subtitle": "Proposed NCM inference · full hierarchy retained",
        "description": "Proposed generation-local cache: retain prefix keys and values during level zero, "
        "reuse them while computing each later concept level, then decode once. "
        "The implementation and speed benefit remain untested.",
        "cards": [
            ("01  FIRST LEVEL", "Build prefix K/V", ["Retain each layer's state", "Fill during level zero", "No separate prefix pass"]),
            ("02  LATER LEVELS", "Reuse fixed prefix", ["Compute current-level queries", "Read prefix + level K/V", "Preserve masks and positions"]),
            ("03  FINAL OUTPUT", "Decode once", ["Same full code hierarchy", "Same checkpoint", "Equivalence must be tested"]),
        ],
        "footer": ["Proposed, unmeasured optimization.", "Reuse requires an unchanged full prefix."],
    },
    "greedy-drafting-proposal": {
        "title": "Propose. Verify. Commit.",
        "subtitle": "Proposed text experiment · greedy target verification",
        "description": "A future greedy text verifier: propose target-token candidates from committed history, "
        "check their longest matching prefix with the autoregressive target, commit matches plus the target "
        "correction or bonus, and rewind rejected cache entries. No local speed result is established.",
        "cards": [
            ("01  CHEAP PROPOSAL", "Draft the next block", ["Use committed history", "Propose target token IDs", "Count all drafting work"]),
            ("02  TARGET CHECK", "Verify causally", ["Check the matching prefix", "Stop at the first mismatch", "Later states are provisional"]),
            ("03  STATE UPDATE", "Commit and rewind", ["Keep matched tokens", "Add target correction/bonus", "Discard rejected cache state"]),
        ],
        "footer": ["Honor EOS and the remaining output budget.", "Greedy identity is a separate contract from stochastic sampling."],
    },
}


def render(name, spec, theme, mobile):
    d = Drawing(420 if mobile else 1080, 1070 if mobile else 485, theme,
                spec["title"], spec["description"])
    d.text(28 if mobile else 32, 49, spec["title"], 24 if mobile else 32, weight=650)
    if mobile:
        for index, label in enumerate(spec["subtitle"].split(" · ")):
            d.text(28, 89 + index * 26, label, 18, muted=True)
    else:
        d.text(32, 93, spec["subtitle"], 20, muted=True)
    for index, (label, title, lines) in enumerate(spec["cards"]):
        x, y = (28, 146 + index * 265) if mobile else (32 + index * 352, 135)
        width = 364 if mobile else 312
        d.box(x, y, width, 215)
        d.text(x + 20, y + 32, label, 16, muted=True)
        d.text(x + 20, y + 77, title, 25, weight=600)
        for row, line in enumerate(lines):
            d.text(x + 20, y + 119 + row * 31, line, 18 if mobile else 17, muted=True)
        if index < 2:
            d.arrow(210 if mobile else x + width + 5, y + 228 if mobile else y + 101, mobile)
    if mobile:
        d.rule(28, 933, 392, 933)
        row = 0
        for footer in spec["footer"]:
            for line in wrap(footer, 39):
                d.text(28, 970 + row * 27, line, 17, muted=True)
                row += 1
    else:
        d.rule(32, 389, 1048, 389)
        for row, line in enumerate(spec["footer"]):
            d.text(32, 427 + row * 30, line, 18, muted=True)
    suffix = "-mobile" if mobile else ""
    d.save(ROOT / f"docs/assets/{name}{suffix}-{theme}.svg")


def main():
    for name, spec in FIGURES.items():
        for theme in PALETTES:
            for mobile in (False, True):
                render(name, spec, theme, mobile)
    print("Rendered proposed prefix-cache and greedy-verification flows")


if __name__ == "__main__":
    main()
