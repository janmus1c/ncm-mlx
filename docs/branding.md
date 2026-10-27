---
title: Project mark and visual assets
description: The NCM-MLX monochrome mark, accessible usage and reproducible cover assets.
---

# Project mark and visual assets

The NCM-MLX mark shows one coarse block refining into two, then four parallel blocks. It is a visual reference to coarse-to-fine generation, rather than an exact diagram of every model configuration. The seven solid blocks remain clear at 24 pixels; no border, outline, gradient or additional symbol is required.

The canonical source is [`assets/logo.svg`](../assets/logo.svg), with a transparent background and one charcoal fill, `#20252b`. Use charcoal on white or a light neutral background. On dark backgrounds, display the mark in white, for example with CSS `filter: brightness(0) invert(1)`. Keep its square proportions and leave at least one small block's width of clear space around it. Use descriptive alternative text when the mark identifies the project, or empty alternative text when adjacent text already gives the name.

The project name is **NCM-MLX**. Its descriptive title is **Next-Concept Modeling in MLX**. Keep benchmark plots' scientific color encodings independent from the monochrome project mark.

## Rebuild assets

```bash
uv run --group paper python scripts/render_branding.py
uv run python scripts/render_diagrams.py
uv run --group paper python scripts/render_research_visuals.py
```

This requires Typst and the development dependencies. The script copies the canonical vector mark to `paper/assets/logo.svg` and renders the EPUB cover and 1200 × 630 social preview using Typst. The cover reads its title, author and version from `paper/manuscript.md`; update the manuscript rather than editing the PNG. Rendering uses Helvetica Neue; another machine needs that font for matching typography. No font files are redistributed.

The mark and layout were drawn for this repository from native vector geometry. They use no external logo, stock image or generated illustration and are included under the repository's [MIT license](../LICENSE). The mark does not imply affiliation with the NCM authors, HazyResearch or MLX maintainers.

## Architecture visuals

The README and architecture guide use native SVG illustrations rather than diagram-renderer syntax. `scripts/render_diagrams.py` produces light and dark assets, with separate mobile compositions, from the same labels and level schedule. The figures describe the two training stages and TinyWorld's six generation levels; they do not imply that every configuration has six levels. Each SVG includes a text title and description.

## Research visuals

`scripts/render_research_visuals.py` extends the same palettes and vector primitives with Sudoku inference, the E19/E20 seed comparison and the research evidence gates. The Sudoku schedule is read from `configs/sudoku-e20-seed1.json`; the bar chart reads `docs/research/e20-seed-assessment.json` and is rendered with Matplotlib. These commands do not run a model or add experimental evidence.

Each figure has light/dark and desktop/mobile compositions. Grid icons are schematic. The result chart preserves the 0–100% scale, both training seeds, sample counts, historical audit limits and the unmatched Original NCM reference. The research overview explicitly distinguishes measured work from planned text validation.
