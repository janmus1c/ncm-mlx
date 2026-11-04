# Paper

An unreviewed working paper on this independent implementation. Version 0.13 adds a TinyStories scaling study (generator size and data), bf16 parity, follow-up runs and the released XSum generator; version 0.12 added an inference-parity study of the released checkpoints, the TinyStories corruption ablation with two generator seeds, and a roadmap. It retains the TinyWorld and Sudoku results, negative outcomes, experimental limitations and the disclosure of OpenAI Codex assistance with implementation, experiments, analysis, figures, documentation and manuscript drafting and revision.

| File | What it is |
|---|---|
| `manuscript.md` | The paper: Pandoc Markdown with YAML metadata, LaTeX maths and `[@key]` citations. |
| `references.bib` | BibTeX references. |
| `make_figures.py` | Collects results from `runs/` into `data/results.json` and draws `figures/`. |
| `data/results.json` | The numbers behind the Sudoku and latency figures, committed so figures rebuild without `runs/`. |
| `data/tinystories-corruption.json` | The numbers behind the corruption-ablation figure, copied from the saved TinyStories evaluation reports. |
| `figures/` | SVG, PDF and PNG figures. |
| `style.typ`, `web.css`, `assets/` | PDF layout (NeurIPS conventions), HTML/EPUB style, logo and e-book cover. |
| `dist/` | Built editions: PDF, HTML, EPUB, LaTeX, JATS XML, DOCX. Committed so they can be read without a toolchain; rebuild with `./build.sh` in the same commit as any manuscript change. |

## Rebuild the replication assessment

```bash
python paper/analyze_replication.py  # standard library; committed aggregate inputs
```

This derives counts, Wilson intervals, descriptive two-seed means and gaps to the quoted reference. It does not rerun models or infer paired significance, population seed variance or causal mechanisms.

## Rebuild the figures

```bash
uv sync --group paper
.venv/bin/python paper/make_figures.py          # collect local measured runs (requires runs/)
.venv/bin/python paper/make_figures.py --plot   # fresh checkout: committed data/results.json only
```

`figures/project-roadmap.*` is the light edition of `docs/assets/roadmap-2026-11-light.svg`, drawn by `scripts/render_research_visuals.py` (PDF and PNG printed from the SVG).

## Build the editions

Needs [Pandoc](https://pandoc.org) 3.1 or later and [Typst](https://typst.app), both open source.

```bash
paper/build.sh              # every edition into paper/dist/
paper/build.sh pdf html     # selected editions: pdf html epub tex jats docx
```

| Edition | Use |
|---|---|
| `ncm-mlx.pdf` | Reading and printing; NeurIPS-style layout. |
| `ncm-mlx.html` | One self-contained web page with MathML maths; follows light and dark mode. |
| `ncm-mlx.epub` | E-readers (Apple Books, Kobo, Calibre). |
| `ncm-mlx.tex` | LaTeX source for submission preparation; compile and inspect it separately from the Typst PDF. |
| `ncm-mlx.jats.xml` | JATS archiving XML, the format journal archives such as PubMed Central ingest. |
| `ncm-mlx.docx` | Word, for reviewers who comment in Word. |

## Rules for this document

- Every number comes from a run in this repository or is quoted from a cited paper and labelled as such.
- Experiments that have not finished are marked *in progress*, never estimated.
- Negative results stay in.

The PDF edition uses Typst. The LaTeX export includes its formatted bibliography and references PDF figures in `figures/`; upload only the main TeX source, referenced figures and required license notices after a separate compilation check. A local build is not evidence of arXiv acceptance or independent human review.
