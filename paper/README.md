# Paper

A working paper on this implementation, written to become a proper PDF.

| File | What it is |
|---|---|
| `manuscript.md` | The paper: Pandoc Markdown with YAML metadata, LaTeX maths and `[@key]` citations. |
| `references.bib` | BibTeX references. |
| `make_figures.py` | Collects results from `runs/` into `data/results.json` and draws `figures/`. |
| `data/results.json` | The numbers behind every figure, committed so figures rebuild without `runs/`. |
| `figures/` | SVG, PDF and PNG figures. |
| `style.typ`, `web.css`, `assets/` | PDF layout (NeurIPS conventions), HTML/EPUB style, logo and e-book cover. |
| `dist/` | Built editions: PDF, HTML, EPUB, LaTeX, JATS XML, DOCX. Committed so they can be read without a toolchain; rebuild with `./build.sh` in the same commit as any manuscript change. |

## Rebuild the figures

```bash
uv sync --group paper
.venv/bin/python paper/make_figures.py          # from runs/
.venv/bin/python paper/make_figures.py --plot   # from data/results.json only
```

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
| `ncm-mlx.tex` | Starting point for an arXiv or journal LaTeX submission. |
| `ncm-mlx.jats.xml` | JATS archiving XML, the format journal archives such as PubMed Central ingest. |
| `ncm-mlx.docx` | Word, for reviewers who comment in Word. |

## Rules for this document

- Every number comes from a run in this repository or is quoted from a cited paper and labelled as such.
- Experiments that have not finished are marked *in progress*, never estimated.
- Negative results stay in.
