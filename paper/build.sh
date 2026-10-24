#!/usr/bin/env bash
# Build every edition of the paper into paper/dist/ (needs pandoc >= 3.1 and typst).
#
#   paper/build.sh            all formats
#   paper/build.sh pdf html   selected formats: pdf html epub tex jats docx
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p dist
formats=${*:-pdf html epub tex jats docx}
../.venv/bin/python check.py || { echo "fix the paper check failures above before building" >&2; exit 1; }
common=(--citeproc --resource-path=.)
# Browsers, e-readers and Word cannot show PDF figures; other editions use SVG or PNG.
with_figures() { sed -E "s#(figures/[a-z0-9-]+)\.pdf#\1.$1#g" manuscript.md; }

for f in $formats; do
  case $f in
    pdf)  pandoc manuscript.md "${common[@]}" -V template=style.typ --pdf-engine=typst -o dist/ncm-mlx.pdf ;;
    html) with_figures svg | pandoc "${common[@]}" --number-sections --math-method=mathml --embed-resources --standalone \
            --css=web.css --toc --toc-depth=1 -o dist/ncm-mlx.html ;;
    epub) with_figures svg | pandoc "${common[@]}" --number-sections --math-method=mathml --css=web.css \
            --epub-cover-image=assets/cover.png -o dist/ncm-mlx.epub ;;
    tex)  pandoc manuscript.md "${common[@]}" --number-sections -s -o dist/ncm-mlx.tex ;;
    jats) with_figures svg | pandoc "${common[@]}" -s -t jats_archiving -o dist/ncm-mlx.jats.xml ;;
    docx) with_figures png | pandoc "${common[@]}" --number-sections -o dist/ncm-mlx.docx ;;
    *) echo "unknown format: $f" >&2; exit 2 ;;
  esac
  echo "built dist/ncm-mlx.$( [ "$f" = jats ] && echo jats.xml || echo "$f" )"
done
