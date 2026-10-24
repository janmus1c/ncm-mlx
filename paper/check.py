"""Layout and consistency checks for the paper; run by build.sh after every build.

    .venv/bin/python paper/check.py

Fails (exit 1) when the version history is not in ascending order, the title-block version
is not the newest row, a referenced figure is missing, a page other than the last is nearly
empty, or anything other than the page number is printed in the bottom margin (content
running past the text block, e.g. a table that does not break across pages).
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

PAPER = Path(__file__).resolve().parent
PPI = 50
MARGIN_IN = 1.0


def check_text(md: str) -> list[str]:
    errors = []
    rows = re.findall(r"^\| (\d+)\.(\d+) \| (\d{4}-\d{2}-\d{2}) \|", md, flags=re.MULTILINE)
    versions = [(int(a), int(b)) for a, b, _ in rows]
    if versions != sorted(versions) or len(set(versions)) != len(versions):
        errors.append(f"version history not strictly ascending: {versions}")
    title = re.search(r'^date: "v(\d+)\.(\d+)', md, flags=re.MULTILINE)
    if title and versions and (int(title[1]), int(title[2])) != versions[-1]:
        errors.append(f"title block says v{title[1]}.{title[2]}, newest history row is v{versions[-1][0]}.{versions[-1][1]}")
    for fig in re.findall(r"\]\((figures/[^)]+)\)", md):
        if not (PAPER / fig).exists():
            errors.append(f"missing figure {fig}")
    return errors


def render_pages(tmp: Path) -> list[np.ndarray]:
    typ = PAPER / ".check.typ"
    try:
        subprocess.run(["pandoc", "manuscript.md", "--citeproc", "--resource-path=.", "-V", "template=style.typ",
                        "-t", "typst", "-s", "-o", typ.name], cwd=PAPER, check=True)
        subprocess.run(["typst", "compile", typ.name, str(tmp / "page-{n}.png"), "--ppi", str(PPI)],
                       cwd=PAPER, check=True, capture_output=True)
    finally:
        typ.unlink(missing_ok=True)
    pages = sorted(tmp.glob("page-*.png"), key=lambda p: int(p.stem.split("-")[1]))
    return [mpimg.imread(p)[..., :3].mean(axis=-1) for p in pages]


def check_pages(pages: list[np.ndarray]) -> list[str]:
    errors = []
    band = int(MARGIN_IN * PPI)
    for n, page in enumerate(pages, start=1):
        h, w = page.shape
        ink = page < 0.6
        body = ink[band:h - band, int(0.17 * w):int(0.83 * w)]
        if n < len(pages) and body.mean() < 0.004:
            errors.append(f"page {n} is nearly empty")
        bottom = ink[h - band + 4:h - 2].copy()
        bottom[:, int(0.42 * w):int(0.58 * w)] = False
        if bottom.sum() > 3:
            errors.append(f"page {n}: content printed in the bottom margin")
    return errors


def main() -> int:
    errors = check_text((PAPER / "manuscript.md").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        errors += check_pages(render_pages(Path(tmp)))
    for e in errors:
        print(f"paper check: {e}", file=sys.stderr)
    print(f"paper check: {'FAILED' if errors else 'ok'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
