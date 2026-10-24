"""Render the project mark, EPUB cover and social preview from native vector sources."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
INK = "#20252b"


def render(source: str, target: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".typ", mode="w", dir=ROOT / "assets") as f:
        f.write(source)
        f.flush()
        subprocess.run(
            ["typst", "compile", "--root", str(ROOT), "--ppi", "72", f.name, str(target)],
            check=True,
        )


def main() -> None:
    frontmatter = (ROOT / "paper/manuscript.md").read_text().split("---", 2)[1]
    metadata = yaml.safe_load(frontmatter)
    title, _, subtitle = metadata["title"].partition(": ")
    title = json.dumps(title, ensure_ascii=False)
    subtitle = json.dumps(subtitle, ensure_ascii=False)
    author = json.dumps(metadata["author"][0]["name"], ensure_ascii=False)
    date = json.dumps("Working paper " + metadata["date"], ensure_ascii=False)
    shutil.copyfile(ROOT / "assets/logo.svg", ROOT / "paper/assets/logo.svg")
    common = f'''#set page(margin: 0pt, fill: white)
#set text(font: "Helvetica Neue", fill: rgb("{INK}"))
#set par(leading: 0.3em)
'''
    render(
        common + f'''#set page(width: 900pt, height: 1350pt)
#place(top + left, dx: 90pt, dy: 112pt, image("logo.svg", width: 84pt))
#place(top + left, dx: 90pt, dy: 272pt, text(size: 24pt, weight: "medium")[NCM-MLX])
#place(top + left, dx: 90pt, dy: 398pt, block(width: 720pt,
  text(size: 59pt, weight: "bold", hyphenate: false, {title})))
#place(top + left, dx: 90pt, dy: 656pt, block(width: 680pt,
  text(size: 31pt, hyphenate: false, {subtitle})))
#place(top + left, dx: 90pt, dy: 1116pt, text(size: 23pt, {author}))
#place(top + left, dx: 90pt, dy: 1171pt, text(size: 18pt,
  {date}))
''',
        ROOT / "paper/assets/cover.png",
    )
    render(
        common + '''#set page(width: 1200pt, height: 630pt)
#place(top + left, dx: 88pt, dy: 74pt, image("logo.svg", width: 76pt))
#place(top + left, dx: 88pt, dy: 232pt, text(size: 80pt, weight: "bold")[NCM-MLX])
#place(top + left, dx: 88pt, dy: 354pt,
  text(size: 36pt)[Next-Concept Modeling in MLX])
#place(top + left, dx: 88pt, dy: 536pt,
  text(size: 21pt)[github.com/janmus1c/ncm-mlx])
''',
        ROOT / "assets/social-preview.png",
    )
    shutil.copyfile(ROOT / "assets/social-preview.png", ROOT / "docs/assets/social-preview.png")
    print("Rendered the paper logo, cover and social preview")


if __name__ == "__main__":
    main()
