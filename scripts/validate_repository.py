"""Validate public metadata, experiment configs and local Markdown links without MLX."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

from ncm_mlx.config import Config


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    citation = yaml.safe_load((root / "CITATION.cff").read_text())
    assert citation["cff-version"] == "1.2.0"
    assert citation["type"] == "software"
    assert citation["version"] == project["version"]
    assert citation["license"] == project["license"] == "MIT"
    assert citation["title"] and citation["authors"] and citation["message"]
    assert project["urls"]["Repository"] == "https://github.com/janmus1c/ncm-mlx"
    configs = list((root / "configs").glob("*.json"))
    for config in configs:
        Config.load(config)
    pages = [*root.glob("*.md"), *(root / "docs").rglob("*.md")]
    pages = [page for page in pages if not page.name.endswith(".local.md")]
    failures = []
    for page in pages:
        text = re.sub(r"```.*?```", "", page.read_text(), flags=re.DOTALL)
        destinations = re.findall(r"\[[^\]]*\]\(([^)]+)\)", text)
        destinations += re.findall(r'(?:src|srcset)="([^"\s]+)"', text)
        for raw in destinations:
            destination = raw.split(' "', 1)[0].strip("<>")
            parsed = urlsplit(destination)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = (page.parent / unquote(parsed.path)).resolve()
            if not target.is_relative_to(root) or not target.exists():
                failures.append(f"{page.relative_to(root)}: {destination}")
    if failures:
        raise SystemExit("Invalid local documentation links:\n" + "\n".join(failures))
    print(f"Validated citation, {len(configs)} configs and links in {len(pages)} Markdown files")


if __name__ == "__main__":
    main()
