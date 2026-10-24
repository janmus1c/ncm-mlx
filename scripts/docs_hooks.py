from __future__ import annotations

import re
import shutil
from pathlib import Path
from urllib.parse import urlsplit

from mkdocs.utils import get_relative_url

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/janmus1c/ncm-mlx/blob/main/"


def on_page_markdown(markdown, page, config, files):
    def rewrite(match):
        label, target = match.groups()
        parsed = urlsplit(target)
        if parsed.scheme or target.startswith("#"):
            return match.group(0)
        resolved = (ROOT / "docs" / page.file.src_path).parent / parsed.path
        resolved = resolved.resolve()
        if not resolved.is_relative_to(ROOT / "docs") and resolved.is_relative_to(ROOT):
            relative = resolved.relative_to(ROOT).as_posix()
            suffix = f"#{parsed.fragment}" if parsed.fragment else ""
            return f"[{label}]({REPOSITORY}{relative}{suffix})"
        return match.group(0)

    markdown = re.sub(r"\[([^\]]+)\]\(([^\s)]+)\)", rewrite, markdown)

    def rewrite_image(match):
        attribute, target = match.groups()
        parsed = urlsplit(target)
        if parsed.scheme or target.startswith(("#", "/")):
            return match.group(0)
        resolved = (ROOT / "docs" / page.file.src_path).parent / parsed.path
        resolved = resolved.resolve()
        if resolved.is_relative_to(ROOT / "docs"):
            destination = resolved.relative_to(ROOT / "docs").as_posix()
            return f'{attribute}="{get_relative_url(destination, page.url)}"'
        return match.group(0)

    return re.sub(r'(src|srcset)="([^"\s]+)"', rewrite_image, markdown)


def on_post_build(config):
    destination = Path(config["site_dir"])
    for name in ("llms.txt", "CITATION.cff"):
        shutil.copyfile(ROOT / name, destination / name)
    paper = destination / "paper"
    paper.mkdir(exist_ok=True)
    for edition in (ROOT / "paper/dist").glob("ncm-mlx.*"):
        shutil.copyfile(edition, paper / edition.name)
