from __future__ import annotations

import json
import struct
import sys
from pathlib import Path


def generator_budget(directory: Path) -> int:
    for suffix in ("", ".previous"):
        path = directory / f"generator-training{suffix}.safetensors"
        try:
            with path.open("rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                if size > path.stat().st_size - 8:
                    raise ValueError("incomplete header")
                header = json.loads(f.read(size))
            return int(json.loads(header["__metadata__"]["training"])["total_steps"])
        except (OSError, ValueError, KeyError, struct.error):
            continue
    if any(directory.glob("codec-training*.safetensors")):
        return int(json.loads((directory / "config.json").read_text())["train"]["generator_steps"])
    raise ValueError(f"no full training snapshot in {directory}")


if __name__ == "__main__":
    print(generator_budget(Path(sys.argv[1])))
