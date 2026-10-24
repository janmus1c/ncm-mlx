import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("producer", ["ldd", "driver", "cap"])
def test_preflight_drains_version_output_before_installing(tmp_path, producer):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / "check_gpu.sh"
    shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts/check_gpu.sh", script)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name in ["nvidia-smi", "ldd"]:
        path = binaries / name
        path.write_text(
            f"#!{shutil.which('python3')}\n"
            "import os, signal, sys\n"
            "signal.signal(signal.SIGPIPE, signal.SIG_DFL)\n"
            "kind = 'ldd' if os.path.basename(sys.argv[0]) == 'ldd' else "
            "('driver' if '--query-gpu=driver_version' in sys.argv else "
            "'cap' if '--query-gpu=compute_cap' in sys.argv else 'report')\n"
            "line = {'ldd': 'ldd (GNU libc) 2.39', 'driver': '595.91.07', "
            "'cap': '8.6', 'report': 'NVIDIA A40, 46068 MiB, 0 MiB, 595.91.07, 8.6'}[kind] + '\\n'\n"
            f"count = 65536 if kind == {producer!r} else 1\n"
            "payload = (line * count).encode()\n"
            "while payload:\n"
            "    payload = payload[os.write(1, payload):]\n"
        )
        path.chmod(0o755)
    uv = binaries / "uv"
    uv.write_text("#!/bin/sh\necho install-reached\nexit 42\n")
    uv.chmod(0o755)
    result = subprocess.run(
        ["bash", str(script)],
        env={**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 42, result.stdout + result.stderr
    assert "OK: driver 595.91.07, compute capability 8.6, glibc 2.39" in result.stdout
    assert "install-reached" in result.stdout
