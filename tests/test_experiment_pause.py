import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest


@pytest.mark.parametrize("watchdog,stop_signal", [(False, signal.SIGTERM), (True, signal.SIGUSR1)])
def test_experiment_wrapper_forwards_pause_and_skips_eval(tmp_path, watchdog, stop_signal):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "scripts").mkdir()
    (tmp_path / "runs").mkdir()
    (tmp_path / ".venv/bin").mkdir(parents=True)
    for name in ("run_experiment.sh", "guarded_run.sh"):
        shutil.copy(root / "scripts" / name, tmp_path / "scripts" / name)
    py = tmp_path / ".venv/bin/python"
    py.write_text("""#!/usr/bin/python3
import os, signal, sys, time
from pathlib import Path
def pause(sig, frame):
    print('paused mock training', flush=True)
    sys.exit(75)
signal.signal(signal.SIGUSR1, pause)
signal.signal(signal.SIGTERM, pause)
Path('ready').write_text(str(os.getpid()))
while True:
    time.sleep(0.01)
""")
    py.chmod(0o755)
    pgrep = tmp_path / ".venv/bin/pgrep"
    pgrep.write_text("#!/bin/sh\nexit 1\n")
    pgrep.chmod(0o755)
    command = ["bash", "scripts/run_experiment.sh", "--worker", "configs/mock.json"]
    if watchdog:
        command = ["bash", "scripts/guarded_run.sh", "-i", "0.05", "-l", "runs/mem.log", "--", *command]
    env = {**os.environ, "PATH": str(py.parent) + os.pathsep + os.environ["PATH"]}
    process = subprocess.Popen(command, cwd=tmp_path, env=env, start_new_session=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(0.01)
        assert (tmp_path / "ready").exists()
        process.send_signal(stop_signal)
        assert process.wait(timeout=5) == 75
        assert "paused mock training" in (tmp_path / "runs/mock-train.log").read_text()
        assert not (tmp_path / "runs/mock-eval.log").exists()
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
