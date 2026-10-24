import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("train_exit", [0, 42])
def test_remote_resume_uploads_state_retains_budget_and_copies_back_on_failure(tmp_path, train_exit):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "scripts").mkdir(parents=True)
    for name in ("remote_run.sh", "checkpoint_steps.py"):
        shutil.copy(root / "scripts" / name, tmp_path / "scripts" / name)
    (tmp_path / "configs").mkdir()
    (tmp_path / "runs").mkdir()
    checkpoint = tmp_path / "artifacts/example"
    checkpoint.mkdir(parents=True)
    cfg = {"generator": {"n_embd": 8}, "train": {"batch_size": 2, "generator_steps": 100}}
    (tmp_path / "configs/example.json").write_text(json.dumps(cfg))
    meta = json.dumps({"__metadata__": {"training": json.dumps({"total_steps": 12345})}}).encode()
    (checkpoint / "generator-training.safetensors").write_bytes(struct.pack("<Q", len(meta)) + meta)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    mock = binaries / "mock"
    mock.write_text("""#!/usr/bin/python3
import json, os, sys
from pathlib import Path
with open(os.environ['COMMAND_LOG'], 'a') as f:
    f.write(json.dumps([Path(sys.argv[0]).name, *sys.argv[1:]]) + '\\n')
if Path(sys.argv[0]).name == 'ssh':
    command = sys.argv[-1]
    if 'pgrep -f' in command:
        sys.exit(1)
    if 'timeout ' in command:
        sys.exit(int(os.environ['TRAIN_EXIT']))
""")
    mock.chmod(0o755)
    for name in ("ssh", "rsync"):
        (binaries / name).symlink_to(mock)
    logfile = tmp_path / "commands.jsonl"
    env = {**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
           "COMMAND_LOG": str(logfile), "TRAIN_EXIT": str(train_exit)}
    result = subprocess.run(["bash", "scripts/remote_run.sh", "fake-host", "configs/example.json",
                             "--resume", "--train-minutes", "999", "--max-hours", "2"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == train_exit, result.stderr
    calls = [json.loads(line) for line in logfile.read_text().splitlines()]
    train = next(c[-1] for c in calls if c[0] == "ssh" and "timeout " in c[-1])
    assert "NCM_MLX_GENERATOR_STEPS=12345" in train
    assert "configs/example.json 24000 --resume" in train
    transfers = [c for c in calls if c[0] == "rsync"]
    assert any("artifacts/example" in c for c in transfers)
    assert any("fake-host:ncm-mlx/artifacts/example" in c for c in transfers)
    assert any("fake-host:ncm-mlx/runs/example-*" in c for c in transfers)
