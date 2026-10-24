from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


class TrainingPaused(SystemExit):
    def __init__(self, step: int):
        super().__init__(75)
        self.step = step


class StopRequest:
    def __enter__(self):
        self.requested = False
        self.previous = {}
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
            self.previous[sig] = signal.signal(sig, self._request)
        return self

    def _request(self, signum, frame):
        self.requested = True

    def __exit__(self, *args):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)


def atomic_weights(model, path: Path):
    atomic_arrays(dict(tree_flatten(model.parameters())), path)


def atomic_arrays(arrays: dict, path: Path, metadata: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.safetensors")
    mx.save_safetensors(str(tmp), arrays, metadata)
    _replace(tmp, path)


def _replace(tmp: Path, path: Path):
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def save_checkpoint(path: Path, model, opt, metadata: dict, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.parameters(), "optimizer": opt.state, "random": list(mx.random.state), **extra}
    flat = dict(tree_flatten(state))
    mx.eval(list(flat.values()))
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.safetensors")
    mx.save_safetensors(str(tmp), flat, {"training": json.dumps({"version": 1, **metadata})})
    if path.exists():
        try:
            mx.load(str(path))
        except (OSError, ValueError, RuntimeError):
            pass
        else:
            previous = path.with_name(f"{path.stem}.previous.safetensors")
            link = previous.with_name(f".{previous.name}.{os.getpid()}")
            os.link(path, link)
            os.replace(link, previous)
    _replace(tmp, path)


def read_checkpoint(path: Path, cfg, stage: str, steps: int, log=print):
    previous = path.with_name(f"{path.stem}.previous.safetensors")
    errors = []
    for candidate in (path, previous):
        try:
            arrays, meta = mx.load(str(candidate), return_metadata=True)
            meta = json.loads(meta["training"])
            state = tree_unflatten(list(arrays.items()))
            mx.eval(list(arrays.values()))
        except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        if meta.get("version") != 1 or meta["config"] != cfg.to_dict():
            raise ValueError(f"checkpoint configuration/version mismatch: {candidate}")
        if meta["stage"] != stage or meta["total_steps"] != steps:
            raise ValueError(f"checkpoint stage/step budget mismatch: {candidate}")
        if candidate != path:
            log(f"recovering previous checkpoint {candidate}; latest was unreadable: {errors[0]}")
        return state, meta
    raise ValueError("no readable training checkpoint: " + "; ".join(errors))


def restore_checkpoint(state, model, opt):
    model.load_weights(list(tree_flatten(state["model"])))
    opt.state = state["optimizer"]
    words = state["random"][0].tolist()
    mx.random.seed((int(words[0]) << 32) | int(words[1]))
    mx.eval(model.parameters(), opt.state, mx.random.state)
