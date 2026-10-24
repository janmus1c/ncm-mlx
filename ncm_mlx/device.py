from __future__ import annotations

import platform

import mlx.core as mx


def runtime_info() -> dict:
    info = {
        "mlx": mx.__version__,
        "device": str(mx.default_device()),
        "metal": bool(mx.metal.is_available()),
        "cuda": bool(hasattr(mx, "cuda") and mx.cuda.is_available()),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
    }
    if info["metal"] or info["cuda"]:
        d = mx.device_info()
        info["chip"] = d.get("device_name")
        info["memory_gb"] = round(d.get("memory_size", 0) / 2**30, 1)
    return info


def require_gpu() -> dict:
    """Runtime info, or an error if MLX is not running on a GPU (Metal on macOS, CUDA on Linux)."""
    info = runtime_info()
    if not (info["metal"] or info["cuda"]) or mx.default_device() != mx.gpu:
        raise RuntimeError(f"MLX is not running on a GPU: {info}")
    return info
