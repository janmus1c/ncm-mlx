import importlib.util

import pytest

HAS_MLX = importlib.util.find_spec("mlx") is not None


def pytest_collection_modifyitems(config, items):
    if HAS_MLX:
        return
    skip = pytest.mark.skip(reason="MLX is not installed (use Metal or the CUDA dependency group)")
    for item in items:
        if "mlx" in item.keywords:
            item.add_marker(skip)
