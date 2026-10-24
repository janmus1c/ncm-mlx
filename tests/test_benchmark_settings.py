import json
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")

from ncm_mlx import cli

pytestmark = pytest.mark.mlx


@pytest.mark.parametrize("options, temperature", [([], 1.0), (["--temperature", "0"], 0.0)])
def test_benchmark_records_the_sampling_settings_it_executes(tmp_path, monkeypatch, options, temperature):
    from ncm_mlx.eval import benchmark

    calls = []
    monkeypatch.setattr(cli, "require_gpu", lambda: {"device": "test"})
    monkeypatch.setattr(cli, "_load", lambda cfg: (None, None, SimpleNamespace(val_prefix=None)))
    monkeypatch.setattr(cli.mx, "device_info", dict)
    monkeypatch.setattr(cli.mx, "set_memory_limit", lambda limit: None)
    monkeypatch.setattr(cli.mx, "set_cache_limit", lambda limit: None)

    def sample(gen, codec, batch_sizes, **settings):
        calls.append((batch_sizes, settings))
        return [{"batch": 1, "p50_ms": 10.0}]

    monkeypatch.setattr(benchmark, "bench_generation", sample)
    output = tmp_path / "bench.json"
    assert cli.main(["bench", "--batch-sizes", "1", "--iters", "7", "--out", str(output), *options]) == 0
    result = json.loads(output.read_text())
    assert calls == [((1,), {"warmup": 3, "iters": 7, "temperature": temperature, "prefixes": None})]
    assert result["temperature"] == temperature
    assert result["warmup"] == 3
    assert result["iterations"] == 7
    assert result["sampler_compiled"] is False
    assert result["synchronization"] == "numpy_host_copy"
