"""tools/model_benchmark/runner.py のユニットテスト（タイミング数値はアサートしない）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from eyeliner.models.eyeliner_cnn import EyelinerCNN  # noqa: E402
from tools.model_benchmark import runner  # noqa: E402


# ---------------------------------------------------------------------------
# compute_latency_stats — 数学的正しさ
# ---------------------------------------------------------------------------

class TestComputeLatencyStats:
    def test_known_values_1_to_100(self) -> None:
        lats = [float(i) for i in range(1, 101)]  # 1.0 .. 100.0
        stats = runner.compute_latency_stats(lats)
        assert stats.mean_ms == pytest.approx(50.5)
        assert stats.min_ms == pytest.approx(1.0)
        assert stats.p50_ms == pytest.approx(50.5)
        # numpy の線形補間: (n-1)*q の小数部分で補間される
        assert stats.p95_ms == pytest.approx(95.05)
        assert stats.p99_ms == pytest.approx(99.01)
        assert stats.std_ms == pytest.approx(float(np.std(lats)))
        assert stats.cv == pytest.approx(float(np.std(lats)) / 50.5)

    def test_constant_series_cv_zero(self) -> None:
        stats = runner.compute_latency_stats([2.0] * 10)
        assert stats.mean_ms == pytest.approx(2.0)
        assert stats.std_ms == pytest.approx(0.0)
        assert stats.cv == pytest.approx(0.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="空です"):
            runner.compute_latency_stats([])


# ---------------------------------------------------------------------------
# measure_callable — スモーク（時間の絶対値は検証しない）
# ---------------------------------------------------------------------------

class TestMeasureCallable:
    def test_smoke_ordering(self) -> None:
        stats = runner.measure_callable(lambda: None, warmup=2, iterations=5)
        assert stats.min_ms >= 0.0
        assert stats.min_ms <= stats.p50_ms <= stats.p99_ms

    def test_runs_exactly_iterations_times(self) -> None:
        count = {"n": 0}

        def fn() -> None:
            count["n"] += 1

        runner.measure_callable(fn, warmup=1, iterations=7)
        assert count["n"] == 8  # warmup 1 + iterations 7


# ---------------------------------------------------------------------------
# run_torch_benchmark — 極小構成のスモーク
# ---------------------------------------------------------------------------

class TestRunTorchBenchmark:
    def test_tiny_model_cpu(self) -> None:
        model = EyelinerCNN(num_conv_layers=2)
        config = runner.BenchmarkConfig(
            batch_sizes=(1,), warmup=1, iterations=2, seed=42,
        )
        results = runner.run_torch_benchmark(model, "single", torch.device("cpu"), config)
        assert len(results) == 1
        r = results[0]
        assert r.runtime == "pytorch"
        assert r.device == "cpu"
        assert r.batch_size == 1
        assert r.iterations == 2
        assert r.throughput_samples_per_s > 0.0


class TestBenchmarkConfigDefaults:
    def test_defaults(self) -> None:
        config = runner.BenchmarkConfig()
        assert config.batch_sizes == (1, 8, 32)
        assert config.warmup == 10
        assert config.iterations == 100
        assert config.seed == 42
