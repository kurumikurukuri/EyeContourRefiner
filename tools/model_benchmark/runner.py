"""タイミングループとレイテンシ統計量の計算。

測定プロトコル:
    1. warmup 回を非計測で実行
    2. 計測 iterations 回を iteration ごとに perf_counter で個別計測
       （per-iteration の分布を得るため）
    3. CUDA 時は各 iteration 前後に torch.cuda.synchronize()
    4. 変動係数 (cv = std/mean) が閾値超過なら警告
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

from tools.model_benchmark.inputs import DEFAULT_PATCH_SIZE, build_feed_for_session, build_inputs
from tools.model_benchmark.loader import ModelFamily

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZES: tuple[int, ...] = (1, 8, 32)
DEFAULT_WARMUP = 10
DEFAULT_ITERATIONS = 100
CV_WARN_THRESHOLD = 0.15


@dataclass(frozen=True)
class LatencyStats:
    """レイテンシ統計量（ミリ秒）。"""

    mean_ms: float
    std_ms: float
    min_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    cv: float


@dataclass(frozen=True)
class BenchmarkConfig:
    """ベンチマーク実行設定。"""

    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES
    warmup: int = DEFAULT_WARMUP
    iterations: int = DEFAULT_ITERATIONS
    seed: int = 42
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE


@dataclass(frozen=True)
class RunResult:
    """1ランタイム×1バッチサイズの測定結果。"""

    runtime: str
    device: str
    batch_size: int
    iterations: int
    stats: LatencyStats
    throughput_samples_per_s: float


def compute_latency_stats(latencies_ms: Sequence[float]) -> LatencyStats:
    """レイテンシ列（ms）から統計量を計算する。"""
    if len(latencies_ms) == 0:
        raise ValueError("latencies_ms が空です")
    arr = np.asarray(latencies_ms, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    cv = float(std / mean) if mean > 0 else 0.0
    return LatencyStats(
        mean_ms=mean,
        std_ms=std,
        min_ms=float(arr.min()),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        cv=cv,
    )


def make_sync(device: torch.device) -> Callable[[], None]:
    """デバイスに応じた同期関数を返す（CPU は no-op）。"""
    if device.type == "cuda":
        def _sync() -> None:
            torch.cuda.synchronize(device)
        return _sync
    return lambda: None


def measure_callable(
    fn: Callable[[], None],
    warmup: int,
    iterations: int,
    sync: Callable[[], None] | None = None,
) -> LatencyStats:
    """関数を warmup → iterations 回実行してレイテンシ統計を返す。"""
    _sync = sync if sync is not None else (lambda: None)
    for _ in range(warmup):
        fn()
    _sync()
    latencies_ms: list[float] = []
    for _ in range(iterations):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    stats = compute_latency_stats(latencies_ms)
    if stats.cv > CV_WARN_THRESHOLD:
        logger.warning(
            "レイテンシの変動係数が大きいです (cv=%.3f > %.2f)。"
            "--threads 固定やバックグラウンド負荷の除去を推奨します",
            stats.cv, CV_WARN_THRESHOLD,
        )
    return stats


def _to_run_result(
    runtime: str,
    device: str,
    batch_size: int,
    iterations: int,
    stats: LatencyStats,
) -> RunResult:
    mean_s = stats.mean_ms / 1000.0
    throughput = batch_size / mean_s if mean_s > 0 else 0.0
    return RunResult(
        runtime=runtime,
        device=device,
        batch_size=batch_size,
        iterations=iterations,
        stats=stats,
        throughput_samples_per_s=throughput,
    )


def run_torch_benchmark(
    model: nn.Module,
    family: ModelFamily,
    device: torch.device,
    config: BenchmarkConfig,
) -> list[RunResult]:
    """PyTorch モデルをバッチサイズ別に測定する。"""
    model = model.to(device)
    model.eval()
    sync = make_sync(device)
    results: list[RunResult] = []
    for batch_size in config.batch_sizes:
        inputs = build_inputs(model, family, batch_size, device, config.seed, config.patch_size)
        args = inputs.torch_args

        def fn() -> None:
            with torch.inference_mode():
                model(*args)

        stats = measure_callable(fn, config.warmup, config.iterations, sync)
        results.append(
            _to_run_result("pytorch", str(device), batch_size, config.iterations, stats)
        )
    return results


def run_onnx_benchmark(
    session: Any,
    device: str,
    config: BenchmarkConfig,
    runtime_label: str = "onnx",
) -> list[RunResult]:
    """ONNX Runtime セッションをバッチサイズ別に測定する。

    runtime_label: 結果行の runtime 列に入れる文字列（量子化版は "onnx_int8" 等を指定）。
    """
    results: list[RunResult] = []
    for batch_size in config.batch_sizes:
        feed = build_feed_for_session(session, batch_size, config.seed)

        def fn() -> None:
            session.run(None, feed)

        stats = measure_callable(fn, config.warmup, config.iterations)
        results.append(
            _to_run_result(runtime_label, device, batch_size, config.iterations, stats)
        )
    return results
