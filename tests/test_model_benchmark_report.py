"""tools/model_benchmark/report.py のユニットテスト（CSV/JSON round-trip）。"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tools.model_benchmark import report  # noqa: E402


def _timing_row() -> dict:
    return {
        "runtime": "pytorch",
        "device": "cpu",
        "batch_size": 1,
        "iterations": 100,
        "latency_mean_ms": 1.5,
        "latency_p50_ms": 1.4,
        "latency_p95_ms": 2.0,
        "latency_p99_ms": 2.5,
        "latency_min_ms": 1.0,
        "latency_std_ms": 0.2,
        "latency_cv": 0.13,
        "throughput_samples_per_s": 666.7,
        "artifact_path": "fold1.pt",
        "artifact_size_bytes": 12345,
    }


def _model_report() -> dict:
    return {
        "model_id": "SD0_baseline/fold1",
        "family": "dual",
        "source_path": "experiments/x/outputs/SD0_baseline/eyeliner_detection_fold1.pt",
        "params": 123456,
        "macs": 987654321,
        "macs_estimator": "thop",
        "notes": "",
        "file_size_bytes": {"pytorch": 12345, "onnx": 12000},
        "results": [_timing_row()],
    }


class TestRowsFromModels:
    def test_flattens_metadata_into_rows(self) -> None:
        rows = report.rows_from_models([_model_report()])
        assert len(rows) == 1
        assert rows[0]["model_id"] == "SD0_baseline/fold1"
        assert rows[0]["family"] == "dual"
        assert rows[0]["latency_mean_ms"] == 1.5


class TestCsvRoundTrip:
    def test_write_and_read_back(self, tmp_path: Path) -> None:
        rows = report.rows_from_models([_model_report()])
        path = tmp_path / "bench.csv"
        report.write_csv(rows, path)
        with path.open(encoding="utf-8", newline="") as f:
            read_rows = list(csv.DictReader(f))
        assert len(read_rows) == 1
        assert read_rows[0]["model_id"] == "SD0_baseline/fold1"
        assert float(read_rows[0]["latency_p95_ms"]) == 2.0
        assert int(read_rows[0]["params"]) == 123456
        # 全カラムがヘッダに存在する
        assert set(report.CSV_COLUMNS) == set(read_rows[0].keys())

    def test_empty_rows_raise(self, tmp_path: Path) -> None:
        import pytest

        with pytest.raises(ValueError, match="書き込む行がありません"):
            report.write_csv([], tmp_path / "empty.csv")


class TestJsonRoundTrip:
    def test_write_and_read_back(self, tmp_path: Path) -> None:
        payload = {"environment": {"torch": "2.0"}, "models": [_model_report()]}
        path = tmp_path / "bench.json"
        report.write_json(payload, path)
        with path.open(encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["environment"]["torch"] == "2.0"
        model = loaded["models"][0]
        assert model["file_size_bytes"]["onnx"] == 12000
        assert model["results"][0]["throughput_samples_per_s"] == 666.7


class TestFormatConsoleTable:
    def test_contains_model_and_rows(self) -> None:
        text = report.format_console_table([_model_report()])
        assert "SD0_baseline/fold1" in text
        assert "pytorch" in text
        assert "params=123,456" in text

    def test_notes_rendered(self) -> None:
        model = {**_model_report(), "notes": "正規化空間平均で測定"}
        assert "正規化空間平均で測定" in report.format_console_table([model])

    def test_empty_models(self) -> None:
        assert "測定対象がありません" in report.format_console_table([])
