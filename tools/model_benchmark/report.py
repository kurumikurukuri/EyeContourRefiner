"""ベンチマーク結果の出力（コンソールテーブル・CSV・JSON）。

データ構造（cli.py が構築）:

    model_report = {
        "model_id": str, "family": str, "source_path": str,
        "params": int | None, "macs": int | None, "macs_estimator": str | None,
        "notes": str,
        "file_size_bytes": {"pytorch": int | None, "onnx": int | None},
        "results": [timing_row, ...],   # flat な測定行
    }
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

CSV_COLUMNS = [
    "model_id",
    "family",
    "runtime",
    "device",
    "batch_size",
    "iterations",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "latency_min_ms",
    "latency_std_ms",
    "latency_cv",
    "throughput_samples_per_s",
    "params",
    "macs",
    "artifact_size_bytes",
    "artifact_path",
    "source_path",
]

_TABLE_HEADER = (
    f"{'runtime':<8} {'device':<6} {'batch':>5} {'iters':>5} "
    f"{'mean[ms]':>10} {'p50[ms]':>10} {'p95[ms]':>10} {'p99[ms]':>10} "
    f"{'cv':>7} {'thr[s/s]':>10}"
)


def rows_from_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """モデルレポート列を CSV 用の flat 行列に変換する。"""
    rows: list[dict[str, Any]] = []
    for m in models:
        for r in m["results"]:
            rows.append({
                "model_id": m["model_id"],
                "family": m["family"],
                "params": m["params"],
                "macs": m["macs"],
                "source_path": m["source_path"],
                **r,
            })
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """測定行を CSV（long format）で書き込む。"""
    if not rows:
        raise ValueError("書き込む行がありません")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(report: dict[str, Any], path: Path) -> None:
    """レポート全体（環境メタデータ込み）を JSON で書き込む。"""
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)


def _fmt_int(value: int | None) -> str:
    return f"{value:,}" if value is not None else "-"


def format_console_table(models: list[dict[str, Any]]) -> str:
    """モデルごとの測定テーブルを文字列として返す。"""
    lines: list[str] = []
    for m in models:
        header = (
            f"=== {m['model_id']} (family={m['family']}, "
            f"params={_fmt_int(m['params'])}, macs={_fmt_int(m['macs'])}) ==="
        )
        lines.append(header)
        lines.append(_TABLE_HEADER)
        for r in m["results"]:
            lines.append(
                f"{r['runtime']:<8} {r['device']:<6} {r['batch_size']:>5} "
                f"{r['iterations']:>5} {r['latency_mean_ms']:>10.3f} "
                f"{r['latency_p50_ms']:>10.3f} {r['latency_p95_ms']:>10.3f} "
                f"{r['latency_p99_ms']:>10.3f} {r['latency_cv']:>7.3f} "
                f"{r['throughput_samples_per_s']:>10.1f}"
            )
        if m.get("notes"):
            lines.append(f"  note: {m['notes']}")
        lines.append("")
    if not models:
        lines.append("（測定対象がありません）")
    return "\n".join(lines)
