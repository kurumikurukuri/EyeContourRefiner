"""EXP013 データセット分析スクリプト

EXP010 と EXP013 の manifest を比較し、t 分布・pq_dist 分布の
Markdown テーブルを標準出力に表示する。

使い方:
    python experiments/EXP013_hparam_combo/scr/analyze_manifests.py
    python experiments/EXP013_hparam_combo/scr/analyze_manifests.py --output report.md
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
EXP010_DIR = ROOT / "experiments" / "EXP010_data_v2_wide_patch" / "manifests"
EXP013_DIR = ROOT / "experiments" / "EXP013_hparam_combo" / "manifests"

MANIFESTS = {
    "EXP010 train": EXP010_DIR / "manifest_train.csv",
    "EXP013 train": EXP013_DIR / "manifest_train.csv",
    "EXP010 test":  EXP010_DIR / "manifest_test.csv",
    "EXP013 test":  EXP013_DIR / "manifest_test.csv",
}

T_BINS = [
    ("t < -0.5",  None, -0.5),
    ("t [-0.5, 0.0)", -0.5, 0.0),
    ("t [0.0, 0.2)",  0.0,  0.2),
    ("t [0.2, 0.5)",  0.2,  0.5),
    ("t [0.5, 1.0)",  0.5,  1.0),
    ("t [1.0, 1.5)",  1.0,  1.5),
    ("t ≥ 1.5",   1.5,  None),
]

PQ_BINS = [
    ("pq [0, 10)",   0,   10),
    ("pq [10, 20)",  10,  20),
    ("pq [20, 40)",  20,  40),
    ("pq [40, 60)",  40,  60),
    ("pq ≥ 60",      60,  None),
]


def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pq_dist(row: dict) -> float:
    return math.hypot(float(row["qx"]) - float(row["px"]),
                      float(row["qy"]) - float(row["py"]))


def quantile_stats(vals: list[float]) -> dict:
    s = sorted(vals)
    n = len(s)
    return {
        "n":      n,
        "min":    s[0],
        "Q1":     s[n // 4],
        "median": s[n // 2],
        "Q3":     s[3 * n // 4],
        "max":    s[-1],
        "mean":   statistics.mean(vals),
        "std":    statistics.stdev(vals) if n > 1 else 0.0,
    }


def bin_counts(vals: list[float], bins: list[tuple]) -> list[tuple[str, int, float]]:
    n = len(vals)
    results = []
    for label, lo, hi in bins:
        cnt = sum(
            1 for v in vals
            if (lo is None or v >= lo) and (hi is None or v < hi)
        )
        results.append((label, cnt, 100 * cnt / n if n else 0.0))
    return results


def md_row(cells: list) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def fmt(v: float, decimals: int = 3) -> str:
    return f"{v:.{decimals}f}"


def build_report(data: dict[str, dict]) -> str:
    keys = list(data.keys())
    lines: list[str] = []

    # ── 基本統計 ──────────────────────────────────────────
    lines.append("## サンプル数・画像数")
    lines.append("")
    header = [""] + keys
    lines.append(md_row(header))
    lines.append(md_row(["---"] + [":---:"] * len(keys)))
    lines.append(md_row(["サンプル数"] + [f"{data[k]['n']:,}"         for k in keys]))
    lines.append(md_row(["画像数"]     + [f"{data[k]['n_images']:,}"  for k in keys]))
    lines.append("")

    # ── t 分布統計 ────────────────────────────────────────
    lines.append("## t 分布（記述統計）")
    lines.append("")
    stat_keys = ["min", "Q1", "median", "Q3", "max", "mean", "std"]
    lines.append(md_row([""] + keys))
    lines.append(md_row(["---"] + [":---:"] * len(keys)))
    for sk in stat_keys:
        lines.append(md_row([sk] + [fmt(data[k]["t_stats"][sk]) for k in keys]))
    lines.append("")

    # ── t ビン分布 ────────────────────────────────────────
    lines.append("## t 分布（ビン別割合）")
    lines.append("")
    lines.append(md_row([""] + keys))
    lines.append(md_row(["---"] + [":---:"] * len(keys)))
    for i, (label, _, _) in enumerate(T_BINS):
        cells = []
        for k in keys:
            cnt, pct = data[k]["t_bins"][i][1], data[k]["t_bins"][i][2]
            cells.append(f"{cnt} ({pct:.1f}%)")
        lines.append(md_row([label] + cells))
    lines.append("")

    # ── pq_dist 統計 ──────────────────────────────────────
    lines.append("## pq_dist 分布（記述統計, 単位: px）")
    lines.append("")
    lines.append(md_row([""] + keys))
    lines.append(md_row(["---"] + [":---:"] * len(keys)))
    for sk in stat_keys:
        lines.append(md_row([sk] + [fmt(data[k]["pq_stats"][sk], 1) for k in keys]))
    lines.append("")

    # ── pq_dist ビン分布 ──────────────────────────────────
    lines.append("## pq_dist 分布（ビン別割合）")
    lines.append("")
    lines.append(md_row([""] + keys))
    lines.append(md_row(["---"] + [":---:"] * len(keys)))
    for i, (label, _, _) in enumerate(PQ_BINS):
        cells = []
        for k in keys:
            cnt, pct = data[k]["pq_bins"][i][1], data[k]["pq_bins"][i][2]
            cells.append(f"{cnt} ({pct:.1f}%)")
        lines.append(md_row([label] + cells))
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="EXP010/EXP013 manifest 比較レポート生成")
    parser.add_argument("--output", type=Path, default=None,
                        help="出力先 Markdown ファイル（省略時は標準出力）")
    args = parser.parse_args()

    data: dict[str, dict] = {}
    for label, path in MANIFESTS.items():
        if not path.exists():
            print(f"[WARN] manifest が見つかりません: {path}", file=sys.stderr)
            continue
        rows = load_rows(path)
        t_vals  = [float(r["t"]) for r in rows]
        pq_vals = [pq_dist(r) for r in rows]
        images  = {r["image_path"] for r in rows}
        data[label] = {
            "n":        len(rows),
            "n_images": len(images),
            "t_stats":  quantile_stats(t_vals),
            "t_bins":   bin_counts(t_vals,  T_BINS),
            "pq_stats": quantile_stats(pq_vals),
            "pq_bins":  bin_counts(pq_vals, PQ_BINS),
        }

    report = build_report(data)

    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"レポートを保存しました: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
