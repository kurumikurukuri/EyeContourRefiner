"""EXP019: アンサンブル後の指標を帯別に算出する。

単体 X0/X2(best) と、アンサンブル ENS4(=X0+X1+X2+X3平均) / X0+X2 を、
元 t 帯・pq 帯ごとに比較し、ベースライン(X0) からの改善/改悪幅を出す。

指標 = raw t MAE = |true_t_raw - pred_t_raw|（experiment.json test_mae と同一スケール）。
pred_t_raw は元 PQ 座標へ逆変換済み（全条件で共通座標 = X0換算）。
アンサンブルは pred_t_raw（元座標）の単純平均。

実行:
    .venv/Scripts/python.exe experiments/EXP019_pq_transform/scr/ensemble/analyze_ensemble_bands.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean
from typing import Callable

EXP_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = EXP_DIR / "outputs"
MEAN_OUTPUT_DIR = EXP_DIR / "outputs" / "_ensemble" / "mean"
VARIANTS = ["X0_baseline", "X1_eye_height", "X2_eye_scale", "X3_mean_pq"]
BASELINE = "X0_baseline"

# 評価する系列: 表示名 -> 平均する条件リスト（単体は1要素）
SERIES: dict[str, list[str]] = {
    "X0": ["X0_baseline"],
    "X2(best単体)": ["X2_eye_scale"],
    "X0+X2": ["X0_baseline", "X2_eye_scale"],
    "ENS4(全平均)": VARIANTS,
}


def load() -> tuple[list[tuple], dict[str, dict[tuple, float]],
                    dict[tuple, float], dict[tuple, float]]:
    preds: dict[str, dict[tuple, float]] = {v: {} for v in VARIANTS}
    true: dict[tuple, float] = {}
    pq: dict[tuple, float] = {}
    for v in VARIANTS:
        with open(OUTPUT_DIR / v / "test_predictions.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["idx"], row["part_name"], row["sub_key"], row["point_index"])
                preds[v][key] = float(row["pred_t_raw"])
                if key not in true:
                    px, py = float(row["px"]), float(row["py"])
                    qx, qy = float(row["qx"]), float(row["qy"])
                    true[key] = float(row["true_t_raw"])
                    pq[key] = math.hypot(qx - px, qy - py)
    common = set.intersection(*(set(preds[v]) for v in VARIANTS))
    keys = [k for k in preds[VARIANTS[0]] if k in common]
    return keys, preds, true, pq


def series_err(members: list[str], key, preds, true) -> float:
    pred = mean(preds[v][key] for v in members)
    return abs(pred - true[key])


def quartile_edges(values: list[float]) -> list[float]:
    s = sorted(values)
    return [s[int(len(s) * q)] for q in (0.25, 0.50, 0.75)]


def t_bands() -> list[tuple[str, Callable[[float], bool]]]:
    return [
        ("t<0 (極端低)",      lambda t: t < 0),
        ("[0,0.5)",          lambda t: 0 <= t < 0.5),
        ("[0.5,1.0)",        lambda t: 0.5 <= t < 1.0),
        ("[1.0,1.2] (正常高)", lambda t: 1.0 <= t <= 1.2),
        ("t>1.2 (極端高)",    lambda t: t > 1.2),
    ]


def pq_bands(edges: list[float]) -> list[tuple[str, Callable[[float], bool]]]:
    q1, q2, q3 = edges
    return [
        (f"pq<{q1:.1f} (Q1)",        lambda pq: pq < q1),
        (f"[{q1:.1f},{q2:.1f}) (Q2)", lambda pq: q1 <= pq < q2),
        (f"[{q2:.1f},{q3:.1f}) (Q3)", lambda pq: q2 <= pq < q3),
        (f"pq>={q3:.1f} (Q4)",       lambda pq: pq >= q3),
    ]


def table(keys, preds, true, get, bands, title) -> list[str]:
    """X0 mae と各系列の Δ% を帯別に出す（負=改善）。"""
    lines = [f"### {title}", ""]
    names = list(SERIES)
    header = f"| {'band':<18} | {'N':>5} | {'X0 mae':>8} |" + \
        "|".join(f" {n} Δ% " for n in names if n != "X0") + "|"
    lines += [header, "|" + "---|" * (3 + (len(names) - 1))]
    for bname, pred in bands:
        bk = [k for k in keys if pred(get(k))]
        if not bk:
            continue
        base = mean(series_err(SERIES["X0"], k, preds, true) for k in bk)
        row = f"| {bname:<18} | {len(bk):>5} | {base:>8.4f} |"
        vals = []
        for n in names:
            if n == "X0":
                continue
            e = mean(series_err(SERIES[n], k, preds, true) for k in bk)
            vals.append(f" {(e-base)/base*100:+5.1f} ")
        lines.append(row + "|".join(vals) + "|")
    lines.append("")
    return lines


def main() -> None:
    keys, preds, true, pq = load()
    edges = quartile_edges([pq[k] for k in keys])

    out = ["# EXP019 アンサンブル後 帯別指標 (指標 = raw t MAE = test_mae スケール)", ""]
    out += [
        "アンサンブル = pred_t_raw（元PQ座標, X0換算）の単純平均。",
        "凡例: Δ% = (系列 − X0) / X0 ×100（**負 = 改善 / 正 = 改悪**）",
        "",
        "### 全体 (holdout)",
        "",
        f"- N = {len(keys)}",
    ]
    for n, members in SERIES.items():
        e = mean(series_err(members, k, preds, true) for k in keys)
        base = mean(series_err(SERIES["X0"], k, preds, true) for k in keys)
        tag = "" if n == "X0" else f"  ({(e-base)/base*100:+.1f}% vs X0)"
        out.append(f"- {n:<12}: {e:.4f}{tag}")
    out += [""]

    out += table(keys, preds, true, lambda k: true[k], t_bands(), "元 t 帯ごと")
    out += table(keys, preds, true, lambda k: pq[k], pq_bands(edges), "元 pq_dist 帯ごと (四分位)")

    text = "\n".join(out)
    print(text)
    MEAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (MEAN_OUTPUT_DIR / "ensemble_bands_results.md").write_text(text, encoding="utf-8")
    print(f"\n[saved] {MEAN_OUTPUT_DIR / 'ensemble_bands_results.md'}")


if __name__ == "__main__":
    main()
