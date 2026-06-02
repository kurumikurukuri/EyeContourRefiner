"""EXP019: 4 条件 (X0/X1/X2/X3) のクロス座標アンサンブル効果を実測する。

各条件は再スケール座標が違う = CNN が見るパッチ(FOV/ズーム/アスペクト)が違う
= 別視点。共通指標で予測を結合し、単体モデルを上回るか検証する。

共通指標 = raw t の MAE（experiment.json の test_mae と同一スケール）:
    err = |true_t_raw - pred_t_raw|
pred_t_raw は予測を元 PQ 座標へ逆変換した raw t（全条件で共通座標 = X0 換算）。
全 holdout 平均は各条件の test_mae に一致する（X0 = 0.0932）。結合は raw t（元座標）の平均で行う。

検証する結合:
  - 単体 (recap)
  - 単純平均 (全4 / 各サブセット)
  - oracle  : サンプルごとに最良条件を選ぶ上限値 (実現不能、参考)
  - pq ルーティング: pq_dist (推論時に既知) の四分位ごとに最良条件を選ぶ (実現可能)
  - 誤差脱相関の度合い (pairwise Pearson)

実行:
    .venv/Scripts/python.exe experiments/EXP019_pq_transform/scr/ensemble/analyze_ensemble.py
"""
from __future__ import annotations

import csv
import math
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Iterable

EXP_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = EXP_DIR / "outputs"
MEAN_OUTPUT_DIR = EXP_DIR / "outputs" / "_ensemble" / "mean"
VARIANTS = ["X0_baseline", "X1_eye_height", "X2_eye_scale", "X3_mean_pq"]
BASELINE = "X0_baseline"
SHORT = {v: v.split("_")[0] for v in VARIANTS}


def load() -> tuple[list[tuple], dict[str, dict[tuple, float]],
                    dict[tuple, float], dict[tuple, float]]:
    """Returns (keys, pred[variant][key]=pred_t_raw, true[key]=true_t_raw, pq[key])."""
    preds: dict[str, dict[tuple, float]] = {v: {} for v in VARIANTS}
    true: dict[tuple, float] = {}
    pq: dict[tuple, float] = {}
    for v in VARIANTS:
        with open(OUTPUT_DIR / v / "test_predictions.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["idx"], row["part_name"], row["sub_key"], row["point_index"])
                preds[v][key] = float(row["pred_t_raw"])     # 元 PQ 座標へ逆変換済み
                if key not in true:
                    px, py = float(row["px"]), float(row["py"])
                    qx, qy = float(row["qx"]), float(row["qy"])
                    true[key] = float(row["true_t_raw"])
                    pq[key] = math.hypot(qx - px, qy - py)
    common = set.intersection(*(set(preds[v]) for v in VARIANTS))
    keys = [k for k in preds[VARIANTS[0]] if k in common]
    return keys, preds, true, pq


def err(pred_t: float, true_t: float) -> float:
    """X0換算 raw t 誤差（= test_mae スケール）。"""
    return abs(pred_t - true_t)


def avg_pred(variants: Iterable[str], preds, key) -> float:
    vs = list(variants)
    return mean(preds[v][key] for v in vs)


def mean_err(variants, keys, preds, true) -> float:
    return mean(err(avg_pred(variants, preds, k), true[k]) for k in keys)


def pearson(a: list[float], b: list[float]) -> float:
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else 0.0


def main() -> None:
    keys, preds, true, pq = load()
    out: list[str] = ["# EXP019 クロス座標アンサンブル効果 (指標 = raw t MAE = test_mae スケール)", ""]
    out += [
        "指標 = |true_t_raw − pred_t_raw|（experiment.json の test_mae と同一スケール）。結合は raw t（元座標）平均。",
        "pred_t_raw は元 PQ 座標へ逆変換済み（全条件で共通座標 = X0換算）。",
        f"N = {len(keys)}", "",
    ]

    single = {v: mean(err(preds[v][k], true[k]) for k in keys) for v in VARIANTS}
    best_single = min(single, key=single.get)
    out += ["### 単体 (recap, = experiment.json test_mae)"]
    for v in VARIANTS:
        out.append(f"- {SHORT[v]:<3}: {single[v]:.4f}")
    out += ["", f"best single = {SHORT[best_single]} ({single[best_single]:.4f})", ""]

    # --- simple averages: all subsets of size >=2 ---
    out += ["### 単純平均アンサンブル (raw t 平均)"]
    out += [f"| {'構成':<22} | {'mae':>8} | {'vs best single':>16} |", "|---|---|---|"]
    rows = []
    for r in range(2, len(VARIANTS) + 1):
        for combo in combinations(VARIANTS, r):
            e = mean_err(combo, keys, preds, true)
            rows.append(("+".join(SHORT[v] for v in combo), e))
    for name, e in sorted(rows, key=lambda x: x[1]):
        d = e - single[best_single]
        out.append(f"| {name:<22} | {e:>8.4f} | {d:+.4f} ({d/single[best_single]*100:+.1f}%) |")
    out += [""]

    # --- oracle upper bound (per-sample best variant) ---
    oracle = mean(min(err(preds[v][k], true[k]) for v in VARIANTS) for k in keys)
    out += ["### oracle 上限 (サンプルごとに最良条件を選択 / 実現不能)",
            f"- oracle = {oracle:.4f}  ({(oracle-single[best_single])/single[best_single]*100:+.1f}% vs best single)",
            ""]

    # --- pq-routing (feasible: pq known at inference) ---
    s = sorted(pq[k] for k in keys)
    edges = [s[int(len(s) * q)] for q in (0.25, 0.5, 0.75)]

    def band_of(p: float) -> int:
        return sum(p >= e for e in edges)  # 0..3

    routed_total = []
    route_choice = {}
    for b in range(4):
        bk = [k for k in keys if band_of(pq[k]) == b]
        be = {v: mean(err(preds[v][k], true[k]) for k in bk) for v in VARIANTS}
        bestv = min(be, key=be.get)
        route_choice[b] = (bestv, be[bestv], len(bk))
        routed_total += [err(preds[bestv][k], true[k]) for k in bk]
    routed = mean(routed_total)
    out += ["### pq ルーティング (pq四分位ごとに最良単体を選択 / 推論時に実現可能, in-sample最適)"]
    labels = [f"pq<{edges[0]:.1f}", f"[{edges[0]:.1f},{edges[1]:.1f})",
              f"[{edges[1]:.1f},{edges[2]:.1f})", f"pq>={edges[2]:.1f}"]
    for b in range(4):
        v, e, n = route_choice[b]
        out.append(f"- {labels[b]:<14} -> {SHORT[v]:<3} ({e:.4f}, n={n})")
    out += [f"- routed total = {routed:.4f} ({(routed-single[best_single])/single[best_single]*100:+.1f}% vs best single)", ""]

    # --- error decorrelation ---
    out += ["### 誤差の脱相関 (per-sample 正規化t誤差の Pearson 相関)",
            "低いほどアンサンブル効果大。", ""]
    errvec = {v: [err(preds[v][k], true[k]) for k in keys] for v in VARIANTS}
    hdr = "| corr |" + "|".join(f" {SHORT[v]:>4} " for v in VARIANTS) + "|"
    out += [hdr, "|" + "---|" * (len(VARIANTS) + 1)]
    for vi in VARIANTS:
        cells = "|".join(f" {pearson(errvec[vi], errvec[vj]):>4.2f} " for vj in VARIANTS)
        out.append(f"| {SHORT[vi]:<4} |" + cells + "|")
    out += [""]

    text = "\n".join(out)
    print(text)
    MEAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (MEAN_OUTPUT_DIR / "ensemble_results.md").write_text(text, encoding="utf-8")
    print(f"\n[saved] {MEAN_OUTPUT_DIR / 'ensemble_results.md'}")


if __name__ == "__main__":
    main()
