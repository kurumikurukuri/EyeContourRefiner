"""EXP019: 元の t 帯 / pq 帯ごとの精度をベースライン(X0)と比較する分析スクリプト。

各条件 (X0/X1/X2/X3) の holdout 予測 (test_predictions.csv) を共通指標で評価する。
共通指標 = raw t の MAE（experiment.json の test_mae と同一スケール）:
    err = |true_t_raw - pred_t_raw|
ここで pred_t_raw は予測を元 PQ 座標へ逆変換した raw t（= 全条件で共通座標 = X0 換算）。
全 holdout 平均をとると各条件の experiment.json test_mae に一致する（X0 = 0.0932）。
条件間で公平に比較でき、holdout_mae スケールが欲しければ全体を baseline label 幅(=3) で割ればよい。

t 帯  : 元の raw t 値でビニング（極端 t = t<0 or t>1.2 を分離）
pq 帯 : 元の pq_dist (=|PQ|, px) の四分位でビニング

出力:
    - 標準出力に t 帯 / pq 帯 / 交差表のテーブル
    - band_breakdown_results.md に同内容を保存

実行:
    .venv/Scripts/python.exe experiments/EXP019_pq_transform/scr/ensemble/analyze_band_breakdown.py
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Callable

OUTPUT_DIR = Path(__file__).parent
VARIANTS = ["X0_baseline", "X1_eye_height", "X2_eye_scale", "X3_mean_pq"]
BASELINE = "X0_baseline"


@dataclass(frozen=True)
class Sample:
    """holdout 1 点分の評価結果。"""
    t: float          # 元の raw t
    pq: float         # 元の pq_dist (px)
    err: float        # |true_t_raw - pred_t_raw| （raw t MAE = experiment.json test_mae と同一スケール, X0換算）


def load_samples(variant: str) -> dict[tuple, Sample]:
    """test_predictions.csv を読み、サンプルキー -> Sample の dict を返す。

    err = |true_t_raw - pred_t_raw|。pred_t_raw は予測を元 PQ 座標へ逆変換した raw t
    （全条件で共通座標 = X0 換算）。全 holdout 平均は experiment.json の test_mae に一致する。
    """
    path = OUTPUT_DIR / variant / "test_predictions.csv"
    out: dict[tuple, Sample] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            px, py = float(row["px"]), float(row["py"])
            qx, qy = float(row["qx"]), float(row["qy"])
            t = float(row["true_t_raw"])
            pred = float(row["pred_t_raw"])          # 元 PQ 座標へ逆変換済み（全条件で共通座標）
            err = abs(t - pred)                       # raw t 誤差 (= test_mae スケール)
            pq = math.hypot(qx - px, qy - py)
            key = (row["idx"], row["part_name"], row["sub_key"], row["point_index"])
            out[key] = Sample(t=t, pq=pq, err=err)
    return out


def quartile_edges(values: list[float]) -> list[float]:
    """昇順 25/50/75% の境界値を返す。"""
    s = sorted(values)
    return [s[int(len(s) * q)] for q in (0.25, 0.50, 0.75)]


def build_t_bands() -> list[tuple[str, Callable[[float], bool]]]:
    return [
        ("t<0 (極端低)",      lambda t: t < 0),
        ("[0,0.5)",          lambda t: 0 <= t < 0.5),
        ("[0.5,1.0)",        lambda t: 0.5 <= t < 1.0),
        ("[1.0,1.2] (正常高)", lambda t: 1.0 <= t <= 1.2),
        ("t>1.2 (極端高)",    lambda t: t > 1.2),
    ]


def build_pq_bands(edges: list[float]) -> list[tuple[str, Callable[[float], bool]]]:
    q1, q2, q3 = edges
    return [
        (f"pq<{q1:.1f} (Q1)",        lambda pq: pq < q1),
        (f"[{q1:.1f},{q2:.1f}) (Q2)", lambda pq: q1 <= pq < q2),
        (f"[{q2:.1f},{q3:.1f}) (Q3)", lambda pq: q2 <= pq < q3),
        (f"pq>={q3:.1f} (Q4)",       lambda pq: pq >= q3),
    ]


def band_table(
    data: dict[str, dict[tuple, Sample]],
    keys: list[tuple],
    bands: list[tuple[str, Callable[[float], bool]]],
    selector: Callable[[Sample], float],
    title: str,
) -> list[str]:
    """1 軸ビニングのテーブルを行リスト(markdown)で返す。

    Δ は variant_mae - X0_mae（負=改善 / 正=改悪）。
    """
    lines: list[str] = [f"### {title}", ""]
    header = (
        f"| {'band':<18} | {'N':>5} | {'X0 mae':>8} | "
        + " | ".join(f"{v.split('_')[0]} Δmae" for v in VARIANTS[1:])
        + " | "
        + " | ".join(f"{v.split('_')[0]} Δ%" for v in VARIANTS[1:])
        + " |"
    )
    sep = "|" + "---|" * (3 + 2 * (len(VARIANTS) - 1))
    lines += [header, sep]

    for band_name, pred in bands:
        band_keys = [k for k in keys if pred(selector(data[BASELINE][k]))]
        n = len(band_keys)
        if n == 0:
            continue
        base_err = mean(data[BASELINE][k].err for k in band_keys)
        cells = [f"| {band_name:<18} | {n:>5} | {base_err:>8.4f} "]
        deltas = []
        for v in VARIANTS[1:]:
            v_err = mean(data[v][k].err for k in band_keys)
            deltas.append((v_err - base_err, (v_err - base_err) / base_err * 100))
        cells.append("| " + " | ".join(f"{d:+8.4f}" for d, _ in deltas))
        cells.append(" | " + " | ".join(f"{p:+5.1f}" for _, p in deltas) + " |")
        lines.append("".join(cells))
    lines.append("")
    return lines


def cross_table(
    data: dict[str, dict[tuple, Sample]],
    keys: list[tuple],
    t_bands: list[tuple[str, Callable[[float], bool]]],
    pq_bands: list[tuple[str, Callable[[float], bool]]],
    variant: str,
) -> list[str]:
    """t帯 × pq帯 の交差表（variant の Δmae vs X0、負=改善）。"""
    lines = [f"### 交差表 {variant.split('_')[0]} Δmae vs X0  (負=改善 / 正=改悪)", ""]
    corner = "t帯 / pq帯"
    header = f"| {corner:<18} |" + "|".join(f" {pn.split(' ')[0]:>12} " for pn, _ in pq_bands) + "|"
    sep = "|" + "---|" * (1 + len(pq_bands))
    lines += [header, sep]
    for tn, tp in t_bands:
        row = [f"| {tn:<18} "]
        for _, pp in pq_bands:
            cell_keys = [
                k for k in keys
                if tp(data[BASELINE][k].t) and pp(data[BASELINE][k].pq)
            ]
            if len(cell_keys) < 5:  # サンプル過少帯は除外
                row.append(f" {'--':>12} ")
                continue
            base = mean(data[BASELINE][k].err for k in cell_keys)
            v = mean(data[variant][k].err for k in cell_keys)
            row.append(f" {v - base:+8.4f}({len(cell_keys):>3}) ")
        lines.append("|".join(row) + "|")
    lines.append("")
    return lines


def overall_line(data: dict[str, dict[tuple, Sample]], keys: list[tuple]) -> list[str]:
    lines = ["### 全体 (holdout)", ""]
    base_err = mean(data[BASELINE][k].err for k in keys)
    base_med = median([data[BASELINE][k].err for k in keys])
    lines.append(f"- N = {len(keys)}")
    lines.append(f"- X0_baseline: mae={base_err:.4f} (= experiment.json test_mae)  median={base_med:.4f}")
    for v in VARIANTS[1:]:
        v_err = mean(data[v][k].err for k in keys)
        v_med = median([data[v][k].err for k in keys])
        d = v_err - base_err
        lines.append(
            f"- {v.split('_')[0]}: mae={v_err:.4f} ({d:+.4f}, {d / base_err * 100:+.1f}%)  "
            f"median={v_med:.4f}"
        )
    lines.append("")
    return lines


def main() -> None:
    data = {v: load_samples(v) for v in VARIANTS}
    # 全条件で共通のキーのみ使用（順序は X0 基準）
    common = set.intersection(*(set(d.keys()) for d in data.values()))
    keys = [k for k in data[BASELINE].keys() if k in common]

    pq_edges = quartile_edges([data[BASELINE][k].pq for k in keys])
    t_bands = build_t_bands()
    pq_bands = build_pq_bands(pq_edges)

    out: list[str] = ["# EXP019 帯別 精度ブレークダウン (指標 = raw t MAE = experiment.json test_mae スケール)", ""]
    out += [
        "指標 = |true_t_raw − pred_t_raw|（experiment.json の test_mae と同一スケール。X0=0.0932）。",
        "pred_t_raw は元 PQ 座標へ逆変換済み（全条件で共通座標 = X0換算）。",
        "凡例: Δmae = variant_mae − X0_mae（**負 = 改善 / 正 = 改悪**）",
        "",
    ]
    out += overall_line(data, keys)
    out += band_table(data, keys, t_bands, lambda s: s.t, "元 t 帯ごと")
    out += band_table(data, keys, pq_bands, lambda s: s.pq, "元 pq_dist 帯ごと (四分位)")
    for v in VARIANTS[1:]:
        out += cross_table(data, keys, t_bands, pq_bands, v)

    text = "\n".join(out)
    print(text)
    (OUTPUT_DIR / "band_breakdown_results.md").write_text(text, encoding="utf-8")
    print(f"\n[saved] {OUTPUT_DIR / 'band_breakdown_results.md'}")


if __name__ == "__main__":
    main()
