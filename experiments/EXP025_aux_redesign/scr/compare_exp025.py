"""EXP025 config 間比較: config 間の test_predictions.csv を paired t-test + t-band breakdown で比較。

train_exp025.py から比較ロジックを分離したもの（EXP024 の compare_exp024.py に相当）。
推論済みの outputs/<exp_name>/test_predictions.csv（生 t）を入力とし:
  - 全体 MAE の paired t-test
  - t-band 別 MAE breakdown + 帯域内 paired t-test
結果を標準出力し、exp025_summary.json の comparisons ブロックに追記する。

使い方:
    python experiments/EXP025_aux_redesign/scr/compare_exp025.py
    python experiments/EXP025_aux_redesign/scr/compare_exp025.py --configs A0 A1 A2
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from scipy.stats import ttest_rel

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
OUTPUT_DIR = EXP_DIR / "outputs"

sys.path.insert(0, str(ROOT))

CONFIGS = {
    "A0": "A0_baseline",
    "A1": "A1_pitch_polarcos",
    "A2": "A2_add_polarsin",
    "A3": "A3_aux_face_shape",
    "A4": "A4_no_features",
}

# 対比リスト: (new_name, base_name)
COMPARISONS = [
    ("A1", "A0"), ("A2", "A0"), ("A3", "A0"),
    ("A2", "A1"), ("A3", "A2"),
    ("A4", "A0"), ("A4", "A2"),
]


def load_pred(exp_name: str) -> dict[int, dict]:
    path = OUTPUT_DIR / exp_name / "test_predictions.csv"
    out: dict[int, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[int(r["idx"])] = {"t_true": float(r["t_true"]), "t_pred": float(r["t_pred"])}
    return out


def band_label(t: float) -> str:
    if t < 0.0:
        return "t<0"
    if t >= 1.2:
        return "t>=1.2"
    b = (t // 0.3) * 0.3
    return f"{b:.1f}-{b+0.3:.1f}"


def paired_compare(new_name: str, base_name: str) -> dict:
    """new_name と base_name の per-sample 絶対誤差を paired t-test で比較する。"""
    new_p, base_p = load_pred(new_name), load_pred(base_name)
    common = sorted(set(new_p) & set(base_p))
    err_new = [abs(new_p[i]["t_pred"] - new_p[i]["t_true"]) for i in common]
    err_base = [abs(base_p[i]["t_pred"] - base_p[i]["t_true"]) for i in common]
    mae_new = sum(err_new) / len(err_new)
    mae_base = sum(err_base) / len(err_base)

    stat = pval = None
    if len(common) >= 10:
        s, p = ttest_rel(err_new, err_base)
        stat, pval = float(s), float(p)

    # t-band breakdown
    bands_new: dict[str, list[float]] = {}
    bands_base: dict[str, list[float]] = {}
    for i in common:
        b = band_label(new_p[i]["t_true"])
        bands_new.setdefault(b, []).append(abs(new_p[i]["t_pred"] - new_p[i]["t_true"]))
        bands_base.setdefault(b, []).append(abs(base_p[i]["t_pred"] - base_p[i]["t_true"]))

    band_order = ["t<0", "0.0-0.3", "0.3-0.6", "0.6-0.9", "0.9-1.2", "t>=1.2"]
    band_summary = {}
    for b in band_order:
        if b not in bands_new:
            continue
        bm_new = sum(bands_new[b]) / len(bands_new[b])
        bm_base = sum(bands_base[b]) / len(bands_base[b])
        bp = None
        if len(bands_new[b]) >= 10:
            _, bp = ttest_rel(bands_new[b], bands_base[b])
        band_summary[b] = {
            "n": len(bands_new[b]),
            "mae_base": round(bm_base, 6),
            "mae_new": round(bm_new, 6),
            "ttest_p": round(float(bp), 6) if bp is not None else None,
        }

    return {
        "n": len(common),
        "mae_base": round(mae_base, 6),
        "mae_new": round(mae_new, 6),
        "improvement_pct": round((mae_base - mae_new) / mae_base * 100, 3),
        "t_stat": round(stat, 4) if stat is not None else None,
        "ttest_p": round(pval, 6) if pval is not None else None,
        "bands": band_summary,
    }


def sig_label(p: float | None) -> str:
    if p is None:
        return "n/a"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def print_table(results: dict[str, dict]) -> None:
    print(f"\n  paired t-test（test per-sample 生t誤差）:")
    print(f"  {'対比':<10}{'n':>7}{'MAE_base':>11}{'MAE_new':>11}{'改善%':>9}{'p':>11}{'':>5}")
    for key, c in results.items():
        p = c["ttest_p"]
        ps = f"{p:.5f}" if p is not None else "n/a"
        print(f"  {key:<10}{c['n']:>7}{c['mae_base']:>11.6f}"
              f"{c['mae_new']:>11.6f}{c['improvement_pct']:>+8.2f}%{ps:>11} {sig_label(p)}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="EXP025 config 間 paired t-test 比較")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS),
                        choices=list(CONFIGS), help="比較に含める config")
    parser.add_argument("--no-write", action="store_true",
                        help="exp025_summary.json への追記をスキップ")
    args = parser.parse_args()

    results: dict[str, dict] = {}
    for new_name, base_name in COMPARISONS:
        if new_name not in args.configs or base_name not in args.configs:
            continue
        pred_new = OUTPUT_DIR / new_name / "test_predictions.csv"
        pred_base = OUTPUT_DIR / base_name / "test_predictions.csv"
        if not pred_new.exists() or not pred_base.exists():
            print(f"  [skip] test_predictions.csv が見つかりません: {new_name} or {base_name}")
            continue
        key = f"{new_name}_vs_{base_name}"
        results[key] = paired_compare(new_name, base_name)

    print_table(results)

    if not args.no_write:
        summary_path = OUTPUT_DIR / "exp025_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        summary["comparisons"] = results
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  保存: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
