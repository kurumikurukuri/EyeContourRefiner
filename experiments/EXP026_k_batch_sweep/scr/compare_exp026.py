"""EXP026 全 6 config の統計集計: passrate / t帯別 MAE / paired t-test vs baseline。

使い方（プロジェクトルートから）:
    python experiments/EXP026_k_batch_sweep/scr/compare_exp026.py
"""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
OUTPUT_DIR = EXP_DIR / "outputs"
DB_PATH = ROOT / "data" / "samples.db"

sys.path.insert(0, str(ROOT))

CONFIGS = ["k3_b32", "k3_b64", "k4_b32", "k4_b64", "k5_b32", "k5_b64"]
BASELINE = "k3_b32"

THRESHOLD_STD = 0.01788
THRESHOLD_STRICT = 0.01604

BANDS = [
    ("t<0",     lambda t: t < 0.0),
    ("0.0-0.3", lambda t: (t >= 0.0) & (t < 0.3)),
    ("0.3-0.6", lambda t: (t >= 0.3) & (t < 0.6)),
    ("0.6-0.9", lambda t: (t >= 0.6) & (t < 0.9)),
    ("0.9-1.2", lambda t: (t >= 0.9) & (t < 1.2)),
    ("t>=1.2",  lambda t: t >= 1.2),
]


def load_pred(config: str) -> dict[int, dict]:
    path = OUTPUT_DIR / config / "test_predictions.csv"
    out: dict[int, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[int(r["idx"])] = {
                "t_true": float(r["t_true"]),
                "t_pred": float(r["t_pred"]),
            }
    return out


def load_db_meta() -> dict[int, dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT idx, px, py, qx, qy, eye_scale FROM samples")
    result = {}
    for idx, px, py, qx, qy, eye_scale in cur.fetchall():
        result[idx] = {
            "pq_dist": float(np.sqrt((px - qx) ** 2 + (py - qy) ** 2)),
            "eye_scale": float(eye_scale) if eye_scale else None,
        }
    conn.close()
    return result


def sig(p: float | None) -> str:
    if p is None:
        return "n/a"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    preds = {c: load_pred(c) for c in CONFIGS}
    db_meta = load_db_meta()

    # --- passrate ---
    print(f"\n{'='*70}")
    print("  passrate (hold-out)")
    print(f"{'='*70}")
    print(f"  {'config':<10} {'n':>6} {'pass_std%':>11} {'pass_strict%':>13}")

    passrate_results = {}
    for cfg in CONFIGS:
        p = preds[cfg]
        rows = []
        for idx, d in p.items():
            meta = db_meta.get(idx, {})
            eye_scale = meta.get("eye_scale")
            pq_dist = meta.get("pq_dist")
            if eye_scale and eye_scale > 0 and pq_dist:
                err_ratio = abs(d["t_pred"] - d["t_true"]) * pq_dist / eye_scale
                rows.append(err_ratio)
        arr = np.array(rows)
        ps = float((arr <= THRESHOLD_STD).mean() * 100)
        pst = float((arr <= THRESHOLD_STRICT).mean() * 100)
        passrate_results[cfg] = {"n": len(arr), "pass_std": ps, "pass_strict": pst}
        print(f"  {cfg:<10} {len(arr):>6} {ps:>10.2f}% {pst:>12.2f}%")

    # --- t帯別 MAE ---
    print(f"\n{'='*70}")
    print("  t帯別 raw-t MAE（baseline = k3_b32）")
    print(f"{'='*70}")
    base_p = preds[BASELINE]
    common_all = sorted(set.intersection(*[set(preds[c]) for c in CONFIGS]))

    band_results: dict[str, dict] = {}
    header = f"  {'band':<10} {'n':>5} " + " ".join(f"{c:>10}" for c in CONFIGS)
    print(header)
    for bname, bfn in BANDS:
        t_arr = np.array([base_p[i]["t_true"] for i in common_all])
        mask = bfn(t_arr)
        idxs = [i for i, m in zip(common_all, mask) if m]
        if not idxs:
            continue
        maes = {}
        for cfg in CONFIGS:
            errs = [abs(preds[cfg][i]["t_pred"] - preds[cfg][i]["t_true"]) for i in idxs]
            maes[cfg] = float(np.mean(errs))
        band_results[bname] = {"n": len(idxs), **maes}
        row = f"  {bname:<10} {len(idxs):>5} " + " ".join(f"{maes[c]:>10.6f}" for c in CONFIGS)
        print(row)

    # --- paired t-test vs baseline ---
    print(f"\n{'='*70}")
    print(f"  paired t-test vs {BASELINE}（全 test, 生t誤差）")
    print(f"{'='*70}")
    print(f"  {'config':<10} {'n':>7} {'MAE_base':>11} {'MAE_new':>11} {'改善%':>9} {'p':>10} {'sig':>5}")

    base_errs = {i: abs(base_p[i]["t_pred"] - base_p[i]["t_true"]) for i in common_all}
    ttest_results = {}
    for cfg in CONFIGS:
        if cfg == BASELINE:
            continue
        new_errs = [abs(preds[cfg][i]["t_pred"] - preds[cfg][i]["t_true"]) for i in common_all]
        b_errs = [base_errs[i] for i in common_all]
        mae_new = float(np.mean(new_errs))
        mae_base = float(np.mean(b_errs))
        _, pval = ttest_rel(new_errs, b_errs)
        impr = (mae_base - mae_new) / mae_base * 100
        ttest_results[f"{cfg}_vs_{BASELINE}"] = {
            "n": len(common_all),
            "mae_base": round(mae_base, 6),
            "mae_new": round(mae_new, 6),
            "improvement_pct": round(impr, 3),
            "ttest_p": round(float(pval), 6),
        }
        print(f"  {cfg:<10} {len(common_all):>7} {mae_base:>11.6f} {mae_new:>11.6f} "
              f"{impr:>+8.2f}% {pval:>10.6f} {sig(pval):>5}")

    # --- summary.json 更新 ---
    summary_path = OUTPUT_DIR / "exp026_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary["passrate"] = passrate_results
    summary["band_mae"] = band_results
    summary["comparisons"] = ttest_results
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  保存: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
