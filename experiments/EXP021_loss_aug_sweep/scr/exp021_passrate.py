"""EXP021: 全条件の pass_rate（pass_std/pass_strict）計算。

tools.annotation_support.variation_tolerance.passrate の compute_pass_rate を再利用し、
EXP021 全条件（Phase1 損失sweep + Phase2 augsweep + Phase3 dual移植）の test_predictions.csv
を採点する。passrate.py 本体（全実験を走査し outputs/best_per_exp.md を上書きする）とは
独立に EXP021 のみを対象にするスクリプト。

Phase3 dual ベスト（phase3_X1_t_l1_woflip）は EXP020 ベスト（X1_independent_none_eye_scale）
と同一 holdout（idx 揃い）のため、error_ratio の paired 検定も併せて行う
（CLAUDE.md: 手法比較は統計的検定必須）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from tools.annotation_support.variation_tolerance.passrate import (  # noqa: E402
    THRESHOLD_STD, THRESHOLD_STRICT, compute_pass_rate, load_db_index,
)

OUT = Path(__file__).parent.parent / "outputs"
EXP020_BEST_CSV = ROOT / "experiments/EXP020_dual_branch/outputs/X1_independent_none_eye_scale/test_predictions.csv"
PHASE3_BEST = "phase3_X1_t_l1_woflip"


def collect_conditions() -> list[tuple[str, Path]]:
    """outputs/*/test_predictions.csv を名前順に返す（無いディレクトリはスキップ）。"""
    result = []
    for d in sorted(OUT.iterdir()):
        if not d.is_dir():
            continue
        csv_path = d / "test_predictions.csv"
        if csv_path.exists():
            result.append((d.name, csv_path))
        else:
            print(f"  SKIP (test_predictions.csv なし): {d.name}")
    return result


def error_ratio_series(csv_path: Path, db_df) -> np.ndarray:
    """compute_pass_rate と同じ定義の error_ratio を per-sample で返す（paired 検定用）。"""
    import pandas as pd
    pred = pd.read_csv(csv_path)
    joined = pred.join(db_df, on="idx", rsuffix="_db")
    joined = joined[joined["idx"].isin(db_df.index)]
    pq_dist = np.sqrt((joined["px"] - joined["qx"]) ** 2 + (joined["py"] - joined["qy"]) ** 2)
    eye_scale = joined["eye_scale"]
    valid = eye_scale.notna() & (eye_scale > 0) & pq_dist.notna()
    err = (joined["t_pred"] - joined["t_true"]).abs()[valid]
    ratio = err * pq_dist[valid] / eye_scale[valid]
    return joined.loc[valid, "idx"].to_numpy(), ratio.to_numpy()


def sig(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def main() -> None:
    db_df = load_db_index()
    conditions = collect_conditions()
    print(f"対象条件数: {len(conditions)}\n")

    rows = []
    for name, csv_path in conditions:
        result = compute_pass_rate(csv_path, db_df)
        if result is None:
            print(f"  SKIP (形式不明): {name}")
            continue
        rows.append({
            "condition": name,
            "n_patches": result["n_patches"],
            "n_missing": result["n_missing"],
            "pass_std": round(result["pass_rate_std"], 2),
            "pass_strict": round(result["pass_rate_strict"], 2),
        })

    header = f"{'condition':<32}{'n_patches':>10}{'n_missing':>10}{'pass_std':>10}{'pass_strict':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['condition']:<32}{r['n_patches']:>10}{r['n_missing']:>10}"
              f"{r['pass_std']:>10.2f}{r['pass_strict']:>12.2f}")

    print(f"\n--- Markdown（閾値: std={THRESHOLD_STD}, strict={THRESHOLD_STRICT}）---")
    print(f"| condition | n_patches | pass_std_{THRESHOLD_STD} % | pass_strict_{THRESHOLD_STRICT} % |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| {r['condition']} | {r['n_patches']} | {r['pass_std']:.2f} | {r['pass_strict']:.2f} |")

    best = max(rows, key=lambda r: r["pass_std"])
    print(f"\nベスト条件（pass_std基準）: {best['condition']} "
          f"(pass_std={best['pass_std']:.2f}, pass_strict={best['pass_strict']:.2f}, n={best['n_patches']})")

    # --- Phase3 dual vs EXP020 best の paired 検定（error_ratio, 同一 idx） ---
    phase3_csv = OUT / PHASE3_BEST / "test_predictions.csv"
    if phase3_csv.exists() and EXP020_BEST_CSV.exists():
        idx_a, ratio_a = error_ratio_series(phase3_csv, db_df)
        idx_b, ratio_b = error_ratio_series(EXP020_BEST_CSV, db_df)
        common = np.intersect1d(idx_a, idx_b)
        map_a = dict(zip(idx_a, ratio_a))
        map_b = dict(zip(idx_b, ratio_b))
        ea = np.array([map_a[i] for i in common])
        eb = np.array([map_b[i] for i in common])
        t_stat, p_t = stats.ttest_rel(ea, eb)
        delta = float(ea.mean() - eb.mean())
        print(f"\n=== {PHASE3_BEST} vs EXP020 X1_independent_none_eye_scale (error_ratio, n={len(common)}) ===")
        print(f"  delta(mean error_ratio) = {delta:+.5f}  t={t_stat:.3f}  "
              f"p(t)={p_t:.2e}  {sig(p_t)}")


if __name__ == "__main__":
    main()
