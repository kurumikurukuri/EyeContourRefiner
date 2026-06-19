"""EXP022: t帯別 MAE比較 / pass rate比較 / baseline との paired t-test。

各 config の test_predictions.csv（生t, idx付き）を data/samples.db と join し、
tools/annotation_support/variation_tolerance/passrate.py と同じ t帯・閾値定義で
- MAE（生t MAE）
- pass rate（error_ratio = |t_pred-t_true|*pq_dist/eye_scale 基準）
- baseline（既定: bc32_pw32）との paired t-test（abs error, 帯ごと）
を一覧表示する。

使い方:
    python experiments/EXP022_channel_pool_sweep/scr/t_band_compare.py
    python experiments/EXP022_channel_pool_sweep/scr/t_band_compare.py \
        --configs bc16_pw8 bc32_pw8 --baseline bc32_pw32
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
OUTPUT_DIR = EXP_DIR / "outputs"
DB_PATH = ROOT / "data" / "samples.db"

ALL_CONFIGS = [
    "bc16_pw1", "bc16_pw8", "bc16_pw16", "bc16_pw32",
    "bc32_pw1", "bc32_pw8", "bc32_pw16", "bc32_pw32",
]
DEFAULT_BASELINE = "bc32_pw32"

# tools/annotation_support/variation_tolerance/passrate.py と同一定義
THRESHOLD_STD = 0.01788
THRESHOLD_STRICT = 0.01604

BANDS = [
    ("t<0", lambda t: t < 0),
    ("0-0.3", lambda t: (t >= 0) & (t < 0.3)),
    ("0.3-0.7", lambda t: (t >= 0.3) & (t < 0.7)),
    ("0.7-1.0", lambda t: (t >= 0.7) & (t < 1.0)),
    ("1.0-1.2", lambda t: (t >= 1.0) & (t < 1.2)),
    ("t>=1.2", lambda t: t >= 1.2),
]


def load_db_index() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT idx, px, py, qx, qy, eye_scale FROM samples", conn
    )
    conn.close()
    return df.set_index("idx")


def load_predictions(config: str, db_df: pd.DataFrame) -> pd.DataFrame:
    """test_predictions.csv を読み込み、abs_err / error_ratio / key 列を付与する。"""
    csv_path = OUTPUT_DIR / config / "test_predictions.csv"
    df = pd.read_csv(csv_path)

    missing = ~df["idx"].isin(db_df.index)
    if missing.any():
        print(f"  [WARN] {config}: DB に存在しない idx を {int(missing.sum())} 件除外")
    df = df[~missing].copy()

    joined = df.join(db_df, on="idx")
    pq_dist = np.sqrt((joined["px"] - joined["qx"]) ** 2 + (joined["py"] - joined["qy"]) ** 2)
    valid = joined["eye_scale"].notna() & (joined["eye_scale"] > 0) & pq_dist.notna()
    joined = joined[valid].copy()
    pq_dist = pq_dist[valid]

    joined["abs_err"] = (joined["t_true"] - joined["t_pred"]).abs()
    joined["error_ratio"] = joined["abs_err"] * pq_dist / joined["eye_scale"]
    # idx は samples.db の主キーでサンプルごとに一意（image_path 等の複合キーは
    # 同一画像内の複数 face_index で重複するため使えない）
    return joined.set_index("idx")


def band_row(df: pd.DataFrame, base: pd.DataFrame, band_name: str, mask_fn) -> dict:
    """1帯分の MAE / pass rate / baseline比較の統計をまとめる。"""
    mask = mask_fn(df["t_true"])
    sub = df[mask]
    n = len(sub)
    if n == 0:
        return {"band": band_name, "n": 0}

    base_sub = base.loc[sub.index]

    mae_cfg = sub["abs_err"].mean()
    mae_base = base_sub["abs_err"].mean()
    pass_std_cfg = (sub["error_ratio"] <= THRESHOLD_STD).mean() * 100
    pass_std_base = (base_sub["error_ratio"] <= THRESHOLD_STD).mean() * 100
    pass_strict_cfg = (sub["error_ratio"] <= THRESHOLD_STRICT).mean() * 100
    pass_strict_base = (base_sub["error_ratio"] <= THRESHOLD_STRICT).mean() * 100

    if n >= 2 and not np.allclose(sub["abs_err"].values, base_sub["abs_err"].values):
        t_stat, t_p = stats.ttest_rel(sub["abs_err"].values, base_sub["abs_err"].values)
    else:
        t_p = float("nan")

    return {
        "band": band_name,
        "n": n,
        "mae_cfg": mae_cfg,
        "mae_base": mae_base,
        "mae_diff": mae_cfg - mae_base,
        "pass_std_cfg": pass_std_cfg,
        "pass_std_base": pass_std_base,
        "pass_std_diff": pass_std_cfg - pass_std_base,
        "pass_strict_cfg": pass_strict_cfg,
        "pass_strict_base": pass_strict_base,
        "pass_strict_diff": pass_strict_cfg - pass_strict_base,
        "t_p": t_p,
    }


def compare_config(config: str, df: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    rows = [band_row(df, base, name, fn) for name, fn in BANDS]
    rows.append(band_row(df, base, "ALL", lambda t: pd.Series(True, index=t.index)))
    return pd.DataFrame(rows)


def print_table(config: str, table: pd.DataFrame) -> None:
    print(f"\n=== {config} vs baseline ===")
    cols = [
        ("band", "<10"), ("n", ">6"),
        ("MAE(cfg)", ">10"), ("MAE(base)", ">10"), ("diff", ">10"),
        ("pass_std(cfg)", ">14"), ("pass_std(base)", ">15"), ("diff", ">7"),
        ("pass_strict(cfg)", ">17"), ("pass_strict(base)", ">18"), ("diff", ">7"),
        ("t_p", ">10"),
    ]
    header = " | ".join(f"{name:{fmt}}" for name, fmt in cols)
    print(header)
    print("-" * len(header))
    for _, r in table.iterrows():
        if r["n"] == 0:
            print(f"{r['band']:<10} | {0:>6}  (no samples)")
            continue
        sig = "*" if (not np.isnan(r["t_p"]) and r["t_p"] < 0.05) else " "
        cells = [
            f"{r['band']:<10}", f"{int(r['n']):>6}",
            f"{r['mae_cfg']:>10.6f}", f"{r['mae_base']:>10.6f}", f"{r['mae_diff']:>+10.6f}",
            f"{r['pass_std_cfg']:>14.1f}", f"{r['pass_std_base']:>15.1f}", f"{r['pass_std_diff']:>+7.1f}",
            f"{r['pass_strict_cfg']:>17.1f}", f"{r['pass_strict_base']:>18.1f}", f"{r['pass_strict_diff']:>+7.1f}",
            f"{r['t_p']:>10.4g}{sig}",
        ]
        print(" | ".join(cells))


def main() -> int:
    parser = argparse.ArgumentParser(description="EXP022 t帯別 MAE/passrate比較 + baseline paired t-test")
    parser.add_argument("--configs", nargs="+", default=[c for c in ALL_CONFIGS if c != DEFAULT_BASELINE],
                        help="比較対象 config 名のリスト（baseline自身は除く）")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, help="比較基準とする config 名")
    parser.add_argument("--csv-out", type=Path, default=None,
                        help="結果をまとめて保存する CSV パス（省略時は保存しない）")
    args = parser.parse_args()

    db_df = load_db_index()
    base_df = load_predictions(args.baseline, db_df)

    all_rows = []
    for config in args.configs:
        df = load_predictions(config, db_df)
        table = compare_config(config, df, base_df)
        print_table(config, table)
        table.insert(0, "config", config)
        all_rows.append(table)

    if args.csv_out:
        out = pd.concat(all_rows, ignore_index=True)
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.csv_out, index=False)
        print(f"\nSaved -> {args.csv_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
