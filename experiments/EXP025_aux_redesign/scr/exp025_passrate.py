"""EXP025 A0–A4 の hold-out pass rate を全体 + t帯別に集計する。

error_ratio = |t_pred - t_true| * pq_dist / eye_scale
pass_rate(threshold) = #{error_ratio <= threshold} / total
閾値は experiments/Completion_Criteria.md（Standard=p10, Strict=p5）。
exp022_passrate.py と同一ロジック（条件名のみ A0–A4 に差し替え）。

使い方（プロジェクトルートから）:
    python experiments/EXP025_aux_redesign/scr/exp025_passrate.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
OUTPUT_DIR = EXP_DIR / "outputs"
DB_PATH = ROOT / "data" / "samples.db"

THRESHOLD_STD = 0.01788
THRESHOLD_STRICT = 0.01604

CONFIGS = ["A0", "A1", "A2", "A3", "A4"]

BANDS = [
    ("t<0",     lambda t: t < 0),
    ("0-0.3",   lambda t: (t >= 0) & (t < 0.3)),
    ("0.3-0.7", lambda t: (t >= 0.3) & (t < 0.7)),
    ("0.7-1.0", lambda t: (t >= 0.7) & (t < 1.0)),
    ("1.0-1.2", lambda t: (t >= 1.0) & (t < 1.2)),
    ("t>=1.2",  lambda t: t >= 1.2),
]


def load_db_index() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT idx, px, py, qx, qy, eye_scale FROM samples", conn)
    conn.close()
    return df.set_index("idx")


def load_joined(config: str, db_df: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_csv(OUTPUT_DIR / config / "test_predictions.csv")
    joined = pred.join(db_df, on="idx")
    pq_dist = np.sqrt((joined["px"] - joined["qx"]) ** 2 + (joined["py"] - joined["qy"]) ** 2)
    valid = joined["eye_scale"].notna() & (joined["eye_scale"] > 0) & pq_dist.notna()
    joined = joined[valid].copy()
    joined["pq_dist"] = pq_dist[valid]
    joined["error_ratio"] = (joined["t_pred"] - joined["t_true"]).abs() * joined["pq_dist"] / joined["eye_scale"]
    return joined


def pass_rate(error_ratio: pd.Series, threshold: float) -> float:
    return float((error_ratio <= threshold).mean() * 100)


def main() -> None:
    db_df = load_db_index()

    overall_rows = []
    band_rows = []

    for config in CONFIGS:
        joined = load_joined(config, db_df)
        n = len(joined)
        overall_rows.append({
            "config": config,
            "n": n,
            "pass_std_%": round(pass_rate(joined["error_ratio"], THRESHOLD_STD), 2),
            "pass_strict_%": round(pass_rate(joined["error_ratio"], THRESHOLD_STRICT), 2),
        })

        for bname, bfn in BANDS:
            mask = bfn(joined["t_true"])
            sub = joined.loc[mask, "error_ratio"]
            band_rows.append({
                "config": config,
                "band": bname,
                "n": int(mask.sum()),
                "pass_std_%": round(pass_rate(sub, THRESHOLD_STD), 1) if len(sub) else float("nan"),
                "pass_strict_%": round(pass_rate(sub, THRESHOLD_STRICT), 1) if len(sub) else float("nan"),
            })

    overall_df = pd.DataFrame(overall_rows)
    band_df = pd.DataFrame(band_rows)

    print("=== Overall pass rate (hold-out) ===")
    print(overall_df.to_string(index=False))

    print("\n=== t帯別 pass_std (%) ===")
    pivot_std = band_df.pivot(index="config", columns="band", values="pass_std_%")
    pivot_std = pivot_std[[b for b, _ in BANDS]].reindex(CONFIGS)
    print(pivot_std.to_string())

    print("\n=== t帯別 pass_strict (%) ===")
    pivot_strict = band_df.pivot(index="config", columns="band", values="pass_strict_%")
    pivot_strict = pivot_strict[[b for b, _ in BANDS]].reindex(CONFIGS)
    print(pivot_strict.to_string())

    print("\n=== n per band ===")
    pivot_n = band_df.pivot(index="config", columns="band", values="n")
    pivot_n = pivot_n[[b for b, _ in BANDS]].reindex(CONFIGS)
    print(pivot_n.to_string())

    overall_df.to_csv(OUTPUT_DIR / "passrate_overall.csv", index=False)
    band_df.to_csv(OUTPUT_DIR / "passrate_by_band.csv", index=False)
    print(f"\nSaved -> {OUTPUT_DIR / 'passrate_overall.csv'}")
    print(f"Saved -> {OUTPUT_DIR / 'passrate_by_band.csv'}")


if __name__ == "__main__":
    main()
