"""EXP025 A0-A4 の hold-out 生t誤差を t帯別・点別(14点=point_index×sub_key)に集計する。

各 config（A1〜A4）の A0 からの差分がどこに集中しているか
（特定の点・特定t帯に偏るか、全体に薄く広がるか）を切り出す。

使い方（プロジェクトルートから）:
    python experiments/EXP025_aux_redesign/scr/exp025_band_point_breakdown.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
OUTPUT_DIR = EXP_DIR / "outputs"
MANIFEST_TEST = EXP_DIR / "manifests" / "manifest_test.csv"

CONFIGS = ["A0", "A1", "A2", "A3", "A4"]

BANDS = [
    ("t<0", lambda t: t < 0),
    ("0-0.3", lambda t: (t >= 0) & (t < 0.3)),
    ("0.3-0.7", lambda t: (t >= 0.3) & (t < 0.7)),
    ("0.7-1.0", lambda t: (t >= 0.7) & (t < 1.0)),
    ("1.0-1.2", lambda t: (t >= 1.0) & (t < 1.2)),
    ("t>=1.2", lambda t: t >= 1.2),
]


def load_manifest_keys() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST_TEST)
    return df[["idx", "sub_key", "point_index"]].set_index("idx")


def load_joined(config: str, keys: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_csv(OUTPUT_DIR / config / "test_predictions.csv")
    joined = pred.join(keys, on="idx")
    joined["abs_err"] = (joined["t_pred"] - joined["t_true"]).abs()
    joined["point"] = joined["sub_key"] + "_" + joined["point_index"].astype(str)
    return joined


def main() -> None:
    keys = load_manifest_keys()
    point_order = [f"{sk}_{pi}" for sk in ("upper", "lower") for pi in range(7)]

    band_rows = []
    point_rows = []
    for config in CONFIGS:
        joined = load_joined(config, keys)

        for bname, bfn in BANDS:
            mask = bfn(joined["t_true"])
            sub = joined.loc[mask, "abs_err"]
            band_rows.append({
                "config": config, "band": bname, "n": int(mask.sum()),
                "mae": round(float(sub.mean()), 5) if len(sub) else float("nan"),
            })

        for pt in point_order:
            sub = joined.loc[joined["point"] == pt, "abs_err"]
            point_rows.append({
                "config": config, "point": pt, "n": int(len(sub)),
                "mae": round(float(sub.mean()), 5) if len(sub) else float("nan"),
            })

    band_df = pd.DataFrame(band_rows)
    point_df = pd.DataFrame(point_rows)

    print("=== t帯別 生t MAE ===")
    pivot_band = band_df.pivot(index="config", columns="band", values="mae")
    pivot_band = pivot_band[[b for b, _ in BANDS]].reindex(CONFIGS)
    print(pivot_band.to_string())

    print("\n=== t帯別 n ===")
    pivot_band_n = band_df.pivot(index="config", columns="band", values="n")
    pivot_band_n = pivot_band_n[[b for b, _ in BANDS]].reindex(CONFIGS)
    print(pivot_band_n.to_string())

    print("\n=== 点別(14点) 生t MAE ===")
    pivot_pt = point_df.pivot(index="config", columns="point", values="mae")
    pivot_pt = pivot_pt[point_order].reindex(CONFIGS)
    print(pivot_pt.to_string())

    print("\n=== 点別(14点) n ===")
    pivot_pt_n = point_df.pivot(index="config", columns="point", values="n")
    pivot_pt_n = pivot_pt_n[point_order].reindex(CONFIGS)
    print(pivot_pt_n.to_string())

    # A0 を基準にした差分（A1〜A4 各 config - A0）を点別・帯別で可視化。
    print("\n=== 点別 MAE 差分（各config - A0）===")
    diff_pt = pivot_pt.subtract(pivot_pt.loc["A0"], axis=1)
    print(diff_pt.loc[[c for c in CONFIGS if c != "A0"]].to_string())

    print("\n=== t帯別 MAE 差分（各config - A0）===")
    diff_band = pivot_band.subtract(pivot_band.loc["A0"], axis=1)
    print(diff_band.loc[[c for c in CONFIGS if c != "A0"]].to_string())

    band_df.to_csv(OUTPUT_DIR / "band_breakdown.csv", index=False)
    point_df.to_csv(OUTPUT_DIR / "point_breakdown.csv", index=False)
    print(f"\nSaved -> {OUTPUT_DIR / 'band_breakdown.csv'}")
    print(f"Saved -> {OUTPUT_DIR / 'point_breakdown.csv'}")


if __name__ == "__main__":
    main()
