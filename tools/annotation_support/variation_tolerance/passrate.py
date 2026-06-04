"""
Pass rate computation across all experiment conditions.

error_ratio = |t_pred - t_true| * pq_dist / eye_scale
pass_rate(threshold) = #{error_ratio <= threshold} / total

Thresholds from experiments/Completion_Criteria.md:
  Standard : 0.01788 (p10)
  Strict   : 0.01604 (p5)

Rows whose idx is absent from the DB are excluded and reported.

Usage (run from project root):
  python tools/annotation_support/variation_tolerance/passrate.py
  python tools/annotation_support/variation_tolerance/passrate.py --band EXP021
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENTS_DIR = Path("experiments")
DB_PATH = Path("data/samples.db")
OUTPUT_DIR = Path(__file__).parent / "outputs"

THRESHOLD_STD = 0.01788
THRESHOLD_STRICT = 0.01604

BANDS = [
    ("t<0",     lambda t: t < 0),
    ("0-0.3",   lambda t: (t >= 0) & (t < 0.3)),
    ("0.3-0.7", lambda t: (t >= 0.3) & (t < 0.7)),
    ("0.7-1.0", lambda t: (t >= 0.7) & (t < 1.0)),
    ("1.0-1.2", lambda t: (t >= 1.0) & (t < 1.2)),
    ("t>=1.2",  lambda t: t >= 1.2),
]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def load_db_index() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT idx, px, py, qx, qy, eye_scale FROM samples",
        conn,
    )
    conn.close()
    return df.set_index("idx")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename column variants so the rest of the code sees t_true / t_pred."""
    rename = {}
    if "true_t_raw" in df.columns:
        rename["true_t_raw"] = "t_true"
    if "pred_t_raw" in df.columns:
        rename["pred_t_raw"] = "t_pred"
    return df.rename(columns=rename) if rename else df


def compute_pass_rate(
    csv_path: Path,
    db_df: pd.DataFrame,
) -> dict | None:
    """
    Returns a result dict, or None if the CSV format is unrecognised.
    Rows missing from the DB are excluded; their image_paths are logged.
    """
    pred = pd.read_csv(csv_path)
    pred = normalize_columns(pred)

    if "t_true" not in pred.columns or "t_pred" not in pred.columns:
        return None

    has_pq_in_csv = all(c in pred.columns for c in ("px", "py", "qx", "qy"))

    # --- detect rows missing from DB ---
    missing_mask = ~pred["idx"].isin(db_df.index)
    if missing_mask.any():
        missing_paths = pred.loc[missing_mask, "image_path"].unique().tolist()
    else:
        missing_paths = []

    pred = pred[~missing_mask].copy()

    if pred.empty:
        return {
            "n_patches": 0,
            "n_missing": int(missing_mask.sum()),
            "missing_image_paths": missing_paths,
            "pass_rate_std": float("nan"),
            "pass_rate_strict": float("nan"),
        }

    # --- join DB columns ---
    joined = pred.join(db_df, on="idx", rsuffix="_db")

    if has_pq_in_csv:
        px, py, qx, qy = joined["px"], joined["py"], joined["qx"], joined["qy"]
    else:
        px = joined["px_db"] if "px_db" in joined.columns else joined["px"]
        py = joined["py_db"] if "py_db" in joined.columns else joined["py"]
        qx = joined["qx_db"] if "qx_db" in joined.columns else joined["qx"]
        qy = joined["qy_db"] if "qy_db" in joined.columns else joined["qy"]

    pq_dist = np.sqrt((px - qx) ** 2 + (py - qy) ** 2)
    eye_scale = joined["eye_scale"]

    valid = eye_scale.notna() & (eye_scale > 0) & pq_dist.notna()
    pq_dist = pq_dist[valid]
    eye_scale = eye_scale[valid]
    t_err = (joined["t_pred"] - joined["t_true"]).abs()[valid]
    t_true_valid = joined["t_true"][valid]

    error_ratio = t_err * pq_dist / eye_scale

    n = len(error_ratio)
    pass_std = float((error_ratio <= THRESHOLD_STD).mean() * 100)
    pass_strict = float((error_ratio <= THRESHOLD_STRICT).mean() * 100)

    band_stats = {}
    for bname, bfn in BANDS:
        mask = bfn(t_true_valid)
        bn = int(mask.sum())
        if bn == 0:
            band_stats[bname] = {"n": 0, "std": float("nan"), "strict": float("nan")}
        else:
            sub = error_ratio[mask]
            band_stats[bname] = {
                "n": bn,
                "std": round(float((sub <= THRESHOLD_STD).mean() * 100), 1),
                "strict": round(float((sub <= THRESHOLD_STRICT).mean() * 100), 1),
            }

    return {
        "n_patches": n,
        "n_missing": int(missing_mask.sum()),
        "missing_image_paths": missing_paths,
        "pass_rate_std": pass_std,
        "pass_rate_strict": pass_strict,
        "band_stats": band_stats,
    }


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def collect_prediction_files(experiments_dir: Path) -> list[tuple[str, str, Path]]:
    """
    Return (exp_name, condition, path) for every test/holdout prediction CSV.
    train_predictions.csv is ignored.
    """
    candidates: dict[tuple[str, str], Path] = {}

    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        outputs_dir = exp_dir / "outputs"
        if not outputs_dir.exists():
            continue

        for csv_path in sorted(outputs_dir.rglob("*.csv")):
            if csv_path.name != "test_predictions.csv":
                continue
            condition = csv_path.relative_to(outputs_dir).parts[0]
            key = (exp_dir.name, condition)
            candidates[key] = csv_path

    return [(exp, cond, path) for (exp, cond), path in sorted(candidates.items())]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def best_per_exp_markdown(summary: pd.DataFrame) -> str:
    """各 EXP の pass_std 最良条件のみを抽出した markdown 表を返す。"""
    std_col = f"pass_std_{THRESHOLD_STD:.5f}_%"
    strict_col = f"pass_strict_{THRESHOLD_STRICT:.5f}_%"

    best = (
        summary.loc[summary.groupby("exp")[std_col].idxmax()]
        .sort_values("exp")
        .reset_index(drop=True)
    )

    lines = [
        f"| exp | best_condition | n_patches | pass_std_{THRESHOLD_STD} % | pass_strict_{THRESHOLD_STRICT} % |",
        "|---|---|---|---|---|",
    ]
    for _, row in best.iterrows():
        lines.append(
            f"| {row['exp']} | {row['condition']} | {row['n_patches']} | "
            f"{row[std_col]:.2f} | {row[strict_col]:.2f} |"
        )
    return "\n".join(lines)


def print_band_table(band_rows: list[dict], threshold_key: str, title: str) -> None:
    band_names = [b for b, _ in BANDS]
    header = f"{'condition':<32}" + "".join(f"{b:>10}" for b in band_names) + f"{'ALL':>8}{'n':>7}"
    print(f"\n=== {title} ===")
    print(header)
    print("-" * len(header))
    for row in band_rows:
        line = f"{row['condition']:<32}"
        for b in band_names:
            v = row["band_stats"][b][threshold_key]
            line += f"{v:>10.1f}" if not (v != v) else f"{'—':>10}"
        line += f"{row['pass_overall']:>8.1f}{row['n_patches']:>7}"
        print(line)
    # サンプル数行
    first = band_rows[0]["band_stats"] if band_rows else {}
    if first:
        counts = "  n: " + " ".join(f"{b}={first[b]['n']}" for b, _ in BANDS)
        print(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--band", metavar="EXP", default=None,
                        help="指定した実験名のt帯別 pass rate を表示 (例: EXP021)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    db_df = load_db_index()
    files = collect_prediction_files(EXPERIMENTS_DIR)
    print(f"Found {len(files)} prediction files\n")

    rows = []
    band_rows: list[dict] = []
    all_missing_paths: dict[str, list[str]] = {}  # key = "exp/condition"

    for exp_name, condition, csv_path in files:
        label = f"{exp_name}/{condition}"
        result = compute_pass_rate(csv_path, db_df)

        if result is None:
            print(f"  SKIP (unrecognised format): {label}")
            continue

        if result["n_missing"]:
            all_missing_paths[label] = result["missing_image_paths"]

        rows.append(
            {
                "exp": exp_name,
                "condition": condition,
                "n_patches": result["n_patches"],
                "n_missing": result["n_missing"],
                f"pass_std_{THRESHOLD_STD:.5f}_%": round(result["pass_rate_std"], 2),
                f"pass_strict_{THRESHOLD_STRICT:.5f}_%": round(result["pass_rate_strict"], 2),
            }
        )

        if args.band and args.band.lower() in exp_name.lower():
            band_rows.append({
                "condition": condition,
                "n_patches": result["n_patches"],
                "band_stats": result["band_stats"],
                "pass_overall": round(result["pass_rate_std"], 1),
            })

    # --- report missing rows ---
    if all_missing_paths:
        print("=" * 60)
        print("WARNING: rows excluded (idx absent from DB)")
        print("=" * 60)
        for label, paths in sorted(all_missing_paths.items()):
            print(f"\n  [{label}]  {len(paths)} image(s):")
            for p in paths:
                print(f"    {p}")
        print()

    # --- summary table ---
    summary = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "pass_rate_summary.csv"
    summary.to_csv(out_path, index=False)

    print("=" * 60)
    print("PASS RATE SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))
    print(f"\nSaved → {out_path}")

    # --- best condition per EXP (markdown) ---
    best_md = best_per_exp_markdown(summary)
    best_md_path = OUTPUT_DIR / "best_per_exp.md"
    best_md_path.write_text(best_md, encoding="utf-8")

    print("\n=== EXP別ベスト条件 (pass_std基準) ===")
    print(best_md)
    print(f"\nSaved → {best_md_path}")

    if band_rows:
        print_band_table(band_rows, "std",    f"t帯別 pass_std ({THRESHOLD_STD}) %")
        print_band_table(band_rows, "strict", f"t帯別 pass_strict ({THRESHOLD_STRICT}) %")

        band_out_path = OUTPUT_DIR / f"band_pass_rate_{args.band}.csv"
        band_csv_rows = []
        for row in band_rows:
            for thr_key in ("std", "strict"):
                for bname, _ in BANDS:
                    band_csv_rows.append({
                        "condition": row["condition"],
                        "threshold": thr_key,
                        "band": bname,
                        "n": row["band_stats"][bname]["n"],
                        "pass_%": row["band_stats"][bname][thr_key],
                    })
        pd.DataFrame(band_csv_rows).to_csv(band_out_path, index=False)
        print(f"\nBand stats saved → {band_out_path}")


if __name__ == "__main__":
    main()
