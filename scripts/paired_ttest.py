"""決定論的 paired t-test ツール。

同一実験内の config 間、または異なる実験間で per-sample 誤差の paired t-test を実行する。
共通 idx で突き合わせ、全体 + t帯別で比較する。

使用例:
  python scripts/paired_ttest.py --exp EXP026 --baseline k3_b32
  python scripts/paired_ttest.py --exp EXP026 --baseline k3_b32 --targets k5_b64
  python scripts/paired_ttest.py --exp EXP026 --baseline k3_b32 --metric error_ratio
  python scripts/paired_ttest.py --exp EXP025 --baseline A0 --targets A1 A2 A3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from _common import (
    DB_PATH,
    PREDICTIONS_CSV_NAMES,
    T_BANDS,
    THRESHOLD_STD,
    compute_error_ratio,
    find_exp_dir,
    find_predictions_csv,
    load_db_index,
    locate_config_file,
    read_predictions_csv,
    sanitize_for_json,
)


# ---------------------------------------------------------------------------
# Experiment discovery (paired_ttest-specific)
# ---------------------------------------------------------------------------

def list_configs(exp_dir: Path) -> list[str]:
    """実験ディレクトリの outputs/ 以下にある config 名一覧を返す。

    outputs/{config}/ と outputs/{config}/{config}/ の両方の構成を扱う（_common.config_output_dirs）。
    """
    outputs_dir = exp_dir / "outputs"
    if not outputs_dir.exists():
        return []
    return sorted(
        d.name for d in outputs_dir.iterdir()
        if d.is_dir() and locate_config_file(exp_dir, d.name, PREDICTIONS_CSV_NAMES) is not None
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def sig(p: float) -> str:
    """p 値から有意性マークを返す。"""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def paired_comparison(
    baseline_csv: Path,
    target_csv: Path,
    db_df: pd.DataFrame,
    metric: str = "raw_t",
) -> dict:
    """2 つの predictions.csv 間で paired t-test を実行する。

    Args:
        baseline_csv: ベースラインの predictions.csv
        target_csv:   比較対象の predictions.csv
        db_df:        DB index (load_db_index())
        metric:       "raw_t" (|t_pred - t_true|) or "error_ratio"

    Returns:
        {
            "baseline": str, "target": str, "metric": str,
            "n_common": int,
            "overall": { "mae_base": float, "mae_target": float, "improvement_pct": float,
                         "t_stat": float, "ttest_p": float, "sig": str },
            "by_band": { band_name: { "n": int, "mae_base": float, "mae_target": float,
                                      "improvement_pct": float, "ttest_p": float, "sig": str } },
        }
    """
    base_df = read_predictions_csv(baseline_csv)
    tgt_df = read_predictions_csv(target_csv)

    # 共通 idx で突き合わせ
    common_idx = sorted(set(base_df["idx"]) & set(tgt_df["idx"]))
    if len(common_idx) < 2:
        return {
            "baseline": str(baseline_csv), "target": str(target_csv),
            "metric": metric, "n_common": len(common_idx),
            "error": "共通サンプルが不足しています（n < 2）",
        }

    base_df = base_df.set_index("idx").loc[common_idx].reset_index()
    tgt_df = tgt_df.set_index("idx").loc[common_idx].reset_index()

    if metric == "error_ratio":
        base_err = compute_error_ratio(base_df, db_df)
        tgt_err = compute_error_ratio(tgt_df, db_df)
        # error_ratio が有効なサンプルのみに絞る
        valid = base_err.notna() & tgt_err.notna()
        base_err = base_err[valid]
        tgt_err = tgt_err[valid]
        common_idx_valid = [idx for i, idx in enumerate(common_idx) if valid.iloc[i]]
    else:
        base_err = (base_df["t_pred"] - base_df["t_true"]).abs()
        tgt_err = (tgt_df["t_pred"] - tgt_df["t_true"]).abs()
        common_idx_valid = common_idx

    n_common = len(common_idx_valid)
    if n_common < 2:
        return {
            "baseline": str(baseline_csv), "target": str(target_csv),
            "metric": metric, "n_common": n_common,
            "error": "有効サンプルが不足しています（n < 2）",
        }

    base_arr = base_err.values
    tgt_arr = tgt_err.values

    mae_base = float(np.mean(base_arr))
    mae_target = float(np.mean(tgt_arr))
    improvement_pct = float((mae_base - mae_target) / mae_base * 100) if mae_base > 0 else 0.0

    t_stat, ttest_p = ttest_rel(tgt_arr, base_arr)
    if np.isnan(t_stat) or np.isnan(ttest_p):
        t_stat, ttest_p = 0.0, 1.0  # degenerate: identical distributions

    result: dict = {
        "baseline": str(baseline_csv),
        "target": str(target_csv),
        "metric": metric,
        "n_common": n_common,
        "overall": {
            "mae_base": round(mae_base, 6),
            "mae_target": round(mae_target, 6),
            "improvement_pct": round(improvement_pct, 3),
            "t_stat": round(float(t_stat), 4),
            "ttest_p": round(float(ttest_p), 6),
            "sig": sig(ttest_p),
        },
        "by_band": {},
    }

    if metric == "error_ratio":
        result["overall"]["pass_std_base"] = float((base_arr <= THRESHOLD_STD).mean() * 100)
        result["overall"]["pass_std_target"] = float((tgt_arr <= THRESHOLD_STD).mean() * 100)

    t_true = base_df["t_true"].values
    if metric == "error_ratio":
        t_true = t_true[valid.values]

    for bname, bfn in T_BANDS:
        mask = bfn(t_true)
        bn = int(mask.sum())
        if bn < 2:
            result["by_band"][bname] = {
                "n": bn, "mae_base": None, "mae_target": None,
                "improvement_pct": None, "ttest_p": None, "sig": "n/a",
            }
            continue

        b_base = base_arr[mask]
        b_tgt = tgt_arr[mask]
        b_mae_base = float(np.mean(b_base))
        b_mae_target = float(np.mean(b_tgt))
        b_impr = float((b_mae_base - b_mae_target) / b_mae_base * 100) if b_mae_base > 0 else 0.0
        _, b_ttest_p = ttest_rel(b_tgt, b_base)
        if np.isnan(b_ttest_p):
            b_ttest_p = 1.0  # degenerate: identical distributions

        result["by_band"][bname] = {
            "n": bn,
            "mae_base": round(b_mae_base, 6),
            "mae_target": round(b_mae_target, 6),
            "improvement_pct": round(b_impr, 3),
            "ttest_p": round(float(b_ttest_p), 6),
            "sig": sig(b_ttest_p),
        }

    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_ttest_table(result: dict) -> None:
    """paired t-test 結果を標準出力に markdown 表形式で表示する。"""
    overall = result["overall"]
    band_names = [b for b, _ in T_BANDS]

    print(f"\n{'='*80}")
    print(f"  Paired t-test: {Path(result['target']).parent.name} vs {Path(result['baseline']).parent.name}")
    print(f"  metric={result['metric']}  n={result['n_common']}")
    print(f"{'='*80}")

    # overall
    print("\n  Overall:")
    print(f"    MAE base={overall['mae_base']:.6f}  target={overall['mae_target']:.6f}  "
          f"改善={overall['improvement_pct']:+.2f}%  "
          f"t={overall['t_stat']:.4f}  p={overall['ttest_p']:.6f}  {overall['sig']}")

    # by band
    print("\n  --- by t-band ---")
    header = f"  {'band':<10} {'n':>5} {'MAE_base':>10} {'MAE_target':>10} {'改善%':>8} {'p':>10} {'sig':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for bname in band_names:
        b = result["by_band"][bname]
        if b["n"] == 0:
            continue
        mae_b_str = f"{b['mae_base']:>10.6f}" if b['mae_base'] is not None else f"{'—':>10}"
        mae_t_str = f"{b['mae_target']:>10.6f}" if b['mae_target'] is not None else f"{'—':>10}"
        impr_str = f"{b['improvement_pct']:>+8.2f}%" if b['improvement_pct'] is not None else f"{'—':>9}"
        p_str = f"{b['ttest_p']:>10.6f}" if b['ttest_p'] is not None else f"{'—':>10}"
        print(f"  {bname:<10} {b['n']:>5} {mae_b_str} {mae_t_str} {impr_str} {p_str} {b['sig']:>5}")

    # markdown copy-paste
    print("\n  --- Markdown (copy-paste) ---")
    print("  | band | n | MAE(base) | MAE(target) | diff% | t_p | sig |")
    print("  |---|---|---|---|---|---|---|")
    print(f"  | **ALL** | {result['n_common']} | {overall['mae_base']:.6f} | "
          f"{overall['mae_target']:.6f} | {overall['improvement_pct']:+.2f}% | "
          f"{overall['ttest_p']:.6f} | {overall['sig']} |")
    for bname in band_names:
        b = result["by_band"][bname]
        if b["n"] < 2:
            continue
        mae_b = b['mae_base'] if b['mae_base'] is not None else float('nan')
        mae_t = b['mae_target'] if b['mae_target'] is not None else float('nan')
        impr = b['improvement_pct'] if b['improvement_pct'] is not None else float('nan')
        pval = b['ttest_p'] if b['ttest_p'] is not None else float('nan')
        print(f"  | {bname} | {b['n']} | {mae_b:.6f} | {mae_t:.6f} | "
              f"{impr:+.2f}% | {pval:.6f} | {b['sig']} |")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(
        description="決定論的 paired t-test ツール"
    )
    ap.add_argument(
        "--exp", type=str, required=True,
        help="実験番号（例: 026）",
    )
    ap.add_argument(
        "--baseline", type=str, required=True,
        help="ベースライン config 名",
    )
    ap.add_argument(
        "--targets", nargs="*", default=None,
        help="比較対象 config 名（省略時は baseline 以外の全 config）",
    )
    ap.add_argument(
        "--metric", choices=["raw_t", "error_ratio"], default="raw_t",
        help="比較メトリック: raw_t（|t_pred-t_true|）or error_ratio（default: raw_t）",
    )
    ap.add_argument(
        "--db", type=Path, default=DB_PATH,
        help="samples.db のパス",
    )
    args = ap.parse_args()

    exp_dir = find_exp_dir(args.exp)
    print(f"実験ディレクトリ: {exp_dir}")

    baseline_csv = find_predictions_csv(exp_dir, args.baseline)

    if args.targets:
        targets = args.targets
    else:
        all_configs = list_configs(exp_dir)
        targets = [c for c in all_configs if c != args.baseline]
        if not targets:
            print("[ERROR] baseline 以外の config が見つかりません")
            return 1

    db_df = load_db_index(args.db)

    all_results = []
    out_dir = exp_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        target_csv = find_predictions_csv(exp_dir, target)
        print(f"\n比較: {target} vs {args.baseline}  (metric={args.metric})")

        try:
            result = paired_comparison(baseline_csv, target_csv, db_df, metric=args.metric)
        except pd.errors.ParserError as e:
            print(f"  [ERROR] CSV のパースに失敗しました: {e}")
            continue
        if "error" in result:
            print(f"  [SKIP] {result['error']}")
            continue

        all_results.append(result)
        print_ttest_table(result)

        # save JSON
        json_path = out_dir / f"ttest_{target}_vs_{args.baseline}.json"
        json_path.write_text(
            json.dumps(sanitize_for_json(result), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if all_results:
        # save combined
        combined = {
            "exp": exp_dir.name,
            "baseline": args.baseline,
            "metric": args.metric,
            "comparisons": all_results,
        }
        combined_path = out_dir / f"ttest_all_vs_{args.baseline}.json"
        combined_path.write_text(
            json.dumps(sanitize_for_json(combined), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n保存: {combined_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
