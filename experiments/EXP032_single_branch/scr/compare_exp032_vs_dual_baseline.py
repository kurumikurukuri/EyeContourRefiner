"""EXP032 vs 現行dual標準ベースラインの paired t-test。

scripts/paired_ttest.py の --exp は同一実験ディレクトリ内の config 名しか解決できない
（find_predictions_csv(exp_dir, config) が exp_dir/outputs/{config}/ 固定）ため、
比較用に EXP031 SD0_baseline の予測 CSV を読み取り専用で参照する。

デフォルトの比較対象:
    baseline = experiments/EXP031_source_space_aug/outputs/SD0_baseline/test_predictions.csv
               （現行dual標準ベースライン）
    target   = experiments/EXP032_single_branch/outputs/SB_eyescale/test_predictions.csv
               （単独ブランチ本命条件）

使い方:
    python experiments/EXP032_single_branch/scr/compare_exp032_vs_dual_baseline.py
    python experiments/EXP032_single_branch/scr/compare_exp032_vs_dual_baseline.py --target SB_none
    python experiments/EXP032_single_branch/scr/compare_exp032_vs_dual_baseline.py --metric error_ratio
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
OUTPUT_DIR = EXP_DIR / "outputs"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _common import load_db_index, sanitize_for_json  # noqa: E402
from paired_ttest import paired_comparison, print_ttest_table  # noqa: E402

DEFAULT_BASELINE = (
    ROOT / "experiments" / "EXP031_source_space_aug" / "outputs" / "SD0_baseline" / "test_predictions.csv"
)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description="EXP032 単独ブランチ vs 現行dual標準ベースライン paired t-test")
    ap.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE,
                    help="dual標準ベースラインの test_predictions.csv（読み取り専用参照）")
    ap.add_argument("--target", type=str, default="SB_eyescale", choices=["SB_eyescale", "SB_none"],
                    help="EXP032側の比較対象 config 名")
    ap.add_argument("--metric", choices=["raw_t", "error_ratio"], default="raw_t")
    args = ap.parse_args()

    target_csv = OUTPUT_DIR / args.target / "test_predictions.csv"
    if not args.baseline_csv.exists():
        raise FileNotFoundError(f"dual標準ベースラインの predictions.csv が見つかりません: {args.baseline_csv}")
    if not target_csv.exists():
        raise FileNotFoundError(f"EXP032側の predictions.csv が見つかりません: {target_csv}")

    print(f"baseline(dual標準): {args.baseline_csv}")
    print(f"target(単独ブランチ {args.target}): {target_csv}")

    db_df = load_db_index()
    result = paired_comparison(args.baseline_csv, target_csv, db_df, metric=args.metric)
    if "error" in result:
        print(f"[ERROR] {result['error']}")
        return 1

    print_ttest_table(result)

    out_path = OUTPUT_DIR / f"ttest_{args.target}_vs_dual_baseline.json"
    out_path.write_text(json.dumps(sanitize_for_json(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n保存: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
