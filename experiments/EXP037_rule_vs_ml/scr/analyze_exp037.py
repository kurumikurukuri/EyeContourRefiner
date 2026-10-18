"""EXP037 精度分析: points_exp037.csv から、評価ツール（compute_passrate / paired_ttest）が出さない観点を集計する。

- 画像単位の対応比較: 同じ画像の点は独立でないため、画像ごとの平均誤差で Wilcoxon 符号順位検定と
  画像単位ブートストラップ（差の 95% CI）を行う
- 上瞼 / 下瞼別の MAE・pixel 誤差・pass_std
- ルールベースの fallback（エッジ未検出で P を返す）の割合と、fallback 有無別の誤差
- ルールベースの参考値: 出力の整数切り捨てを外した場合（rule_untrunc）、入力を小数座標にした場合（rule_float）
- ルールベースの探索範囲（t ∈ (0.5, 1]）外に正解がある点の割合
- 失敗要因（fallback・探索範囲外）の重なりと、両方を除いた点での集計

出力: outputs/analysis_exp037.json（標準出力にも表示）

使い方:
    python experiments/EXP037_rule_vs_ml/scr/analyze_exp037.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

EXP_DIR = Path(__file__).parent.parent
POINTS_PATH = EXP_DIR / "outputs" / "points_exp037.csv"
OUTPUT_PATH = EXP_DIR / "outputs" / "analysis_exp037.json"

THRESHOLD_STD = 0.01788
RULE_T_RANGE = (0.5, 1.0)   # face_makeup は P から Q へ距離の 0〜0.5 を探索する（t_model = 1 - t_rule）
N_BOOTSTRAP = 10_000
SEED = 0
# rule = 本番と同じ整数座標入力（主結果）。rule_untrunc = 出力の整数切り捨て前、rule_float = 小数座標入力（いずれも参考）
# ml_int = ML に整数座標を入力した場合（参考）
METHODS = ("rule", "rule_untrunc", "rule_float", "ml", "ml_int")


def summarize(points: pd.DataFrame) -> dict:
    return {
        "n": len(points),
        **{
            method: {
                "mae_t": float(points[f"abs_t_err_{method}"].mean()),
                "px_err_mean": float(points[f"px_err_{method}"].mean()),
                "px_err_median": float(points[f"px_err_{method}"].median()),
                "pass_std_pct": float((points[f"error_ratio_{method}"] <= THRESHOLD_STD).mean() * 100),
            }
            for method in METHODS
        },
    }


def image_level_comparison(points: pd.DataFrame, metric: str) -> dict:
    """画像ごとの平均 {metric} で ML − ルールの差を検定する（負なら ML の方が誤差が小さい）。"""
    per_image = points.groupby("image_path")[[f"{metric}_rule", f"{metric}_ml"]].mean()
    diff = (per_image[f"{metric}_ml"] - per_image[f"{metric}_rule"]).to_numpy()
    rng = np.random.default_rng(SEED)
    boot = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(N_BOOTSTRAP)])
    return {
        "n_images": len(diff),
        "mean_diff_ml_minus_rule": float(diff.mean()),
        "bootstrap_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "ml_better_images_pct": float((diff < 0).mean() * 100),
        "wilcoxon_p": float(wilcoxon(diff).pvalue),
    }


def rule_range_coverage(points: pd.DataFrame) -> dict:
    low, high = RULE_T_RANGE
    outside = (points["t_true"] <= low) | (points["t_true"] > high)
    return {
        "rule_search_range_t": [low, high],
        "t_true_outside_pct": float(outside.mean() * 100),
        "t_rule_min": float(points["t_rule"].min()),
        "t_rule_max": float(points["t_rule"].max()),
        "inside_range": summarize(points[~outside]),
        "outside_range": summarize(points[outside]),
    }


def factor_exclusion(points: pd.DataFrame) -> dict:
    """ルールベースの失敗要因のうち fallback と探索範囲外の重なりと、両方を除いた点での集計。

    整数切り捨ては点を除くのではなく rule_untrunc で見る。
    """
    low, high = RULE_T_RANGE
    fallback = points["rule_fallback"]
    outside = (points["t_true"] <= low) | (points["t_true"] > high)
    return {
        "overlap": {
            "fallback_and_outside": int((fallback & outside).sum()),
            "fallback_only": int((fallback & ~outside).sum()),
            "outside_only": int((~fallback & outside).sum()),
            "either_pct": float((fallback | outside).mean() * 100),
        },
        "exclude_fallback_and_outside": summarize(points[~fallback & ~outside]),
    }


def analyze(points: pd.DataFrame) -> dict:
    return {
        "overall": summarize(points),
        "image_level": {metric: image_level_comparison(points, metric) for metric in ("abs_t_err", "px_err")},
        "by_sub_key": {key: summarize(group) for key, group in points.groupby("sub_key")},
        "rule_fallback": {
            "fallback_pct": float(points["rule_fallback"].mean() * 100),
            "fallback": summarize(points[points["rule_fallback"]]),
            "detected": summarize(points[~points["rule_fallback"]]),
        },
        "rule_range": rule_range_coverage(points),
        "factor_exclusion": factor_exclusion(points),
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    if not POINTS_PATH.exists():
        raise FileNotFoundError(f"{POINTS_PATH} がありません。先に infer_exp037.py を実行する")

    result = analyze(pd.read_csv(POINTS_PATH))
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"保存: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
