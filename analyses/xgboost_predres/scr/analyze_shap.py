"""特徴量重要度 + SHAP 分析。

bin_005 (または指定 config) のモデルとテストデータを対象に:
  - XGBoost feature importance (weight / gain / cover) の CSV 出力
  - SHAP TreeExplainer による値の算出
  - クラス別 beeswarm / summary bar の PNG 保存
  - SHAP mean |値| の CSV 出力
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

LABELS = ["under", "ok", "over"]


# ---------------------------------------------------------------------------
# Feature preparation (infer.py と同じロジック)
# ---------------------------------------------------------------------------
def _prepare_features(
    df: pd.DataFrame,
    feat_cols: List[str],
    train_medians: pd.Series,
    percentile_refs: dict,
    image_cache_path: Optional[str],
    feat_cfg_dict: Dict[str, Any],
) -> pd.DataFrame:
    from analyses.xgboost_baseline.scr.features import (
        FeatureConfig,
        add_dataset_percentile_features,
        enrich_all,
    )

    feat_cfg = FeatureConfig(
        use_geometry=feat_cfg_dict.get("use_geometry", True),
        use_eye_scale=feat_cfg_dict.get("use_eye_scale", True),
        use_face_pose=feat_cfg_dict.get("use_face_pose", True),
        use_categorical=feat_cfg_dict.get("use_categorical", True),
        use_point_index=feat_cfg_dict.get("use_point_index", True),
        use_cyclical_angle=feat_cfg_dict.get("use_cyclical_angle", False),
        use_interactions=feat_cfg_dict.get("use_interactions", False),
        use_niqe_score=feat_cfg_dict.get("use_niqe_score", False),
        use_aggregate_features=feat_cfg_dict.get("use_aggregate_features", False),
        use_extended_aggregate=feat_cfg_dict.get("use_extended_aggregate", False),
        use_pose_composite=feat_cfg_dict.get("use_pose_composite", False),
        use_dataset_percentile=feat_cfg_dict.get("use_dataset_percentile", False),
        use_image_features=feat_cfg_dict.get("use_image_features", False),
        use_eye_brightness=feat_cfg_dict.get("use_eye_brightness", False),
        use_oi_class_features=feat_cfg_dict.get("use_oi_class_features", False),
    )
    enriched = enrich_all(df, feat_cfg, image_cache_path=image_cache_path)
    if feat_cfg.use_dataset_percentile and percentile_refs:
        enriched, _ = add_dataset_percentile_features(enriched, percentile_refs)
    for c in feat_cols:
        if c not in enriched.columns:
            enriched[c] = np.nan
    return enriched[feat_cols].astype(float).fillna(train_medians)


# ---------------------------------------------------------------------------
# XGBoost feature importance (gain / weight / cover)
# ---------------------------------------------------------------------------
def save_xgb_importance(model: Any, feat_cols: List[str], outputs_dir: Path) -> pd.DataFrame:
    booster = model.get_booster() if hasattr(model, "get_booster") else None
    rows = []
    for imp_type in ["weight", "gain", "cover"]:
        if booster is None:
            break
        scores = booster.get_score(importance_type=imp_type)
        for feat in feat_cols:
            rows.append({
                "feature": feat,
                "importance_type": imp_type,
                "value": scores.get(feat, 0.0),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    pivot = df.pivot(index="feature", columns="importance_type", values="value").fillna(0)
    pivot = pivot.sort_values("gain", ascending=False)
    pivot.reset_index().to_csv(outputs_dir / "analysis_xgb_importance.csv", index=False)
    logger.info("wrote analysis_xgb_importance.csv (%d features)", len(pivot))
    return pivot


# ---------------------------------------------------------------------------
# SHAP 分析
# ---------------------------------------------------------------------------
def compute_shap(
    model: Any,
    X: pd.DataFrame,
    outputs_dir: Path,
    n_samples: int = 2000,
    is_classifier: bool = True,
) -> None:
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = outputs_dir / "shap_plots"
    plots_dir.mkdir(exist_ok=True)

    # サンプル数が多い場合はランダムサンプリング
    rng = np.random.default_rng(42)
    if len(X) > n_samples:
        idx = rng.choice(len(X), size=n_samples, replace=False)
        X_sub = X.iloc[idx].reset_index(drop=True)
        logger.info("SHAP: サンプリング %d → %d", len(X), n_samples)
    else:
        X_sub = X.reset_index(drop=True)

    logger.info("SHAP TreeExplainer 計算中 (%d samples × %d features) ...", len(X_sub), X_sub.shape[1])
    explainer = shap.TreeExplainer(model)
    shap_output = explainer(X_sub)

    # shap.Explanation の values shape: (n_samples, n_features) or (n_samples, n_features, n_classes)
    values = shap_output.values
    logger.info("SHAP values shape: %s", values.shape)

    if is_classifier and values.ndim == 3:
        # multi-class: (n_samples, n_features, n_classes)
        n_classes = values.shape[2]

        # --- クラス別 beeswarm ---
        for cls_i in range(n_classes):
            label = LABELS[cls_i] if cls_i < len(LABELS) else str(cls_i)
            exp_cls = shap.Explanation(
                values=values[:, :, cls_i],
                base_values=shap_output.base_values[:, cls_i] if shap_output.base_values.ndim == 2 else shap_output.base_values,
                data=X_sub.values,
                feature_names=list(X_sub.columns),
            )
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots.beeswarm(exp_cls, max_display=20, show=False)
            plt.title(f"SHAP beeswarm — class: {label}")
            plt.tight_layout()
            fig.savefig(plots_dir / f"shap_beeswarm_{label}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            logger.info("wrote shap_beeswarm_%s.png", label)

        # --- 全クラス mean |SHAP| の bar ---
        mean_abs = np.abs(values).mean(axis=(0, 2))  # (n_features,)
        fig, ax = plt.subplots(figsize=(8, 10))
        order = np.argsort(mean_abs)[::-1][:25]
        ax.barh(
            [X_sub.columns[i] for i in order[::-1]],
            mean_abs[order[::-1]],
        )
        ax.set_xlabel("mean |SHAP value| (all classes)")
        ax.set_title("Feature importance (SHAP)")
        plt.tight_layout()
        fig.savefig(plots_dir / "shap_bar_overall.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("wrote shap_bar_overall.png")

        # --- CSV: クラス別 mean |SHAP| ---
        rows = []
        for i, feat in enumerate(X_sub.columns):
            row = {"feature": feat, "mean_abs_shap_all": float(mean_abs[i])}
            for cls_i in range(n_classes):
                label = LABELS[cls_i] if cls_i < len(LABELS) else str(cls_i)
                row[f"mean_abs_shap_{label}"] = float(np.abs(values[:, i, cls_i]).mean())
            rows.append(row)
        shap_df = (
            pd.DataFrame(rows)
            .sort_values("mean_abs_shap_all", ascending=False)
            .reset_index(drop=True)
        )
        shap_df.to_csv(outputs_dir / "analysis_shap_importance.csv", index=False)
        logger.info("wrote analysis_shap_importance.csv")

    else:
        # 回帰 or binary: (n_samples, n_features)
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.beeswarm(shap_output, max_display=20, show=False)
        plt.title("SHAP beeswarm")
        plt.tight_layout()
        fig.savefig(plots_dir / "shap_beeswarm.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("wrote shap_beeswarm.png")

        mean_abs = np.abs(values).mean(axis=0)
        rows = [{"feature": f, "mean_abs_shap": float(v)}
                for f, v in zip(X_sub.columns, mean_abs)]
        pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).to_csv(
            outputs_dir / "analysis_shap_importance.csv", index=False
        )
        logger.info("wrote analysis_shap_importance.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="xgboost_predres SHAP 分析")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analyses/xgboost_predres/config/bin_005.yaml"),
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="分析対象スプリット（デフォルト: test）",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2000,
        help="SHAP 計算のサンプル上限（デフォルト: 2000）",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[ERROR] config が見つかりません: {args.config}", file=sys.stderr)
        sys.exit(1)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    manifests_dir = Path(cfg["data"]["manifests_dir"])
    outputs_dir = Path(cfg["data"]["outputs_dir"])
    image_cache_path: Optional[str] = cfg.get("data", {}).get("image_cache_path", None)
    feat_cfg_dict: Dict[str, Any] = cfg.get("features", {})

    # 分類 or 回帰を model ファイル名で判別
    cls_model_path = outputs_dir / "xgb_cls.joblib"
    reg_model_path = outputs_dir / "xgb_residual.joblib"
    if cls_model_path.exists():
        model_path = cls_model_path
        is_classifier = True
    elif reg_model_path.exists():
        model_path = reg_model_path
        is_classifier = False
    else:
        print(f"[ERROR] モデルが見つかりません: {outputs_dir}", file=sys.stderr)
        sys.exit(1)

    feat_cols: List[str] = json.loads(
        (outputs_dir / "feature_columns.json").read_text(encoding="utf-8")
    )
    train_medians = pd.read_csv(outputs_dir / "train_medians.csv", index_col=0).iloc[:, 0]
    percentile_refs: dict = {}
    pct_path = outputs_dir / "percentile_refs.json"
    if pct_path.exists():
        raw = json.loads(pct_path.read_text(encoding="utf-8"))
        percentile_refs = {k: np.array(v) for k, v in raw.items()}

    model = joblib.load(model_path)
    logger.info("loaded model: %s (classifier=%s)", model_path.name, is_classifier)

    manifest_path = manifests_dir / f"manifest_{args.split}.csv"
    if not manifest_path.exists():
        print(f"[ERROR] manifest が見つかりません: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(manifest_path)
    logger.info("loaded manifest: %s (%d rows)", args.split, len(df))

    X = _prepare_features(
        df, feat_cols, train_medians, percentile_refs, image_cache_path, feat_cfg_dict
    )

    # XGBoost feature importance
    save_xgb_importance(model, feat_cols, outputs_dir)

    # SHAP
    compute_shap(model, X, outputs_dir, n_samples=args.n_samples, is_classifier=is_classifier)

    print("[OK] analyze_shap 完了:", outputs_dir)


if __name__ == "__main__":
    main()
