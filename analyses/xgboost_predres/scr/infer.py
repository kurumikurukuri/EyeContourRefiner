"""学習済みモデルで train/val/test の残差予測を出力する。

出力:
  predictions_{train,val,test}.csv   行毎予測結果 (pred_residual, true_residual, error)
  analysis_feature_importance.csv    特徴量重要度ランキング
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from analyses.xgboost_baseline.scr.features import (
    add_dataset_percentile_features,
    enrich_all,
)
from analyses.xgboost_baseline.scr.train import (
    KFoldEnsemble,
    regression_metrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
def prepare_features(
    df: pd.DataFrame,
    feat_cols: List[str],
    train_medians: pd.Series,
    percentile_refs: dict,
    image_cache_path: Optional[str],
    feat_cfg_dict: Dict[str, Any],
) -> pd.DataFrame:
    from analyses.xgboost_baseline.scr.features import FeatureConfig

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
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="xgboost_predres 推論")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analyses/xgboost_predres/config/baseline.yaml"),
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

    feat_cols_path = outputs_dir / "feature_columns.json"
    medians_path = outputs_dir / "train_medians.csv"
    model_path = outputs_dir / "xgb_residual.joblib"

    for p in [feat_cols_path, medians_path, model_path]:
        if not p.exists():
            print(f"[ERROR] train 成果物がありません: {p}", file=sys.stderr)
            sys.exit(1)

    import joblib

    feat_cols: List[str] = json.loads(feat_cols_path.read_text(encoding="utf-8"))
    train_medians = pd.read_csv(medians_path, index_col=0).iloc[:, 0]
    model = joblib.load(model_path)

    percentile_refs: dict = {}
    pct_refs_path = outputs_dir / "percentile_refs.json"
    if pct_refs_path.exists():
        raw = json.loads(pct_refs_path.read_text(encoding="utf-8"))
        percentile_refs = {k: np.array(v) for k, v in raw.items()}

    id_cols = ["idx", "image_path", "face_index", "part_name", "sub_key", "point_index"]
    all_metrics: Dict[str, Any] = {}

    for split_name in ["train", "val", "test"]:
        manifest_path = manifests_dir / f"manifest_{split_name}.csv"
        if not manifest_path.exists():
            logger.warning("manifest が見つかりません: %s", manifest_path)
            continue

        df = pd.read_csv(manifest_path)
        logger.info("inferring %s (%d rows)", split_name, len(df))

        X = prepare_features(df, feat_cols, train_medians, percentile_refs, image_cache_path, feat_cfg_dict)

        if isinstance(model, KFoldEnsemble):
            pred_residual = model.predict(X)
        else:
            pred_residual = model.predict(X)

        cols = [c for c in id_cols if c in df.columns]
        out = df[cols].reset_index(drop=True).copy()
        out["pred_residual"] = pred_residual
        if "residual" in df.columns:
            y_true = df["residual"].astype(float).to_numpy()
            out["true_residual"] = y_true
            out["error"] = pred_residual - y_true
            m = regression_metrics(y_true, pred_residual)
            all_metrics[split_name] = m
            logger.info("%s metrics: %s", split_name, m)

        out_path = outputs_dir / f"predictions_{split_name}.csv"
        out.to_csv(out_path, index=False)
        logger.info("wrote %s", out_path)

    if all_metrics:
        (outputs_dir / "infer_metrics.json").write_text(
            json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    fi_path = outputs_dir / "feature_importance.json"
    if fi_path.exists():
        fi = json.loads(fi_path.read_text(encoding="utf-8"))
        fi_df = pd.DataFrame(
            [{"feature": k, "importance": v} for k, v in fi.items()]
        ).sort_values("importance", ascending=False)
        fi_df.to_csv(outputs_dir / "analysis_feature_importance.csv", index=False)
        logger.info("wrote analysis_feature_importance.csv")

    print("[OK] infer 完了:", outputs_dir)


if __name__ == "__main__":
    main()
