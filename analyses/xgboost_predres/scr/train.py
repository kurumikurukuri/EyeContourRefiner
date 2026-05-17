"""XGBoost で EXP017 W0_baseline の残差 (pred_t_raw - true_t_raw) を回帰する。

オプション機能:
  - kfold.enabled=true  : train+val を k-fold で学習し KFoldEnsemble として保存
  - optuna.enabled=true : Optuna で XGBoost ハイパーパラメータを k-fold CV で探索
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from analyses.xgboost_baseline.scr.features import (
    FeatureConfig,
    add_dataset_percentile_features,
    build_feature_matrix,
    enrich_all,
)
from analyses.xgboost_baseline.scr.train import (
    KFoldConfig,
    KFoldEnsemble,
    OptunaConfig,
    QuantileConfig,
    XGBConfig,
    _kfold_cfg_from,
    _optuna_cfg_from,
    _quantile_cfg_from,
    _xgb_cfg_from,
    regression_metrics,
    run_optuna,
    train_iforest,
    train_kfold_ensemble,
    train_xgb_median,
    train_xgb_quantile,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _feature_cfg_from(d: Dict[str, Any]) -> FeatureConfig:
    return FeatureConfig(
        use_geometry=bool(d.get("use_geometry", True)),
        use_eye_scale=bool(d.get("use_eye_scale", True)),
        use_face_pose=bool(d.get("use_face_pose", True)),
        use_categorical=bool(d.get("use_categorical", True)),
        use_point_index=bool(d.get("use_point_index", True)),
        use_cyclical_angle=bool(d.get("use_cyclical_angle", False)),
        use_interactions=bool(d.get("use_interactions", False)),
        use_niqe_score=bool(d.get("use_niqe_score", False)),
        use_aggregate_features=bool(d.get("use_aggregate_features", False)),
        use_extended_aggregate=bool(d.get("use_extended_aggregate", False)),
        use_pose_composite=bool(d.get("use_pose_composite", False)),
        use_dataset_percentile=bool(d.get("use_dataset_percentile", False)),
        use_image_features=bool(d.get("use_image_features", False)),
        use_eye_brightness=bool(d.get("use_eye_brightness", False)),
        use_oi_class_features=bool(d.get("use_oi_class_features", False)),
    )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _load_manifests(manifests_dir: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for name in ["train", "val", "test"]:
        path = manifests_dir / f"manifest_{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"manifest が見つかりません: {path}")
        out[name] = pd.read_csv(path)
    return out


def _save_model(model: Any, path: Path) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def _prepare_split(
    df: pd.DataFrame,
    feat_cfg: FeatureConfig,
    feat_cols: List[str],
    train_medians: pd.Series,
    percentile_refs: dict,
    image_cache_path: Optional[str],
) -> pd.DataFrame:
    enriched = enrich_all(df, feat_cfg, image_cache_path=image_cache_path)
    if feat_cfg.use_dataset_percentile:
        enriched, _ = add_dataset_percentile_features(enriched, percentile_refs)
    for c in feat_cols:
        if c not in enriched.columns:
            enriched[c] = np.nan
    return enriched[feat_cols].astype(float).fillna(train_medians)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="xgboost_predres 学習")
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
    outputs_dir.mkdir(parents=True, exist_ok=True)
    image_cache_path: Optional[str] = cfg.get("data", {}).get("image_cache_path", None)

    feat_cfg = _feature_cfg_from(cfg.get("features", {}))
    xgb_cfg = _xgb_cfg_from(cfg["xgboost"])
    q_cfg = _quantile_cfg_from(cfg["quantile"])
    kfold_cfg = _kfold_cfg_from(cfg.get("kfold", {}))
    optuna_cfg = _optuna_cfg_from(cfg.get("optuna", {}))

    # --- データ読み込み ---
    manifests = _load_manifests(manifests_dir)
    logger.info(
        "loaded manifests: train=%d val=%d test=%d",
        len(manifests["train"]),
        len(manifests["val"]),
        len(manifests["test"]),
    )

    # --- 目的変数 ---
    y_tr = manifests["train"]["residual"].astype(float).to_numpy()
    y_val = manifests["val"]["residual"].astype(float).to_numpy()
    y_te = manifests["test"]["residual"].astype(float).to_numpy()

    # --- 特徴量行列 ---
    train_feat, feat_cols, percentile_refs = build_feature_matrix(
        manifests["train"], feat_cfg, image_cache_path=image_cache_path
    )
    train_medians = train_feat.median(numeric_only=True)
    if feat_cfg.use_niqe_score and "niqe_score" in feat_cols:
        train_medians["niqe_score"] = manifests["train"]["niqe_score"].max()

    X_val = _prepare_split(
        manifests["val"], feat_cfg, feat_cols, train_medians, percentile_refs, image_cache_path
    )
    X_te = _prepare_split(
        manifests["test"], feat_cfg, feat_cols, train_medians, percentile_refs, image_cache_path
    )

    X_tv = pd.concat([train_feat, X_val], axis=0, ignore_index=True)
    y_tv = np.concatenate([y_tr, y_val])

    # --- Optuna HPO ---
    optuna_summary: Dict[str, Any] = {}
    if optuna_cfg.enabled:
        logger.info("Optuna HPO 開始 (%d trials) ...", optuna_cfg.n_trials)
        xgb_cfg, optuna_summary = run_optuna(X_tv, y_tv, xgb_cfg, kfold_cfg, optuna_cfg)
        logger.info("Optuna 完了: best_cv_mae=%.4f", optuna_summary["best_cv_mae"])

    # --- XGBoost 学習 ---
    oof_preds: Optional[np.ndarray] = None
    if kfold_cfg.enabled:
        logger.info("k-fold 学習 (%d splits) ...", kfold_cfg.n_splits)
        m_med, m_lo, m_hi, oof_preds = train_kfold_ensemble(
            X_tv, y_tv, xgb_cfg, q_cfg, kfold_cfg
        )
    else:
        logger.info("通常学習 ...")
        m_med = train_xgb_median(train_feat, y_tr, X_val, y_val, xgb_cfg)
        m_lo = train_xgb_quantile(train_feat, y_tr, X_val, y_val, q_cfg.q_lo, xgb_cfg)
        m_hi = train_xgb_quantile(train_feat, y_tr, X_val, y_val, q_cfg.q_hi, xgb_cfg)

    # --- 評価 ---
    pred_tr = m_med.predict(train_feat)
    pred_val = m_med.predict(X_val)
    pred_te = m_med.predict(X_te)

    metrics: Dict[str, Any] = {
        "train": regression_metrics(y_tr, pred_tr),
        "val": regression_metrics(y_val, pred_val),
        "test": regression_metrics(y_te, pred_te),
        "n_train": int(len(train_feat)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_te)),
        "feature_columns": feat_cols,
    }
    if oof_preds is not None:
        metrics["oof"] = regression_metrics(y_tv, oof_preds)

    logger.info("train metrics: %s", metrics["train"])
    logger.info("val   metrics: %s", metrics["val"])
    logger.info("test  metrics: %s", metrics["test"])
    if "oof" in metrics:
        logger.info("oof   metrics: %s", metrics["oof"])

    # --- predictions CSV 出力 ---
    id_cols = ["idx", "image_path", "face_index", "part_name", "sub_key", "point_index"]

    def _pred_df(manifest: pd.DataFrame, X: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        cols = [c for c in id_cols if c in manifest.columns]
        out = manifest[cols].reset_index(drop=True).copy()
        out["true_residual"] = y_true
        out["pred_residual"] = y_pred
        out["error"] = y_pred - y_true
        return out

    pred_tr_df = _pred_df(manifests["train"], train_feat, y_tr, pred_tr)
    pred_val_df = _pred_df(manifests["val"], X_val, y_val, pred_val)
    pred_te_df = _pred_df(manifests["test"], X_te, y_te, pred_te)

    pred_tr_df.to_csv(outputs_dir / "predictions_train.csv", index=False)
    pred_val_df.to_csv(outputs_dir / "predictions_val.csv", index=False)
    pred_te_df.to_csv(outputs_dir / "predictions_test.csv", index=False)

    # --- モデル・メタデータ保存 ---
    _save_model(m_med, outputs_dir / "xgb_residual.joblib")
    _save_model(m_lo, outputs_dir / "xgb_q_lo.joblib")
    _save_model(m_hi, outputs_dir / "xgb_q_hi.joblib")
    train_medians.to_csv(outputs_dir / "train_medians.csv")

    serializable_refs = {k: v.tolist() for k, v in percentile_refs.items()}
    (outputs_dir / "percentile_refs.json").write_text(
        json.dumps(serializable_refs, ensure_ascii=False), encoding="utf-8"
    )
    (outputs_dir / "feature_columns.json").write_text(
        json.dumps(feat_cols, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 特徴量重要度
    if hasattr(m_med, "get_booster"):
        fi = m_med.get_booster().get_fscore()
        (outputs_dir / "feature_importance.json").write_text(
            json.dumps(fi, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    (outputs_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outputs_dir / "experiment.json").write_text(
        json.dumps(
            {
                "experiment": "xgboost_predres",
                "config_path": str(args.config),
                "target": "residual = pred_t_raw - true_t_raw (EXP017 W0_baseline)",
                "xgboost": asdict(xgb_cfg),
                "quantile": asdict(q_cfg),
                "kfold": asdict(kfold_cfg),
                "optuna": {**asdict(optuna_cfg), **optuna_summary},
                "feature_config": asdict(feat_cfg),
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[OK] train 完了:", outputs_dir)


if __name__ == "__main__":
    main()
