"""擬似ラベル + ノイジーラベル除去で再学習する。

ステップ:
1. ensemble (or 通常) モデルで unlabeled に予測 → 高確信 (prob>=high_thr or <=low_thr) を pseudo-label として train に追加
2. train OOF prob と true label の乖離が大きい上位 noise_drop_ratio 行を train から除外
3. ベース config と同じ特徴量設定で XGB を再学習し評価

Usage:
    python -m analyses.xgboost_predskip.scr.train_pseudo \
        --config analyses/xgboost_predskip/config/pseudo.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from analyses.xgboost_baseline.scr.features import (
    add_dataset_percentile_features,
    build_feature_matrix,
    enrich_all,
)
from analyses.xgboost_baseline.scr.image_features import build_image_features
from analyses.xgboost_predskip.scr.train import (
    KFoldEnsemble,
    classification_metrics,
    train_kfold_ensemble,
    train_xgb_classifier,
    _feature_cfg_from,
    _kfold_cfg_from,
    _load_full_cfg,
    _load_manifests,
    _xgb_cfg_from,
)

logger = logging.getLogger(__name__)


def _predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    if isinstance(model, KFoldEnsemble):
        return model.predict_proba(X)
    return model.predict_proba(X)[:, 1]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="pseudo-label + noisy label drop")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analyses/xgboost_predskip/config/pseudo.yaml"),
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[ERROR] config が見つかりません: {args.config}", file=sys.stderr)
        sys.exit(1)

    cfg = _load_full_cfg(args.config)
    manifests_dir = Path(cfg["data"]["manifests_dir"])
    outputs_dir = Path(cfg["data"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    feat_cfg = _feature_cfg_from(cfg.get("features", {}))
    xgb_cfg = _xgb_cfg_from(cfg["xgboost"])
    kfold_cfg = _kfold_cfg_from(cfg.get("kfold", {}))
    image_cache_path = cfg.get("data", {}).get(
        "image_cache_path",
        "analyses/xgboost_predskip/outputs/image_features.parquet",
    )

    pseudo_cfg = cfg.get("pseudo", {})
    high_thr = float(pseudo_cfg.get("high_thr", 0.99))
    low_thr = float(pseudo_cfg.get("low_thr", 0.01))
    noise_drop_ratio = float(pseudo_cfg.get("noise_drop_ratio", 0.05))
    use_pseudo = bool(pseudo_cfg.get("use_pseudo", True))
    use_noise_drop = bool(pseudo_cfg.get("use_noise_drop", True))

    # --- データ読み込み ---
    manifests = _load_manifests(manifests_dir)
    unlabeled_path = manifests_dir / "manifest_unlabeled.csv"
    unlabeled_df: Optional[pd.DataFrame] = None
    if unlabeled_path.exists() and use_pseudo:
        unlabeled_df = pd.read_csv(unlabeled_path)
        logger.info("loaded unlabeled: %d rows", len(unlabeled_df))

    # --- 画像特徴量 ---
    if feat_cfg.use_image_features:
        all_mfs = list(manifests.values())
        if unlabeled_df is not None:
            all_mfs.append(unlabeled_df)
        all_mf = pd.concat(all_mfs, ignore_index=True)
        logger.info("pre-computing image features on %d rows ...", len(all_mf))
        build_image_features(all_mf, cache_path=image_cache_path)

    # --- 学習用特徴量 ---
    train_feat, feat_cols, percentile_refs = build_feature_matrix(
        manifests["train"], feat_cfg, image_cache_path=image_cache_path,
    )
    train_medians = train_feat.median(numeric_only=True)
    if feat_cfg.use_niqe_score and "niqe_score" in feat_cols:
        train_medians["niqe_score"] = manifests["train"]["niqe_score"].max()
    y_tr = manifests["train"]["skip_flag"].astype(int).to_numpy()

    def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        enriched = enrich_all(df, feat_cfg, image_cache_path=image_cache_path)
        if feat_cfg.use_dataset_percentile and percentile_refs:
            enriched, _ = add_dataset_percentile_features(enriched, percentile_refs)
        for c in feat_cols:
            if c not in enriched.columns:
                enriched[c] = np.nan
        return enriched[feat_cols].astype(float).fillna(train_medians)

    X_val = _prepare_df(manifests["val"])
    X_te = _prepare_df(manifests["test"])
    y_val = manifests["val"]["skip_flag"].astype(int).to_numpy()
    y_te = manifests["test"]["skip_flag"].astype(int).to_numpy()

    # --- Stage 1: ベースモデルで pseudo-label と OOF を取得 ---
    logger.info("stage 1: base model k-fold for pseudo-label + OOF ...")
    X_tv = pd.concat([train_feat, X_val], axis=0, ignore_index=True)
    y_tv = np.concatenate([y_tr, y_val])
    groups_tv = np.concatenate([
        manifests["train"]["image_path"].to_numpy(),
        manifests["val"]["image_path"].to_numpy(),
    ])
    base_model, oof_probs = train_kfold_ensemble(
        X_tv, y_tv, xgb_cfg, kfold_cfg, groups=groups_tv,
    )
    base_metrics = classification_metrics(
        y_te,
        (_predict_proba(base_model, X_te) >= 0.5).astype(int),
        _predict_proba(base_model, X_te),
    )
    logger.info("base model test: %s", base_metrics)

    # --- Stage 2: ノイジーラベル除去 (train 部分のみ) ---
    train_oof = oof_probs[: len(train_feat)]
    if use_noise_drop and noise_drop_ratio > 0:
        # disagreement: |y_tr - oof_prob|
        disagreement = np.abs(y_tr - train_oof)
        n_drop = int(len(train_feat) * noise_drop_ratio)
        drop_idx = np.argsort(disagreement)[-n_drop:]
        keep_mask = np.ones(len(train_feat), dtype=bool)
        keep_mask[drop_idx] = False
        logger.info(
            "noise drop: %d rows (ratio=%.3f, max_disagree=%.3f)",
            n_drop, noise_drop_ratio, float(disagreement.max()),
        )
        train_feat_clean = train_feat[keep_mask].reset_index(drop=True)
        y_tr_clean = y_tr[keep_mask]
        train_mf_clean = manifests["train"][keep_mask].reset_index(drop=True)
    else:
        train_feat_clean = train_feat
        y_tr_clean = y_tr
        train_mf_clean = manifests["train"]

    # --- Stage 3: pseudo-label を unlabeled から取得 ---
    pseudo_X = None
    pseudo_y = None
    if use_pseudo and unlabeled_df is not None and len(unlabeled_df) > 0:
        X_unl = _prepare_df(unlabeled_df)
        unl_prob = _predict_proba(base_model, X_unl)
        high_mask = unl_prob >= high_thr
        low_mask = unl_prob <= low_thr
        n_high = int(high_mask.sum())
        n_low = int(low_mask.sum())
        logger.info(
            "pseudo: high(>=%.2f)=%d, low(<=%.2f)=%d / %d",
            high_thr, n_high, low_thr, n_low, len(unlabeled_df),
        )
        keep = high_mask | low_mask
        if keep.any():
            pseudo_X = X_unl[keep].reset_index(drop=True)
            pseudo_y = np.where(unl_prob[keep] >= high_thr, 1, 0).astype(int)

    # --- Stage 4: 拡張 train で再学習 ---
    if pseudo_X is not None:
        logger.info("retrain with pseudo (+%d) and clean train (%d)", len(pseudo_X), len(train_feat_clean))
        X_tr_aug = pd.concat([train_feat_clean, pseudo_X], axis=0, ignore_index=True)
        y_tr_aug = np.concatenate([y_tr_clean, pseudo_y])
    else:
        logger.info("retrain with clean train only (%d)", len(train_feat_clean))
        X_tr_aug = train_feat_clean
        y_tr_aug = y_tr_clean

    # 通常学習: val を eval set として早期停止
    final_model = train_xgb_classifier(X_tr_aug, y_tr_aug, X_val, y_val, xgb_cfg)
    val_prob = final_model.predict_proba(X_val)[:, 1]
    test_prob = final_model.predict_proba(X_te)[:, 1]
    val_metrics = classification_metrics(y_val, (val_prob >= 0.5).astype(int), val_prob)
    test_metrics = classification_metrics(y_te, (test_prob >= 0.5).astype(int), test_prob)
    logger.info("final val: %s", val_metrics)
    logger.info("final test: %s", test_metrics)

    # --- 保存 ---
    metrics = {
        "base_test": base_metrics,
        "final_val": val_metrics,
        "final_test": test_metrics,
        "n_train_clean": int(len(train_feat_clean)),
        "n_pseudo": int(len(pseudo_X)) if pseudo_X is not None else 0,
        "feature_columns": feat_cols,
    }
    joblib.dump(final_model, outputs_dir / "xgb_classifier.joblib")
    train_medians.to_csv(outputs_dir / "train_medians.csv")
    if percentile_refs:
        (outputs_dir / "percentile_refs.json").write_text(
            json.dumps(
                {k: v.tolist() for k, v in percentile_refs.items()},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    (outputs_dir / "feature_columns.json").write_text(
        json.dumps(feat_cols, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outputs_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outputs_dir / "experiment.json").write_text(
        json.dumps(
            {
                "experiment": "xgboost_predskip_pseudo",
                "config_path": str(args.config),
                "xgboost": asdict(xgb_cfg),
                "feature_config": asdict(feat_cfg),
                "kfold": asdict(kfold_cfg),
                "pseudo": {
                    "high_thr": high_thr,
                    "low_thr": low_thr,
                    "noise_drop_ratio": noise_drop_ratio,
                    "use_pseudo": use_pseudo,
                    "use_noise_drop": use_noise_drop,
                },
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[OK] pseudo 完了:", outputs_dir)


if __name__ == "__main__":
    main()
