"""XGBoost + LightGBM + CatBoost のヘテロアンサンブル学習。

各モデルを GroupKFold (groups=image_path) で k-fold 学習し、
OOF 確率と test 確率を 3 モデル × seed 数の単純平均で統合する。

Usage:
    python -m analyses.xgboost_predskip.scr.train_ensemble \
        --config <config.yaml>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from analyses.xgboost_baseline.scr.features import (
    add_dataset_percentile_features,
    build_feature_matrix,
    enrich_all,
)
from analyses.xgboost_baseline.scr.image_features import build_image_features
from analyses.xgboost_predskip.scr.train import (
    _feature_cfg_from,
    _kfold_cfg_from,
    _load_full_cfg,
    _load_manifests,
    classification_metrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EnsembleConfig:
    seeds: Tuple[int, ...]
    use_xgb: bool
    use_lgbm: bool
    use_cat: bool
    xgb_params: Dict[str, Any]
    lgbm_params: Dict[str, Any]
    cat_params: Dict[str, Any]


# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------
def _train_xgb(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Dict[str, Any],
    seed: int,
):
    import xgboost as xgb

    p = {**params, "random_state": seed}
    p.setdefault("objective", "binary:logistic")
    p.setdefault("eval_metric", "logloss")
    p.setdefault("verbosity", 0)
    early = p.pop("early_stopping_rounds", 50)
    model = xgb.XGBClassifier(early_stopping_rounds=early, **p)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def _train_lgbm(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Dict[str, Any],
    seed: int,
):
    import lightgbm as lgb

    p = {**params, "random_state": seed, "seed": seed}
    p.setdefault("objective", "binary")
    p.setdefault("metric", "binary_logloss")
    p.setdefault("verbosity", -1)
    early = p.pop("early_stopping_rounds", 50)
    model = lgb.LGBMClassifier(**p)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(early, verbose=False)],
    )
    return model


def _train_cat(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Dict[str, Any],
    seed: int,
):
    from catboost import CatBoostClassifier

    p = {**params, "random_seed": seed}
    p.setdefault("loss_function", "Logloss")
    p.setdefault("eval_metric", "AUC")
    p.setdefault("verbose", False)
    p.setdefault("allow_writing_files", False)
    early = p.pop("early_stopping_rounds", 50)
    model = CatBoostClassifier(early_stopping_rounds=early, **p)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    return model


# ---------------------------------------------------------------------------
# OOF & test predictions per model family
# ---------------------------------------------------------------------------
def _kfold_oof_test(
    trainer,
    X_tv: pd.DataFrame,
    y_tv: np.ndarray,
    X_te: pd.DataFrame,
    groups: np.ndarray,
    n_splits: int,
    params: Dict[str, Any],
    seed: int,
    name: str,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """k-fold で OOF + test 予測を返す。test は fold 平均。"""
    kf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(X_tv))
    test_probs = np.zeros((n_splits, len(X_te)))
    fitted: list = []

    for fold_i, (tr_idx, va_idx) in enumerate(kf.split(X_tv, y_tv, groups=groups)):
        logger.info("%s seed=%d fold %d/%d ...", name, seed, fold_i + 1, n_splits)
        m = trainer(
            X_tv.iloc[tr_idx],
            y_tv[tr_idx],
            X_tv.iloc[va_idx],
            y_tv[va_idx],
            params,
            seed,
        )
        oof[va_idx] = m.predict_proba(X_tv.iloc[va_idx])[:, 1]
        test_probs[fold_i] = m.predict_proba(X_te)[:, 1]
        fitted.append(m)

    return oof, test_probs.mean(axis=0), fitted


# ---------------------------------------------------------------------------
# Threshold optimization
# ---------------------------------------------------------------------------
def _optimal_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Youden's J / F0.5 / F2 / specificity>=0.8 制約下の recall 最大の閾値を返す。"""
    from sklearn.metrics import precision_recall_curve, roc_curve

    fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    j_idx = int(np.argmax(youden))
    youden_thr = float(roc_thr[j_idx])

    prec, rec, pr_thr = precision_recall_curve(y_true, y_prob)
    eps = 1e-12

    def fbeta(p, r, beta):
        return (1 + beta * beta) * p * r / (beta * beta * p + r + eps)

    f05 = fbeta(prec[:-1], rec[:-1], 0.5)
    f2 = fbeta(prec[:-1], rec[:-1], 2.0)
    f05_idx = int(np.argmax(f05))
    f2_idx = int(np.argmax(f2))
    f05_thr = float(pr_thr[f05_idx])
    f2_thr = float(pr_thr[f2_idx])

    spec_min = 0.8
    feasible = np.where(1 - fpr >= spec_min)[0]
    if len(feasible) > 0:
        best_in_feasible = feasible[int(np.argmax(tpr[feasible]))]
        spec_thr = float(roc_thr[best_in_feasible])
    else:
        spec_thr = 0.5

    return {
        "youden_j": youden_thr,
        "f0.5": f05_thr,
        "f2": f2_thr,
        "specificity_ge_0.8": spec_thr,
    }


def _metrics_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, thr: float,
) -> Dict[str, Any]:
    pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "specificity": float(tn / (tn + fp + 1e-12)),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="XGB+LGBM+CatBoost ensemble")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
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
    kfold_cfg = _kfold_cfg_from(cfg.get("kfold", {}))
    image_cache_path = cfg.get("data", {}).get(
        "image_cache_path",
        "analyses/xgboost_predskip/outputs/image_features.parquet",
    )

    ens_cfg_raw = cfg.get("ensemble", {})
    seeds = tuple(int(s) for s in ens_cfg_raw.get("seeds", [42]))
    use_xgb = bool(ens_cfg_raw.get("use_xgb", True))
    use_lgbm = bool(ens_cfg_raw.get("use_lgbm", True))
    use_cat = bool(ens_cfg_raw.get("use_cat", True))
    xgb_params = ens_cfg_raw.get("xgb_params", {}) or {}
    lgbm_params = ens_cfg_raw.get("lgbm_params", {}) or {}
    cat_params = ens_cfg_raw.get("cat_params", {}) or {}

    # --- データ・特徴量 ---
    manifests = _load_manifests(manifests_dir)
    logger.info(
        "loaded manifests: train=%d val=%d test=%d",
        len(manifests["train"]), len(manifests["val"]), len(manifests["test"]),
    )

    if feat_cfg.use_image_features:
        mf_list = list(manifests.values())
        unlabeled_path = manifests_dir / "manifest_unlabeled.csv"
        if unlabeled_path.exists():
            mf_list.append(pd.read_csv(unlabeled_path))
        all_mf = pd.concat(mf_list, ignore_index=True)
        logger.info("pre-computing image features on %d rows ...", len(all_mf))
        build_image_features(all_mf, cache_path=image_cache_path)

    train_feat, feat_cols, percentile_refs = build_feature_matrix(
        manifests["train"], feat_cfg, image_cache_path=image_cache_path,
    )
    train_medians = train_feat.median(numeric_only=True)
    if feat_cfg.use_niqe_score and "niqe_score" in feat_cols:
        train_medians["niqe_score"] = manifests["train"]["niqe_score"].max()
    y_tr = manifests["train"]["skip_flag"].astype(int).to_numpy()

    def _prepare(name: str) -> pd.DataFrame:
        df = manifests[name]
        enriched = enrich_all(df, feat_cfg, image_cache_path=image_cache_path)
        if feat_cfg.use_dataset_percentile and percentile_refs:
            enriched, _ = add_dataset_percentile_features(enriched, percentile_refs)
        for c in feat_cols:
            if c not in enriched.columns:
                enriched[c] = np.nan
        return enriched[feat_cols].astype(float).fillna(train_medians)

    X_val = _prepare("val")
    X_te = _prepare("test")
    y_val = manifests["val"]["skip_flag"].astype(int).to_numpy()
    y_te = manifests["test"]["skip_flag"].astype(int).to_numpy()

    X_tv = pd.concat([train_feat, X_val], axis=0, ignore_index=True)
    y_tv = np.concatenate([y_tr, y_val])
    groups_tv = np.concatenate([
        manifests["train"]["image_path"].to_numpy(),
        manifests["val"]["image_path"].to_numpy(),
    ])

    # --- 各モデルを seed 別に学習 ---
    all_oof: List[np.ndarray] = []
    all_test: List[np.ndarray] = []
    component_summaries: Dict[str, Any] = {}
    fitted_models: Dict[str, Any] = {}

    families: List[Tuple[str, callable, Dict[str, Any]]] = []
    if use_xgb:
        families.append(("xgb", _train_xgb, xgb_params))
    if use_lgbm:
        families.append(("lgbm", _train_lgbm, lgbm_params))
    if use_cat:
        families.append(("cat", _train_cat, cat_params))

    for name, trainer, params in families:
        for seed in seeds:
            oof, test_p, fitted = _kfold_oof_test(
                trainer,
                X_tv,
                y_tv,
                X_te,
                groups_tv,
                kfold_cfg.n_splits,
                params,
                seed,
                name,
            )
            tag = f"{name}_seed{seed}"
            all_oof.append(oof)
            all_test.append(test_p)
            fitted_models[tag] = fitted
            component_summaries[tag] = {
                "oof_auc": float(roc_auc_score(y_tv, oof)),
                "test_auc": float(roc_auc_score(y_te, test_p)),
            }
            logger.info(
                "%s: oof_auc=%.4f test_auc=%.4f",
                tag,
                component_summaries[tag]["oof_auc"],
                component_summaries[tag]["test_auc"],
            )

    # --- 単純平均アンサンブル ---
    oof_mean = np.mean(np.stack(all_oof, axis=0), axis=0)
    test_mean = np.mean(np.stack(all_test, axis=0), axis=0)

    # --- 閾値最適化 (val 部分の OOF で計算) ---
    n_train = len(train_feat)
    val_oof = oof_mean[n_train:]
    thresholds = _optimal_thresholds(y_val, val_oof)

    metrics: Dict[str, Any] = {
        "components": component_summaries,
        "ensemble_oof_auc": float(roc_auc_score(y_tv, oof_mean)),
        "ensemble_val_oof_auc": float(roc_auc_score(y_val, val_oof)),
        "ensemble_test_at_thresholds": {
            name: _metrics_at_threshold(y_te, test_mean, thr)
            for name, thr in {**thresholds, "default_0.5": 0.5}.items()
        },
        "thresholds": thresholds,
        "feature_columns": feat_cols,
        "n_train": int(len(train_feat)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_te)),
    }
    logger.info(
        "ensemble: oof_auc=%.4f test_auc=%.4f",
        metrics["ensemble_oof_auc"],
        metrics["ensemble_test_at_thresholds"]["default_0.5"]["roc_auc"],
    )

    # --- 保存 ---
    joblib.dump(
        {"models": fitted_models, "feat_cols": feat_cols},
        outputs_dir / "ensemble.joblib",
    )
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
                "experiment": "xgboost_predskip_ensemble",
                "config_path": str(args.config),
                "feature_config": asdict(feat_cfg),
                "kfold": asdict(kfold_cfg),
                "ensemble": {
                    "seeds": list(seeds),
                    "use_xgb": use_xgb,
                    "use_lgbm": use_lgbm,
                    "use_cat": use_cat,
                    "xgb_params": xgb_params,
                    "lgbm_params": lgbm_params,
                    "cat_params": cat_params,
                },
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # OOF / test predictions for stacking
    np.save(outputs_dir / "oof_mean.npy", oof_mean)
    np.save(outputs_dir / "test_mean.npy", test_mean)
    np.save(outputs_dir / "y_tv.npy", y_tv)
    np.save(outputs_dir / "y_te.npy", y_te)

    print("[OK] ensemble 完了:", outputs_dir)


if __name__ == "__main__":
    main()
