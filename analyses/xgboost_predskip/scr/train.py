"""XGBoost で skip_flag を二値分類する。

オプション機能:
  - kfold.enabled=true  : train+val を k-fold で学習し KFoldEnsemble として保存
  - optuna.enabled=true : Optuna で XGBoost ハイパーパラメータを k-fold CV で探索 (AUC 最大化)
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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from analyses.xgboost_baseline.scr.features import (
    FeatureConfig,
    build_feature_matrix,
    enrich_all,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class XGBConfig:
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    min_child_weight: int
    early_stopping_rounds: int
    random_state: int
    verbosity: int
    scale_pos_weight: float


@dataclass(frozen=True)
class KFoldConfig:
    enabled: bool
    n_splits: int
    shuffle: bool
    seed: int


@dataclass(frozen=True)
class OptunaConfig:
    enabled: bool
    n_trials: int
    timeout: Optional[int]


# ---------------------------------------------------------------------------
# KFoldEnsemble
# ---------------------------------------------------------------------------
class KFoldEnsemble:
    """k-fold で学習した XGBClassifier のアンサンブル。"""

    def __init__(self, models: list) -> None:
        self.models = models

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.stack([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        return probs.mean(axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Core training functions
# ---------------------------------------------------------------------------
def train_xgb_classifier(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cfg: XGBConfig,
):
    """binary:logistic の XGBClassifier を学習する。"""
    import xgboost as xgb

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        min_child_weight=cfg.min_child_weight,
        scale_pos_weight=cfg.scale_pos_weight,
        random_state=cfg.random_state,
        verbosity=cfg.verbosity,
        early_stopping_rounds=cfg.early_stopping_rounds,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------
def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# ---------------------------------------------------------------------------
# K-fold ensemble training
# ---------------------------------------------------------------------------
def train_kfold_ensemble(
    X_tv: pd.DataFrame,
    y_tv: np.ndarray,
    xgb_cfg: XGBConfig,
    kfold_cfg: KFoldConfig,
) -> tuple[KFoldEnsemble, np.ndarray]:
    """train+val を k-fold で学習し KFoldEnsemble と OOF 確率を返す。"""
    from sklearn.model_selection import KFold

    kf = KFold(
        n_splits=kfold_cfg.n_splits,
        shuffle=kfold_cfg.shuffle,
        random_state=kfold_cfg.seed,
    )
    models: list = []
    oof_probs = np.zeros(len(X_tv))

    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_tv)):
        logger.info("k-fold: fold %d/%d ...", fold_i + 1, kfold_cfg.n_splits)
        m = train_xgb_classifier(
            X_tv.iloc[tr_idx],
            y_tv[tr_idx],
            X_tv.iloc[val_idx],
            y_tv[val_idx],
            xgb_cfg,
        )
        oof_probs[val_idx] = m.predict_proba(X_tv.iloc[val_idx])[:, 1]
        models.append(m)

    oof_labels = (oof_probs >= 0.5).astype(int)
    oof_metrics = classification_metrics(y_tv, oof_labels, oof_probs)
    logger.info("k-fold OOF metrics: %s", oof_metrics)

    return KFoldEnsemble(models), oof_probs


# ---------------------------------------------------------------------------
# Optuna HPO
# ---------------------------------------------------------------------------
def _optuna_objective(
    trial: Any,
    X_tv: pd.DataFrame,
    y_tv: np.ndarray,
    base_cfg: XGBConfig,
    kfold_cfg: KFoldConfig,
) -> float:
    """1 trial の k-fold CV AUC を返す Optuna 目的関数。"""
    from sklearn.model_selection import KFold

    trial_cfg = XGBConfig(
        n_estimators=base_cfg.n_estimators,
        max_depth=trial.suggest_int("max_depth", 3, 9),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
        early_stopping_rounds=base_cfg.early_stopping_rounds,
        random_state=base_cfg.random_state,
        verbosity=0,
        scale_pos_weight=trial.suggest_float("scale_pos_weight", 0.5, 1.5),
    )
    kf = KFold(
        n_splits=kfold_cfg.n_splits,
        shuffle=kfold_cfg.shuffle,
        random_state=kfold_cfg.seed,
    )
    aucs: List[float] = []
    for tr_idx, val_idx in kf.split(X_tv):
        m = train_xgb_classifier(
            X_tv.iloc[tr_idx],
            y_tv[tr_idx],
            X_tv.iloc[val_idx],
            y_tv[val_idx],
            trial_cfg,
        )
        prob = m.predict_proba(X_tv.iloc[val_idx])[:, 1]
        aucs.append(float(roc_auc_score(y_tv[val_idx], prob)))
    return float(np.mean(aucs))


def run_optuna(
    X_tv: pd.DataFrame,
    y_tv: np.ndarray,
    base_cfg: XGBConfig,
    kfold_cfg: KFoldConfig,
    optuna_cfg: OptunaConfig,
) -> tuple[XGBConfig, Dict[str, Any]]:
    """Optuna で XGBoost ハイパーパラメータを AUC 最大化で探索する。"""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=base_cfg.random_state),
    )
    study.optimize(
        lambda trial: _optuna_objective(trial, X_tv, y_tv, base_cfg, kfold_cfg),
        n_trials=optuna_cfg.n_trials,
        timeout=optuna_cfg.timeout,
    )

    best = study.best_params
    logger.info("Optuna best params: %s", best)
    logger.info("Optuna best CV AUC: %.4f", study.best_value)

    best_cfg = XGBConfig(
        n_estimators=base_cfg.n_estimators,
        max_depth=int(best["max_depth"]),
        learning_rate=float(best["learning_rate"]),
        subsample=float(best["subsample"]),
        colsample_bytree=float(best["colsample_bytree"]),
        reg_alpha=float(best["reg_alpha"]),
        reg_lambda=float(best["reg_lambda"]),
        min_child_weight=int(best["min_child_weight"]),
        early_stopping_rounds=base_cfg.early_stopping_rounds,
        random_state=base_cfg.random_state,
        verbosity=base_cfg.verbosity,
        scale_pos_weight=float(best["scale_pos_weight"]),
    )
    study_summary = {
        "best_params": best,
        "best_cv_auc": float(study.best_value),
        "n_trials_completed": len(study.trials),
    }
    return best_cfg, study_summary


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def _load_full_cfg(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _feature_cfg_from(d: Dict[str, Any]) -> FeatureConfig:
    return FeatureConfig(
        use_geometry=bool(d.get("use_geometry", True)),
        use_eye_scale=bool(d.get("use_eye_scale", True)),
        use_face_pose=bool(d.get("use_face_pose", True)),
        use_categorical=bool(d.get("use_categorical", True)),
        use_point_index=bool(d.get("use_point_index", True)),
        use_cyclical_angle=bool(d.get("use_cyclical_angle", False)),
        use_interactions=bool(d.get("use_interactions", False)),
    )


def _xgb_cfg_from(d: Dict[str, Any]) -> XGBConfig:
    return XGBConfig(
        n_estimators=int(d["n_estimators"]),
        max_depth=int(d["max_depth"]),
        learning_rate=float(d["learning_rate"]),
        subsample=float(d["subsample"]),
        colsample_bytree=float(d["colsample_bytree"]),
        reg_alpha=float(d.get("reg_alpha", 0.0)),
        reg_lambda=float(d.get("reg_lambda", 1.0)),
        min_child_weight=int(d.get("min_child_weight", 1)),
        early_stopping_rounds=int(d.get("early_stopping_rounds", 50)),
        random_state=int(d.get("random_state", 42)),
        verbosity=int(d.get("verbosity", 0)),
        scale_pos_weight=float(d.get("scale_pos_weight", 1.0)),
    )


def _kfold_cfg_from(d: Dict[str, Any]) -> KFoldConfig:
    return KFoldConfig(
        enabled=bool(d.get("enabled", False)),
        n_splits=int(d.get("n_splits", 5)),
        shuffle=bool(d.get("shuffle", True)),
        seed=int(d.get("seed", 42)),
    )


def _optuna_cfg_from(d: Dict[str, Any]) -> OptunaConfig:
    timeout_raw = d.get("timeout", None)
    return OptunaConfig(
        enabled=bool(d.get("enabled", False)),
        n_trials=int(d.get("n_trials", 50)),
        timeout=None if timeout_raw is None else int(timeout_raw),
    )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _load_manifests(manifests_dir: Path) -> Dict[str, pd.DataFrame]:
    names = ["train", "val", "test"]
    out: Dict[str, pd.DataFrame] = {}
    for n in names:
        path = manifests_dir / f"manifest_{n}.csv"
        if not path.exists():
            raise FileNotFoundError(f"manifest が見つかりません: {path}")
        out[n] = pd.read_csv(path)
    return out


def _save_model(model: Any, path: Path) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="xgboost_predskip 学習")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analyses/xgboost_predskip/config/baseline.yaml"),
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
    optuna_cfg = _optuna_cfg_from(cfg.get("optuna", {}))

    # --- データ読み込み ---
    manifests = _load_manifests(manifests_dir)
    logger.info(
        "loaded manifests: train=%d val=%d test=%d",
        len(manifests["train"]),
        len(manifests["val"]),
        len(manifests["test"]),
    )

    # --- train 特徴量（median 統計量を固定）---
    train_feat, feat_cols = build_feature_matrix(manifests["train"], feat_cfg)
    train_medians = train_feat.median(numeric_only=True)
    y_tr = manifests["train"]["skip_flag"].astype(int).to_numpy()

    def _prepare(name: str) -> pd.DataFrame:
        df = manifests[name]
        enriched = enrich_all(df, feat_cfg)
        for c in feat_cols:
            if c not in enriched.columns:
                enriched[c] = np.nan
        return enriched[feat_cols].astype(float).fillna(train_medians)

    X_val = _prepare("val")
    X_te = _prepare("test")
    y_val = manifests["val"]["skip_flag"].astype(int).to_numpy()
    y_te = manifests["test"]["skip_flag"].astype(int).to_numpy()

    # --- Optuna / k-fold 用に train+val をマージ ---
    X_tv = pd.concat([train_feat, X_val], axis=0, ignore_index=True)
    y_tv = np.concatenate([y_tr, y_val])

    # --- Optuna HPO ---
    optuna_summary: Dict[str, Any] = {}
    if optuna_cfg.enabled:
        logger.info("Optuna HPO 開始 (%d trials) ...", optuna_cfg.n_trials)
        xgb_cfg, optuna_summary = run_optuna(X_tv, y_tv, xgb_cfg, kfold_cfg, optuna_cfg)
        logger.info("Optuna 完了: best_cv_auc=%.4f", optuna_summary["best_cv_auc"])

    # --- XGBoost 学習（k-fold or 通常）---
    oof_probs: Optional[np.ndarray] = None
    if kfold_cfg.enabled:
        logger.info("k-fold 学習 (%d splits) ...", kfold_cfg.n_splits)
        model, oof_probs = train_kfold_ensemble(X_tv, y_tv, xgb_cfg, kfold_cfg)
    else:
        logger.info("通常学習 ...")
        model = train_xgb_classifier(train_feat, y_tr, X_val, y_val, xgb_cfg)

    # --- 評価 ---
    def _predict_metrics(X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        if isinstance(model, KFoldEnsemble):
            prob = model.predict_proba(X)
        else:
            prob = model.predict_proba(X)[:, 1]
        pred = (prob >= 0.5).astype(int)
        return classification_metrics(y, pred, prob)

    val_metrics = _predict_metrics(X_val, y_val)
    test_metrics = _predict_metrics(X_te, y_te)

    metrics: Dict[str, Any] = {
        "val": val_metrics,
        "test": test_metrics,
        "n_train": int(len(train_feat)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_te)),
        "feature_columns": feat_cols,
    }
    if oof_probs is not None:
        oof_labels = (oof_probs >= 0.5).astype(int)
        metrics["oof"] = classification_metrics(y_tv, oof_labels, oof_probs)
    logger.info("val metrics: %s", val_metrics)
    logger.info("test metrics: %s", test_metrics)
    if "oof" in metrics:
        logger.info("oof metrics: %s", metrics["oof"])

    # --- 特徴量重要度 ---
    if isinstance(model, KFoldEnsemble):
        importances = np.mean(
            [m.feature_importances_ for m in model.models], axis=0
        )
    else:
        importances = model.feature_importances_
    feat_importance = {
        col: float(imp) for col, imp in sorted(
            zip(feat_cols, importances),
            key=lambda x: x[1],
            reverse=True,
        )
    }

    # --- 保存 ---
    _save_model(model, outputs_dir / "xgb_classifier.joblib")
    train_medians.to_csv(outputs_dir / "train_medians.csv")
    (outputs_dir / "feature_columns.json").write_text(
        json.dumps(feat_cols, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outputs_dir / "feature_importance.json").write_text(
        json.dumps(feat_importance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outputs_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outputs_dir / "experiment.json").write_text(
        json.dumps(
            {
                "experiment": "xgboost_predskip",
                "config_path": str(args.config),
                "xgboost": asdict(xgb_cfg),
                "feature_config": asdict(feat_cfg),
                "kfold": asdict(kfold_cfg),
                "optuna": {**asdict(optuna_cfg), **optuna_summary},
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
