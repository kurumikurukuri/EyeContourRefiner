"""XGBoost で true_t_raw を回帰 + Quantile 回帰 + IsolationForest で外れ値スコアを学習。

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


@dataclass(frozen=True)
class QuantileConfig:
    q_lo: float
    q_hi: float


@dataclass(frozen=True)
class IForestConfig:
    n_estimators: int
    contamination: Any
    max_samples: Any
    random_state: int


@dataclass(frozen=True)
class TargetConfig:
    t_resolution: Optional[float]


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
    timeout: Optional[int]  # seconds; None = no limit


# ---------------------------------------------------------------------------
# KFoldEnsemble — 複数 fold モデルの predict() を平均する薄いラッパー
# ---------------------------------------------------------------------------
class KFoldEnsemble:
    """k-fold で学習したモデルのアンサンブル。joblib で保存・読み込み可能。"""

    def __init__(self, models: list) -> None:
        self.models = models

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.stack([m.predict(X) for m in self.models], axis=0)
        return preds.mean(axis=0)


# ---------------------------------------------------------------------------
# Core training functions
# ---------------------------------------------------------------------------
def train_xgb_median(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cfg: XGBConfig,
):
    """squared-error 回帰モデル（median 相当として扱う）。"""
    import xgboost as xgb

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        min_child_weight=cfg.min_child_weight,
        random_state=cfg.random_state,
        verbosity=cfg.verbosity,
        early_stopping_rounds=cfg.early_stopping_rounds,
        eval_metric="rmse",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_xgb_quantile(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    alpha: float,
    cfg: XGBConfig,
):
    """quantile loss の XGBoost（XGBoost >= 1.7 を要求）。"""
    import xgboost as xgb

    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        min_child_weight=cfg.min_child_weight,
        random_state=cfg.random_state,
        verbosity=cfg.verbosity,
        early_stopping_rounds=cfg.early_stopping_rounds,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_iforest(X: pd.DataFrame, cfg: IForestConfig):
    """特徴空間の異常度を学習（unlabeled を含む全データで fit する想定）。"""
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(
        n_estimators=cfg.n_estimators,
        contamination=cfg.contamination,
        max_samples=cfg.max_samples,
        random_state=cfg.random_state,
    )
    model.fit(X)
    return model


# ---------------------------------------------------------------------------
# K-fold ensemble training
# ---------------------------------------------------------------------------
def train_kfold_ensemble(
    X_tv: pd.DataFrame,
    y_tv: np.ndarray,
    xgb_cfg: XGBConfig,
    q_cfg: QuantileConfig,
    kfold_cfg: KFoldConfig,
) -> tuple[KFoldEnsemble, KFoldEnsemble, KFoldEnsemble, np.ndarray]:
    """train+val を k-fold で学習し median/q_lo/q_hi の KFoldEnsemble を返す。

    Returns:
        (ensemble_median, ensemble_q_lo, ensemble_q_hi, oof_predictions_median)
    """
    from sklearn.model_selection import KFold

    kf = KFold(
        n_splits=kfold_cfg.n_splits,
        shuffle=kfold_cfg.shuffle,
        random_state=kfold_cfg.seed,
    )
    med_models, lo_models, hi_models = [], [], []
    oof_preds = np.zeros(len(X_tv))

    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_tv)):
        logger.info("k-fold: fold %d/%d ...", fold_i + 1, kfold_cfg.n_splits)
        X_tr_f = X_tv.iloc[tr_idx]
        X_val_f = X_tv.iloc[val_idx]
        y_tr_f = y_tv[tr_idx]
        y_val_f = y_tv[val_idx]

        m_med = train_xgb_median(X_tr_f, y_tr_f, X_val_f, y_val_f, xgb_cfg)
        m_lo = train_xgb_quantile(X_tr_f, y_tr_f, X_val_f, y_val_f, q_cfg.q_lo, xgb_cfg)
        m_hi = train_xgb_quantile(X_tr_f, y_tr_f, X_val_f, y_val_f, q_cfg.q_hi, xgb_cfg)

        oof_preds[val_idx] = m_med.predict(X_val_f)
        med_models.append(m_med)
        lo_models.append(m_lo)
        hi_models.append(m_hi)

    oof_mae = float(np.mean(np.abs(oof_preds - y_tv)))
    logger.info("k-fold OOF MAE: %.4f", oof_mae)

    return (
        KFoldEnsemble(med_models),
        KFoldEnsemble(lo_models),
        KFoldEnsemble(hi_models),
        oof_preds,
    )


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
    """1 trial の k-fold CV MAE を返す Optuna 目的関数。"""
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
    )
    kf = KFold(
        n_splits=kfold_cfg.n_splits,
        shuffle=kfold_cfg.shuffle,
        random_state=kfold_cfg.seed,
    )
    maes: List[float] = []
    for tr_idx, val_idx in kf.split(X_tv):
        X_tr_f = X_tv.iloc[tr_idx]
        X_val_f = X_tv.iloc[val_idx]
        y_tr_f = y_tv[tr_idx]
        y_val_f = y_tv[val_idx]
        model = train_xgb_median(X_tr_f, y_tr_f, X_val_f, y_val_f, trial_cfg)
        pred = model.predict(X_val_f)
        maes.append(float(np.mean(np.abs(pred - y_val_f))))
    return float(np.mean(maes))


def run_optuna(
    X_tv: pd.DataFrame,
    y_tv: np.ndarray,
    base_cfg: XGBConfig,
    kfold_cfg: KFoldConfig,
    optuna_cfg: OptunaConfig,
) -> tuple[XGBConfig, Dict[str, Any]]:
    """Optuna で XGBoost ハイパーパラメータを探索し (best_cfg, study_summary) を返す。"""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=base_cfg.random_state),
    )
    study.optimize(
        lambda trial: _optuna_objective(trial, X_tv, y_tv, base_cfg, kfold_cfg),
        n_trials=optuna_cfg.n_trials,
        timeout=optuna_cfg.timeout,
    )

    best = study.best_params
    logger.info("Optuna best params: %s", best)
    logger.info("Optuna best CV MAE: %.4f", study.best_value)

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
    )
    study_summary = {
        "best_params": best,
        "best_cv_mae": float(study.best_value),
        "n_trials_completed": len(study.trials),
    }
    return best_cfg, study_summary


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    y_mean = float(np.mean(y_true))
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


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
    )


def _quantile_cfg_from(d: Dict[str, Any]) -> QuantileConfig:
    return QuantileConfig(q_lo=float(d["q_lo"]), q_hi=float(d["q_hi"]))


def _target_cfg_from(d: Dict[str, Any]) -> TargetConfig:
    raw = d.get("t_resolution", None)
    step = None if raw is None else float(raw)
    if step is not None and step <= 0:
        raise ValueError(f"target.t_resolution は正の値を指定してください: {raw}")
    return TargetConfig(t_resolution=step)


def _iforest_cfg_from(d: Dict[str, Any]) -> IForestConfig:
    return IForestConfig(
        n_estimators=int(d["n_estimators"]),
        contamination=d.get("contamination", "auto"),
        max_samples=d.get("max_samples", "auto"),
        random_state=int(d.get("random_state", 42)),
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


def apply_t_resolution(y: np.ndarray, step: Optional[float]) -> np.ndarray:
    """step が None なら生値、指定があれば最近傍の倍数に丸める。"""
    if step is None:
        return y
    return np.round(y / step) * step


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _load_manifests(manifests_dir: Path) -> Dict[str, pd.DataFrame]:
    names = ["train", "val", "test", "unlabeled"]
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
    parser = argparse.ArgumentParser(description="xgboost_baseline 学習")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analyses/xgboost_baseline/config/baseline.yaml"),
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
    q_cfg = _quantile_cfg_from(cfg["quantile"])
    if_cfg = _iforest_cfg_from(cfg["iforest"])
    target_cfg = _target_cfg_from(cfg.get("target", {}))
    kfold_cfg = _kfold_cfg_from(cfg.get("kfold", {}))
    optuna_cfg = _optuna_cfg_from(cfg.get("optuna", {}))

    # --- データ読み込み ---
    manifests = _load_manifests(manifests_dir)
    logger.info(
        "loaded manifests: train=%d val=%d test=%d unlabeled=%d",
        len(manifests["train"]),
        len(manifests["val"]),
        len(manifests["test"]),
        len(manifests["unlabeled"]),
    )

    # --- train 特徴量（median 統計量を固定）---
    train_feat, feat_cols = build_feature_matrix(manifests["train"], feat_cfg)
    train_medians = train_feat.median(numeric_only=True)
    y_tr_raw = manifests["train"]["true_t_raw"].astype(float).to_numpy()
    y_tr = apply_t_resolution(y_tr_raw, target_cfg.t_resolution)

    # val / test / unlabeled: 同じカラム集合 + train の median で埋める
    def _prepare(name: str) -> pd.DataFrame:
        df = manifests[name]
        enriched = enrich_all(df, feat_cfg)
        for c in feat_cols:
            if c not in enriched.columns:
                enriched[c] = np.nan
        return enriched[feat_cols].astype(float).fillna(train_medians)

    X_val = _prepare("val")
    X_te = _prepare("test")
    X_unl = _prepare("unlabeled")
    y_val_raw = manifests["val"]["true_t_raw"].astype(float).to_numpy()
    y_te_raw = manifests["test"]["true_t_raw"].astype(float).to_numpy()
    y_val = apply_t_resolution(y_val_raw, target_cfg.t_resolution)
    y_te = apply_t_resolution(y_te_raw, target_cfg.t_resolution)

    # --- Optuna / k-fold 用に train+val をマージ ---
    X_tv = pd.concat([train_feat, X_val], axis=0, ignore_index=True)
    y_tv = np.concatenate([y_tr, y_val])

    # --- Optuna HPO ---
    optuna_summary: Dict[str, Any] = {}
    if optuna_cfg.enabled:
        logger.info("Optuna HPO 開始 (%d trials) ...", optuna_cfg.n_trials)
        xgb_cfg, optuna_summary = run_optuna(X_tv, y_tv, xgb_cfg, kfold_cfg, optuna_cfg)
        logger.info("Optuna 完了: best_cv_mae=%.4f", optuna_summary["best_cv_mae"])

    # --- XGBoost 学習（k-fold or 通常）---
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

    # --- IsolationForest ---
    logger.info("training IsolationForest on all features ...")
    X_all_for_if = pd.concat(
        [X_tv if kfold_cfg.enabled else train_feat, X_te, X_unl],
        axis=0,
        ignore_index=True,
    )
    m_if = train_iforest(X_all_for_if, if_cfg)

    # --- 評価 ---
    pred_val = m_med.predict(X_val)
    pred_te = m_med.predict(X_te)
    metrics: Dict[str, Any] = {
        "val": regression_metrics(y_val, pred_val),
        "test": regression_metrics(y_te, pred_te),
        "n_train": int(len(train_feat)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_te)),
        "n_unlabeled": int(len(X_unl)),
        "feature_columns": feat_cols,
    }
    if oof_preds is not None:
        metrics["oof"] = regression_metrics(y_tv, oof_preds)
    logger.info("val metrics: %s", metrics["val"])
    logger.info("test metrics: %s", metrics["test"])
    if "oof" in metrics:
        logger.info("oof metrics: %s", metrics["oof"])

    # --- 保存 ---
    _save_model(m_med, outputs_dir / "xgb_median.joblib")
    _save_model(m_lo, outputs_dir / "xgb_q_lo.joblib")
    _save_model(m_hi, outputs_dir / "xgb_q_hi.joblib")
    _save_model(m_if, outputs_dir / "iforest.joblib")
    train_medians.to_csv(outputs_dir / "train_medians.csv")
    (outputs_dir / "feature_columns.json").write_text(
        json.dumps(feat_cols, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outputs_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outputs_dir / "experiment.json").write_text(
        json.dumps(
            {
                "experiment": "xgboost_baseline",
                "config_path": str(args.config),
                "xgboost": asdict(xgb_cfg),
                "quantile": asdict(q_cfg),
                "target": asdict(target_cfg),
                "iforest": {
                    "n_estimators": if_cfg.n_estimators,
                    "contamination": if_cfg.contamination,
                    "max_samples": if_cfg.max_samples,
                    "random_state": if_cfg.random_state,
                },
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
