"""残差を 3 クラス (under/ok/over) に bin 化して XGBoost で多値分類する。

bin ラベル:
  0 = under : residual < -threshold
  1 = ok    : |residual| <= threshold
  2 = over  : residual >  threshold

オプション機能:
  - kfold.enabled=true  : StratifiedKFold で k-fold アンサンブル
  - optuna.enabled=true : Optuna で macro-F1 を最大化
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
    classification_report,
    confusion_matrix,
    f1_score,
)

from analyses.xgboost_baseline.scr.features import (
    FeatureConfig,
    add_dataset_percentile_features,
    build_feature_matrix,
    enrich_all,
)
from analyses.xgboost_baseline.scr.train import (
    KFoldConfig,
    OptunaConfig,
    XGBConfig,
    _kfold_cfg_from,
    _optuna_cfg_from,
    _xgb_cfg_from,
)

logger = logging.getLogger(__name__)

N_CLASSES = 3
LABELS = ["under", "ok", "over"]


# ---------------------------------------------------------------------------
# KFoldEnsemble (分類版)
# ---------------------------------------------------------------------------
class KFoldEnsemble:
    def __init__(self, models: list) -> None:
        self.models = models

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.stack([m.predict_proba(X) for m in self.models], axis=0)
        return probs.mean(axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


# ---------------------------------------------------------------------------
# Bin utilities
# ---------------------------------------------------------------------------
def make_bin_labels(residual: np.ndarray, threshold: float) -> np.ndarray:
    labels = np.ones(len(residual), dtype=int)  # デフォルト: ok (1)
    labels[residual < -threshold] = 0           # under
    labels[residual > threshold] = 2            # over
    return labels


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """各サンプルに逆頻度重みを付ける。"""
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    weights_per_class = len(y) / (N_CLASSES * counts)
    return weights_per_class[y]


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
# Training
# ---------------------------------------------------------------------------
def train_xgb_cls(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cfg: XGBConfig,
    use_class_weight: bool,
) -> Any:
    import xgboost as xgb

    sample_weight = compute_class_weights(y_tr) if use_class_weight else None
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=N_CLASSES,
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
        eval_metric="mlogloss",
    )
    model.fit(
        X_tr, y_tr,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def train_kfold_ensemble(
    X_tv: pd.DataFrame,
    y_tv: np.ndarray,
    xgb_cfg: XGBConfig,
    kfold_cfg: KFoldConfig,
    use_class_weight: bool,
) -> tuple[KFoldEnsemble, np.ndarray]:
    from sklearn.model_selection import StratifiedKFold

    kf = StratifiedKFold(
        n_splits=kfold_cfg.n_splits,
        shuffle=kfold_cfg.shuffle,
        random_state=kfold_cfg.seed,
    )
    models = []
    oof_preds = np.zeros(len(X_tv), dtype=int)

    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_tv, y_tv)):
        logger.info("k-fold: fold %d/%d ...", fold_i + 1, kfold_cfg.n_splits)
        m = train_xgb_cls(
            X_tv.iloc[tr_idx], y_tv[tr_idx],
            X_tv.iloc[val_idx], y_tv[val_idx],
            xgb_cfg, use_class_weight,
        )
        oof_preds[val_idx] = m.predict(X_tv.iloc[val_idx])
        models.append(m)

    oof_f1 = float(f1_score(y_tv, oof_preds, average="macro"))
    logger.info("k-fold OOF macro-F1: %.4f", oof_f1)
    return KFoldEnsemble(models), oof_preds


def _optuna_objective(
    trial: Any,
    X_tv: pd.DataFrame,
    y_tv: np.ndarray,
    base_cfg: XGBConfig,
    kfold_cfg: KFoldConfig,
    use_class_weight: bool,
) -> float:
    from sklearn.model_selection import StratifiedKFold

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
    kf = StratifiedKFold(
        n_splits=kfold_cfg.n_splits,
        shuffle=kfold_cfg.shuffle,
        random_state=kfold_cfg.seed,
    )
    f1s: List[float] = []
    for tr_idx, val_idx in kf.split(X_tv, y_tv):
        m = train_xgb_cls(
            X_tv.iloc[tr_idx], y_tv[tr_idx],
            X_tv.iloc[val_idx], y_tv[val_idx],
            trial_cfg, use_class_weight,
        )
        pred = m.predict(X_tv.iloc[val_idx])
        f1s.append(float(f1_score(y_tv[val_idx], pred, average="macro")))
    return -float(np.mean(f1s))  # minimize → maximize macro-F1


def run_optuna(
    X_tv: pd.DataFrame,
    y_tv: np.ndarray,
    base_cfg: XGBConfig,
    kfold_cfg: KFoldConfig,
    optuna_cfg: OptunaConfig,
    use_class_weight: bool,
) -> tuple[XGBConfig, Dict[str, Any]]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=base_cfg.random_state),
    )
    study.optimize(
        lambda t: _optuna_objective(t, X_tv, y_tv, base_cfg, kfold_cfg, use_class_weight),
        n_trials=optuna_cfg.n_trials,
        timeout=optuna_cfg.timeout,
    )
    best = study.best_params
    logger.info("Optuna best params: %s", best)
    logger.info("Optuna best macro-F1: %.4f", -study.best_value)
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
    return best_cfg, {"best_params": best, "best_macro_f1": -study.best_value}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def cls_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    report = classification_report(
        y_true, y_pred,
        labels=[0, 1, 2],
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label in LABELS
        },
    }


def _log_confusion_matrix(name: str, cm: list) -> None:
    header = f"{'':>10}" + "".join(f"{l:>8}" for l in LABELS)
    logger.info("[%s] confusion matrix:", name)
    logger.info(header)
    for i, label in enumerate(LABELS):
        row = f"{label:>10}" + "".join(f"{cm[i][j]:>8}" for j in range(N_CLASSES))
        logger.info(row)


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
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="xgboost_predres 分類学習（bin_01）")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analyses/xgboost_predres/config/bin_01.yaml"),
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

    threshold = float(cfg["bin"]["threshold"])
    feat_cfg = _feature_cfg_from(cfg.get("features", {}))
    xgb_cfg = _xgb_cfg_from(cfg["xgboost"])
    kfold_cfg = _kfold_cfg_from(cfg.get("kfold", {}))
    optuna_cfg = _optuna_cfg_from(cfg.get("optuna", {}))
    use_class_weight = bool(cfg["xgboost"].get("use_class_weight", True))

    # --- データ読み込み ---
    manifests = _load_manifests(manifests_dir)
    logger.info(
        "loaded manifests: train=%d val=%d test=%d",
        len(manifests["train"]), len(manifests["val"]), len(manifests["test"]),
    )

    # --- bin ラベル生成 ---
    y_tr = make_bin_labels(manifests["train"]["residual"].astype(float).to_numpy(), threshold)
    y_val = make_bin_labels(manifests["val"]["residual"].astype(float).to_numpy(), threshold)
    y_te = make_bin_labels(manifests["test"]["residual"].astype(float).to_numpy(), threshold)

    for split_name, y in [("train", y_tr), ("val", y_val), ("test", y_te)]:
        counts = np.bincount(y, minlength=N_CLASSES)
        logger.info(
            "[%s] under=%d ok=%d over=%d",
            split_name, counts[0], counts[1], counts[2],
        )

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
        xgb_cfg, optuna_summary = run_optuna(
            X_tv, y_tv, xgb_cfg, kfold_cfg, optuna_cfg, use_class_weight
        )

    # --- 学習 ---
    oof_preds: Optional[np.ndarray] = None
    if kfold_cfg.enabled:
        logger.info("k-fold 学習 (%d splits) ...", kfold_cfg.n_splits)
        model, oof_preds = train_kfold_ensemble(
            X_tv, y_tv, xgb_cfg, kfold_cfg, use_class_weight
        )
    else:
        logger.info("通常学習 ...")
        model = train_xgb_cls(train_feat, y_tr, X_val, y_val, xgb_cfg, use_class_weight)

    # --- 評価 ---
    pred_tr = model.predict(train_feat) if not kfold_cfg.enabled else model.predict(train_feat)
    pred_val = model.predict(X_val)
    pred_te = model.predict(X_te)

    m_tr = cls_metrics(y_tr, pred_tr)
    m_val = cls_metrics(y_val, pred_val)
    m_te = cls_metrics(y_te, pred_te)

    logger.info("train  acc=%.3f macro-F1=%.3f", m_tr["accuracy"], m_tr["macro_f1"])
    logger.info("val    acc=%.3f macro-F1=%.3f", m_val["accuracy"], m_val["macro_f1"])
    logger.info("test   acc=%.3f macro-F1=%.3f", m_te["accuracy"], m_te["macro_f1"])
    _log_confusion_matrix("val", m_val["confusion_matrix"])
    _log_confusion_matrix("test", m_te["confusion_matrix"])

    if oof_preds is not None:
        m_oof = cls_metrics(y_tv, oof_preds)
        logger.info("oof    acc=%.3f macro-F1=%.3f", m_oof["accuracy"], m_oof["macro_f1"])

    # --- predictions CSV 出力 ---
    id_cols = ["idx", "image_path", "face_index", "part_name", "sub_key", "point_index"]

    def _pred_df(
        manifest: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        proba: np.ndarray,
    ) -> pd.DataFrame:
        cols = [c for c in id_cols if c in manifest.columns]
        out = manifest[cols].reset_index(drop=True).copy()
        out["residual"] = manifest["residual"].reset_index(drop=True)
        out["true_bin"] = y_true
        out["true_label"] = [LABELS[i] for i in y_true]
        out["pred_bin"] = y_pred
        out["pred_label"] = [LABELS[i] for i in y_pred]
        out["correct"] = y_true == y_pred
        for i, label in enumerate(LABELS):
            out[f"prob_{label}"] = proba[:, i]
        return out

    def _get_proba(X: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        return np.zeros((len(X), N_CLASSES))

    pred_tr_df = _pred_df(manifests["train"], y_tr, pred_tr, _get_proba(train_feat))
    pred_val_df = _pred_df(manifests["val"], y_val, pred_val, _get_proba(X_val))
    pred_te_df = _pred_df(manifests["test"], y_te, pred_te, _get_proba(X_te))

    pred_tr_df.to_csv(outputs_dir / "predictions_train.csv", index=False)
    pred_val_df.to_csv(outputs_dir / "predictions_val.csv", index=False)
    pred_te_df.to_csv(outputs_dir / "predictions_test.csv", index=False)

    # --- モデル・メタデータ保存 ---
    import joblib
    joblib.dump(model, outputs_dir / "xgb_cls.joblib")
    train_medians.to_csv(outputs_dir / "train_medians.csv")

    serializable_refs = {k: v.tolist() for k, v in percentile_refs.items()}
    (outputs_dir / "percentile_refs.json").write_text(
        json.dumps(serializable_refs, ensure_ascii=False), encoding="utf-8"
    )
    (outputs_dir / "feature_columns.json").write_text(
        json.dumps(feat_cols, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if hasattr(model, "get_booster"):
        fi = model.get_booster().get_fscore()
        (outputs_dir / "feature_importance.json").write_text(
            json.dumps(fi, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    all_metrics = {
        "threshold": threshold,
        "train": m_tr,
        "val": m_val,
        "test": m_te,
        "n_train": int(len(train_feat)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_te)),
        "feature_columns": feat_cols,
    }
    if oof_preds is not None:
        all_metrics["oof"] = cls_metrics(y_tv, oof_preds)

    (outputs_dir / "metrics.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outputs_dir / "experiment.json").write_text(
        json.dumps(
            {
                "experiment": "xgboost_predres_cls",
                "config_path": str(args.config),
                "target": f"bin(residual, threshold={threshold}): under/ok/over",
                "xgboost": asdict(xgb_cfg),
                "kfold": asdict(kfold_cfg),
                "optuna": {**asdict(optuna_cfg), **optuna_summary},
                "feature_config": asdict(feat_cfg),
                "metrics": all_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[OK] train_cls 完了:", outputs_dir)


if __name__ == "__main__":
    main()
