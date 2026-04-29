"""学習済みモデルで train/val/test/unlabeled の予測を出し分類分析を行う。

出力:
  predictions_{train,val,test,unlabeled}.csv   行毎予測結果
  analysis_subgroup.csv                        サブグループ別メトリクス
  analysis_feature_importance.csv              特徴量重要度ランキング
  analysis_confusion_matrix.json               val/test 混同行列
  misclassified_topk.csv                       高確信度誤分類 top-K
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

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
    add_categorical_features,
    add_cyclical_features,
    add_geometry_features,
    add_interaction_features,
)
from analyses.xgboost_predskip.scr.train import KFoldEnsemble

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AnalysisConfig:
    top_k_misclassified: int
    subgroup_columns: List[str]


def _analysis_cfg_from(d: Dict[str, Any]) -> AnalysisConfig:
    return AnalysisConfig(
        top_k_misclassified=int(d.get("top_k_misclassified", 500)),
        subgroup_columns=d.get(
            "subgroup_columns",
            ["part_name", "sub_key", "point_index"],
        ),
    )


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
def prepare_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    train_medians: pd.Series,
) -> pd.DataFrame:
    enriched = add_geometry_features(df)
    enriched = add_categorical_features(enriched)
    enriched = add_cyclical_features(enriched)
    enriched = add_interaction_features(enriched)
    for c in feature_columns:
        if c not in enriched.columns:
            enriched[c] = np.nan
    return enriched[feature_columns].astype(float).fillna(train_medians)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _identity_cols(df: pd.DataFrame) -> List[str]:
    candidates = [
        "idx",
        "image_path",
        "face_index",
        "part_name",
        "sub_key",
        "point_index",
        "px",
        "py",
        "qx",
        "qy",
        "true_t_raw",
        "raw_t",
        "skip_flag",
    ]
    return [c for c in candidates if c in df.columns]


def infer_split(
    df: pd.DataFrame,
    feat_cols: List[str],
    train_medians: pd.Series,
    model: Any,
) -> pd.DataFrame:
    feat = prepare_features(df, feat_cols, train_medians)
    if isinstance(model, KFoldEnsemble):
        prob = model.predict_proba(feat)
    else:
        prob = model.predict_proba(feat)[:, 1]
    pred = (prob >= 0.5).astype(int)
    id_cols = _identity_cols(df)
    out = pd.concat(
        [
            df[id_cols].reset_index(drop=True),
            pd.DataFrame(
                {
                    "pred_prob": prob,
                    "pred_label": pred,
                }
            ).reset_index(drop=True),
        ],
        axis=1,
    )
    if "skip_flag" in out.columns:
        out["correct"] = out["pred_label"] == out["skip_flag"]
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def compute_subgroup_metrics(
    pred_df: pd.DataFrame,
    subgroup_columns: List[str],
) -> pd.DataFrame:
    rows: list = []
    for col in subgroup_columns:
        if col not in pred_df.columns:
            continue
        for val, grp in pred_df.groupby(col):
            y_true = grp["skip_flag"].astype(int).to_numpy()
            y_pred = grp["pred_label"].astype(int).to_numpy()
            y_prob = grp["pred_prob"].to_numpy()
            n_pos = int(y_true.sum())
            n_total = len(y_true)
            if n_pos == 0 or n_pos == n_total:
                auc = float("nan")
            else:
                auc = float(roc_auc_score(y_true, y_prob))
            rows.append(
                {
                    "subgroup_type": col,
                    "subgroup_value": val,
                    "n": n_total,
                    "n_pos": n_pos,
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "precision": float(
                        precision_score(y_true, y_pred, zero_division=0)
                    ),
                    "recall": float(
                        recall_score(y_true, y_pred, zero_division=0)
                    ),
                    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                    "roc_auc": auc,
                }
            )
    return pd.DataFrame(rows)


def extract_misclassified_topk(
    pred_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    pred_df = pred_df.copy()
    pred_df["confidence"] = np.abs(
        pred_df["pred_prob"] - pred_df["skip_flag"]
    )
    wrong = pred_df[~pred_df["correct"]]
    return wrong.sort_values(
        "confidence",
        ascending=False,
    ).head(top_k)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_model(outputs_dir: Path) -> Any:
    import joblib

    return joblib.load(outputs_dir / "xgb_classifier.joblib")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="xgboost_predskip 推論 + 分析"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analyses/xgboost_predskip/config/baseline.yaml"),
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(
            f"[ERROR] config が見つかりません: {args.config}",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    manifests_dir = Path(cfg["data"]["manifests_dir"])
    outputs_dir = Path(cfg["data"]["outputs_dir"])
    analysis_cfg = _analysis_cfg_from(cfg.get("analysis", {}))

    feat_cols_path = outputs_dir / "feature_columns.json"
    medians_path = outputs_dir / "train_medians.csv"
    if not feat_cols_path.exists() or not medians_path.exists():
        print(
            f"[ERROR] train 成果物がありません: "
            f"{feat_cols_path}, {medians_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    feat_cols: List[str] = json.loads(
        feat_cols_path.read_text(encoding="utf-8")
    )
    train_medians = pd.read_csv(
        medians_path,
        index_col=0,
    ).iloc[:, 0]
    model = _load_model(outputs_dir)

    # --- split 毎の推論 ---
    all_preds: Dict[str, pd.DataFrame] = {}
    confusion: Dict[str, Any] = {}
    for split_name in ["train", "val", "test", "unlabeled"]:
        manifest_path = manifests_dir / f"manifest_{split_name}.csv"
        if not manifest_path.exists():
            logger.warning("manifest が見つかりません: %s", manifest_path)
            continue
        df = pd.read_csv(manifest_path)
        logger.info(
            "inferring %s (%d rows)",
            split_name,
            len(df),
        )
        pred_df = infer_split(
            df,
            feat_cols,
            train_medians,
            model,
        )

        all_preds[split_name] = pred_df

        out_path = outputs_dir / f"predictions_{split_name}.csv"
        pred_df.to_csv(out_path, index=False)
        logger.info("wrote %s", out_path)

        # confusion matrix は labeled split のみ
        if split_name in ("val", "test") and "skip_flag" in pred_df.columns:
            cm = confusion_matrix(
                pred_df["skip_flag"].astype(int),
                pred_df["pred_label"].astype(int),
                labels=[0, 1],
            )
            tn, fp, fn, tp = cm.ravel()
            confusion[split_name] = {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
    labeled_splits = [
        all_preds[s]
        for s in ["train", "val", "test"]
        if s in all_preds
    ]
    if labeled_splits:
        combined = pd.concat(labeled_splits, ignore_index=True)

        # --- サブグループ分析 ---
        subgroup_df = compute_subgroup_metrics(
            combined,
            analysis_cfg.subgroup_columns,
        )

        subgroup_df.to_csv(
            outputs_dir / "analysis_subgroup.csv",
            index=False,
        )

        logger.info("wrote analysis_subgroup.csv")

        # --- 誤分類 top-K ---
        topk = extract_misclassified_topk(
            combined,
            analysis_cfg.top_k_misclassified,
        )

        topk.to_csv(
            outputs_dir / "misclassified_topk.csv",
            index=False,
        )

        logger.info(
            "wrote misclassified_topk.csv (%d rows)",
            len(topk),
        )

    # --- 特徴量重要度コピー ---
    fi_path = outputs_dir / "feature_importance.json"

    if fi_path.exists():
        fi = json.loads(fi_path.read_text(encoding="utf-8"))

        fi_df = pd.DataFrame(
            [
                {
                    "feature": k,
                    "importance": v,
                }
                for k, v in fi.items()
            ]
        ).sort_values("importance", ascending=False)
        fi_df.to_csv(
            outputs_dir / "analysis_feature_importance.csv",
            index=False,
        )
        logger.info("wrote analysis_feature_importance.csv")

    # --- 混同行列 ---
    if confusion:
        (outputs_dir / "analysis_confusion_matrix.json").write_text(
            json.dumps(
                confusion,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("wrote analysis_confusion_matrix.json")

    print("[OK] infer 完了:", outputs_dir)


if __name__ == "__main__":
    main()