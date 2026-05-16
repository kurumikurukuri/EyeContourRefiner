"""学習済みモデルで train/val/test/unlabeled の予測を出し外れ値スコアを計算する。

出力:
  predictions_train.csv / predictions_val.csv / predictions_test.csv / predictions_unlabeled.csv
  outlier_topk.csv            # unlabeled の外れ値スコア上位 top_k
  outlier_topk_annotated.csv  # train+val+test の外れ値スコア上位 top_k（ラベルミス探索用）

外れ値スコア: rank-normalize した (uncertainty, range_violation, iforest) の重み付き和。
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

from analyses.xgboost_baseline.scr.features import (
    FeatureConfig,
    add_dataset_percentile_features,
    enrich_all,
)
from analyses.xgboost_baseline.scr.train import (  # noqa: F401
    KFoldEnsemble,
    _feature_cfg_from,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OutlierConfig:
    top_k: int
    weight_uncertainty: float
    weight_range_violation: float
    weight_iforest: float
    t_valid_min: float
    t_valid_max: float


def _outlier_cfg_from(d: Dict[str, Any]) -> OutlierConfig:
    return OutlierConfig(
        top_k=int(d.get("top_k", 500)),
        weight_uncertainty=float(d.get("weight_uncertainty", 0.4)),
        weight_range_violation=float(d.get("weight_range_violation", 0.3)),
        weight_iforest=float(d.get("weight_iforest", 0.3)),
        t_valid_min=float(d.get("t_valid_min", 0.0)),
        t_valid_max=float(d.get("t_valid_max", 1.5)),
    )


# ---------------------------------------------------------------------------
# Outlier scoring
# ---------------------------------------------------------------------------
def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """0..1 にランク正規化する（大きいほど高スコア扱い）。同値は平均順位。"""
    if len(x) == 0:
        return x
    s = pd.Series(x)
    ranks = s.rank(method="average").to_numpy()
    return (ranks - 1.0) / max(1.0, len(x) - 1)


def compute_outlier_score(
    pred_lo: np.ndarray,
    pred_hi: np.ndarray,
    pred_med: np.ndarray,
    iforest_score: np.ndarray,
    cfg: OutlierConfig,
) -> pd.DataFrame:
    """3 種のスコアを rank-normalize + 重み付き和した DataFrame を返す。

    Returns:
        columns = [score, score_uncertainty, score_range_violation, score_iforest,
                   pred_lo, pred_med, pred_hi]
    """
    uncertainty = pred_hi - pred_lo
    range_violation = np.maximum(
        cfg.t_valid_min - pred_med,
        pred_med - cfg.t_valid_max,
    ).clip(min=0.0)
    # IsolationForest: score_samples は高いほど正常。外れ値ほど高いスコアにするため符号反転。
    iforest_outlier = -iforest_score

    r_u = _rank_normalize(uncertainty)
    r_r = _rank_normalize(range_violation)
    r_i = _rank_normalize(iforest_outlier)

    score = (
        cfg.weight_uncertainty * r_u
        + cfg.weight_range_violation * r_r
        + cfg.weight_iforest * r_i
    )
    return pd.DataFrame(
        {
            "pred_lo": pred_lo,
            "pred_med": pred_med,
            "pred_hi": pred_hi,
            "score_uncertainty": r_u,
            "score_range_violation": r_r,
            "score_iforest": r_i,
            "score": score,
        }
    )


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
def prepare_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    train_medians: pd.Series,
    feat_cfg: FeatureConfig | None = None,
    percentile_refs: dict | None = None,
    image_cache_path: str | None = None,
) -> pd.DataFrame:
    if feat_cfg is not None:
        enriched = enrich_all(df, feat_cfg, image_cache_path=image_cache_path)
        if feat_cfg.use_dataset_percentile and percentile_refs:
            enriched, _ = add_dataset_percentile_features(enriched, percentile_refs)
    else:
        # 後方互換: feat_cfg がない場合は geometry + categorical のみ
        from analyses.xgboost_baseline.scr.features import (
            add_categorical_features,
            add_cyclical_features,
            add_geometry_features,
            add_interaction_features,
        )
        enriched = add_geometry_features(df)
        enriched = add_categorical_features(enriched)
        enriched = add_cyclical_features(enriched)
        enriched = add_interaction_features(enriched)
    for c in feature_columns:
        if c not in enriched.columns:
            enriched[c] = np.nan
    return enriched[feature_columns].astype(float).fillna(train_medians)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_models(outputs_dir: Path):
    import joblib

    return {
        "median": joblib.load(outputs_dir / "xgb_median.joblib"),
        "q_lo": joblib.load(outputs_dir / "xgb_q_lo.joblib"),
        "q_hi": joblib.load(outputs_dir / "xgb_q_hi.joblib"),
        "iforest": joblib.load(outputs_dir / "iforest.joblib"),
    }


def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


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
        "skip_flag",
    ]
    return [c for c in candidates if c in df.columns]


def infer_split(
    df: pd.DataFrame,
    feat_cols: List[str],
    train_medians: pd.Series,
    models: Dict[str, Any],
    outlier_cfg: OutlierConfig,
    feat_cfg: FeatureConfig | None = None,
    percentile_refs: dict | None = None,
    image_cache_path: str | None = None,
) -> pd.DataFrame:
    feat = prepare_features(
        df, feat_cols, train_medians,
        feat_cfg=feat_cfg, percentile_refs=percentile_refs,
        image_cache_path=image_cache_path,
    )
    pred_med = models["median"].predict(feat)
    pred_lo = models["q_lo"].predict(feat)
    pred_hi = models["q_hi"].predict(feat)
    if_score = models["iforest"].score_samples(feat)
    scores = compute_outlier_score(pred_lo, pred_hi, pred_med, if_score, outlier_cfg)
    id_cols = _identity_cols(df)
    out = pd.concat(
        [df[id_cols].reset_index(drop=True), scores.reset_index(drop=True)], axis=1
    )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="xgboost_baseline 推論 + 外れ値スコア")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analyses/xgboost_baseline/config/baseline.yaml"),
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[ERROR] config が見つかりません: {args.config}", file=sys.stderr)
        sys.exit(1)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    manifests_dir = Path(cfg["data"]["manifests_dir"])
    outputs_dir = Path(cfg["data"]["outputs_dir"])
    outlier_cfg = _outlier_cfg_from(cfg.get("outlier", {}))

    feat_cols_path = outputs_dir / "feature_columns.json"
    medians_path = outputs_dir / "train_medians.csv"
    if not feat_cols_path.exists() or not medians_path.exists():
        print(
            f"[ERROR] train 成果物がありません: {feat_cols_path}, {medians_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    feat_cols: List[str] = json.loads(feat_cols_path.read_text(encoding="utf-8"))
    train_medians = pd.read_csv(medians_path, index_col=0).iloc[:, 0]
    models = _load_models(outputs_dir)

    # feat_cfg / percentile_refs / image_cache_path（存在する場合のみ使用）
    feat_cfg: FeatureConfig | None = None
    percentile_refs: dict | None = None
    image_cache_path: str | None = cfg.get("data", {}).get("image_cache_path", None)
    exp_path = outputs_dir / "experiment.json"
    if exp_path.exists():
        exp = json.loads(exp_path.read_text(encoding="utf-8"))
        fc_dict = exp.get("feature_config", {})
        if fc_dict:
            feat_cfg = _feature_cfg_from(fc_dict)
    refs_path = outputs_dir / "percentile_refs.json"
    if refs_path.exists():
        raw = json.loads(refs_path.read_text(encoding="utf-8"))
        percentile_refs = {k: np.array(v) for k, v in raw.items()}

    for split_name in ["train", "val", "test", "unlabeled"]:
        df = _load_manifest(manifests_dir / f"manifest_{split_name}.csv")

        # skip_flag を除去
        if "skip_flag" in df.columns:
            df = df[df["skip_flag"] == 0]

        logger.info("inferring %s (%d rows)", split_name, len(df))
        pred_df = infer_split(
            df, feat_cols, train_medians, models, outlier_cfg,
            feat_cfg=feat_cfg, percentile_refs=percentile_refs,
            image_cache_path=image_cache_path,
        )
        out_path = outputs_dir / f"predictions_{split_name}.csv"
        pred_df.to_csv(out_path, index=False)
        logger.info("wrote %s", out_path)

    # --- 外れ値 top-K 抽出 ---
    unl_df = pd.read_csv(outputs_dir / "predictions_unlabeled.csv")
    topk_unl = unl_df.sort_values("score", ascending=False).head(outlier_cfg.top_k)
    topk_unl.to_csv(outputs_dir / "outlier_topk.csv", index=False)
    logger.info("wrote outlier_topk.csv (%d rows)", len(topk_unl))

    # ラベル済みでもスコアが高いものは「ラベルミスの可能性」として別途抽出
    annotated = pd.concat(
        [
            pd.read_csv(outputs_dir / "predictions_train.csv"),
            pd.read_csv(outputs_dir / "predictions_val.csv"),
            pd.read_csv(outputs_dir / "predictions_test.csv"),
        ],
        ignore_index=True,
    )
    topk_ann = annotated.sort_values("score", ascending=False).head(outlier_cfg.top_k)
    topk_ann.to_csv(outputs_dir / "outlier_topk_annotated.csv", index=False)
    logger.info("wrote outlier_topk_annotated.csv (%d rows)", len(topk_ann))

    print("[OK] infer 完了:", outputs_dir)


if __name__ == "__main__":
    main()
