"""samples.csv → XGBoost 用特徴量 DataFrame に変換。

純粋関数群で pandas DataFrame を in → out で返す。学習/推論の双方から参照される。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

_ALL_NULL_THRESHOLD = 0.99


@dataclass(frozen=True)
class FeatureConfig:
    use_geometry: bool = True
    use_eye_scale: bool = True
    use_face_pose: bool = True
    use_categorical: bool = True
    use_point_index: bool = True
    use_cyclical_angle: bool = False   # pq_angle を sin/cos に分解して ±π 不連続を解消
    use_interactions: bool = False      # is_upper × 幾何量 のインタラクション特徴
    use_niqe_score: bool = False        # niqe_score (画質スコア)


def add_geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    """px/py/qx/qy/eye_scale/point_index から派生特徴を追加する（非破壊）。"""
    out = df.copy()
    dx = out["qx"] - out["px"]
    dy = out["qy"] - out["py"]
    out["pq_dist"] = np.sqrt(dx * dx + dy * dy)
    out["pq_angle"] = np.arctan2(dy, dx)
    out["pq_ratio"] = np.where(
        out["eye_scale"].fillna(0.0) > 1e-6,
        out["pq_dist"] / out["eye_scale"],
        np.nan,
    )
    pi = out["point_index"].fillna(3).astype(int)
    out["dist_from_center"] = np.minimum(pi, 6 - pi) / 3.0
    return out


def add_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """part_name / sub_key を 0/1 整数特徴に変換する。"""
    out = df.copy()
    out["is_upper"] = (out["sub_key"] == "upper").astype(int)
    out["is_left_eye"] = out["part_name"].str.startswith("eyeliner_left").astype(int)
    return out


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """pq_angle を sin/cos に分解する（add_geometry_features 後に呼ぶこと）。"""
    out = df.copy()
    out["pq_angle_sin"] = np.sin(out["pq_angle"])
    out["pq_angle_cos"] = np.cos(out["pq_angle"])
    return out


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """is_upper × 幾何量 のインタラクション特徴を追加する（categorical + geometry 後に呼ぶこと）。"""
    out = df.copy()
    # pq_ratio が NaN の場合は interaction も NaN のまま（後段で median 埋め）
    out["is_upper_x_pq_ratio"] = out["is_upper"] * out["pq_ratio"]
    out["is_upper_x_dist_from_center"] = out["is_upper"] * out["dist_from_center"]
    return out


def enrich_all(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """cfg に基づいて全エンリッチメントを適用する（非破壊）。"""
    out = add_geometry_features(df)
    out = add_categorical_features(out)
    if cfg.use_cyclical_angle:
        out = add_cyclical_features(out)
    if cfg.use_interactions:
        out = add_interaction_features(out)
    return out


def select_feature_columns(
    df: pd.DataFrame,
    cfg: FeatureConfig,
) -> List[str]:
    """設定に応じて使用する特徴量カラム名のリストを返す（CSV 内の存在チェック + 全 NULL 除外）。"""
    candidates: List[str] = []
    if cfg.use_point_index:
        candidates.append("point_index")
    if cfg.use_categorical:
        candidates.extend(["is_upper", "is_left_eye"])
    if cfg.use_geometry:
        if cfg.use_cyclical_angle:
            # pq_angle を sin/cos に分解（生の pq_angle は除外）
            candidates.extend(
                ["pq_dist", "pq_angle_sin", "pq_angle_cos", "pq_ratio", "dist_from_center"]
            )
        else:
            candidates.extend(["pq_dist", "pq_angle", "pq_ratio", "dist_from_center"])
    if cfg.use_eye_scale:
        candidates.extend(
            ["eye_scale", "norm_eye_scale", "eye_bbox_w", "eye_bbox_h", "eye_aspect"]
        )
    if cfg.use_face_pose:
        candidates.extend(["face_yaw", "face_pitch", "face_roll"])
    if cfg.use_niqe_score:
        candidates.append("niqe_score")
    if cfg.use_interactions:
        candidates.extend(["is_upper_x_pq_ratio", "is_upper_x_dist_from_center"])

    selected: List[str] = []
    for col in candidates:
        if col not in df.columns:
            continue
        null_ratio = df[col].isna().mean() if len(df) > 0 else 1.0
        if null_ratio >= _ALL_NULL_THRESHOLD:
            continue
        selected.append(col)
    return selected


def build_feature_matrix(
    df: pd.DataFrame,
    cfg: FeatureConfig,
) -> tuple[pd.DataFrame, List[str]]:
    """df（samples.csv 相当）から特徴量 DataFrame と使用カラム名を返す。"""
    enriched = enrich_all(df, cfg)
    cols = select_feature_columns(enriched, cfg)
    if not cols:
        raise ValueError("使用可能な特徴量カラムが見つかりません（全て欠落または全 NULL）。")
    feat = enriched[cols].astype(float)
    fill_values = feat.median(numeric_only=True)
    if cfg.use_niqe_score and "niqe_score" in cols:
        fill_values["niqe_score"] = feat["niqe_score"].max()
    feat = feat.fillna(fill_values)
    return feat, cols


def fillna_with_reference(
    df: pd.DataFrame, reference_medians: pd.Series
) -> pd.DataFrame:
    """推論時に学習セットの median で欠損補完する。"""
    return df.fillna(reference_medians)
