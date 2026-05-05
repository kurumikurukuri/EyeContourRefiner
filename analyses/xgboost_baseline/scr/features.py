"""samples.csv → XGBoost 用特徴量 DataFrame に変換。

純粋関数群で pandas DataFrame を in → out で返す。学習/推論の双方から参照される。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

_ALL_NULL_THRESHOLD = 0.99

_IMAGE_FEATURE_COLS = [
    "eye_gray_mean",
    "eye_gray_std",
    "eye_gray_skewness",
    "eye_gray_entropy",
    "eye_gray_p10",
    "eye_gray_p90",
    "eye_laplacian_var",
    "eye_sobel_mean",
    "eye_sobel_std",
    "eye_canny_density",
    "eye_hog_mean",
    "eye_hog_std",
    "eye_lbp_uniform_ratio",
    "eye_lbp_entropy",
]

_DEFAULT_IMAGE_CACHE = "analyses/xgboost_predskip/outputs/image_features.parquet"


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
    use_aggregate_features: bool = False  # 眼内ばらつき・左右非対称・顔数など画像/顔単位の集約特徴
    use_extended_aggregate: bool = False  # 顔単位 range / 画像単位 niqe 統計 / 顔サイズ正規化
    use_pose_composite: bool = False     # pose_magnitude, niqe_score_x_pose_magnitude
    use_dataset_percentile: bool = False  # niqe_percentile, eye_scale_percentile
    use_image_features: bool = False      # 目領域の局所画像特徴量 (HOG/LBP/blur/histogram)
    exclude_features: tuple[str, ...] = ()  # 除外する特徴量カラム名


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


def add_pose_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """顔向き合成特徴量と niqe_score との交互作用を追加する。"""
    out = df.copy()
    out["pose_magnitude"] = np.sqrt(
        out["face_yaw"] ** 2 + out["face_pitch"] ** 2 + out["face_roll"] ** 2
    )
    if "niqe_score" in out.columns:
        out["niqe_score_x_pose_magnitude"] = out["niqe_score"] * out["pose_magnitude"]
    return out


def add_image_features(
    df: pd.DataFrame,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """目領域の局所画像特徴量をキャッシュからロードしてマージする。

    キャッシュは train.py の事前計算で作成済みであること。
    """
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)

    out = df.copy()
    cpath = Path(cache_path) if cache_path else Path(_DEFAULT_IMAGE_CACHE)
    if not cpath.exists():
        logger.warning("image features cache not found: %s", cpath)
        return out
    img_feat = pd.read_parquet(cpath)
    if len(img_feat) == 0:
        return out
    merge_keys = ["image_path", "face_index", "part_name"]
    return out.merge(img_feat, on=merge_keys, how="left")


def add_dataset_percentile_features(
    df: pd.DataFrame,
    references: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """データセット内パーセンタイル特徴量を追加する。

    references が None の場合は df 自身から参照分布を計算して返す。
    推論時は学習時に保存した references を渡す。
    """
    out = df.copy()
    new_refs: dict = {}

    configs = [
        ("niqe_percentile", "niqe_score"),
        ("eye_scale_percentile", "eye_scale"),
    ]

    for out_col, src_col in configs:
        if src_col not in out.columns:
            continue
        if references is not None and src_col in references:
            ref_sorted = references[src_col]
        else:
            ref_sorted = np.sort(out[src_col].dropna().values)
            new_refs[src_col] = ref_sorted
        ranks = np.searchsorted(ref_sorted, out[src_col].fillna(ref_sorted[-1]).values)
        out[out_col] = ranks / len(ref_sorted) * 100.0

    return out, new_refs


def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """眼内ばらつき・左右非対称・顔数など画像/顔単位の集約特徴を追加する。

    add_geometry_features / add_categorical_features の後に呼ぶこと。
    """
    out = df.copy()
    eye_key = ["image_path", "face_index", "part_name"]
    face_key = ["image_path", "face_index"]
    img_key = ["image_path"]

    # 眼内点群のばらつき (14点固定だが、点の広がりは画像品質で変動)
    for col in ["pq_dist", "pq_ratio", "dist_from_center"]:
        grp = out.groupby(eye_key)[col]
        out[f"{col}_eye_std"] = grp.transform("std").fillna(0.0)

    # 左右眼の非対称性
    eye_means = out.groupby(eye_key).agg(
        eye_scale_mean=("eye_scale", "mean"),
        eye_aspect_mean=("eye_aspect", "mean"),
        norm_eye_scale_mean=("norm_eye_scale", "mean"),
        eye_bbox_w_mean=("eye_bbox_w", "mean"),
        eye_bbox_h_mean=("eye_bbox_h", "mean"),
    ).reset_index()

    for col in ["eye_scale", "eye_aspect", "norm_eye_scale", "eye_bbox_w", "eye_bbox_h"]:
        pivot = eye_means.pivot_table(
            index=face_key,
            columns="part_name",
            values=f"{col}_mean",
        )
        left_col = [c for c in pivot.columns if "left" in str(c).lower()]
        right_col = [c for c in pivot.columns if "right" in str(c).lower()]
        if left_col and right_col:
            left_vals = pivot[left_col[0]]
            right_vals = pivot[right_col[0]]
            denom = left_vals.combine(right_vals, max)
            asymmetry = ((left_vals - right_vals).abs() / denom.replace(0, 1)).fillna(0)
            out = out.merge(
                asymmetry.rename(f"{col}_asymmetry"),
                left_on=face_key,
                right_index=True,
                how="left",
            )
        else:
            out[f"{col}_asymmetry"] = 0.0

    # 画像内の顔数
    out["n_faces"] = out.groupby(img_key)["face_index"].transform("nunique")

    # 複数顔間の姿勢ばらつき
    face_pose = out.groupby(face_key)[["face_yaw", "face_pitch", "face_roll"]].mean()
    for col in ["face_yaw", "face_pitch", "face_roll"]:
        cross_std = face_pose.groupby("image_path")[col].std().reindex(out["image_path"])
        out[f"{col}_cross_face_std"] = cross_std.fillna(0.0).values

    return out


def add_extended_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 2B: 顔単位 range / 画像単位 niqe 統計 / 顔サイズ正規化を追加する。

    add_aggregate_features の後に呼ぶこと。
    """
    out = df.copy()
    eye_key = ["image_path", "face_index", "part_name"]
    face_key = ["image_path", "face_index"]
    img_key = ["image_path"]

    # ランドマーク検出信頼性プロキシ：眼内 7 点での pq_dist の max-min
    if "pq_dist" in out.columns:
        grp = out.groupby(eye_key)["pq_dist"]
        eye_max = grp.transform("max")
        eye_min = grp.transform("min")
        out["pq_dist_eye_range"] = (eye_max - eye_min).fillna(0.0)

    # 顔単位 eye_scale range / mean
    if "eye_scale" in out.columns:
        face_grp = out.groupby(face_key)["eye_scale"]
        out["eye_scale_face_max"] = face_grp.transform("max")
        out["eye_scale_face_min"] = face_grp.transform("min")
        out["eye_scale_face_range"] = (
            out["eye_scale_face_max"] - out["eye_scale_face_min"]
        ).fillna(0.0)

    # 画像単位 niqe 統計
    if "niqe_score" in out.columns:
        img_grp = out.groupby(img_key)["niqe_score"]
        out["niqe_in_img_std"] = img_grp.transform("std").fillna(0.0)
        out["niqe_in_img_min"] = img_grp.transform("min")
        out["niqe_in_img_max"] = img_grp.transform("max")
        out["niqe_in_img_range"] = (
            out["niqe_in_img_max"] - out["niqe_in_img_min"]
        ).fillna(0.0)

    # 顔サイズ正規化：eye_bbox_w を顔の eye_bbox_w 最大値で正規化
    if "eye_bbox_w" in out.columns:
        face_grp = out.groupby(face_key)["eye_bbox_w"]
        face_max_bbox = face_grp.transform("max").replace(0, np.nan)
        out["eye_bbox_w_norm_face"] = (out["eye_bbox_w"] / face_max_bbox).fillna(1.0)

    # 顔ごとの aspect 範囲
    if "eye_aspect" in out.columns:
        face_grp = out.groupby(face_key)["eye_aspect"]
        out["eye_aspect_face_range"] = (
            face_grp.transform("max") - face_grp.transform("min")
        ).fillna(0.0)

    return out


def enrich_all(
    df: pd.DataFrame,
    cfg: FeatureConfig,
    image_cache_path: str | None = None,
) -> pd.DataFrame:
    """cfg に基づいて全エンリッチメントを適用する（非破壊）。"""
    out = add_geometry_features(df)
    out = add_categorical_features(out)
    if cfg.use_cyclical_angle:
        out = add_cyclical_features(out)
    if cfg.use_image_features:
        out = add_image_features(out, cache_path=image_cache_path)
    if cfg.use_interactions:
        out = add_interaction_features(out)
    if cfg.use_pose_composite:
        out = add_pose_composite_features(out)
    if cfg.use_aggregate_features:
        out = add_aggregate_features(out)
    if cfg.use_extended_aggregate:
        out = add_extended_aggregate_features(out)
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
    if cfg.use_pose_composite:
        candidates.extend(["pose_magnitude", "niqe_score_x_pose_magnitude"])
    if cfg.use_dataset_percentile:
        candidates.extend(["niqe_percentile", "eye_scale_percentile"])
    if cfg.use_image_features:
        candidates.extend(_IMAGE_FEATURE_COLS)
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
    if cfg.use_aggregate_features:
        candidates.extend([
            "pq_dist_eye_std", "pq_ratio_eye_std", "dist_from_center_eye_std",
            "eye_scale_asymmetry", "eye_aspect_asymmetry", "norm_eye_scale_asymmetry",
            "eye_bbox_w_asymmetry", "eye_bbox_h_asymmetry",
            "n_faces",
            "face_yaw_cross_face_std", "face_pitch_cross_face_std", "face_roll_cross_face_std",
        ])
    if cfg.use_extended_aggregate:
        candidates.extend([
            "pq_dist_eye_range",
            "eye_scale_face_max", "eye_scale_face_min", "eye_scale_face_range",
            "niqe_in_img_std", "niqe_in_img_min", "niqe_in_img_max", "niqe_in_img_range",
            "eye_bbox_w_norm_face",
            "eye_aspect_face_range",
        ])

    selected: List[str] = []
    exclude = set(cfg.exclude_features)
    for col in candidates:
        if col in exclude:
            continue
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
    percentile_refs: dict | None = None,
    image_cache_path: str | None = None,
) -> tuple[pd.DataFrame, List[str], dict]:
    """df（samples.csv 相当）から特徴量 DataFrame と使用カラム名、percentile参照分布を返す。"""
    enriched = enrich_all(df, cfg, image_cache_path=image_cache_path)
    new_refs: dict = {}
    if cfg.use_dataset_percentile:
        enriched, new_refs = add_dataset_percentile_features(enriched, percentile_refs)
    cols = select_feature_columns(enriched, cfg)
    if not cols:
        raise ValueError("使用可能な特徴量カラムが見つかりません（全て欠落または全 NULL）。")
    feat = enriched[cols].astype(float)
    fill_values = feat.median(numeric_only=True)
    if cfg.use_niqe_score and "niqe_score" in cols:
        fill_values["niqe_score"] = feat["niqe_score"].max()
    feat = feat.fillna(fill_values)
    return feat, cols, new_refs


def fillna_with_reference(
    df: pd.DataFrame, reference_medians: pd.Series
) -> pd.DataFrame:
    """推論時に学習セットの median で欠損補完する。"""
    return df.fillna(reference_medians)
