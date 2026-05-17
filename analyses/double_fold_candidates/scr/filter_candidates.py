"""predictions_unlabeled_upper.csv から二重幅誤認候補を抽出する。

基本方針:
  upper アイラインの予測 t が想定分布（中央値 ~0.73）から大きく下振れしているケース
  = 実際のアイラインより内側（瞼側）を検出してしまっている候補
  = 二重線をアイラインと誤認している可能性が高い

出力:
  candidates_points.csv : 全ポイント単位の候補（pred_t_raw 昇順）
  candidates_faces.csv  : (image_path, face_index) 単位に集約した候補
                          面ごとの pred_t 最小値・平均値を記録
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
PRED_CSV = ROOT / "analyses" / "double_fold_candidates" / "outputs" / "predictions_unlabeled_upper.csv"
SAMPLES_CSV = ROOT / "analyses" / "samples.csv"
OUT_POINTS = ROOT / "analyses" / "double_fold_candidates" / "outputs" / "candidates_points.csv"
OUT_FACES = ROOT / "analyses" / "double_fold_candidates" / "outputs" / "candidates_faces.csv"

# 二重幅誤認しきい値:
#   upper の pred_t 中央値は約 0.73。0.30 未満は明らかに下振れしており
#   二重線側に検出位置が引き寄せられている有力候補
POINT_T_THRESHOLD = 0.30

# 面単位: 少なくとも N 点以上が閾値を割った face のみを "face 候補" とする
MIN_POINTS_PER_FACE = 2


def main() -> None:
    pred = pd.read_csv(PRED_CSV)
    samples = pd.read_csv(SAMPLES_CSV)

    meta_cols = [
        "idx",
        "image_path",
        "face_index",
        "point_index",
        "sub_key",
        "eye_scale",
        "norm_eye_scale",
        "eye_bbox_w",
        "eye_bbox_h",
        "eye_aspect",
        "face_yaw",
        "face_pitch",
        "face_roll",
    ]
    meta = samples[meta_cols].copy()

    merged = pred.merge(
        meta,
        on=["idx", "image_path", "point_index", "sub_key"],
        how="left",
        suffixes=("", "_meta"),
    )
    logger.info("merged rows: %d", len(merged))

    pts = merged.loc[merged["pred_t_raw"] < POINT_T_THRESHOLD].copy()
    pts = pts.sort_values("pred_t_raw", ascending=True).reset_index(drop=True)
    logger.info(
        "point candidates (pred_t_raw<%.2f): %d (unique images=%d)",
        POINT_T_THRESHOLD,
        len(pts),
        pts["image_path"].nunique(),
    )

    point_cols = [
        "rank",
        "pred_t_raw",
        "pred_t_norm",
        "image_path",
        "face_index",
        "part_name",
        "sub_key",
        "point_index",
        "px",
        "py",
        "qx",
        "qy",
        "rx_pred",
        "ry_pred",
        "eye_scale",
        "norm_eye_scale",
        "eye_aspect",
        "face_yaw",
        "face_pitch",
        "face_roll",
    ]
    pts.insert(0, "rank", range(1, len(pts) + 1))
    pts[point_cols].to_csv(OUT_POINTS, index=False)
    logger.info("wrote: %s", OUT_POINTS)

    low = merged.loc[merged["pred_t_raw"] < POINT_T_THRESHOLD]
    faces = (
        low.groupby(["image_path", "face_index", "part_name"], as_index=False)
        .agg(
            n_low=("pred_t_raw", "size"),
            pred_t_min=("pred_t_raw", "min"),
            pred_t_mean=("pred_t_raw", "mean"),
            eye_aspect=("eye_aspect", "first"),
            face_yaw=("face_yaw", "first"),
            face_pitch=("face_pitch", "first"),
        )
    )
    faces = faces.loc[faces["n_low"] >= MIN_POINTS_PER_FACE].copy()
    faces = faces.sort_values(["pred_t_min", "pred_t_mean"], ascending=[True, True])
    faces.insert(0, "rank", range(1, len(faces) + 1))
    faces.to_csv(OUT_FACES, index=False)
    logger.info(
        "face candidates (>=%d low points / face): %d (unique images=%d)",
        MIN_POINTS_PER_FACE,
        len(faces),
        faces["image_path"].nunique(),
    )
    logger.info("wrote: %s", OUT_FACES)

    print("\n[TOP 15 FACE CANDIDATES]")
    print(faces.head(15).to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
