"""サンプル重み計算（目スケール正規化）とaux特徴量計算。"""
from __future__ import annotations

import math
from collections import defaultdict


def compute_eye_meta(records: list[dict]) -> None:
    """各レコードに weight と val_weight を付与する（in-place）。

    weight     = pq_dist / (eye_scale * n_pq)  — 訓練損失用（目スケール正規化）
    val_weight = pq_dist / n_pq                — 検証ピクセル誤差用

    同一画像・同一 part_name の点群から目スケール（目の対角長）を算出する。
    """
    eye_points: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    eye_count: dict[tuple, int] = defaultdict(int)
    pq_dist_sum: dict[tuple, float] = defaultdict(float)

    for rec in records:
        key = (rec["image_path"], rec.get("face_index", 0), rec["part_name"])
        eye_points[key].append((float(rec["px"]), float(rec["py"])))
        eye_points[key].append((float(rec["qx"]), float(rec["qy"])))
        eye_count[key] += 1
        pq_dist_sum[key] += math.sqrt(
            (float(rec["qx"]) - float(rec["px"])) ** 2
            + (float(rec["qy"]) - float(rec["py"])) ** 2
        )

    eye_scale_map: dict[tuple, float] = {}
    eye_height_map: dict[tuple, float] = {}
    mean_pq_dist_map: dict[tuple, float] = {}
    for key, pts in eye_points.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        scale = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)
        eye_scale_map[key] = max(scale, 1.0)
        eye_height_map[key] = max(max(ys) - min(ys), 1.0)
        mean_pq_dist_map[key] = max(pq_dist_sum[key] / eye_count[key], 1.0)

    for rec in records:
        key = (rec["image_path"], rec.get("face_index", 0), rec["part_name"])
        pq_dist = math.sqrt(
            (float(rec["qx"]) - float(rec["px"])) ** 2
            + (float(rec["qy"]) - float(rec["py"])) ** 2
        )
        n_pq = eye_count[key]
        eye_scale = eye_scale_map[key]
        rec["weight"] = pq_dist / (eye_scale * n_pq)
        rec["val_weight"] = pq_dist / n_pq
        rec["pq_ratio"] = pq_dist / eye_scale
        rec["eye_scale"] = eye_scale
        rec["eye_height"] = eye_height_map[key]
        rec["mean_pq_dist"] = mean_pq_dist_map[key]


def compute_aux_face_shape_meta(records: list[dict], db_path: "str | Path | None" = None) -> None:
    """各レコードに aux_face_shape 特徴量を付与する（in-place）。

    追加フィールド:
        pq_angle_cos     : PQ ベクトルの cos（(qx-px) / pq_dist）
        pq_angle_sin     : PQ ベクトルの sin（(qy-py) / pq_dist）
        position_polar_cos: 点番号の極座標 cos（データセットと同一計算式）
        position_polar_sin: 点番号の極座標 sin
        face_yaw_sin, face_yaw_cos
        face_pitch_sin, face_pitch_cos
        face_roll_sin, face_roll_cos

    Args:
        records: 対象レコード（in-place で更新）。
        db_path: face_yaw/pitch/roll を引く DB。None なら中立値で埋める。
    """
    import sqlite3

    for rec in records:
        px, py = float(rec["px"]), float(rec["py"])
        qx, qy = float(rec["qx"]), float(rec["qy"])
        pq_dist = math.sqrt((qx - px) ** 2 + (qy - py) ** 2)

        rec["pq_angle_cos"] = (qx - px) / pq_dist if pq_dist > 0 else 1.0
        rec["pq_angle_sin"] = (qy - py) / pq_dist if pq_dist > 0 else 0.0

        pi = int(rec.get("point_index", 3))
        is_upper = 1.0 if rec.get("sub_key", "") == "upper" else 0.0
        degree = (pi + 1) * 22.5 if is_upper == 1.0 else (pi + 9) * 22.5
        rad = degree * math.pi / 180.0
        rec["position_polar_cos"] = math.cos(rad)
        rec["position_polar_sin"] = math.sin(rad)

    if db_path is not None:
        keys = {(rec["image_path"], int(rec.get("face_index", 0))) for rec in records}

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        placeholders = ",".join("(?,?)" for _ in keys)
        flat_keys = [v for pair in keys for v in pair]
        rows = conn.execute(
            f"SELECT image_path, face_index, face_yaw, face_pitch, face_roll "
            f"FROM samples WHERE (image_path, face_index) IN ({placeholders})",
            flat_keys,
        ).fetchall()
        conn.close()

        pose_map: dict[tuple, tuple[float, float, float]] = {}
        for row in rows:
            k = (row["image_path"], int(row["face_index"]))
            pose_map[k] = (
                float(row["face_yaw"]   or 0.0),
                float(row["face_pitch"] or 0.0),
                float(row["face_roll"]  or 0.0),
            )

        for rec in records:
            k = (rec["image_path"], int(rec.get("face_index", 0)))
            yaw, pitch, roll = pose_map.get(k, (0.0, 0.0, 0.0))
            yaw_r   = math.radians(yaw)
            pitch_r = math.radians(pitch)
            roll_r  = math.radians(roll)
            rec["face_yaw_sin"]   = math.sin(yaw_r)
            rec["face_yaw_cos"]   = math.cos(yaw_r)
            rec["face_pitch_sin"] = math.sin(pitch_r)
            rec["face_pitch_cos"] = math.cos(pitch_r)
            rec["face_roll_sin"]  = math.sin(roll_r)
            rec["face_roll_cos"]  = math.cos(roll_r)
    else:
        for rec in records:
            rec.setdefault("face_yaw_sin",   0.0)
            rec.setdefault("face_yaw_cos",   1.0)
            rec.setdefault("face_pitch_sin", 0.0)
            rec.setdefault("face_pitch_cos", 1.0)
            rec.setdefault("face_roll_sin",  0.0)
            rec.setdefault("face_roll_cos",  1.0)
