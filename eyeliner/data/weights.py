"""サンプル重み計算（目スケール正規化）。"""
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

    for rec in records:
        key = (rec["image_path"], rec["part_name"])
        eye_points[key].append((float(rec["px"]), float(rec["py"])))
        eye_points[key].append((float(rec["qx"]), float(rec["qy"])))
        eye_count[key] += 1

    eye_scale_map: dict[tuple, float] = {}
    for key, pts in eye_points.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        scale = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)
        eye_scale_map[key] = max(scale, 1.0)  # ゼロ除算防止

    for rec in records:
        key = (rec["image_path"], rec["part_name"])
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
