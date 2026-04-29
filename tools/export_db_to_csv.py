"""DB を指定カラム形式の CSV に出力する。

eye_scale / norm_eye_scale / eye_bbox_w / eye_bbox_h / eye_aspect はデータから計算。
face_yaw / face_pitch / face_roll は DB に格納済みの値を読む（preprocessor が付与）。
pred_* / true_t_norm / rx_pred / ry_pred は推論が必要なため空。
"""
import math
import sqlite3
from collections import defaultdict
from csv import DictWriter
from typing import Optional

import numpy as np
from PIL import Image

from skimage import io, img_as_float
from skvideo.measure import niqe

DB_PATH = "data/samples.db"
OUTPUT_PATH = "data/samples.csv"

OUTPUT_COLS = [
    "idx", "image_path", "face_index", "part_name", "sub_key", "point_index",
    "px", "py", "qx", "qy",
    "true_t_raw",
    "eye_scale", "norm_eye_scale",
    "eye_bbox_w", "eye_bbox_h", "eye_aspect",
    "face_yaw", "face_pitch", "face_roll",
    "niqe_score",
    "skip_flag",
]


def load_records() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM samples")
    records = [dict(r) for r in cur.fetchall()]
    conn.close()
    return records


def _collect_eye_points(
    records: list[dict],
    image_path: str,
    face_index: int,
    part_name: str,
) -> list[tuple[float, float]]:
    """指定 key の PQ 点群を収集する共通処理。"""
    key = (image_path, face_index, part_name)

    pts: list[tuple[float, float]] = []
    for rec in records:
        rec_key = (
            rec["image_path"],
            rec.get("face_index", 0),
            rec["part_name"],
        )
        if rec_key != key:
            continue

        pts.append((float(rec["px"]), float(rec["py"])))
        pts.append((float(rec["qx"]), float(rec["qy"])))

    return pts


def _compute_bbox_from_points(
    pts: list[tuple[float, float]],
) -> Optional[dict[str, float]]:
    """点群から BBOX を計算する共通処理。"""
    if not pts:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    w = x_max - x_min
    h = y_max - y_min

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "w": w,
        "h": h,
        "diag": max(math.sqrt(w * w + h * h), 1.0),
    }


def compute_eye_bbox(records: list[dict]) -> dict[tuple, dict[str, float]]:
    """(image_path, face_index, part_name) ごとに PQ 点群 BBOX を計算する。"""
    grouped: dict[tuple, list[tuple[float, float]]] = defaultdict(list)

    for rec in records:
        key = (
            rec["image_path"],
            rec.get("face_index", 0),
            rec["part_name"],
        )

        grouped[key].append((float(rec["px"]), float(rec["py"])))
        grouped[key].append((float(rec["qx"]), float(rec["qy"])))

    result: dict[tuple, dict[str, float]] = {}

    for key, pts in grouped.items():
        bbox = _compute_bbox_from_points(pts)
        if bbox:
            result[key] = bbox

    return result


def _compute_niqe_score(
    records: list[dict],
    image_path: str,
    face_index: int,
    part_name: str,
    margin_ratio: float = 0.15,
) -> Optional[float]:
    """指定 eye BBOX 内の NIQE スコアを計算する。"""
    try:
        pts = _collect_eye_points(records, image_path, face_index, part_name)
        bbox = _compute_bbox_from_points(pts)

        if not bbox:
            return None

        img = io.imread(image_path)
        img = img_as_float(img)

        img_h, img_w = img.shape[:2]

        mx = bbox["w"] * margin_ratio
        my = bbox["h"] * margin_ratio

        x1 = max(int(math.floor(bbox["x_min"] - mx)), 0)
        y1 = max(int(math.floor(bbox["y_min"] - my)), 0)
        x2 = min(int(math.ceil(bbox["x_max"] + mx)), img_w)
        y2 = min(int(math.ceil(bbox["y_max"] + my)), img_h)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = img[y1:y2, x1:x2]

        # NIQE は小さすぎる画像で不安定
        if crop.shape[0] < 32 or crop.shape[1] < 32:
            return None

        # Gray 化
        if crop.ndim == 3:
            crop = (
                0.299 * crop[..., 0]
                + 0.587 * crop[..., 1]
                + 0.114 * crop[..., 2]
            )

        # skvideo.measure.niqe:
        # 入力 shape = (T, H, W)
        # 0-255 uint8 推奨
        crop_uint8 = np.clip(crop * 255.0, 0, 255).astype(np.uint8)
        crop_batch = crop_uint8[np.newaxis, :, :]

        score = niqe(crop_batch)

        # 戻り値は ndarray の場合あり
        if isinstance(score, np.ndarray):
            score = float(score.mean())

        if score is None or math.isnan(score) or math.isinf(score):
            return None

        return float(score)

    except Exception:
        return None


def build_niqe_map(records: list[dict]) -> dict[tuple, Optional[float]]:
    """(image_path, face_index, part_name) ごとの NIQE を事前計算。"""
    niqe_map: dict[tuple, Optional[float]] = {}

    unique_keys = {
        (
            rec["image_path"],
            rec.get("face_index", 0),
            rec["part_name"],
        )
        for rec in records
    }

    for image_path, face_index, part_name in unique_keys:
        niqe_map[(image_path, face_index, part_name)] = _compute_niqe_score(
            records,
            image_path,
            face_index,
            part_name,
        )

    return niqe_map


def load_img_diag_map(image_paths: list[str]) -> dict[str, float]:
    """画像ごとの対角長をロード。"""
    diag_map: dict[str, float] = {}

    for path in image_paths:
        try:
            with Image.open(path) as img:
                w, h = img.size
                diag_map[path] = math.sqrt(w * w + h * h)
        except Exception:
            diag_map[path] = 0.0

    return diag_map


def _fmt(value: Optional[float], precision: int = 6) -> str:
    """None は空文字、float は指定桁数で整形。"""
    if value is None:
        return ""
    return f"{value:.{precision}f}"


def main() -> None:
    records = load_records()
    print(f"Loaded {len(records)} records")

    # BBOX
    eye_bbox_map = compute_eye_bbox(records)

    # NIQE
    print("Computing NIQE scores...")
    niqe_map = build_niqe_map(records)

    # 画像サイズ
    unique_paths = list({rec["image_path"] for rec in records})
    print(f"Loading image sizes for {len(unique_paths)} images...")
    diag_map = load_img_diag_map(unique_paths)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = DictWriter(f, fieldnames=OUTPUT_COLS, extrasaction="ignore")
        writer.writeheader()

        for rec in records:
            key = (
                rec["image_path"],
                rec.get("face_index", 0),
                rec["part_name"],
            )

            bbox = eye_bbox_map.get(key)
            niqe_score = niqe_map.get(key)

            eye_scale: Optional[float] = bbox["diag"] if bbox else None
            eye_bbox_w: Optional[float] = bbox["w"] if bbox else None
            eye_bbox_h: Optional[float] = bbox["h"] if bbox else None

            eye_aspect: Optional[float] = None
            if bbox and bbox["h"] > 1e-6:
                eye_aspect = bbox["w"] / bbox["h"]

            img_diag = diag_map.get(rec["image_path"], 0.0)

            norm_eye_scale: Optional[float] = None
            if img_diag > 0 and eye_scale is not None:
                norm_eye_scale = eye_scale / img_diag

            writer.writerow({
                "idx":            rec["idx"],
                "image_path":     rec["image_path"],
                "face_index":     rec.get("face_index", 0),
                "part_name":      rec["part_name"],
                "sub_key":        rec["sub_key"],
                "point_index":    rec["point_index"],
                "px":             rec["px"],
                "py":             rec["py"],
                "qx":             rec["qx"],
                "qy":             rec["qy"],
                "true_t_raw":     rec["t"] if rec["t"] is not None else "",
                "eye_scale":      _fmt(eye_scale),
                "norm_eye_scale": _fmt(norm_eye_scale),
                "eye_bbox_w":     _fmt(eye_bbox_w),
                "eye_bbox_h":     _fmt(eye_bbox_h),
                "eye_aspect":     _fmt(eye_aspect),
                "face_yaw":       _fmt(rec.get("face_yaw"), precision=4),
                "face_pitch":     _fmt(rec.get("face_pitch"), precision=4),
                "face_roll":      _fmt(rec.get("face_roll"), precision=4),
                "niqe_score":     _fmt(niqe_score),
                "skip_flag":      rec.get("skip_flag", 0),
            })

    print(f"Exported {len(records)} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()