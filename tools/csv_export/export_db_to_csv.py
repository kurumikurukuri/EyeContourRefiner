"""
DB を指定カラム形式の CSV に出力する。

eye_scale / norm_eye_scale / eye_bbox_w / eye_bbox_h / eye_aspect はデータから計算。
face_yaw / face_pitch / face_roll は DB に格納済みの値を読む。
pred_* / true_t_norm / rx_pred / ry_pred は空。
"""

import logging
import math
import sqlite3
from collections import defaultdict
from csv import DictWriter
from typing import Optional

import cv2
import numpy as np
import pyiqa
import scipy
import torch
from PIL import Image
from skimage import io, img_as_float
from skimage.transform import resize as sk_resize



DB_PATH = "data/samples.db"
OUTPUT_PATH = "analyses/samples.csv"

OUTPUT_COLS = [
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
    "eye_scale",
    "norm_eye_scale",
    "eye_bbox_w",
    "eye_bbox_h",
    "eye_aspect",
    "face_yaw",
    "face_pitch",
    "face_roll",
    "niqe_score",
    "skip_flag",
]

# -----------------------------
# pyiqa
# -----------------------------
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_NIQE_METRIC = pyiqa.create_metric("niqe", device=_DEVICE)


# -----------------------------
# scipy.misc.imresize compatibility
# -----------------------------
def _imresize(arr, size, interp="bicubic", mode=None):
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(arr)

    interp_map = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
    }

    if isinstance(size, float):
        new_size = (
            max(1, int(arr.shape[1] * size)),
            max(1, int(arr.shape[0] * size)),
        )
    else:
        # size = (H, W)
        new_size = (int(size[1]), int(size[0]))

    return np.array(
        pil_img.resize(
            new_size,
            interp_map.get(interp, Image.BICUBIC),
        )
    )


if not hasattr(scipy, "misc"):
    class _Misc:
        pass

    scipy.misc = _Misc()

scipy.misc.imresize = _imresize


# -----------------------------
# DB
# -----------------------------
def load_records() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM samples")
    records = [dict(r) for r in cur.fetchall()]
    conn.close()
    return records


# -----------------------------
# Eye points / bbox
# -----------------------------
def _collect_eye_points(
    records: list[dict],
    image_path: str,
    face_index: int,
    part_name: str,
) -> list[tuple[float, float]]:
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


# -----------------------------
# Debug
# -----------------------------
def debug_eye_bbox_opencv(
    records: list[dict],
    image_path: str,
    face_index: int,
    part_name: str,
    margin_ratio: float = 0.15,
) -> None:
    print("=" * 80)
    print(f"[DEBUG] image_path={image_path}")
    print(f"[DEBUG] face_index={face_index}")
    print(f"[DEBUG] part_name={part_name}")

    pts = _collect_eye_points(records, image_path, face_index, part_name)

    if not pts:
        print("[DEBUG] No points found")
        return

    bbox = _compute_bbox_from_points(pts)

    if not bbox:
        print("[DEBUG] bbox is None")
        return

    print(
        f"[DEBUG] Raw BBOX: "
        f"x_min={bbox['x_min']:.2f}, y_min={bbox['y_min']:.2f}, "
        f"x_max={bbox['x_max']:.2f}, y_max={bbox['y_max']:.2f}"
    )

    img = cv2.imread(image_path)

    if img is None:
        print("[DEBUG] Failed to load image")
        return

    img_h, img_w = img.shape[:2]

    mx = bbox["w"] * margin_ratio
    my = bbox["h"] * margin_ratio

    x1 = max(int(math.floor(bbox["x_min"] - mx)), 0)
    y1 = max(int(math.floor(bbox["y_min"] - my)), 0)
    x2 = min(int(math.ceil(bbox["x_max"] + mx)), img_w)
    y2 = min(int(math.ceil(bbox["y_max"] + my)), img_h)

    print(
        f"[DEBUG] Margin BBOX: "
        f"x1={x1}, y1={y1}, x2={x2}, y2={y2}, "
        f"crop_w={x2 - x1}, crop_h={y2 - y1}"
    )

    print("=" * 80)


# -----------------------------
# NIQE
# -----------------------------
def _compute_niqe_from_img(
    img_uint8: np.ndarray,
    image_path: str,
    bbox: dict[str, float],
    margin_ratio: float = 0.20,
) -> Optional[float]:
    img_h, img_w = img_uint8.shape[:2]

    mx = bbox["w"] * margin_ratio
    my = bbox["h"] * margin_ratio

    x1 = max(int(math.floor(bbox["x_min"] - mx)), 0)
    y1 = max(int(math.floor(bbox["y_min"] - my)), 0)
    x2 = min(int(math.ceil(bbox["x_max"] + mx)), img_w)
    y2 = min(int(math.ceil(bbox["y_max"] + my)), img_h)

    if x2 <= x1 or y2 <= y1:
        return None

    # uint8 のままクロップしてから float32 変換（大画像の OOM を防ぐ）
    crop = img_as_float(img_uint8[y1:y2, x1:x2]).astype(np.float32)

    if crop.size == 0:
        return None

    if crop.ndim == 2:
        crop = np.stack([crop, crop, crop], axis=-1)

    if crop.ndim == 3 and crop.shape[2] == 4:
        crop = crop[..., :3]

    min_size = 224
    h, w = crop.shape[:2]

    if min(h, w) < min_size:
        scale = min_size / min(h, w)
        crop = sk_resize(
            crop,
            (max(1, int(h * scale)), max(1, int(w * scale))),
            order=3,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)

    tensor = (
        torch.from_numpy(crop)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(_DEVICE)
    )
    tensor = torch.clamp(tensor, 0.0, 1.0)

    try:
        with torch.no_grad():
            score = _NIQE_METRIC(tensor)
        if torch.is_tensor(score):
            return float(score.item())
        return float(score)
    except Exception as e:
        logging.warning(f"NIQE failed: image={image_path}, err={e}")
        return None


def build_niqe_map(
    eye_bbox_map: dict[tuple, dict[str, float]],
) -> dict[tuple, Optional[float]]:
    # 画像パスごとにグループ化して 1 枚につき 1 回だけ imread する
    grouped: dict[str, list[tuple]] = defaultdict(list)
    for (image_path, face_index, part_name) in eye_bbox_map:
        grouped[image_path].append((face_index, part_name))

    niqe_map: dict[tuple, Optional[float]] = {}
    total_images = len(grouped)

    for img_idx, (image_path, entries) in enumerate(grouped.items()):
        if img_idx % 200 == 0:
            print(f"  NIQE {img_idx}/{total_images} images ...")

        try:
            img_uint8 = io.imread(image_path)
        except Exception:
            for face_index, part_name in entries:
                niqe_map[(image_path, face_index, part_name)] = None
            continue

        for face_index, part_name in entries:
            bbox = eye_bbox_map[(image_path, face_index, part_name)]
            niqe_map[(image_path, face_index, part_name)] = _compute_niqe_from_img(
                img_uint8, image_path, bbox
            )

    return niqe_map


# -----------------------------
# Image diag
# -----------------------------
def load_img_diag_map(image_paths: list[str]) -> dict[str, float]:
    diag_map: dict[str, float] = {}

    for path in image_paths:
        try:
            with Image.open(path) as img:
                w, h = img.size
                diag_map[path] = math.sqrt(w * w + h * h)
        except Exception:
            diag_map[path] = 0.0

    return diag_map


# -----------------------------
# Utils
# -----------------------------
def _fmt(value: Optional[float], precision: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{precision}f}"


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-niqe", action="store_true", help="NIQE 計算をスキップ（高速エクスポート）")
    args = parser.parse_args()

    records = load_records()
    print(f"Loaded {len(records)} records")

    print("Computing eye bbox...")
    eye_bbox_map = compute_eye_bbox(records)

    if args.skip_niqe:
        print("Skipping NIQE computation.")
        niqe_map: dict = {}
    else:
        print("Computing NIQE scores...")
        niqe_map = build_niqe_map(eye_bbox_map)

    unique_paths = list({rec["image_path"] for rec in records})

    print(f"Loading image sizes for {len(unique_paths)} images...")
    diag_map = load_img_diag_map(unique_paths)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = DictWriter(
            f,
            fieldnames=OUTPUT_COLS,
            extrasaction="ignore",
        )

        writer.writeheader()

        for rec in records:
            key = (
                rec["image_path"],
                rec.get("face_index", 0),
                rec["part_name"],
            )

            bbox = eye_bbox_map.get(key)
            niqe_score = niqe_map.get(key)

            eye_scale = bbox["diag"] if bbox else None
            eye_bbox_w = bbox["w"] if bbox else None
            eye_bbox_h = bbox["h"] if bbox else None

            eye_aspect = None
            if bbox and bbox["h"] > 1e-6:
                eye_aspect = bbox["w"] / bbox["h"]

            img_diag = diag_map.get(rec["image_path"], 0.0)

            norm_eye_scale = None
            if img_diag > 0 and eye_scale is not None:
                norm_eye_scale = eye_scale / img_diag

            writer.writerow(
                {
                    "idx": rec["idx"],
                    "image_path": rec["image_path"],
                    "face_index": rec.get("face_index", 0),
                    "part_name": rec["part_name"],
                    "sub_key": rec["sub_key"],
                    "point_index": rec["point_index"],
                    "px": rec["px"],
                    "py": rec["py"],
                    "qx": rec["qx"],
                    "qy": rec["qy"],
                    "true_t_raw": rec["t"] if rec["t"] is not None else "",
                    "eye_scale": _fmt(eye_scale),
                    "norm_eye_scale": _fmt(norm_eye_scale),
                    "eye_bbox_w": _fmt(eye_bbox_w),
                    "eye_bbox_h": _fmt(eye_bbox_h),
                    "eye_aspect": _fmt(eye_aspect),
                    "face_yaw": _fmt(rec.get("face_yaw"), precision=4),
                    "face_pitch": _fmt(rec.get("face_pitch"), precision=4),
                    "face_roll": _fmt(rec.get("face_roll"), precision=4),
                    "niqe_score": _fmt(niqe_score),
                    "skip_flag": rec.get("skip_flag", 0),
                }
            )

    print(f"Exported {len(records)} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()