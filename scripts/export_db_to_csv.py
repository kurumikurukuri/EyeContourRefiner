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

from PIL import Image

DB_PATH = "data/samples.db"
OUTPUT_PATH = "data/samples.csv"

OUTPUT_COLS = [
    "idx", "image_path", "face_index", "part_name", "sub_key", "point_index",
    "px", "py", "qx", "qy",
    "true_t_raw",
    "eye_scale", "norm_eye_scale",
    "eye_bbox_w", "eye_bbox_h", "eye_aspect",
    "face_yaw", "face_pitch", "face_roll",
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


def compute_eye_bbox(records: list[dict]) -> dict[tuple, dict[str, float]]:
    """(image_path, face_index, part_name) ごとに PQ 点群 BBOX を計算する。

    Returns:
        key -> {"w": 横幅, "h": 縦幅, "diag": 対角長}。対角長は 1.0 以上にクランプ。
    """
    eye_points: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for rec in records:
        key = (rec["image_path"], rec.get("face_index", 0), rec["part_name"])
        eye_points[key].append((float(rec["px"]), float(rec["py"])))
        eye_points[key].append((float(rec["qx"]), float(rec["qy"])))

    result: dict[tuple, dict[str, float]] = {}
    for key, pts in eye_points.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        diag = math.sqrt(w * w + h * h)
        result[key] = {
            "w": w,
            "h": h,
            "diag": max(diag, 1.0),
        }
    return result


def load_img_diag_map(image_paths: list[str]) -> dict[str, float]:
    diag_map: dict[str, float] = {}
    for path in image_paths:
        try:
            with Image.open(path) as img:
                w, h = img.size
                diag_map[path] = math.sqrt(w * h)
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

    eye_bbox_map = compute_eye_bbox(records)

    unique_paths = list({rec["image_path"] for rec in records})
    print(f"Loading image sizes for {len(unique_paths)} images...")
    diag_map = load_img_diag_map(unique_paths)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = DictWriter(f, fieldnames=OUTPUT_COLS, extrasaction="ignore")
        writer.writeheader()

        for rec in records:
            key = (rec["image_path"], rec.get("face_index", 0), rec["part_name"])
            bbox = eye_bbox_map.get(key)
            eye_scale = bbox["diag"] if bbox else None
            eye_bbox_w = bbox["w"] if bbox else None
            eye_bbox_h = bbox["h"] if bbox else None
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
                "skip_flag":      rec.get("skip_flag", 0),
            })

    print(f"Exported {len(records)} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
