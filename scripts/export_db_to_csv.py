"""DB を指定カラム形式の CSV に出力する。

eye_scale / norm_eye_scale はデータから計算。
pred_* / true_t_norm / rx_pred / ry_pred は推論が必要なため空。
"""
import math
import os
import sqlite3
from collections import defaultdict
from csv import DictWriter

from PIL import Image

DB_PATH = "data/samples.db"
OUTPUT_PATH = "data/samples.csv"

OUTPUT_COLS = [
    "idx", "image_path","face_index", "part_name", "sub_key", "point_index",
    "px", "py", "qx", "qy",
    "true_t_raw", 
    "eye_scale", "norm_eye_scale",
    "skip_flag"
]


def load_records() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM samples")
    records = [dict(r) for r in cur.fetchall()]
    conn.close()
    return records


def compute_eye_scale(records: list[dict]) -> dict[tuple, float]:
    eye_points: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for rec in records:
        key = (rec["image_path"], rec.get("face_index", 0), rec["part_name"])
        eye_points[key].append((float(rec["px"]), float(rec["py"])))
        eye_points[key].append((float(rec["qx"]), float(rec["qy"])))

    eye_scale_map: dict[tuple, float] = {}
    for key, pts in eye_points.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        scale = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)
        eye_scale_map[key] = max(scale, 1.0)
    return eye_scale_map


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


def main() -> None:
    records = load_records()
    print(f"Loaded {len(records)} records")

    eye_scale_map = compute_eye_scale(records)

    unique_paths = list({rec["image_path"] for rec in records})
    print(f"Loading image sizes for {len(unique_paths)} images...")
    diag_map = load_img_diag_map(unique_paths)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = DictWriter(f, fieldnames=OUTPUT_COLS, extrasaction="ignore")
        writer.writeheader()

        for rec in records:
            key = (rec["image_path"], rec.get("face_index", 0), rec["part_name"])
            eye_scale = eye_scale_map.get(key, "")
            img_diag = diag_map.get(rec["image_path"], 0.0)
            norm_eye_scale = eye_scale / img_diag if img_diag and eye_scale != "" else ""

            writer.writerow({
                "idx":            rec["idx"],
                "image_path":     rec["image_path"],
                "face_index":     rec.get("face_index", 0),  # ←追加
                "part_name":      rec["part_name"],
                "sub_key":        rec["sub_key"],
                "point_index":    rec["point_index"],
                "px":             rec["px"],
                "py":             rec["py"],
                "qx":             rec["qx"],
                "qy":             rec["qy"],
                "true_t_raw":     rec["t"] if rec["t"] is not None else "",
                "eye_scale":      f"{eye_scale:.6f}" if eye_scale != "" else "",
                "norm_eye_scale": f"{norm_eye_scale:.6f}" if norm_eye_scale != "" else "",
                "skip_flag":      rec.get("skip_flag", 0), 
            })

    print(f"Exported {len(records)} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
