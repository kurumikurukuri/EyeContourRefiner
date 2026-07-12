"""EXP027 用: manifest 内の全画像から MediaPipe face mesh landmark #4（鼻先）を抽出し
DB の nose_x/nose_y カラムを UPDATE する。

使い方:
    python experiments/EXP027_perspective_aug/scr/extract_nose_coords.py
"""
from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
MANIFEST_DIR = EXP_DIR / "manifests"
DB_PATH = ROOT / "data" / "samples.db"
IMAGES_DIR = ROOT / "data" / "images"

LANDMARK_NOSE_TIP = 4  # MediaPipe face mesh index #4 = 鼻先


def collect_unique_images() -> set[str]:
    """train + test manifest からユニークな image_path を収集。"""
    images: set[str] = set()
    for manifest_name in ["manifest_train.csv", "manifest_test.csv"]:
        path = MANIFEST_DIR / manifest_name
        if not path.exists():
            print(f"[WARN] {path} が見つかりません。スキップ。")
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                images.add(row["image_path"])
    return images


def extract_nose_from_image(image_path_relative: str) -> dict[int, tuple[float, float]] | None:
    """1枚の画像から鼻先座標を抽出し face_index → (nose_x, nose_y) の dict で返す。

    MediaPipe が検出した顔の順序（multi_face_landmarks のインデックス）を
    そのまま face_index とみなす（DB の face_index と一致することが前提）。
    失敗時は None。
    """
    full_path = ROOT / image_path_relative
    if not full_path.exists():
        print(f"  [WARN] 画像なし: {full_path}")
        return None

    try:
        import mediapipe as mp
    except ImportError:
        print("[ERROR] mediapipe がインストールされていません。pip install mediapipe")
        return None

    image_bgr = cv2.imread(str(full_path))
    if image_bgr is None:
        print(f"  [WARN] 読込失敗: {full_path}")
        return None

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=5,
        refine_landmarks=True,
    ) as face_mesh:
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        print(f"  [WARN] 顔未検出: {image_path_relative}")
        return None

    nose_by_face: dict[int, tuple[float, float]] = {}
    for face_index, face_landmarks in enumerate(results.multi_face_landmarks):
        lm = face_landmarks.landmark[LANDMARK_NOSE_TIP]
        nose_by_face[face_index] = (lm.x * w, lm.y * h)

    if len(nose_by_face) > 1:
        print(f"  [INFO] {len(nose_by_face)} 顔検出: {image_path_relative}")
    return nose_by_face


def main() -> int:
    print("EXP027 鼻先座標抽出 & DB 更新")
    print(f"  DB: {DB_PATH}")
    print(f"  images dir: {IMAGES_DIR}")

    images = collect_unique_images()
    print(f"  対象画像: {len(images)} 枚")

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    # nose_x, nose_y カラムの存在確認
    existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(samples)")}
    for col in ("nose_x", "nose_y"):
        if col not in existing_cols:
            print(f"[ERROR] DB に {col} カラムがありません。先に annotator/preprocessor.py を実行してください。")
            return 1

    done = 0
    errors = 0
    total = len(images)

    for i, image_path in enumerate(sorted(images)):
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{total} (errors={errors})")

        nose_by_face = extract_nose_from_image(image_path)
        if nose_by_face is None:
            errors += 1
            continue

        for face_index, (nose_x, nose_y) in nose_by_face.items():
            conn.execute(
                "UPDATE samples SET nose_x = ?, nose_y = ? WHERE image_path = ? AND face_index = ?",
                (nose_x, nose_y, image_path, face_index),
            )
        done += 1

    conn.commit()
    conn.close()

    print(f"\n  完了: {done} 枚更新, {errors} エラー")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
