"""復元系テスト（test_restore_face_pose / test_restore_dataset）共通の合成データ。

実際の FaceMesh・画像の代わりに、目の輪郭が楕円上にある合成ランドマークと、その周囲に
P/Q を置いた manifest 行、画像幅で応答を切り替えるフェイク FaceMesh を提供する。
"""
from __future__ import annotations

import gc
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from annotator import db as sample_db  # noqa: E402
from annotator.face_pose import compute_face_orientation  # noqa: E402
from tools.data_management.manifest_source import ManifestRow  # noqa: E402
from tools.sampledb_maintenance import rebuild_samples_db as rb  # noqa: E402
from tools.sampledb_maintenance import restore_face_pose as rfp  # noqa: E402

NUM_LANDMARKS = 478
IMG_H = 240
BASE_WIDTH = 320
EYE_RX, EYE_RY = 0.04, 0.02          # 目の輪郭の楕円半径（正規化座標）
EYE_CENTERS = {"eyeliner_left": (0.30, 0.40), "eyeliner_right": (0.70, 0.40)}


# ---------------------------------------------------------------------------
# 合成データ
# ---------------------------------------------------------------------------

def make_face(dx=0.0, dy=0.0, yaw_z=0.0, chin_z=0.0) -> list[SimpleNamespace]:
    """478 点のランドマーク。目の輪郭 16 点は楕円上、姿勢用の 5 点は幾何が縮退しない位置に置く。"""
    lm = [SimpleNamespace(x=0.5 + dx, y=0.5 + dy, z=0.0) for _ in range(NUM_LANDMARKS)]
    for part, contour in rfp.EYE_CONTOUR_LANDMARKS.items():
        cx, cy = EYE_CENTERS[part]
        for k, index in enumerate(contour):
            angle = 2 * math.pi * k / len(contour)
            lm[index] = SimpleNamespace(
                x=cx + dx + EYE_RX * math.cos(angle), y=cy + dy + EYE_RY * math.sin(angle), z=0.0)
    lm[263].z = yaw_z                                        # 右目外側が奥 → yaw
    lm[10] = SimpleNamespace(x=0.5 + dx, y=0.2 + dy, z=0.0)  # 額
    lm[152] = SimpleNamespace(x=0.5 + dx, y=0.8 + dy, z=chin_z)  # 顎 → pitch
    lm[1] = SimpleNamespace(x=0.5 + dx, y=0.5 + dy, z=0.0)   # 鼻先
    return lm


def expected_pose(landmarks, width):
    xyz = np.array([[p.x * width, p.y * IMG_H, p.z * width] for p in landmarks], dtype=np.float64)
    return compute_face_orientation(xyz, width, IMG_H)


def manifest_rows(image_id, width, *, face_index=0, dx=0.0, dy=0.0, offset_px=(0.0, 0.0),
                  start_idx=0, parts=("eyeliner_left", "eyeliner_right")) -> list[ManifestRow]:
    """目の輪郭の周囲に P/Q 14 点ずつ。重心は輪郭中心 + offset_px になる。"""
    rows, idx = [], start_idx
    for part in parts:
        cx = (EYE_CENTERS[part][0] + dx) * width + offset_px[0]
        cy = (EYE_CENTERS[part][1] + dy) * IMG_H + offset_px[1]
        for sub_key, base_angle in (("upper", math.pi), ("lower", 0.0)):
            for k in range(7):
                angle = base_angle + math.pi * (k + 0.5) / 7
                rx, ry = EYE_RX * width, EYE_RY * IMG_H
                rows.append(ManifestRow(
                    idx=idx, image_id=image_id, image_path=f"data/images/{image_id}.jpg",
                    face_index=face_index, part_name=part, sub_key=sub_key, point_index=k,
                    px=cx + 0.9 * rx * math.cos(angle), py=cy + 0.9 * ry * math.sin(angle),
                    qx=cx + 1.4 * rx * math.cos(angle), qy=cy + 1.4 * ry * math.sin(angle),
                    t=0.5, nose_x=None, nose_y=None))
                idx += 1
    return rows


def write_image(images_dir: Path, image_id: str, width: int, seed=0) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(IMG_H, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(images_dir / f"{image_id}.jpg"), pixels)


class FakeFaceMesh:
    """画像幅で応答を切り替える FaceMesh。`faces_by_width[width]` = 検出する顔のリスト。"""

    def __init__(self, faces_by_width):
        self._faces_by_width = faces_by_width
        self.processed_widths: list[int] = []

    def process(self, rgb):
        width = rgb.shape[1]
        self.processed_widths.append(width)
        faces = self._faces_by_width.get(width, [])
        return SimpleNamespace(
            multi_face_landmarks=[SimpleNamespace(landmark=f) for f in faces] if faces else None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_factory(faces_by_width):
    mesh = FakeFaceMesh(faces_by_width)
    factory = lambda: mesh  # noqa: E731
    factory.mesh = mesh
    return factory


def build_db(tmp_path: Path, rows) -> Path:
    db_path = tmp_path / "samples.db"
    rb.rebuild_samples_db(rows, db_path, images_dir=None)
    return db_path


def db_rows(db_path: Path) -> dict[int, dict]:
    rows = {s["idx"]: s for s in sample_db.load_all_samples(db_path)}
    gc.collect()
    return rows
