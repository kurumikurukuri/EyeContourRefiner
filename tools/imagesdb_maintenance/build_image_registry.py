"""画像ディレクトリ全ファイルのレジストリ DB を構築する。

既知画像（samples.db に存在）は DB からデータを移植。
未知画像のみ MediaPipe で顔検出 + 特徴計算する。
一度作れば以降のフィルタ操作は SQL 一発、MediaPipe 不要。

Usage:
  python tools/imagesdb_maintenance/build_image_registry.py                          # 全画像スキャン
  python tools/imagesdb_maintenance/build_image_registry.py --workers 4              # 並列数指定
  python tools/imagesdb_maintenance/build_image_registry.py --force                  # 再スキャン（既存DB破棄）
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from annotator import db as _db  # noqa: E402
from annotator.face_pose import compute_face_orientation  # noqa: E402

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
IMAGES_DIR = ROOT / "data" / "images"
SAMPLES_DB = ROOT / "data" / "samples.db"
ENRICHED_CSV = ROOT / "data" / "samples_enriched.csv"
REGISTRY_DB = ROOT / "data" / "images.db"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_LEFT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
_RIGHT_EYE_CONTOUR = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    image_path  TEXT PRIMARY KEY,
    image_stem  TEXT NOT NULL,
    has_face    INTEGER NOT NULL DEFAULT 0,
    max_eye_scale REAL,
    max_lap_var   REAL,
    min_face_yaw  REAL,
    min_face_pitch REAL,
    min_face_roll  REAL,
    n_faces        INTEGER,
    min_eye_brightness REAL,
    max_eye_brightness REAL,
    source      TEXT
);
CREATE INDEX IF NOT EXISTS idx_has_face ON images(has_face);
CREATE INDEX IF NOT EXISTS idx_source ON images(source);
"""


# ---------------------------------------------------------------------------
# Step 1: list all image files
# ---------------------------------------------------------------------------

def list_all_images() -> list[Path]:
    paths = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    log.info("image files in dir: %d", len(paths))
    return paths


# ---------------------------------------------------------------------------
# Step 2: populate from samples.db
# ---------------------------------------------------------------------------

def populate_from_db(conn: sqlite3.Connection) -> int:
    """samples.db から既知画像の has_face=1, max_eye_scale を UPDATE する。"""
    if not SAMPLES_DB.exists():
        log.warning("samples.db not found, skipping")
        return 0

    max_scale = _db.get_max_eye_scale_per_image(SAMPLES_DB)
    if not max_scale:
        return 0

    # image_path は "data/images/xxx.jpg" → basename だけ抽出してマッチ
    rows = [(float(scale), Path(ip).name) for ip, scale in max_scale.items()]
    conn.executemany(
        "UPDATE images SET has_face = 1, max_eye_scale = ?, source = 'db' WHERE image_path = ?",
        rows,
    )
    conn.commit()
    log.info("populated from samples.db: %d images", len(rows))
    return len(rows)


# ---------------------------------------------------------------------------
# Step 3: populate from enriched CSV
# ---------------------------------------------------------------------------

def populate_from_csv(conn: sqlite3.Connection) -> int:
    """enriched CSV から max_lap_var を UPDATE する。"""
    if not ENRICHED_CSV.exists():
        log.warning("enriched CSV not found, skipping")
        return 0

    import pandas as pd
    df = pd.read_csv(ENRICHED_CSV, usecols=["image_path", "eye_laplacian_var"])
    # image_path は relative、basename でマッチさせる
    # DB内の image_path は "data/images/xxx.jpg" 形式

    per_img = df.groupby("image_path")["eye_laplacian_var"].max()

    # basename → max_lap_var のマップ
    basename_to_lap: dict[str, float] = {}
    for ip, val in per_img.items():
        bn = Path(ip).name
        if bn not in basename_to_lap or val > basename_to_lap[bn]:
            basename_to_lap[bn] = val

    # UPDATE by basename
    cur = conn.cursor()
    updated = 0
    for bn, val in basename_to_lap.items():
        cur.execute(
            "UPDATE images SET max_lap_var = ? WHERE image_path = ?",
            (float(val), bn),
        )
        if cur.rowcount > 0:
            updated += 1
    conn.commit()
    log.info("populated max_lap_var from CSV: %d images", updated)
    return updated


# ---------------------------------------------------------------------------
# Step 4: MediaPipe scan for unknown images (2-phase)
# ---------------------------------------------------------------------------

def scan_phase1_detect_faces(conn: sqlite3.Connection, workers: int = 1) -> int:
    """Phase 1: source IS NULL の画像に FaceDetection のみ実行 → has_face を設定。

    MediaPipe を一度だけ初期化し、全画像をループで処理する。
    """
    try:
        import mediapipe as mp
    except ImportError:
        log.error("mediapipe not installed")
        return 0

    cur = conn.cursor()
    unknowns = cur.execute(
        "SELECT image_path FROM images WHERE source IS NULL"
    ).fetchall()

    if not unknowns:
        log.info("no unknown images for phase 1")
        return 0

    paths = [row[0] for row in unknowns]
    total = len(paths)
    log.info("Phase 1: FaceDetection on %d images", total)

    mp_face_detection = mp.solutions.face_detection
    detector = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5,
    )

    updated = 0
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        image_path = IMAGES_DIR / p
        bgr = cv2.imread(str(image_path))
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            det = detector.process(rgb)
            has_face = 1 if det.detections else 0
        else:
            has_face = 0

        conn.execute(
            "UPDATE images SET has_face = ?, source = 'mp-phase1' WHERE image_path = ?",
            (has_face, p),
        )
        updated += 1

        if i % 500 == 0:
            conn.commit()
            elapsed = time.time() - t0
            rate = i / elapsed
            eta_min = (total - i) / rate / 60 if rate > 0 else 0
            log.info("  phase1: %d/%d (%.0f%%)  %.1f img/s  ETA %.0f min", i, total, i / total * 100, rate, eta_min)

    detector.close()
    conn.commit()

    n_faces = conn.execute(
        "SELECT COUNT(*) FROM images WHERE source = 'mp-phase1' AND has_face = 1"
    ).fetchone()[0]
    elapsed = time.time() - t0
    log.info("Phase 1 done: %d scanned, %d faces, %.1f min", updated, n_faces, elapsed / 60)
    return updated


def _extract_features_one(image_path_str: str) -> dict:
    """1 画像から全特徴量を抽出する（FaceMesh + eye_scale + lap + yaw）。"""
    image_path = IMAGES_DIR / image_path_str
    result = {
        "image_path": image_path_str,
        "has_face": 1,
        "max_eye_scale": None,
        "max_lap_var": None,
        "min_face_yaw": None,
        "min_face_pitch": None,
        "min_face_roll": None,
        "n_faces": None,
        "source": "mediapipe",
    }
    try:
        import mediapipe as mp
    except ImportError:
        return result

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        result["has_face"] = 0
        return result

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = rgb.shape[:2]

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=10,
        refine_landmarks=True,
    ) as face_mesh:
        fmesh = face_mesh.process(rgb)

    if not fmesh.multi_face_landmarks:
        result["has_face"] = 0
        return result

    face_landmarks_list = list(fmesh.multi_face_landmarks)

    # eye_scale
    eye_scales: list[float] = []
    for lm_list in face_landmarks_list:
        for eye_indices in (_LEFT_EYE_CONTOUR, _RIGHT_EYE_CONTOUR):
            pts = np.array(
                [[lm_list.landmark[i].x * img_w, lm_list.landmark[i].y * img_h]
                 for i in eye_indices], dtype=np.float64)
            xs = pts[:, 0]; ys = pts[:, 1]
            scale = np.sqrt((xs.max() - xs.min()) ** 2 + (ys.max() - ys.min()) ** 2)
            eye_scales.append(max(float(scale), 1.0))
    if eye_scales:
        result["max_eye_scale"] = max(eye_scales)

    # laplacian
    lap_vars: list[float] = []
    for lm_list in face_landmarks_list:
        for eye_indices in (_LEFT_EYE_CONTOUR, _RIGHT_EYE_CONTOUR):
            pts = np.array(
                [[lm_list.landmark[i].x * img_w, lm_list.landmark[i].y * img_h]
                 for i in eye_indices], dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            margin = int(min(w, h) * 0.2)
            x = max(0, x - margin); y = max(0, y - margin)
            w = min(img_w - x, w + 2 * margin); h = min(img_h - y, h + 2 * margin)
            if w <= 0 or h <= 0:
                continue
            patch = rgb[y:y + h, x:x + w]
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            lap_vars.append(float(lap.var()))
    if lap_vars:
        result["max_lap_var"] = max(lap_vars)

    # face_yaw / pitch / roll / n_faces
    yaws: list[float] = []
    pitches: list[float] = []
    rolls: list[float] = []
    for lm_list in face_landmarks_list:
        landmarks_xyz = np.array(
            [[lm.x * img_w, lm.y * img_h, lm.z * img_w] for lm in lm_list.landmark],
            dtype=np.float64)
        orient = compute_face_orientation(landmarks_xyz, img_w, img_h)
        if orient is not None:
            yaws.append(abs(orient[0]))
            pitches.append(abs(orient[1]))
            rolls.append(abs(orient[2]))
    if yaws:
        result["min_face_yaw"] = min(yaws)
    if pitches:
        result["min_face_pitch"] = min(pitches)
    if rolls:
        result["min_face_roll"] = min(rolls)
    result["n_faces"] = len(face_landmarks_list)

    return result


def _extract_features_from_rgb(rgb: np.ndarray, face_mesh) -> dict:
    """Pre-loaded RGB 画像から特徴量抽出（FaceMesh インスタンスは外部から注入）。"""
    img_h, img_w = rgb.shape[:2]
    result = {
        "has_face": 1,
        "max_eye_scale": None,
        "max_lap_var": None,
        "min_face_yaw": None,
        "min_face_pitch": None,
        "min_face_roll": None,
        "n_faces": None,
        "min_eye_brightness": None,
        "max_eye_brightness": None,
    }

    fmesh = face_mesh.process(rgb)

    if not fmesh.multi_face_landmarks:
        result["has_face"] = 0
        return result

    face_landmarks_list = list(fmesh.multi_face_landmarks)

    # eye_scale
    eye_scales: list[float] = []
    for lm_list in face_landmarks_list:
        for eye_indices in (_LEFT_EYE_CONTOUR, _RIGHT_EYE_CONTOUR):
            pts = np.array(
                [[lm_list.landmark[i].x * img_w, lm_list.landmark[i].y * img_h]
                 for i in eye_indices], dtype=np.float64)
            xs = pts[:, 0]; ys = pts[:, 1]
            scale = np.sqrt((xs.max() - xs.min()) ** 2 + (ys.max() - ys.min()) ** 2)
            eye_scales.append(max(float(scale), 1.0))
    if eye_scales:
        result["max_eye_scale"] = max(eye_scales)

    # laplacian
    lap_vars: list[float] = []
    for lm_list in face_landmarks_list:
        for eye_indices in (_LEFT_EYE_CONTOUR, _RIGHT_EYE_CONTOUR):
            pts = np.array(
                [[lm_list.landmark[i].x * img_w, lm_list.landmark[i].y * img_h]
                 for i in eye_indices], dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            margin = int(min(w, h) * 0.2)
            x = max(0, x - margin); y = max(0, y - margin)
            w = min(img_w - x, w + 2 * margin); h = min(img_h - y, h + 2 * margin)
            if w <= 0 or h <= 0:
                continue
            patch = rgb[y:y + h, x:x + w]
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            lap_vars.append(float(lap.var()))
    if lap_vars:
        result["max_lap_var"] = max(lap_vars)

    # eye brightness
    #   max_eye_brightness = max(全眼): 最も明るい目でも暗い → サングラス検出
    #   min_eye_brightness = min(全眼): 最も暗い目でも明るい → 露出過多検出
    all_eye_vals: list[float] = []
    for lm_list in face_landmarks_list:
        for eye_indices in (_LEFT_EYE_CONTOUR, _RIGHT_EYE_CONTOUR):
            pts = np.array(
                [[lm_list.landmark[i].x * img_w, lm_list.landmark[i].y * img_h]
                 for i in eye_indices], dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            patch = rgb[y1:y2, x1:x2]
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            all_eye_vals.append(float(gray.mean()))
    if all_eye_vals:
        result["max_eye_brightness"] = max(all_eye_vals)
        result["min_eye_brightness"] = min(all_eye_vals)

    # face_yaw / pitch / roll / n_faces
    yaws: list[float] = []
    pitches: list[float] = []
    rolls: list[float] = []
    for lm_list in face_landmarks_list:
        landmarks_xyz = np.array(
            [[lm.x * img_w, lm.y * img_h, lm.z * img_w] for lm in lm_list.landmark],
            dtype=np.float64)
        orient = compute_face_orientation(landmarks_xyz, img_w, img_h)
        if orient is not None:
            yaws.append(abs(orient[0]))
            pitches.append(abs(orient[1]))
            rolls.append(abs(orient[2]))
    if yaws:
        result["min_face_yaw"] = min(yaws)
    if pitches:
        result["min_face_pitch"] = min(pitches)
    if rolls:
        result["min_face_roll"] = min(rolls)
    result["n_faces"] = len(face_landmarks_list)

    return result


def _process_batch(image_paths: list[str]) -> list[tuple[str, dict]]:
    """1 スレッドで使う FaceMesh を作り、画像バッチを処理する。"""
    import mediapipe as mp

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=10,
        refine_landmarks=True,
    )
    results: list[tuple[str, dict]] = []
    for p in image_paths:
        image_path = IMAGES_DIR / p
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            results.append((p, {"has_face": 0, "max_eye_scale": None, "max_lap_var": None, "min_face_yaw": None}))
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            r = _extract_features_from_rgb(rgb, face_mesh)
            results.append((p, r))
    face_mesh.close()
    return results


def scan_phase2_features(conn: sqlite3.Connection, workers: int = 1) -> int:
    """Phase 2: has_face=1 かつ source='mp-phase1' の画像に特徴量抽出。

    workers >= 2 の場合は ThreadPoolExecutor で並列処理する。
    各スレッドが独自の FaceMesh インスタンスを持つ。
    """
    cur = conn.cursor()
    targets = cur.execute(
        "SELECT image_path FROM images WHERE has_face = 1 AND source = 'mp-phase1'"
    ).fetchall()

    if not targets:
        log.info("no targets for phase 2")
        return 0

    paths = [row[0] for row in targets]
    total = len(paths)
    log.info("Phase 2: feature extraction on %d images (workers=%d)", total, workers)

    updated = 0
    t0 = time.time()

    if workers <= 1:
        # シングルスレッド: 従来通りのループ
        try:
            import mediapipe as mp
        except ImportError:
            log.error("mediapipe not installed")
            return 0

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,
        )
        for i, p in enumerate(paths, 1):
            image_path = IMAGES_DIR / p
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                result: dict = {"has_face": 0, "max_eye_scale": None, "max_lap_var": None, "min_face_yaw": None}
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                result = _extract_features_from_rgb(rgb, face_mesh)

            conn.execute(
                """UPDATE images SET
                    has_face = ?, max_eye_scale = ?, max_lap_var = ?,
                    min_face_yaw = ?, min_face_pitch = ?, min_face_roll = ?,
                    n_faces = ?, min_eye_brightness = ?, max_eye_brightness = ?, source = ?
                WHERE image_path = ?""",
                (result["has_face"], result["max_eye_scale"],
                 result["max_lap_var"], result["min_face_yaw"],
                 result.get("min_face_pitch"), result.get("min_face_roll"),
                 result.get("n_faces"), result.get("min_eye_brightness"),
                 result.get("max_eye_brightness"), "mediapipe", p),
            )
            updated += 1
            if i % 500 == 0:
                conn.commit()
                elapsed = time.time() - t0
                rate = i / elapsed
                eta_min = (total - i) / rate / 60 if rate > 0 else 0
                log.info("  phase2: %d/%d (%.0f%%)  %.1f img/s  ETA %.0f min", i, total, i / total * 100, rate, eta_min)

        face_mesh.close()

    else:
        # マルチスレッド
        from concurrent.futures import ThreadPoolExecutor, as_completed

        chunk_size = max(50, len(paths) // (workers * 10))
        chunks = [paths[i:i + chunk_size] for i in range(0, len(paths), chunk_size)]
        log.info("  %d chunks of ~%d images each", len(chunks), chunk_size)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_batch, chunk): i for i, chunk in enumerate(chunks)}

            for future in as_completed(futures):
                for p, result in future.result():
                    conn.execute(
                        """UPDATE images SET
                            has_face = ?, max_eye_scale = ?, max_lap_var = ?,
                            min_face_yaw = ?, min_face_pitch = ?, min_face_roll = ?,
                            n_faces = ?, min_eye_brightness = ?, max_eye_brightness = ?, source = ?
                        WHERE image_path = ?""",
                        (result["has_face"], result["max_eye_scale"],
                         result["max_lap_var"], result["min_face_yaw"],
                         result.get("min_face_pitch"), result.get("min_face_roll"),
                         result.get("n_faces"), result.get("min_eye_brightness"),
                         result.get("max_eye_brightness"), "mediapipe", p),
                    )
                    updated += 1

                if updated % 1000 == 0:
                    conn.commit()
                    elapsed = time.time() - t0
                    rate = updated / elapsed
                    eta_min = (total - updated) / rate / 60 if rate > 0 else 0
                    log.info("  phase2: %d/%d (%.0f%%)  %.1f img/s  ETA %.0f min", updated, total, updated / total * 100, rate, eta_min)

    conn.commit()

    elapsed = time.time() - t0
    log.info("Phase 2 done: %d images in %.1f min (%.1f img/s)", updated, elapsed / 60, updated / elapsed)
    return updated


# ---------------------------------------------------------------------------
# Phase 3: pitch / roll / n_faces / min_eye_brightness を既存画像に補完
# ---------------------------------------------------------------------------

def _process_batch_pose(image_paths: list[str]) -> list[tuple[str, dict]]:
    """Phase 3 用バッチワーカー。pose / brightness のみを返す。"""
    import mediapipe as mp

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=10,
        refine_landmarks=True,
    )
    results: list[tuple[str, dict]] = []
    for p in image_paths:
        image_path = IMAGES_DIR / p
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            results.append((p, {}))
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            r = _extract_features_from_rgb(rgb, face_mesh)
            results.append((p, r))
    face_mesh.close()
    return results


def scan_phase3_pose(conn: sqlite3.Connection, workers: int = 1) -> int:
    """Phase 3: has_face=1 かつ min_face_pitch IS NULL の画像に pose/brightness を補完。

    既存の has_face / max_eye_scale / max_lap_var / min_face_yaw は上書きしない。
    """
    cur = conn.cursor()
    targets = cur.execute(
        "SELECT image_path FROM images WHERE has_face = 1"
        " AND (min_face_pitch IS NULL OR min_eye_brightness IS NULL OR max_eye_brightness IS NULL)"
    ).fetchall()

    if not targets:
        log.info("no targets for phase 3 (all images already have pose data)")
        return 0

    paths = [row[0] for row in targets]
    total = len(paths)
    log.info("Phase 3: pose/brightness patch on %d images (workers=%d)", total, workers)

    updated = 0
    t0 = time.time()

    def _patch(conn: sqlite3.Connection, p: str, r: dict) -> None:
        conn.execute(
            """UPDATE images SET
                min_face_pitch = ?, min_face_roll = ?,
                n_faces = ?, min_eye_brightness = ?, max_eye_brightness = ?
            WHERE image_path = ?""",
            (r.get("min_face_pitch"), r.get("min_face_roll"),
             r.get("n_faces"), r.get("min_eye_brightness"),
             r.get("max_eye_brightness"), p),
        )

    if workers <= 1:
        try:
            import mediapipe as mp
        except ImportError:
            log.error("mediapipe not installed")
            return 0

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,
        )
        for i, p in enumerate(paths, 1):
            image_path = IMAGES_DIR / p
            bgr = cv2.imread(str(image_path))
            r: dict = {}
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                r = _extract_features_from_rgb(rgb, face_mesh)
            _patch(conn, p, r)
            updated += 1
            if i % 500 == 0:
                conn.commit()
                elapsed = time.time() - t0
                rate = i / elapsed
                eta_min = (total - i) / rate / 60 if rate > 0 else 0
                log.info("  phase3: %d/%d (%.0f%%)  %.1f img/s  ETA %.0f min",
                         i, total, i / total * 100, rate, eta_min)
        face_mesh.close()

    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        chunk_size = max(50, len(paths) // (workers * 10))
        chunks = [paths[i:i + chunk_size] for i in range(0, len(paths), chunk_size)]
        log.info("  %d chunks of ~%d images each", len(chunks), chunk_size)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_batch_pose, chunk): i
                       for i, chunk in enumerate(chunks)}

            for future in as_completed(futures):
                for p, r in future.result():
                    _patch(conn, p, r)
                    updated += 1

                if updated % 1000 == 0:
                    conn.commit()
                    elapsed = time.time() - t0
                    rate = updated / elapsed
                    eta_min = (total - updated) / rate / 60 if rate > 0 else 0
                    log.info("  phase3: %d/%d (%.0f%%)  %.1f img/s  ETA %.0f min",
                             updated, total, updated / total * 100, rate, eta_min)

    conn.commit()

    elapsed = time.time() - t0
    log.info("Phase 3 done: %d images in %.1f min (%.1f img/s)", updated, elapsed / 60, updated / elapsed)
    return updated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_registry(
    force: bool = False,
    workers: int = 1,
    phase: str = "all",
) -> sqlite3.Connection:
    t0 = time.time()

    if force and REGISTRY_DB.exists():
        REGISTRY_DB.unlink()
        log.info("removed existing registry")

    conn = sqlite3.connect(str(REGISTRY_DB))
    conn.executescript(SCHEMA)
    # 既存 DB への列追加マイグレーション
    for col_def in [
        "min_eye_brightness REAL",
        "max_eye_brightness REAL",
        "min_face_pitch REAL",
        "min_face_roll REAL",
        "n_faces INTEGER",
    ]:
        try:
            conn.execute(f"ALTER TABLE images ADD COLUMN {col_def}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # 列がすでに存在する場合は無視
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")

    # Step 1: insert all image files (INSERT OR IGNORE = safe for incremental)
    all_paths = list_all_images()
    conn.executemany(
        "INSERT OR IGNORE INTO images (image_path, image_stem, has_face) VALUES (?, ?, 0)",
        [(p.name, p.stem) for p in all_paths],
    )
    conn.commit()
    log.info("registry: %d images (%d on disk)", conn.execute("SELECT COUNT(*) FROM images").fetchone()[0], len(all_paths))

    # Step 2: populate from samples.db
    populate_from_db(conn)

    # Step 3: populate from enriched CSV
    populate_from_csv(conn)

    # Step 4: scan unknowns
    if phase in ("1", "all"):
        scan_phase1_detect_faces(conn, workers=workers)

    if phase in ("2", "all"):
        scan_phase2_features(conn, workers=workers)

    if phase in ("3", "all"):
        scan_phase3_pose(conn, workers=workers)

    # Stats
    stats = conn.execute(
        "SELECT source, COUNT(*) as cnt FROM images GROUP BY source"
    ).fetchall()
    log.info("registry stats: %s", {s: c for s, c in stats})

    elapsed = time.time() - t0
    log.info("build done in %.1f min", elapsed / 60)

    return conn


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except OSError:
        pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="画像レジストリ DB 構築")
    parser.add_argument("--force", action="store_true", help="既存 DB を破棄して再構築")
    parser.add_argument("--workers", type=int, default=1, help="並列数 (default: 1)")
    parser.add_argument(
        "--phase", choices=["1", "2", "3", "all"], default="all",
        help="1=顔検出のみ, 2=特徴量のみ, 3=pose/brightness補完のみ, all=すべて (default: all)",
    )
    args = parser.parse_args()

    build_registry(force=args.force, workers=args.workers, phase=args.phase)
