"""
MediaPipe FaceMesh 並列前処理スクリプト。

images.db にあり samples.db にない新規画像に対して FaceMesh を実行し、
中間 JSONL ファイルに結果を書き出す。全シャード完了後、samples.db にマージする。

使い方:
    # 4プロセスで全工程実行
    python tools/imagesdb_maintenance/preprocess_new_images.py --workers 4

    # 特定シャードのみ実行（手動並列用）
    python tools/imagesdb_maintenance/preprocess_new_images.py --shard 0 --total 4

    # 中間ファイルのマージのみ（再実行しても samples.db に既にある行は挿入しない）
    python tools/imagesdb_maintenance/preprocess_new_images.py --merge-only

    # マージ件数の確認のみ（samples.db は変更しない）
    python tools/imagesdb_maintenance/preprocess_new_images.py --merge-only --dry-run

    # 小規模テスト
    python tools/imagesdb_maintenance/preprocess_new_images.py --workers 2 --limit 100
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from annotator.face_pose import compute_face_orientation  # noqa: E402
from annotator.preprocessor import EYELINER_SAMPLES_MAP  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
IMAGES_DB = PROJECT_ROOT / "data" / "images.db"
SAMPLES_DB = PROJECT_ROOT / "data" / "samples.db"
SHARD_DIR = PROJECT_ROOT / "data"
BATCH_SIZE = 100


def _normalize_path(p: str) -> str:
    return Path(p).as_posix()


def get_new_image_paths() -> List[str]:
    """images.db にあって samples.db にない画像パスを返す。

    images.db にはファイル名のみ格納されているため、
    samples.db と同じ ``data/images/<name>`` 形式に変換する。
    """
    _require_samples_db()
    if not IMAGES_DB.exists():
        raise FileNotFoundError(f"images.db が見つかりません: {IMAGES_DB}")
    conn_img = sqlite3.connect(str(IMAGES_DB))
    conn_samp = sqlite3.connect(str(SAMPLES_DB))

    cur_s = conn_samp.cursor()
    cur_s.execute("SELECT DISTINCT image_path FROM samples")
    existing = set(r[0] for r in cur_s.fetchall())

    cur_i = conn_img.cursor()
    cur_i.execute("SELECT image_path FROM images WHERE has_face = 1")
    new_paths = [
        f"data/images/{r[0]}" for r in cur_i.fetchall()
        if f"data/images/{r[0]}" not in existing
    ]

    conn_img.close()
    conn_samp.close()
    return new_paths


def load_processed_paths(shard_path: Path) -> set[str]:
    """既存 JSONL から処理済み画像パスを読み込む（レジューム用）。"""
    if not shard_path.exists():
        return set()
    processed = set()
    with open(shard_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                processed.add(obj["image_path"])
            except (json.JSONDecodeError, KeyError):
                continue
    return processed


def process_image(
    face_mesh: Any,
    image_path_str: str,
) -> Optional[Dict[str, Any]]:
    """1枚の画像を FaceMesh で処理し、結果を返す。"""
    image_bgr = cv2.imread(image_path_str)
    if image_bgr is None:
        return None

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None

    all_coords: List[np.ndarray] = []
    all_coords_xyz: List[np.ndarray] = []
    for face_landmarks in results.multi_face_landmarks:
        coords = np.array(
            [[lm.x * w, lm.y * h] for lm in face_landmarks.landmark],
            dtype=np.float32,
        )
        coords_xyz = np.array(
            [[lm.x * w, lm.y * h, lm.z * w] for lm in face_landmarks.landmark],
            dtype=np.float64,
        )
        all_coords.append(coords)
        all_coords_xyz.append(coords_xyz)

    # face_index: x座標昇順の正準順序
    n = len(all_coords)
    centroids = [
        (float(np.mean(c[:, 0])), float(np.mean(c[:, 1]))) for c in all_coords
    ]
    order = sorted(range(n), key=lambda i: (centroids[i][0], centroids[i][1]))
    face_index_map = [0] * n
    for rank, det_i in enumerate(order):
        face_index_map[det_i] = rank

    faces = []
    for det_i, (coords, coords_xyz) in enumerate(zip(all_coords, all_coords_xyz)):
        face_index = face_index_map[det_i]
        pose = compute_face_orientation(coords_xyz, w, h)
        face_yaw = pose[0] if pose else None
        face_pitch = pose[1] if pose else None
        face_roll = pose[2] if pose else None

        samples = []
        for part_name, sub_map in EYELINER_SAMPLES_MAP.items():
            for sub_key, index_list in sub_map.items():
                point_index = 0
                for entry in index_list:
                    _, p_idx, q_idx = entry
                    px, py = coords[p_idx]
                    qx, qy = coords[q_idx]
                    samples.append({
                        "part_name": part_name,
                        "sub_key": sub_key,
                        "point_index": point_index,
                        "px": float(px),
                        "py": float(py),
                        "qx": float(qx),
                        "qy": float(qy),
                    })
                    point_index += 1

        faces.append({
            "face_index": face_index,
            "face_yaw": face_yaw,
            "face_pitch": face_pitch,
            "face_roll": face_roll,
            "samples": samples,
        })

    return {
        "image_path": _normalize_path(image_path_str),
        "faces": faces,
    }


def run_shard(
    shard_idx: int,
    image_paths: List[str],
    shard_path: Path,
    progress_queue: Optional[Queue] = None,
) -> None:
    """1シャードの処理を実行する。"""
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)s [shard-%(process)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        _run_shard_inner(shard_idx, image_paths, shard_path, progress_queue)
    except Exception as exc:
        _logging.getLogger(__name__).error("Shard %d failed: %s", shard_idx, exc, exc_info=True)
        raise


def _run_shard_inner(
    shard_idx: int,
    image_paths: List[str],
    shard_path: Path,
    progress_queue: Optional[Queue],
) -> None:
    import mediapipe as mp

    processed = load_processed_paths(shard_path)
    todo = [p for p in image_paths if _normalize_path(p) not in processed]
    logger.info(
        "Shard %d: %d images total, %d already done, %d to process",
        shard_idx, len(image_paths), len(processed), len(todo),
    )

    if not todo:
        logger.info("Shard %d: nothing to do", shard_idx)
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=10,
        refine_landmarks=True,
    )

    batch: List[str] = []
    done = len(processed)
    t0 = time.time()

    try:
        for i, img_path in enumerate(todo):
            result = process_image(face_mesh, img_path)
            if result is not None:
                batch.append(json.dumps(result, ensure_ascii=False))

            done += 1

            if len(batch) >= BATCH_SIZE:
                with open(shard_path, "a", encoding="utf-8") as f:
                    for line in batch:
                        f.write(line + "\n")
                batch.clear()

            if done % 1000 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(todo) - (done - len(processed))) / rate if rate > 0 else 0
                logger.info(
                    "Shard %d: %d/%d (%.1f%%) %.1f img/s ETA %.0fs",
                    shard_idx, done, len(todo),
                    done / len(todo) * 100, rate, eta,
                )
                if progress_queue is not None:
                    progress_queue.put((shard_idx, done, len(todo)))

        if batch:
            with open(shard_path, "a", encoding="utf-8") as f:
                for line in batch:
                    f.write(line + "\n")

        elapsed = time.time() - t0
        logger.info(
            "Shard %d: complete. %d images in %.1fs (%.1f img/s)",
            shard_idx, done, elapsed, done / elapsed if elapsed > 0 else 0,
        )
        if progress_queue is not None:
            progress_queue.put((shard_idx, done, len(todo)))

    finally:
        face_mesh.close()


NaturalKey = Tuple[str, int, str, str, int]

_INSERT_SQL = (
    "INSERT INTO samples "
    "(idx, image_path, face_index, part_name, sub_key, point_index, "
    "px, py, qx, qy, t, skip_flag, face_yaw, face_pitch, face_roll) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, ?, ?, ?)"
)


def _require_samples_db() -> None:
    """samples.db が無ければ中断する（sqlite3.connect が空 DB を作るのを防ぐ）。"""
    if not SAMPLES_DB.exists():
        raise FileNotFoundError(f"samples.db が見つかりません: {SAMPLES_DB}")


def _iter_shard_samples(shard_paths: List[Path]):
    """JSONL 中間ファイルから (natural_key, (px, py, qx, qy, yaw, pitch, roll)) を順に返す。"""
    for shard_path in shard_paths:
        if not shard_path.exists():
            logger.warning("Shard file not found: %s", shard_path)
            continue
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                for face in obj["faces"]:
                    pose = (face.get("face_yaw"), face.get("face_pitch"), face.get("face_roll"))
                    for s in face["samples"]:
                        key = (obj["image_path"], face["face_index"],
                               s["part_name"], s["sub_key"], s["point_index"])
                        yield key, (s["px"], s["py"], s["qx"], s["qy"], *pose)


def _load_existing_keys(conn: sqlite3.Connection) -> set[NaturalKey]:
    rows = conn.execute(
        "SELECT image_path, face_index, part_name, sub_key, point_index FROM samples"
    ).fetchall()
    return {tuple(r) for r in rows}


def _collect_new_rows(
    shard_paths: List[Path], existing_keys: set[NaturalKey], start_idx: int,
) -> Tuple[List[Tuple[Any, ...]], int]:
    """既存・シャード間重複の natural key を除いた INSERT 行と、スキップ件数を返す。"""
    seen = set(existing_keys)
    rows: List[Tuple[Any, ...]] = []
    skipped = 0
    for key, values in _iter_shard_samples(shard_paths):
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        rows.append((start_idx + len(rows), *key, *values))
    return rows, skipped


def merge_to_samples_db(shard_paths: List[Path], dry_run: bool = False) -> None:
    """JSONL 中間ファイルを samples.db にマージする。

    (image_path, face_index, part_name, sub_key, point_index) が既に samples.db にある行は
    挿入しないため、--merge-only のやり直しや全工程の再実行でも行は重複しない。
    """
    from annotator import db as _db

    _require_samples_db()
    if not dry_run:
        _db.init_db(SAMPLES_DB)  # 旧スキーマの DB に追加カラム（face_yaw 等）を補充する
    uri = f"{SAMPLES_DB.resolve().as_uri()}{'?mode=ro' if dry_run else ''}"
    conn = sqlite3.connect(uri, uri=True)
    try:
        row = conn.execute("SELECT MAX(idx) FROM samples").fetchone()
        next_idx = (row[0] or 0) + 1
        logger.info("Existing max idx: %d, new samples start from %d", next_idx - 1, next_idx)

        rows, skipped = _collect_new_rows(shard_paths, _load_existing_keys(conn), next_idx)
        logger.info("Merge: %d new samples, %d skipped (already in samples.db)", len(rows), skipped)
        if dry_run:
            logger.info("[dry-run] samples.db は変更しません")
            return

        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        with conn:
            conn.executemany(_INSERT_SQL, rows)
    finally:
        conn.close()
    logger.info("Merge complete: %d new samples inserted", len(rows))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(process)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="MediaPipe FaceMesh 並列前処理（中間JSONL → samples.db）"
    )
    parser.add_argument("--workers", type=int, default=4, help="並列プロセス数 (default: 4)")
    parser.add_argument("--shard", type=int, default=None, help="特定シャードのみ実行")
    parser.add_argument("--total", type=int, default=4, help="シャード総数 (default: 4)")
    parser.add_argument("--merge-only", action="store_true", help="中間ファイルのマージのみ")
    parser.add_argument("--limit", type=int, default=None, help="処理画像数上限（テスト用）")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="samples.db に書き込まず、マージされる件数だけ表示する（中間 JSONL は生成する）",
    )
    args = parser.parse_args()

    if args.merge_only:
        shard_paths = [
            SHARD_DIR / f"preprocess_shard_{i}.jsonl"
            for i in range(args.total)
        ]
        merge_to_samples_db(shard_paths, dry_run=args.dry_run)
        return

    logger.info("Getting new image paths...")
    new_paths = get_new_image_paths()
    logger.info("Found %d new images", len(new_paths))

    if args.limit:
        new_paths = new_paths[:args.limit]
        logger.info("Limited to %d images for testing", args.limit)

    if args.shard is not None:
        # 特定シャードのみ実行
        total = args.total
        shard_idx = args.shard
        chunk_size = len(new_paths) // total + (1 if len(new_paths) % total else 0)
        start = shard_idx * chunk_size
        end = min(start + chunk_size, len(new_paths))
        shard_paths_list = [new_paths[start:end]]
        shard_indices = [shard_idx]
    else:
        # 全シャード並列実行
        total = args.workers
        chunk_size = len(new_paths) // total + (1 if len(new_paths) % total else 0)
        shard_paths_list = []
        shard_indices = []
        for i in range(total):
            start = i * chunk_size
            end = min(start + chunk_size, len(new_paths))
            shard_paths_list.append(new_paths[start:end])
            shard_indices.append(i)

    progress_queue = Queue()
    processes = []
    for shard_idx, shard_images in zip(shard_indices, shard_paths_list):
        shard_path = SHARD_DIR / f"preprocess_shard_{shard_idx}.jsonl"
        p = Process(
            target=run_shard,
            args=(shard_idx, shard_images, shard_path, progress_queue),
        )
        p.start()
        processes.append(p)
        logger.info(
            "Started shard %d: %d images → %s",
            shard_idx, len(shard_images), shard_path,
        )

    for p in processes:
        p.join()

    logger.info("All shards complete.")

    # マージ
    shard_paths = [
        SHARD_DIR / f"preprocess_shard_{i}.jsonl"
        for i in range(total)
    ]
    merge_to_samples_db(shard_paths, dry_run=args.dry_run)

    # 最終状態を表示
    conn = sqlite3.connect(str(SAMPLES_DB))
    total_samples = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    total_images = conn.execute("SELECT COUNT(DISTINCT image_path) FROM samples").fetchone()[0]
    annotated = conn.execute("SELECT COUNT(*) FROM samples WHERE t IS NOT NULL").fetchone()[0]
    conn.close()
    logger.info(
        "Final state: %d samples, %d images, %d annotated",
        total_samples, total_images, annotated,
    )


if __name__ == "__main__":
    main()
