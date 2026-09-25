"""
顔画像からアイライン (P, Q) ペアを抽出し、アノテーション用サンプルを生成する。

使い方:
    python annotator/preprocessor.py \\
        --images data/images/ \\
        --db data/samples.db

既存の DB が存在する場合、t / skip_flag 値を保持してマージします（全置換のため、
生成サンプルが 0 件または既存件数から急減する場合は DB を更新せず中断します）。
face_index は既存 DB の PQ 重心と最近傍マッチングにより揃えます。重心から顔スケールの
半分を超えて離れた顔は別人とみなし、既存の face_index には対応付けません。
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from annotator import db as _db  # noqa: E402
from annotator.face_pose import compute_face_orientation  # noqa: E402
from eyeliner.data.eyeliner_landmarks import EYELINER_SAMPLE_PQ  # noqa: E402

logger = logging.getLogger(__name__)


def _normalize_path(p: str) -> str:
    """パス区切りを POSIX 形式（/）に正規化する。"""
    return Path(p).as_posix()


# サンプル生成用: {part_name: {sub_key: ((edge_id, p_idx, q_idx), ...)}}。
# 並び順が point_index（0〜6）に対応する。tools/imagesdb_maintenance も参照する。
EYELINER_SAMPLES_MAP = EYELINER_SAMPLE_PQ


_MP_OUTER_EYE_CORNERS = (33, 263)

# 検出顔を既存 face_index と同一人物とみなす最大距離（顔スケールに対する比）。
# 同じ画像を再検出した顔の重心はほぼ一致し、別の顔との距離は実測で外眼角間距離の 1.03 倍以上
# （data/images の複数顔 92 枚・123 ペアの最小値）だったので、その半分を上限にする。
_FACE_MATCH_MAX_DIST_RATIO = 0.5


def _face_centroid(coords: np.ndarray) -> Tuple[float, float]:
    """検出顔のアイライン重心（全 PQ 中点の平均）を返す。

    DB 側の重心（_build_existing_centroids）と同じ統計量にして、両者を直接比較できるようにする。
    全ランドマーク平均だと同一顔でも外眼角間距離の 0.27〜0.74 倍ずれ、距離閾値が使えない。
    """
    mids = np.array([
        (coords[p] + coords[q]) / 2
        for sub_map in EYELINER_SAMPLES_MAP.values()
        for entries in sub_map.values()
        for _, p, q in entries
    ])
    return float(np.mean(mids[:, 0])), float(np.mean(mids[:, 1]))


def _face_scale(coords: np.ndarray) -> float:
    """検出顔の大きさの目安として、外眼角間距離（ランドマーク 33 - 263）を返す。"""
    left, right = _MP_OUTER_EYE_CORNERS
    return float(np.linalg.norm(coords[left] - coords[right]))


def _match_face_order(
    face_coords_list: List[np.ndarray],
    existing_centroids: Dict[int, Tuple[float, float]],
) -> List[int]:
    """検出された顔を既存 DB の face_index に最近傍マッチングで対応付ける。

    既存 face_index の重心から検出顔の顔スケール × _FACE_MATCH_MAX_DIST_RATIO を超えて離れた
    顔は対応付けない。複数人の画像で一方が検出漏れしたとき、残った別人の顔に既存の
    face_index（と t / skip_flag）が付け替わるのを防ぐ。

    Args:
        face_coords_list: MediaPipe が返した顔ごとの (478, 2) ランドマーク配列リスト。
        existing_centroids: 既存 DB の {face_index: (cx, cy)} マップ。

    Returns:
        detect_to_face_index[i] = 検出順 i の顔に割り当てる face_index。
        既存 face_index に対応付けられなかった顔と、既存データがない画像の顔には、
        既存最大 + 1 以降（既存データがなければ x 座標昇順の 0, 1, 2, …）を割り当てる。
    """
    n = len(face_coords_list)
    new_centroids = [_face_centroid(c) for c in face_coords_list]
    max_dists = [_face_scale(c) * _FACE_MATCH_MAX_DIST_RATIO for c in face_coords_list]

    if not existing_centroids:
        order = sorted(range(n), key=lambda i: (new_centroids[i][0], new_centroids[i][1]))
        result = [0] * n
        for rank, det_i in enumerate(order):
            result[det_i] = rank
        return result

    # 既存 face_index -> 閾値内で最近傍の検出顔 をグリーディに割り当て
    detect_to_face: Dict[int, int] = {}
    used_detect: set = set()

    for ex_idx in sorted(existing_centroids.keys()):
        ex_c = existing_centroids[ex_idx]
        best_i: Optional[int] = None
        best_dist = float("inf")
        for det_i, nc in enumerate(new_centroids):
            if det_i in used_detect:
                continue
            d = (nc[0] - ex_c[0]) ** 2 + (nc[1] - ex_c[1]) ** 2
            if d > max_dists[det_i] ** 2:
                continue  # 顔スケールに対して遠すぎる → 別人
            if d < best_dist:
                best_dist = d
                best_i = det_i
        if best_i is not None:
            detect_to_face[best_i] = ex_idx
            used_detect.add(best_i)

    # マッチしなかった新顔には既存最大 + 1 以降を割り当て
    next_idx = max(existing_centroids.keys(), default=-1) + 1
    result = []
    for i in range(n):
        if i in detect_to_face:
            result.append(detect_to_face[i])
        else:
            result.append(next_idx)
            next_idx += 1
    return result


def _build_existing_centroids(
    image_path_posix: str,
    existing: List[Dict[str, Any]],
) -> Dict[int, Tuple[float, float]]:
    """既存 DB レコードから image_path ごとの {face_index: (cx, cy)} を構築する。

    cx, cy は同一 face の全 PQ 中点の平均。
    """
    face_points: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for s in existing:
        if _normalize_path(s["image_path"]) != image_path_posix:
            continue
        fi = s.get("face_index", 0)
        face_points[fi].append(((s["px"] + s["qx"]) / 2, (s["py"] + s["qy"]) / 2))

    centroids: Dict[int, Tuple[float, float]] = {}
    for fi, pts in face_points.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        centroids[fi] = (sum(xs) / len(xs), sum(ys) / len(ys))
    return centroids


def _extract_samples_from_image(
    image_path: Path,
    existing_centroids: Dict[int, Tuple[float, float]],
) -> Optional[List[Dict[str, Any]]]:
    """
    1枚の画像から全アイライン (P, Q) ペアを抽出する。

    Args:
        image_path: 顔画像のパス。
        existing_centroids: 既存 DB の {face_index: (cx, cy)}。
            空の場合は x 座標昇順で face_index を割り当てる。

    Returns:
        サンプルのリスト、またはランドマーク取得失敗時は None。
    """
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "mediapipe が見つかりません。`pip install mediapipe` でインストールしてください。"
        ) from exc

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        logger.warning("画像を読み込めません: %s", image_path)
        return None

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=10,
        refine_landmarks=True,
    ) as face_mesh:
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        logger.warning("顔が検出されませんでした: %s", image_path)
        return None

    logger.info("  %d 人の顔を検出: %s", len(results.multi_face_landmarks), image_path.name)

    # 全顔の 2D ランドマークを収集
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

    # 既存 DB の face_index に合わせた順序を決定
    face_index_list = _match_face_order(all_coords, existing_centroids)
    unmatched_existing = sorted(set(existing_centroids) - set(face_index_list))
    if unmatched_existing:
        logger.warning(
            "既存の face_index %s に対応する顔が検出されませんでした（t / skip_flag は引き継がれません）: %s",
            unmatched_existing, image_path.name,
        )

    samples: List[Dict[str, Any]] = []
    for det_i, (coords, coords_xyz) in enumerate(zip(all_coords, all_coords_xyz)):
        face_index = face_index_list[det_i]

        pose = compute_face_orientation(coords_xyz, w, h)
        if pose is not None:
            face_yaw, face_pitch, face_roll = pose
        else:
            face_yaw = face_pitch = face_roll = None

        # MediaPipe face mesh landmark #4 = 鼻先（顔中心 pivot 用）
        nose_x = float(coords[4][0])
        nose_y = float(coords[4][1])

        for part_name, sub_map in EYELINER_SAMPLES_MAP.items():
            for sub_key, index_list in sub_map.items():
                point_index = 0
                for entry in index_list:
                    _, p_idx, q_idx = entry
                    px, py = coords[p_idx]
                    qx, qy = coords[q_idx]
                    samples.append({
                        "image_path": _normalize_path(str(image_path)),
                        "face_index": face_index,
                        "part_name": part_name,
                        "sub_key": sub_key,
                        "point_index": point_index,
                        "px": float(px), "py": float(py),
                        "qx": float(qx), "qy": float(qy),
                        "t": None,
                        "skip_flag": False,
                        "face_yaw": face_yaw,
                        "face_pitch": face_pitch,
                        "face_roll": face_roll,
                        "nose_x": nose_x,
                        "nose_y": nose_y,
                    })
                    point_index += 1

    return samples


def _process_chunk(
    image_paths: List[Path],
    existing_by_image: Dict[str, List[Dict[str, Any]]],
    chunk_idx: int,
) -> List[Dict[str, Any]]:
    """1 ワーカーが担当する画像チャンクを処理する。"""
    samples: List[Dict[str, Any]] = []
    total = len(image_paths)
    for i, image_path in enumerate(image_paths):
        image_path_posix = _normalize_path(str(image_path))
        existing_centroids = _build_existing_centroids(
            image_path_posix, existing_by_image.get(image_path_posix, [])
        )
        result = _extract_samples_from_image(image_path, existing_centroids)
        if result:
            samples.extend(result)
        if (i + 1) % 500 == 0:
            logger.info("  chunk %d: %d/%d (%.0f%%)", chunk_idx, i + 1, total, (i + 1) / total * 100)
    return samples


# 既存アノテーション件数に対する生成サンプル数の下限比率。これを下回る急減は
# mediapipe API 互換性の崩れ等による異常とみなし、DB の全置換を中断する。
_MIN_SAMPLE_RETENTION_RATIO = 0.5


def generate_samples(
    images_dir: Path,
    db_path: Path,
    workers: int = 1,
) -> None:
    """
    images_dir 内の全画像からサンプルを生成し db_path (SQLite) に保存する。

    既存の DB が存在する場合、t / skip_flag 値を保持してマージする。
    face_index は既存 DB の PQ 重心との最近傍マッチングにより揃える。

    DB は全置換される。images_dir に既存 DB の画像の一部しか無い場合は生成数が急減し、
    急減ガードで中断する。既存 DB へ画像を追加する用途には、増分マージする
    tools/imagesdb_maintenance/preprocess_new_images.py を使うこと。

    workers >= 2 の場合、ThreadPoolExecutor で並列処理する。
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    )

    if not image_paths:
        logger.error("画像が見つかりません: %s", images_dir)
        return

    logger.info("%d 枚の画像を処理します (workers=%d)...", len(image_paths), workers)

    # 既存の t / skip_flag 値と face_index 対応を保持するため全サンプルを読み込む
    existing_by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    existing_annotations: Dict[tuple, Dict[str, Any]] = {}
    _db.init_db(db_path)
    try:
        existing = _db.load_all_samples(db_path)
        for entry in existing:
            normalized_path = _normalize_path(entry["image_path"])
            existing_by_image[normalized_path].append(entry)
            key = (
                normalized_path,
                entry.get("face_index", 0),
                entry["part_name"],
                entry["sub_key"],
                entry["point_index"],
            )
            existing_annotations[key] = {
                "t": entry.get("t"),
                "skip_flag": entry.get("skip_flag", False),
            }
        logger.info("既存の DB から %d 件のアノテーションを読み込みました。", len(existing_annotations))
    except Exception as exc:
        logger.warning("既存の DB を読み込めませんでした: %s", exc)

    all_samples: List[Dict[str, Any]] = []

    if workers <= 1:
        all_samples = _process_chunk(image_paths, existing_by_image, 0)
    else:
        from concurrent.futures import ThreadPoolExecutor

        chunk_size = max(100, len(image_paths) // workers)
        chunks = [image_paths[i:i + chunk_size] for i in range(0, len(image_paths), chunk_size)]
        logger.info("%d workers, %d chunks", workers, len(chunks))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_chunk, chunk, existing_by_image, idx): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in futures:
                all_samples.extend(future.result())

    # idx を付与し、既存の t / skip_flag 値をマージ
    for idx, sample in enumerate(all_samples):
        sample["idx"] = idx
        key = (
            sample["image_path"],
            sample.get("face_index", 0),
            sample["part_name"],
            sample["sub_key"],
            sample["point_index"],
        )
        if key in existing_annotations:
            ann = existing_annotations[key]
            sample["t"] = ann["t"]
            sample["skip_flag"] = ann["skip_flag"]

    existing_count = len(existing_annotations)
    if not all_samples:
        logger.error(
            "生成されたサンプルが 0 件のため DB を更新しません（既存アノテーション %d 件を保護）。",
            existing_count,
        )
        return
    if existing_count > 0 and len(all_samples) < existing_count * _MIN_SAMPLE_RETENTION_RATIO:
        logger.error(
            "生成サンプル %d 件が既存 %d 件の %d%% を下回るため DB を更新しません（異常な減少を検出）。",
            len(all_samples), existing_count, int(_MIN_SAMPLE_RETENTION_RATIO * 100),
        )
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)
    _db.replace_all_samples(all_samples, db_path)
    logger.info(
        "DB を保存しました: %d サンプル → %s",
        len(all_samples), db_path,
    )


def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="アイラインアノテーション用サンプル生成")
    parser.add_argument("--images", required=True, type=Path, help="顔画像ディレクトリ")
    parser.add_argument(
        "--db", default="data/samples.db", type=Path,
        help="出力 SQLite DB パス (default: data/samples.db)",
    )
    parser.add_argument("--workers", type=int, default=1, help="並列数 (default: 1)")
    args = parser.parse_args()
    generate_samples(args.images, args.db, workers=args.workers)


if __name__ == "__main__":
    main()
