"""
顔画像からアイライン (P, Q) ペアを抽出し、アノテーション用サンプルを生成する。

使い方:
    python annotator/preprocessor.py \\
        --images data/images/ \\
        --db data/samples.db

既存の DB が存在する場合、t / skip_flag 値を保持してマージします（再実行しても上書きしません）。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from annotator import db as _db  # noqa: E402
from eyeliner.data.eyeliner_landmarks import EYELINER_CORRECTION_MAP  # noqa: E402

logger = logging.getLogger(__name__)


def _normalize_path(p: str) -> str:
    """パス区切りを POSIX 形式（/）に正規化する。"""
    return Path(p).as_posix()


# サンプル生成用: {part_name: {sub_key: (int | ('edge', p, q), ...)}}。
# ('edge', p, q) の並び順が point_index（0〜6）に対応する。
EYELINER_SAMPLES_MAP = EYELINER_CORRECTION_MAP


def _extract_samples_from_image(
    image_path: Path,
) -> Optional[List[Dict[str, Any]]]:
    """
    1枚の画像から全アイライン (P, Q) ペアを抽出する。

    Args:
        image_path: 顔画像のパス。

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

    samples: List[Dict[str, Any]] = []
    for face_index, face_landmarks in enumerate(results.multi_face_landmarks):
        coords = np.array(
            [[lm.x * w, lm.y * h] for lm in face_landmarks.landmark],
            dtype=np.float32,
        )
        for part_name, sub_map in EYELINER_SAMPLES_MAP.items():
            for sub_key, index_list in sub_map.items():
                # index_list の各要素は int（アンカー点）か ('edge', p, q) タプル（P-Q ペア）
                # アノテーションが必要なのは ('edge', p, q) のみ
                point_index = 0
                for entry in index_list:
                    if not isinstance(entry, tuple):
                        continue  # アンカー点はスキップ
                    _, p_idx, q_idx = entry  # ('edge', p, q)
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
                    })
                    point_index += 1

    return samples


def generate_samples(
    images_dir: Path,
    db_path: Path,
) -> None:
    """
    images_dir 内の全画像からサンプルを生成し db_path (SQLite) に保存する。

    既存の DB が存在する場合、t / skip_flag 値を保持してマージする。
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    )

    if not image_paths:
        logger.error("画像が見つかりません: %s", images_dir)
        return

    logger.info("%d 枚の画像を処理します...", len(image_paths))

    # 既存の t / skip_flag 値を保持するためのインデックスを構築
    existing_annotations: Dict[tuple, Dict[str, Any]] = {}
    _db.init_db(db_path)
    try:
        existing = _db.load_all_samples(db_path)
        for entry in existing:
            normalized_path = _normalize_path(entry["image_path"])
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
    for image_path in image_paths:
        logger.info("処理中: %s", image_path.name)
        samples = _extract_samples_from_image(image_path)
        if samples:
            all_samples.extend(samples)

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
    args = parser.parse_args()
    generate_samples(args.images, args.db)


if __name__ == "__main__":
    main()
