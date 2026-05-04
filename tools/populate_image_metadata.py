"""samples.db の画像に Open Images メタデータを登録する。

fiftyone キャッシュの detections.csv と classes.csv を読み込み、
samples.db に image_labels テーブルとしてラベル・フィルタフラグを保存する。

使い方:
    python tools/populate_image_metadata.py
"""
from __future__ import annotations

import csv
import json
import logging
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).parent.parent))
from annotator import db as _db

logger = logging.getLogger(__name__)

DB_PATH = Path("data/samples.db")
FO_CACHE_ROOT = Path.home() / "fiftyone" / "open-images-v7"

FLAG_COLUMNS = ("IsDepiction", "IsGroupOf", "IsInside", "IsOccluded", "IsTruncated")


def _load_label_maps(
    classes_csv: Path,
) -> tuple[Dict[str, str], Dict[str, str]]:
    name_to_id: Dict[str, str] = {}
    id_to_name: Dict[str, str] = {}
    with open(classes_csv, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                id_to_name[row[0]] = row[1]
                name_to_id[row[1]] = row[0]
    return name_to_id, id_to_name


def _collect_annotations(
    annotations_csv: Path,
    permissive_ids: Set[str],
    face_label_id: str,
    id_to_name: Dict[str, str],
) -> tuple[Dict[str, Set[str]], Dict[str, Dict[str, int]]]:
    """画像ごとのラベル一覧と、顔bboxフラグのOR集約を返す。"""
    image_labels: Dict[str, Set[str]] = defaultdict(set)
    image_face_flags: Dict[str, Dict[str, int]] = {}

    with open(annotations_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img_id = row["ImageID"]
            if img_id not in permissive_ids:
                continue
            label_id = row["LabelName"]
            image_labels[img_id].add(id_to_name.get(label_id, label_id))

            if label_id == face_label_id:
                existing = image_face_flags.get(img_id, {})
                for flag in FLAG_COLUMNS:
                    val = int(row.get(flag, 0))
                    if val:
                        existing[flag] = 1
                image_face_flags[img_id] = existing

    return image_labels, image_face_flags


def run() -> None:
    _db.init_db(DB_PATH)

    # samples.db 内の全 image_id を取得
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        image_ids = {
            Path(r["image_path"]).stem
            for r in conn.execute("SELECT DISTINCT image_path FROM samples").fetchall()
        }
    logger.info("samples.db 内のユニーク画像: %d", len(image_ids))

    upserted = 0
    for split in ("test", "train", "validation"):
        classes_csv = FO_CACHE_ROOT / split / "metadata" / "classes.csv"
        annotations_csv = FO_CACHE_ROOT / split / "labels" / "detections.csv"

        if not annotations_csv.exists():
            logger.warning("%s: annotations CSV not found, skip", split)
            continue

        name_to_id, id_to_name = _load_label_maps(classes_csv)
        face_label_id = name_to_id.get("Human face")
        if not face_label_id:
            logger.warning("%s: Human face label not found", split)
            continue

        image_labels, image_face_flags = _collect_annotations(
            annotations_csv, image_ids, face_label_id, id_to_name,
        )

        for img_id, labels in image_labels.items():
            flags = image_face_flags.get(img_id, {})
            _db.upsert_image_labels(
                DB_PATH,
                img_id,
                ",".join(sorted(labels)),
                json.dumps(flags, ensure_ascii=False),
            )
            upserted += 1

        logger.info("%s: %d 画像のメタデータを登録", split, len(image_labels))

    logger.info("完了: 合計 %d 画像", upserted)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    run()
