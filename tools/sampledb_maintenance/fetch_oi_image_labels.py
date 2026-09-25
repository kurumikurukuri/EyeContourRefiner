"""
samples.db に含まれる ImageID に絞った Open Images image-level human labels を
ストリーム取得してキャッシュする。

出力: data/.open_images_cache/oi_image_labels.csv
  columns: image_id, label_name  (Confidence=1 のみ)

Usage:
    python tools/sampledb_maintenance/fetch_oi_image_labels.py
    python tools/sampledb_maintenance/fetch_oi_image_labels.py --force   # キャッシュ再取得
"""
from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Iterator, Set

import requests

logger = logging.getLogger(__name__)

IMAGE_LABEL_URLS: Dict[str, str] = {
    "train": (
        "https://storage.googleapis.com/openimages/v5/"
        "train-annotations-human-imagelabels.csv"
    ),
    "validation": (
        "https://storage.googleapis.com/openimages/v5/"
        "validation-annotations-human-imagelabels.csv"
    ),
    "test": (
        "https://storage.googleapis.com/openimages/v5/"
        "test-annotations-human-imagelabels.csv"
    ),
}

DEFAULT_DB = "data/samples.db"
DEFAULT_CACHE = Path("data/.open_images_cache")
OUTPUT_FILE = "oi_image_labels.csv"


def load_db_image_ids(db_path: str) -> Set[str]:
    if not Path(db_path).exists():
        raise FileNotFoundError(f"samples.db が見つかりません: {db_path}")
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT DISTINCT image_path FROM samples").fetchall()
    conn.close()
    return {Path(r[0]).stem for r in rows}


def split_membership(db_ids: Set[str], cache_dir: Path) -> Dict[str, Set[str]]:
    """permissive_meta CSV から、各 split に属する DB 内の ImageID を返す。"""
    result: Dict[str, Set[str]] = {}
    for split in IMAGE_LABEL_URLS:
        meta = cache_dir / f"{split}_permissive_meta.csv"
        if not meta.exists():
            logger.warning("  [%s] permissive_meta が見つかりません。スキップ", split)
            continue
        ids: Set[str] = set()
        with open(meta, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["image_id"] in db_ids:
                    ids.add(row["image_id"])
        if ids:
            result[split] = ids
            logger.info("  [%s] %d 画像が DB と一致", split, len(ids))
    return result


def stream_labels(url: str, target_ids: Set[str]) -> Iterator[tuple[str, str]]:
    """URL から CSV をストリームし、target_ids & Confidence=1 の (image_id, label_name) を yield する。"""
    logger.info("  GET %s", url)
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        lines = (
            line.decode("utf-8", errors="replace")
            for line in resp.iter_lines()
            if line
        )
        reader = csv.DictReader(lines)
        scanned = 0
        for row in reader:
            scanned += 1
            if scanned % 1_000_000 == 0:
                logger.info("    %dM 行スキャン済み...", scanned // 1_000_000)
            if row["ImageID"] in target_ids and row.get("Confidence") == "1":
                yield row["ImageID"], row["LabelName"]


def fetch_all(split_ids: Dict[str, Set[str]], output_path: Path) -> int:
    written = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "label_name"])
        for split, ids in split_ids.items():
            url = IMAGE_LABEL_URLS[split]
            logger.info("[%s] ストリーム取得開始 (%d images)", split, len(ids))
            for image_id, label in stream_labels(url, ids):
                writer.writerow([image_id, label])
                written += 1
    return written


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--cache", default=DEFAULT_CACHE, type=Path)
    parser.add_argument("--force", action="store_true", help="キャッシュを再取得")
    args = parser.parse_args()

    output = args.cache / OUTPUT_FILE
    if output.exists() and not args.force:
        logger.info("キャッシュ済み: %s  (--force で再取得)", output)
        return

    logger.info("=== Step 1: DB の ImageID 取得 ===")
    db_ids = load_db_image_ids(args.db)
    logger.info("  %d ユニーク画像", len(db_ids))

    logger.info("=== Step 2: split 所属判定 ===")
    split_ids = split_membership(db_ids, args.cache)
    covered = sum(len(v) for v in split_ids.values())
    logger.info(
        "  合計カバー: %d / %d 画像 (%.1f%%)",
        covered, len(db_ids), 100 * covered / max(len(db_ids), 1),
    )

    logger.info("=== Step 3: image-level labels ストリーム取得 ===")
    args.cache.mkdir(parents=True, exist_ok=True)
    n = fetch_all(split_ids, output)
    logger.info("=== 完了: %d 行 → %s ===", n, output)


if __name__ == "__main__":
    main()
