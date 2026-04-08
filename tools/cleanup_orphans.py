"""
data/samples.json と data/images/ の孤立エントリを削除するスクリプト。

- samples.json に存在するが画像ファイルが無いエントリを削除する
- data/images/ に存在するが samples.json に参照が無い画像を削除する
- data/open_images_attribution.csv に存在するが samples.json に参照が無い行を削除する

使い方:
    # 確認のみ（削除しない）
    python scripts/cleanup_orphans.py --dry-run

    # 実際に削除する
    python scripts/cleanup_orphans.py
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
SAMPLES_JSON = ROOT / "data" / "samples.json"
IMAGES_DIR = ROOT / "data" / "images"
ATTRIBUTION_CSV = ROOT / "data" / "open_images_attribution.csv"


def load_samples(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_samples(path: Path, samples: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def resolve_image_path(image_path_str: str) -> Path:
    """samples.json の image_path を絶対パスに変換する。"""
    p = Path(image_path_str)
    if p.is_absolute():
        return p
    return ROOT / p


def main(dry_run: bool) -> None:
    if not SAMPLES_JSON.exists():
        log.error("samples.json が見つかりません: %s", SAMPLES_JSON)
        return
    if not IMAGES_DIR.exists():
        log.error("images ディレクトリが見つかりません: %s", IMAGES_DIR)
        return

    samples = load_samples(SAMPLES_JSON)
    log.info("samples.json エントリ数: %d", len(samples))

    # samples.json から参照されている画像パスのセット
    referenced_images: set[Path] = {
        resolve_image_path(s["image_path"]).resolve() for s in samples
    }

    # --- 1. 画像が存在しない samples.json エントリを除去 ---
    valid_samples = [
        s for s in samples
        if resolve_image_path(s["image_path"]).resolve().exists()
    ]
    removed_entries = len(samples) - len(valid_samples)
    log.info("参照先画像が存在しないエントリ: %d 件", removed_entries)

    if not dry_run and removed_entries > 0:
        save_samples(SAMPLES_JSON, valid_samples)
        log.info("samples.json を更新しました（残り %d エントリ）", len(valid_samples))

    # --- 2. samples.json から参照されていない画像ファイルを削除 ---
    image_files = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
    orphan_images = [img for img in image_files if img.resolve() not in referenced_images]
    log.info("samples.json に参照が無い孤立画像: %d 件", len(orphan_images))

    for img in orphan_images:
        log.info("  削除対象: %s", img.name)
        if not dry_run:
            img.unlink()

    # --- 3. open_images_attribution.csv から samples.json に無い行を削除 ---
    referenced_ids: set[str] = {
        Path(s["image_path"]).stem for s in samples
    }

    if ATTRIBUTION_CSV.exists():
        with ATTRIBUTION_CSV.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)

        kept_rows = [r for r in rows if r["image_id"] in referenced_ids]
        removed_csv_rows = len(rows) - len(kept_rows)
        log.info("attribution.csv: 孤立行 %d 件（残り %d 件）", removed_csv_rows, len(kept_rows))

        if not dry_run and removed_csv_rows > 0:
            with ATTRIBUTION_CSV.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(kept_rows)
            log.info("open_images_attribution.csv を更新しました")
    else:
        removed_csv_rows = 0
        log.warning("open_images_attribution.csv が見つかりません: %s", ATTRIBUTION_CSV)

    if dry_run:
        log.info("[dry-run] 削除は実行されていません。--dry-run を外すと実際に削除されます。")
    else:
        log.info(
            "完了: エントリ %d 件削除、画像 %d 件削除、attribution行 %d 件削除",
            removed_entries,
            len(orphan_images),
            removed_csv_rows,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="孤立した samples.json エントリと画像を削除する")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際には削除せず、対象ファイルを表示するだけ",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
