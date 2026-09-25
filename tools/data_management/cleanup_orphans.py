"""
data/samples.json と data/images/ の孤立エントリを削除するスクリプト。

- samples.json に存在するが画像ファイルが無いエントリを削除する
- data/images/ に存在するが samples.json に参照が無い画像を削除する
- data/open_images_attribution.csv に存在するが samples.json に参照が無い行を削除する

data/images/ は git 管理外で削除すると復旧できないため、DB が 0 件のとき、および孤立画像が
画像全体の過半（MAX_ORPHAN_RATIO）を占めるとき（別 DB の取り違えの疑い）は何も削除せず中断する。
後者は --force で上書きできるが、DB が 0 件のときは --force でも中断する。

使い方:
    # 確認のみ（削除しない）
    python tools/data_management/cleanup_orphans.py --dry-run

    # 実際に削除する
    python tools/data_management/cleanup_orphans.py
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from annotator import db as _db  # noqa: E402

ROOT = Path(__file__).parent.parent.parent
DB_PATH = ROOT / "data" / "samples.db"
IMAGES_DIR = ROOT / "data" / "images"
ATTRIBUTION_CSV = ROOT / "data" / "open_images_attribution.csv"

# 孤立画像が data/images/ の画像数に占める割合がこれを超えたら、DB の取り違えを疑って中断する。
MAX_ORPHAN_RATIO = 0.5


def resolve_image_path(image_path_str: str) -> Path:
    """samples.json の image_path を絶対パスに変換する。"""
    p = Path(image_path_str)
    if p.is_absolute():
        return p
    return ROOT / p


def main(dry_run: bool, force: bool = False) -> int:
    """孤立エントリ・画像・attribution 行を削除する。正常終了は 0、中断は 1 を返す。"""
    if not DB_PATH.exists():
        log.error("samples.db が見つかりません: %s", DB_PATH)
        return 1
    if not IMAGES_DIR.exists():
        log.error("images ディレクトリが見つかりません: %s", IMAGES_DIR)
        return 1

    samples = _db.load_all_samples(DB_PATH)
    log.info("DB エントリ数: %d", len(samples))
    if not samples:
        log.error(
            "DB が 0 件です。data/images/ の全画像と attribution 行が孤立扱いになるため、"
            "何も削除せず中断します: %s",
            DB_PATH,
        )
        return 1

    # DB から参照されている画像パスのセット
    referenced_images: set[Path] = {
        resolve_image_path(s["image_path"]).resolve() for s in samples
    }

    image_files = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
    orphan_images = [img for img in image_files if img.resolve() not in referenced_images]
    orphan_ratio = len(orphan_images) / len(image_files) if image_files else 0.0
    if orphan_ratio > MAX_ORPHAN_RATIO:
        # % を含む文字列なので logging の引数書式を使わず、完成した文字列をそのまま渡す。
        message = (
            f"孤立画像が {len(orphan_images)}/{len(image_files)} 件（{orphan_ratio:.0%}）と過半です。"
            f"別の DB を読んでいる可能性があります（DB_PATH: {DB_PATH}）。"
        )
        if not force:
            log.error(message + "何も削除せず中断します。意図した結果なら --force を付けてください。")
            return 1
        log.warning(message + "--force が指定されたため続行します。")

    # --- 1. 画像が存在しない DB エントリを除去 ---
    valid_paths: list[str] = [
        s["image_path"]
        for s in samples
        if resolve_image_path(s["image_path"]).resolve().exists()
    ]
    removed_entries = len(samples) - len(valid_paths)
    log.info("参照先画像が存在しないエントリ: %d 件", removed_entries)

    if not dry_run and removed_entries > 0:
        _db.delete_orphan_samples(valid_paths, DB_PATH)
        log.info("DB を更新しました（残り %d エントリ）", len(valid_paths))

    # --- 2. DB から参照されていない画像ファイルを削除 ---
    log.info("DB に参照が無い孤立画像: %d 件", len(orphan_images))

    for img in orphan_images:
        log.info("  削除対象: %s", img.name)
        if not dry_run:
            img.unlink()

    # --- 3. open_images_attribution.csv から DB に無い行を削除 ---
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
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="孤立した samples.json エントリと画像を削除する")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際には削除せず、対象ファイルを表示するだけ",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"孤立画像が過半（{MAX_ORPHAN_RATIO:.0%} 超）でも削除する。DB が 0 件のときは効かない",
    )
    args = parser.parse_args()
    sys.exit(main(dry_run=args.dry_run, force=args.force))
