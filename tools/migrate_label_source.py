"""
既存データに label_source / confidence フィールドを追加する移行スクリプト。

- samples.json: アノテーション済み(t!=null)に label_source="manual" を付与
- manifest.csv: label_source="manual", confidence="" 列を追加

使用方法:
    python scripts/migrate_label_source.py [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

SAMPLES_PATH = Path("data/samples.json")
MANIFEST_PATH = Path("data/manifest.csv")

MANIFEST_FIELDNAMES = [
    "idx", "image_path", "part_name", "sub_key", "point_index",
    "px", "py", "qx", "qy", "t", "label_source", "confidence",
]


def migrate_samples(dry_run: bool) -> tuple[int, int]:
    """samples.json に label_source/confidence を追加。
    Returns: (updated_count, skipped_count)
    """
    samples: list[dict] = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
    updated = 0
    skipped = 0
    migrated = []
    for s in samples:
        if "label_source" in s:
            skipped += 1
            migrated.append(s)
            continue
        entry = dict(s)
        if entry.get("t") is not None:
            entry["label_source"] = "manual"
        else:
            entry["label_source"] = None
        entry.setdefault("confidence", None)
        migrated.append(entry)
        updated += 1

    if not dry_run:
        SAMPLES_PATH.write_text(
            json.dumps(migrated, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return updated, skipped


def migrate_manifest(dry_run: bool) -> tuple[int, int]:
    """manifest.csv に label_source/confidence 列を追加。
    Returns: (updated_count, skipped_count)
    """
    if not MANIFEST_PATH.exists():
        print("manifest.csv が見つかりません。スキップします。")
        return 0, 0

    with MANIFEST_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        existing_fields = reader.fieldnames or []

    if "label_source" in existing_fields:
        print("manifest.csv はすでに移行済みです。")
        return 0, len(rows)

    updated_rows = []
    for row in rows:
        updated_rows.append({
            **row,
            "label_source": "manual",
            "confidence": "",
        })

    if not dry_run:
        shutil.copy(MANIFEST_PATH, MANIFEST_PATH.with_suffix(".csv.bak"))
        with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
            writer.writeheader()
            writer.writerows(updated_rows)

    return len(updated_rows), 0


def main() -> None:
    parser = argparse.ArgumentParser(description="label_source フィールドを既存データに追加")
    parser.add_argument("--dry-run", action="store_true", help="変更を書き込まずに確認のみ")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] 実際の変更は行いません")

    s_updated, s_skipped = migrate_samples(args.dry_run)
    print(f"samples.json: {s_updated} 件更新, {s_skipped} 件スキップ（既存）")

    m_updated, m_skipped = migrate_manifest(args.dry_run)
    print(f"manifest.csv: {m_updated} 件更新, {m_skipped} 件スキップ（既存）")

    if not args.dry_run and m_updated > 0:
        print("  バックアップ: manifest.csv.bak を作成しました")

    print("移行完了。")


if __name__ == "__main__":
    main()
