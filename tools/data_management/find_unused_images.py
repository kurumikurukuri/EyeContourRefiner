"""全実験の manifest から一度も参照されていない画像を列挙するスクリプト。

data/samples.db（source of truth）に登録されている画像のうち、
experiments/EXP*/manifests/*.csv のどれからも参照されていないものを一覧化する。

使い方:
    python tools/data_management/find_unused_images.py
    python tools/data_management/find_unused_images.py --out unused_images.csv
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent.parent
DB_PATH = ROOT / "data" / "samples.db"
MANIFEST_GLOB = "experiments/*/manifests/*.csv"


def load_db_image_paths(db_path: Path, annotated_only: bool) -> set[str]:
    query = "SELECT DISTINCT image_path FROM samples"
    if annotated_only:
        query += " WHERE t IS NOT NULL"
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(query).fetchall()
    finally:
        con.close()
    return {r[0].replace("\\", "/") for r in rows}


def load_referenced_image_paths(root: Path) -> tuple[set[str], int]:
    referenced: set[str] = set()
    manifest_paths = sorted(root.glob(MANIFEST_GLOB))
    for manifest_path in manifest_paths:
        with manifest_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "image_path" not in reader.fieldnames:
                continue
            for row in reader:
                referenced.add(row["image_path"].replace("\\", "/"))
    return referenced, len(manifest_paths)


def main(out_path: Path | None, annotated_only: bool) -> None:
    if not DB_PATH.exists():
        print(f"samples.db が見つかりません: {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    db_images = load_db_image_paths(DB_PATH, annotated_only)
    referenced_images, manifest_count = load_referenced_image_paths(ROOT)

    unused = sorted(db_images - referenced_images)

    scope = "アノテーション済み (t IS NOT NULL)" if annotated_only else "全て"
    print(f"manifest ファイル数: {manifest_count}")
    print(f"DB 上の画像数 ({scope}): {len(db_images)}")
    print(f"manifest で参照されている画像数: {len(referenced_images & db_images)}")
    print(f"一度も参照されていない画像数: {len(unused)}")

    if out_path is not None:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path"])
            writer.writerows([[p] for p in unused])
        print(f"一覧を書き出しました: {out_path}")
    else:
        for p in unused[:20]:
            print(f"  {p}")
        if len(unused) > 20:
            print(f"  ... 他 {len(unused) - 20} 件")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="どの manifest からも参照されていない画像を列挙する")
    parser.add_argument("--out", type=Path, default=None, help="結果を書き出す CSV パス")
    parser.add_argument(
        "--annotated-only",
        action="store_true",
        help="t IS NOT NULL（アノテーション済み）の画像のみを対象にする",
    )
    args = parser.parse_args()
    main(out_path=args.out, annotated_only=args.annotated_only)
