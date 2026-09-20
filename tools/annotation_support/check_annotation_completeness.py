"""
アノテーション完全性チェックスクリプト

目単位（image_path, face_index, part_name）でアノテーション済み点数（t IS NOT NULL）が
14の倍数でない目を検出して表示する。

使い方:
    python tools/annotation_support/check_annotation_completeness.py
    python tools/annotation_support/check_annotation_completeness.py --db path/to/samples.db
    python tools/annotation_support/check_annotation_completeness.py --skip-flag 0  # skip_flag=0のみ対象
"""

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IncompleteEye:
    image_path: str
    face_index: int
    part_name: str
    annotated: int
    unannotated: int
    upper0_idx: int | None

    @property
    def total(self) -> int:
        return self.annotated + self.unannotated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="アノテーション完全性チェック")
    parser.add_argument(
        "--db",
        default="data/samples.db",
        help="samples.db のパス (default: data/samples.db)",
    )
    parser.add_argument(
        "--skip-flag",
        type=int,
        choices=[0, 1],
        default=None,
        help="skip_flag でフィルタ (0=有効のみ, 1=スキップのみ, 省略=全件)",
    )
    parser.add_argument(
        "--multiple",
        type=int,
        default=14,
        help="チェックする倍数 (default: 14)",
    )
    return parser.parse_args()


def fetch_incomplete_eyes(
    db_path: str,
    multiple: int,
    skip_flag: int | None,
) -> list[IncompleteEye]:
    where_clause = "" if skip_flag is None else f"WHERE skip_flag = {skip_flag}"
    query = f"""
        SELECT
            image_path,
            face_index,
            part_name,
            SUM(CASE WHEN t IS NOT NULL THEN 1 ELSE 0 END) AS annotated,
            SUM(CASE WHEN t IS NULL     THEN 1 ELSE 0 END) AS unannotated,
            MAX(CASE WHEN sub_key = 'upper' AND point_index = 0 THEN idx END) AS upper0_idx
        FROM samples
        {where_clause}
        GROUP BY image_path, face_index, part_name
        HAVING annotated % {multiple} != 0
        ORDER BY annotated, image_path, part_name
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query).fetchall()

    return [
        IncompleteEye(
            image_path=r["image_path"],
            face_index=r["face_index"],
            part_name=r["part_name"],
            annotated=r["annotated"],
            unannotated=r["unannotated"],
            upper0_idx=r["upper0_idx"],
        )
        for r in rows
    ]


def display_results(eyes: list[IncompleteEye], multiple: int) -> None:
    if not eyes:
        print(f"✓ アノテーション数が {multiple} の倍数でない目は見つかりませんでした。")
        return

    col_w = max(len(e.image_path) for e in eyes) + 2
    header = (
        f"{'image_path':<{col_w}} {'face':>4}  {'part_name':<16}"
        f"  {'annotated':>10}  {'unannotated':>12}  {'total':>6}  {'upper0_idx':>12}"
    )
    sep = "-" * len(header)

    print(f"\n{multiple} の倍数でないアノテーションが存在する目: {len(eyes)} 件\n")
    print(header)
    print(sep)
    for e in eyes:
        idx_str = str(e.upper0_idx) if e.upper0_idx is not None else "N/A"
        print(
            f"{e.image_path:<{col_w}} {e.face_index:>4}  {e.part_name:<16}"
            f"  {e.annotated:>10}  {e.unannotated:>12}  {e.total:>6}  {idx_str:>12}"
        )
    print(sep)

    # サマリー
    unique_images = len({e.image_path for e in eyes})
    print(f"\n画像数: {unique_images} 枚  /  目数: {len(eyes)} 目")


def main() -> None:
    args = parse_args()
    db_path = args.db

    if not Path(db_path).exists():
        print(f"ERROR: DB が見つかりません: {db_path}", file=sys.stderr)
        sys.exit(1)

    skip_label = {None: "全件", 0: "skip_flag=0 のみ", 1: "skip_flag=1 のみ"}[args.skip_flag]
    print(f"DB: {db_path}  |  対象: {skip_label}  |  チェック倍数: {args.multiple}")

    eyes = fetch_incomplete_eyes(db_path, args.multiple, args.skip_flag)
    display_results(eyes, args.multiple)


if __name__ == "__main__":
    main()
