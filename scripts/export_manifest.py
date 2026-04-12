"""data/samples.db から data/manifest.csv を生成するスクリプト。

使い方:
    python scripts/export_manifest.py
    python scripts/export_manifest.py --db data/samples.db --output data/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from annotator import db as _db  # noqa: E402

FIELDNAMES = [
    "idx", "image_path", "part_name", "sub_key", "point_index",
    "px", "py", "qx", "qy", "t",
]


def export_manifest(db_path: Path, output_path: Path) -> int:
    """DB のアノテーション済みエントリを manifest.csv に書き出す。

    Returns:
        書き出したサンプル数
    """
    annotated = _db.get_annotated_samples(db_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for s in annotated:
            writer.writerow({
                "idx": s["idx"],
                "image_path": s["image_path"].replace("\\", "/"),
                "part_name": s["part_name"],
                "sub_key": s.get("sub_key", ""),
                "point_index": s.get("point_index", 0),
                "px": s["px"],
                "py": s["py"],
                "qx": s["qx"],
                "qy": s["qy"],
                "t": s["t"],
            })

    return len(annotated)


def main() -> None:
    parser = argparse.ArgumentParser(description="samples.db から manifest.csv を生成する")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/samples.db"),
        help="samples.db のパス（デフォルト: data/samples.db）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifest.csv"),
        help="manifest.csv の出力パス（デフォルト: data/manifest.csv）",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"[ERROR] samples.db が見つかりません: {args.db}")
        sys.exit(1)

    n = export_manifest(args.db, args.output)
    print(f"[OK] manifest.csv 出力: {n} サンプル → {args.output}")


if __name__ == "__main__":
    main()
