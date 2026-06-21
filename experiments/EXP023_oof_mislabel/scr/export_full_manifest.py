"""EXP023: data/samples.db から t IS NOT NULL の全件を manifest 化する。

scripts/split_holdout.py の _load_annotated_rows と同じロジック（annotator.db.get_annotated_samples）
を使い、train/test に分割せず 1 本の manifest として出力する。

使い方:
    python experiments/EXP023_oof_mislabel/scr/export_full_manifest.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
MANIFEST_DIR = EXP_DIR / "manifests"
DEFAULT_OUTPUT = MANIFEST_DIR / "full_manifest.csv"
DEFAULT_DB = ROOT / "data" / "samples.db"

sys.path.insert(0, str(ROOT))
from annotator import db as _db  # noqa: E402

FIELDNAMES = [
    "idx", "image_path", "face_index", "part_name", "sub_key", "point_index",
    "px", "py", "qx", "qy", "t",
]


def export_full_manifest(db_path: Path, out_path: Path) -> int:
    """DB の t IS NOT NULL 全件を manifest CSV に書き出す。

    Returns:
        書き出した行数
    """
    annotated = _db.get_annotated_samples(db_path)
    rows = [
        {
            "idx":         s["idx"],
            "image_path":  s["image_path"].replace("\\", "/"),
            "face_index":  s.get("face_index", 0),
            "part_name":   s["part_name"],
            "sub_key":     s.get("sub_key", ""),
            "point_index": s.get("point_index", 0),
            "px":          s["px"],
            "py":          s["py"],
            "qx":          s["qx"],
            "qy":          s["qy"],
            "t":           s["t"],
        }
        for s in annotated
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="EXP023 full manifest export")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    if not args.db.exists():
        print(f"[ERROR] DB 不在: {args.db}")
        return 1

    n = export_full_manifest(args.db, args.output)
    print(f"[OK] {n} 件 → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
