"""data/samples.json から data/manifest.csv を生成するスクリプト。

使い方:
    python scripts/export_manifest.py
    python scripts/export_manifest.py --samples data/samples.json --output data/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

FIELDNAMES = [
    "idx", "image_path", "part_name", "sub_key", "point_index",
    "px", "py", "qx", "qy", "t", "label_source", "confidence",
]


def export_manifest(samples_path: Path, output_path: Path) -> int:
    """samples.json のアノテーション済みエントリを manifest.csv に書き出す。

    Returns:
        書き出したサンプル数
    """
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    annotated = [s for s in samples if s.get("t") is not None]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for s in annotated:
            writer.writerow({
                "idx": s["idx"],
                "image_path": s["image_path"],
                "part_name": s["part_name"],
                "sub_key": s.get("sub_key", ""),
                "point_index": s.get("point_index", 0),
                "px": s["px"],
                "py": s["py"],
                "qx": s["qx"],
                "qy": s["qy"],
                "t": s["t"],
                "label_source": s.get("label_source") or "manual",
                "confidence": s.get("confidence") or "",
            })

    return len(annotated)


def main() -> None:
    parser = argparse.ArgumentParser(description="samples.json から manifest.csv を生成する")
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("data/samples.json"),
        help="samples.json のパス（デフォルト: data/samples.json）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifest.csv"),
        help="manifest.csv の出力パス（デフォルト: data/manifest.csv）",
    )
    args = parser.parse_args()

    if not args.samples.exists():
        print(f"[ERROR] samples.json が見つかりません: {args.samples}")
        sys.exit(1)

    n = export_manifest(args.samples, args.output)
    print(f"[OK] manifest.csv 出力: {n} サンプル → {args.output}")


if __name__ == "__main__":
    main()
