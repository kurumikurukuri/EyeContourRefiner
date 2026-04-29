"""EXP015: corner / interior manifest 生成スクリプト

data/samples.db の annotated サンプルを画像単位で train/test に分割し、
さらに point_index グループで絞り込んで 4 つの manifest ファイルを生成する。

グループ定義:
    corner   : point_index ∈ {0, 6}（眼角・端点）
    interior : point_index ∈ {1, 2, 3, 4, 5}（中間点）

画像単位の train/test 分割は split_holdout.py の random メソッドと同じロジック
（同 seed・ratio なら他 EXP と test 画像セットが一致する）。

出力:
    experiments/EXP015_interior_corner_model/manifests/
        manifest_train_corner.csv
        manifest_test_corner.csv
        manifest_train_interior.csv
        manifest_test_interior.csv

使い方:
    python experiments/EXP015_interior_corner_model/scr/split_exp015.py
    python experiments/EXP015_interior_corner_model/scr/split_exp015.py \\
        --db data/samples.db --ratio 0.2 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
MANIFEST_DIR = EXP_DIR / "manifests"

sys.path.insert(0, str(ROOT))
from annotator import db as _db  # noqa: E402

CORNER_INDICES = {0, 6}
INTERIOR_INDICES = {1, 2, 3, 4, 5}

FIELDNAMES = [
    "idx", "image_path", "face_index", "part_name", "sub_key", "point_index",
    "px", "py", "qx", "qy", "t",
]


def _load_annotated_rows(db_path: Path) -> list[dict[str, Any]]:
    annotated = _db.get_annotated_samples(db_path)
    return [
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


def _write_manifest(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def generate_manifests(
    db_path: Path,
    manifest_dir: Path,
    ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, int]:
    """DB から 4 manifest（corner/interior × train/test）を生成する。

    Returns:
        dict: ファイル名 → サンプル数
    """
    rows = _load_annotated_rows(db_path)
    if not rows:
        raise ValueError("DB に annotated (t IS NOT NULL) サンプルがありません")

    unique_images = sorted({row["image_path"] for row in rows})
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_images)

    n_test_images = max(1, int(len(unique_images) * ratio))
    test_images = set(unique_images[:n_test_images])
    train_images = set(unique_images[n_test_images:])

    train_rows = [r for r in rows if r["image_path"] in train_images]
    test_rows  = [r for r in rows if r["image_path"] in test_images]

    groups: dict[str, set[int]] = {
        "corner":   CORNER_INDICES,
        "interior": INTERIOR_INDICES,
    }

    results: dict[str, int] = {}
    for split_name, split_rows in [("train", train_rows), ("test", test_rows)]:
        for group_name, indices in groups.items():
            filtered = [r for r in split_rows if r["point_index"] in indices]
            out_path = manifest_dir / f"manifest_{split_name}_{group_name}.csv"
            _write_manifest(filtered, out_path)
            key = f"manifest_{split_name}_{group_name}.csv"
            results[key] = len(filtered)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP015 用 corner / interior manifest を生成する"
    )
    parser.add_argument(
        "--db", type=Path, default=ROOT / "data" / "samples.db",
        help="samples.db のパス（デフォルト: data/samples.db）",
    )
    parser.add_argument(
        "--ratio", type=float, default=0.2,
        help="テストの割合（デフォルト: 0.2 = 20%%）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="乱数シード（デフォルト: 42）",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"[ERROR] samples.db が見つかりません: {args.db}")
        sys.exit(1)

    print(f"db:     {args.db}")
    print(f"output: {MANIFEST_DIR}")
    print(f"ratio:  {args.ratio:.1%}  seed: {args.seed}")
    print()

    counts = generate_manifests(args.db, MANIFEST_DIR, args.ratio, args.seed)

    print("[OK] manifest 生成完了:")
    for filename, n in sorted(counts.items()):
        print(f"  {n:>6} サンプル → {MANIFEST_DIR / filename}")

    total_corner   = counts["manifest_train_corner.csv"]   + counts["manifest_test_corner.csv"]
    total_interior = counts["manifest_train_interior.csv"] + counts["manifest_test_interior.csv"]
    print()
    print(f"  corner   (point_index 0,6): {total_corner} サンプル合計")
    print(f"  interior (point_index 1–5): {total_interior} サンプル合計")
    print()
    print("[注意] manifest_test_*.csv は以降変更しないでください。")
    print("       holdout_mae の計算のみに使用し、ハイパーパラメータ選定には使わないこと。")


if __name__ == "__main__":
    main()
