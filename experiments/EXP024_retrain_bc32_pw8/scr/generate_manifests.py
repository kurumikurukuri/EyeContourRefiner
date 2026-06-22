"""EXP024: 修正後DBから manifests を生成（EXP022 と同一の train/test idx 分割）。

使い方:
    python experiments/EXP024_retrain_bc32_pw8/scr/generate_manifests.py
"""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

EXP022_DIR = Path(__file__).parent.parent.parent.parent / "experiments" / "EXP022_channel_pool_sweep"
EXP024_DIR = Path(__file__).parent.parent
OUTPUT_DIR = EXP024_DIR / "manifests"
DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "samples.db"
MANIFEST_COLUMNS = ["idx", "image_path", "face_index", "part_name", "sub_key", "point_index", "px", "py", "qx", "qy", "t"]


def _load_exp022_split(split: str) -> set[int]:
    path = EXP022_DIR / "manifests" / f"manifest_{split}.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return {int(r["idx"]) for r in csv.DictReader(f)}


def _write_manifest(idx_set: set[int], output_path: Path) -> int:
    conn = sqlite3.connect(str(DB_PATH))
    placeholders = ",".join("?" for _ in idx_set)
    rows = conn.execute(f"""
        SELECT idx, image_path, face_index, part_name, sub_key,
               point_index, px, py, qx, qy, t
        FROM samples
        WHERE idx IN ({placeholders}) AND t IS NOT NULL
        ORDER BY idx
    """, list(idx_set)).fetchall()
    conn.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(MANIFEST_COLUMNS)
        writer.writerows(rows)
    return len(rows)


def main():
    train_idx = _load_exp022_split("train")
    test_idx = _load_exp022_split("test")
    print(f"EXP022 train idx: {len(train_idx)}  test idx: {len(test_idx)}")

    for name, idx_set in [("train", train_idx), ("test", test_idx)]:
        out = OUTPUT_DIR / f"manifest_{name}.csv"
        n = _write_manifest(idx_set, out)
        print(f"  manifest_{name}.csv: {n} 件 → {out}")

    print("完了")


if __name__ == "__main__":
    main()
