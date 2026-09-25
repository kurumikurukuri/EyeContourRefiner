"""Compare samples.db and samples_old.db.

Checks whether images that have t_raw_true (t IS NOT NULL) in samples_old.db
also exist as records in samples.db.
"""

import argparse
import sqlite3
from pathlib import Path

KEY_COLUMNS = ("image_path", "face_index", "part_name", "sub_key", "point_index")


def _require_db(db_path: Path) -> None:
    """sqlite3.connect が存在しないパスに空 DB を作るのを防ぐ。"""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"DB が見つかりません: {db_path}")


def load_keys_with_t(db_path: Path) -> set[tuple, ...]:
    """Load composite keys of rows where t IS NOT NULL."""
    _require_db(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT image_path, face_index, part_name, sub_key, point_index "
        "FROM samples WHERE t IS NOT NULL"
    ).fetchall()
    conn.close()
    return set(rows)


def load_all_keys(db_path: Path) -> set[tuple, ...]:
    """Load all composite keys."""
    _require_db(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT image_path, face_index, part_name, sub_key, point_index "
        "FROM samples"
    ).fetchall()
    conn.close()
    return set(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare samples.db and samples_old.db")
    parser.add_argument("--new", default="data/samples.db", help="Path to new DB")
    parser.add_argument("--old", default="data/samples_old.db", help="Path to old DB")
    args = parser.parse_args()

    new_path = Path(args.new)
    old_path = Path(args.old)

    old_t_keys = load_keys_with_t(old_path)
    new_all_keys = load_all_keys(new_path)

    matched = old_t_keys & new_all_keys
    missing = old_t_keys - new_all_keys

    print(f"=== DB Comparison: t_raw_true check ===")
    print(f"Old DB (t IS NOT NULL): {len(old_t_keys)} records")
    print(f"New DB (all records):   {len(new_all_keys)} records")
    print()
    print(f"Matched in new DB:      {len(matched)} / {len(old_t_keys)}")
    print(f"Missing from new DB:    {len(missing)} / {len(old_t_keys)}")

    if missing:
        print(f"\n--- {len(missing)} missing records (first 20) ---")
        for key in sorted(missing)[:20]:
            print(f"  {key}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

        missing_images = {k[0] for k in missing}
        print(f"\nUnique missing images: {len(missing_images)}")
        for img in sorted(missing_images)[:20]:
            print(f"  {img}")
        if len(missing_images) > 20:
            print(f"  ... and {len(missing_images) - 20} more")
    else:
        print("\nAll t_raw_true records from old DB exist in new DB.")


if __name__ == "__main__":
    main()
