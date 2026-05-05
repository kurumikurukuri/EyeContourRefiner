"""
DB の skip_flag / t (true_t_raw) を CSV に反映する。
NIQE 再計算は行わず、既存 CSV の値を保持する。
"""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB = "data/samples.db"
DEFAULT_CSV = "data/samples.csv"


def update_csv_from_db(
    csv_path: str,
    db_path: str,
    columns: tuple[str, ...] = ("skip_flag", "t"),
) -> pd.DataFrame:
    csv_df = pd.read_csv(csv_path)
    logger.info("CSV loaded: %s (%d rows)", csv_path, len(csv_df))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(f"SELECT idx, {', '.join(columns)} FROM samples")
    db_rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    db_df = pd.DataFrame(db_rows).set_index("idx")
    logger.info("DB loaded: %s (%d rows)", db_path, len(db_df))

    changes = {}
    col_map = {"t": "true_t_raw"}

    for col in columns:
        csv_col = col_map.get(col, col)
        old = csv_df[csv_col].copy()
        new = csv_df["idx"].map(db_df[col])

        if col == "t":
            # Compare numerically (float64 serialization noise を無視)
            old_num = pd.to_numeric(old, errors="coerce")
            new_num = pd.to_numeric(new, errors="coerce")
            old_cmp = old_num.fillna(-999.0)
            new_cmp = new_num.fillna(-999.0)
            new = new.where(new.notna(), "")
        else:
            old_cmp = old.fillna("__NA__")
            new_cmp = new.fillna("__NA__")

        changed = (~np.isclose(old_cmp, new_cmp, rtol=1e-14, atol=1e-14)).sum() if col == "t" else (old_cmp != new_cmp).sum()
        changes[col] = changed
        csv_df[csv_col] = new

    csv_df.to_csv(csv_path, index=False)
    logger.info("Updated CSV written: %s", csv_path)

    for col, n in changes.items():
        logger.info("  %s: %d rows changed", col, n)

    return csv_df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="DB から CSV へ skip_flag / t を反映（NIQE 再計算なし）"
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help=f"CSV path (default: {DEFAULT_CSV})")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"DB path (default: {DEFAULT_DB})")
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"[ERROR] CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.db).exists():
        print(f"[ERROR] DB not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    update_csv_from_db(args.csv, args.db)


if __name__ == "__main__":
    main()
