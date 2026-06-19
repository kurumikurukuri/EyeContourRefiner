"""t_pred スキャン結果 DB アクセス層。

テーブル: t_pred_hints
  (image_path, part_name) を主キーとする1眼1レコード。
  CNN が予測した per-eye の min/max t_pred を保存する。
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

DEFAULT_DB = Path(__file__).parent / "t_pred_hints.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS t_pred_hints (
    image_path  TEXT NOT NULL,
    part_name   TEXT NOT NULL,
    repr_idx    INTEGER,
    min_t_pred  REAL,
    max_t_pred  REAL,
    n_extreme   INTEGER,
    n_points    INTEGER,
    scan_model  TEXT,
    scanned_at  TEXT,
    PRIMARY KEY (image_path, part_name)
)
"""

_CREATE_IDX = """
CREATE INDEX IF NOT EXISTS idx_t_range
ON t_pred_hints(min_t_pred, max_t_pred)
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(db_path: Path = DEFAULT_DB) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_IDX)


def get_processed_keys(db_path: Path = DEFAULT_DB) -> Set[Tuple[str, str]]:
    if not db_path.exists():
        return set()
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT image_path, part_name FROM t_pred_hints WHERE scanned_at IS NOT NULL"
        ).fetchall()
    return {(r["image_path"], r["part_name"]) for r in rows}


def upsert(
    image_path: str,
    part_name: str,
    repr_idx: int,
    min_t_pred: float,
    max_t_pred: float,
    n_extreme: int,
    n_points: int,
    scan_model: str,
    db_path: Path = DEFAULT_DB,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO t_pred_hints
                (image_path, part_name, repr_idx, min_t_pred, max_t_pred,
                 n_extreme, n_points, scan_model, scanned_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_path, part_name) DO UPDATE SET
                repr_idx   = excluded.repr_idx,
                min_t_pred = excluded.min_t_pred,
                max_t_pred = excluded.max_t_pred,
                n_extreme  = excluded.n_extreme,
                n_points   = excluded.n_points,
                scan_model = excluded.scan_model,
                scanned_at = excluded.scanned_at
            """,
            (image_path, part_name, repr_idx,
             min_t_pred, max_t_pred, n_extreme, n_points, scan_model, now),
        )


def count_stats(db_path: Path = DEFAULT_DB) -> Dict[str, int]:
    if not db_path.exists():
        return {"total": 0, "extreme": 0}
    with _connect(db_path) as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM t_pred_hints WHERE scanned_at IS NOT NULL"
        ).fetchone()[0]
        extreme = conn.execute(
            "SELECT COUNT(*) FROM t_pred_hints WHERE n_extreme > 0"
        ).fetchone()[0]
    return {"total": total, "extreme": extreme}


# ソート可能カラムのホワイトリスト（SQL インジェクション対策）
_SORTABLE_COLUMNS = {"min_t_pred", "max_t_pred", "n_extreme", "n_points", "scanned_at"}


def _build_filter_clause(
    min_t_max: Optional[float],
    max_t_min: Optional[float],
    min_n_extreme: Optional[int],
) -> Tuple[str, List[Any]]:
    """scanned_at IS NOT NULL を基点に、指定されたフィルタ条件を AND で連結する。"""
    clauses = ["scanned_at IS NOT NULL"]
    params: List[Any] = []
    if min_t_max is not None:
        clauses.append("min_t_pred <= ?")
        params.append(min_t_max)
    if max_t_min is not None:
        clauses.append("max_t_pred >= ?")
        params.append(max_t_min)
    if min_n_extreme is not None:
        clauses.append("n_extreme >= ?")
        params.append(min_n_extreme)
    return " AND ".join(clauses), params


def list_hints(
    db_path: Path = DEFAULT_DB,
    *,
    min_t_max: Optional[float] = None,
    max_t_min: Optional[float] = None,
    min_n_extreme: Optional[int] = None,
    sort_by: str = "scanned_at",
    descending: bool = True,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """フィルタ・ソート・ページング付きで t_pred_hints を一覧取得する。

    sort_by はホワイトリスト外を許可しない（SQL インジェクション対策）。
    """
    if sort_by not in _SORTABLE_COLUMNS:
        raise ValueError(f"sort_by は {_SORTABLE_COLUMNS} のいずれかである必要があります: {sort_by}")
    if not db_path.exists():
        return []

    where_clause, params = _build_filter_clause(min_t_max, max_t_min, min_n_extreme)
    order_dir = "DESC" if descending else "ASC"

    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM t_pred_hints
            WHERE {where_clause}
            ORDER BY {sort_by} {order_dir}
            LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


def get_hint(
    db_path: Path, image_path: str, part_name: str
) -> Optional[Dict[str, Any]]:
    """(image_path, part_name) で1件取得する。なければ None。"""
    if not db_path.exists():
        return None
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM t_pred_hints WHERE image_path = ? AND part_name = ?",
            (image_path, part_name),
        ).fetchone()
    return dict(row) if row is not None else None


def count_filtered(
    db_path: Path = DEFAULT_DB,
    *,
    min_t_max: Optional[float] = None,
    max_t_min: Optional[float] = None,
    min_n_extreme: Optional[int] = None,
) -> int:
    """list_hints と同じフィルタ条件での総件数（ページング用）。"""
    if not db_path.exists():
        return 0
    where_clause, params = _build_filter_clause(min_t_max, max_t_min, min_n_extreme)
    with _connect(db_path) as conn:
        count = conn.execute(
            f"SELECT COUNT(*) FROM t_pred_hints WHERE {where_clause}",
            params,
        ).fetchone()[0]
    return count
