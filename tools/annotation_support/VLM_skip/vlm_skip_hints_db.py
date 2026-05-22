"""VLM スキップ候補 DB アクセス層。

samples.db とは完全に独立した別 DB (vlm_skip_hints.db) を管理する。
テーブル: image_hints
  - VLM 判定列 (vlm_*)
  - 人間の目視判定列 (human_*)
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

DEFAULT_HINTS_DB = Path(__file__).parent / "vlm_skip_hints.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS image_hints (
    image_path         TEXT PRIMARY KEY,

    vlm_skip           INTEGER,
    vlm_reason         TEXT,
    vlm_confidence     REAL,
    vlm_model          TEXT,
    vlm_processed_at   TEXT,

    human_skip         INTEGER,
    human_note         TEXT,
    human_reviewed_at  TEXT,

    merged_at          TEXT
)
"""

_CREATE_IDX = "CREATE INDEX IF NOT EXISTS idx_vlm_skip ON image_hints(vlm_skip)"
_CREATE_IDX_MERGED = "CREATE INDEX IF NOT EXISTS idx_merged_at ON image_hints(merged_at)"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(hints_db_path: Path = DEFAULT_HINTS_DB) -> None:
    hints_db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(hints_db_path) as conn:
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_IDX)
        # 既存DBへのマイグレーション（インデックス作成より先に列を追加）
        cols = {r[1] for r in conn.execute("PRAGMA table_info(image_hints)")}
        if "merged_at" not in cols:
            conn.execute("ALTER TABLE image_hints ADD COLUMN merged_at TEXT")
        conn.execute(_CREATE_IDX_MERGED)


def get_processed_paths(hints_db_path: Path = DEFAULT_HINTS_DB) -> Set[str]:
    """vlm_processed_at が設定済みの image_path セットを返す。"""
    if not hints_db_path.exists():
        return set()
    with _connect(hints_db_path) as conn:
        rows = conn.execute(
            "SELECT image_path FROM image_hints WHERE vlm_processed_at IS NOT NULL"
        ).fetchall()
    return {r["image_path"] for r in rows}


def upsert_vlm_hint(
    image_path: str,
    vlm_skip: int,
    vlm_reason: Optional[str],
    vlm_confidence: Optional[float],
    vlm_model: str,
    hints_db_path: Path = DEFAULT_HINTS_DB,
) -> None:
    """VLM 判定結果を upsert する。human_* 列は上書きしない。"""
    now = datetime.now(timezone.utc).isoformat()
    with _connect(hints_db_path) as conn:
        conn.execute(
            """
            INSERT INTO image_hints
                (image_path, vlm_skip, vlm_reason, vlm_confidence, vlm_model, vlm_processed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_path) DO UPDATE SET
                vlm_skip         = excluded.vlm_skip,
                vlm_reason       = excluded.vlm_reason,
                vlm_confidence   = excluded.vlm_confidence,
                vlm_model        = excluded.vlm_model,
                vlm_processed_at = excluded.vlm_processed_at
            """,
            (image_path, vlm_skip, vlm_reason, vlm_confidence, vlm_model, now),
        )


def update_human_judgment(
    image_path: str,
    human_skip: int,
    human_note: Optional[str] = None,
    hints_db_path: Path = DEFAULT_HINTS_DB,
) -> None:
    """人間の目視判定を更新する。"""
    now = datetime.now(timezone.utc).isoformat()
    with _connect(hints_db_path) as conn:
        conn.execute(
            """
            INSERT INTO image_hints (image_path, human_skip, human_note, human_reviewed_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(image_path) DO UPDATE SET
                human_skip        = excluded.human_skip,
                human_note        = excluded.human_note,
                human_reviewed_at = excluded.human_reviewed_at
            """,
            (image_path, human_skip, human_note, now),
        )


def get_skip_candidates(
    hints_db_path: Path = DEFAULT_HINTS_DB,
    min_confidence: float = 0.7,
    use: str = "vlm_only",
) -> List[Dict[str, Any]]:
    """スキップ候補を返す。

    use:
        "vlm_only"    — vlm_skip=1 のみ
        "human_only"  — human_skip=1 のみ
    """
    if not hints_db_path.exists():
        return []

    if use == "human_only":
        query = "SELECT * FROM image_hints WHERE human_skip = 1"
        params: tuple = ()
    else:  # vlm_only
        query = """
            SELECT * FROM image_hints
            WHERE vlm_skip = 1 AND (vlm_confidence IS NULL OR vlm_confidence >= ?)
        """
        params = (min_confidence,)

    with _connect(hints_db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def count_stats(hints_db_path: Path = DEFAULT_HINTS_DB) -> Dict[str, int]:
    """DB の集計統計を返す。"""
    if not hints_db_path.exists():
        return {"total": 0, "vlm_processed": 0, "vlm_skip": 0, "human_reviewed": 0, "human_skip": 0, "merged": 0}
    with _connect(hints_db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM image_hints WHERE merged_at IS NULL").fetchone()[0]
        vlm_proc = conn.execute(
            "SELECT COUNT(*) FROM image_hints WHERE vlm_processed_at IS NOT NULL AND merged_at IS NULL"
        ).fetchone()[0]
        vlm_skip = conn.execute(
            "SELECT COUNT(*) FROM image_hints WHERE vlm_skip = 1 AND merged_at IS NULL"
        ).fetchone()[0]
        human_rev = conn.execute(
            "SELECT COUNT(*) FROM image_hints WHERE human_reviewed_at IS NOT NULL AND merged_at IS NULL"
        ).fetchone()[0]
        human_skip = conn.execute(
            "SELECT COUNT(*) FROM image_hints WHERE human_skip = 1 AND merged_at IS NULL"
        ).fetchone()[0]
        merged = conn.execute(
            "SELECT COUNT(*) FROM image_hints WHERE merged_at IS NOT NULL"
        ).fetchone()[0]
    return {
        "total": total,
        "vlm_processed": vlm_proc,
        "vlm_skip": vlm_skip,
        "human_reviewed": human_rev,
        "human_skip": human_skip,
        "merged": merged,
    }
