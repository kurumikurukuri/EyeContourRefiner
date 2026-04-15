"""SQLite データアクセス層。samples テーブルの CRUD を提供する。"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401  (Optional used in callers)

DB_PATH = Path("data/samples.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS samples (
    idx         INTEGER PRIMARY KEY,
    image_path  TEXT    NOT NULL,
    face_index  INTEGER NOT NULL DEFAULT 0,
    part_name   TEXT    NOT NULL,
    sub_key     TEXT    NOT NULL,
    point_index INTEGER NOT NULL,
    px          REAL    NOT NULL,
    py          REAL    NOT NULL,
    qx          REAL    NOT NULL,
    qy          REAL    NOT NULL,
    t           REAL,
    skip_flag   INTEGER NOT NULL DEFAULT 0
)
"""

_CREATE_IDX_IMAGE_PATH = (
    "CREATE INDEX IF NOT EXISTS idx_image_path ON samples(image_path)"
)
_CREATE_IDX_T = (
    "CREATE INDEX IF NOT EXISTS idx_t_annotated ON samples(t)"
)


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["skip_flag"] = bool(d["skip_flag"])
    return d


def init_db(db_path: Path = DB_PATH) -> None:
    """テーブルとインデックスを作成する（冪等）。"""
    with _connect(db_path) as conn:
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_IDX_IMAGE_PATH)
        conn.execute(_CREATE_IDX_T)


def load_all_samples(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """全サンプルを idx 順で返す。"""
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM samples ORDER BY idx").fetchall()
    return [_row_to_dict(r) for r in rows]


def get_sample_by_idx(idx: int, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    """idx で 1 件取得する。存在しなければ None。"""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM samples WHERE idx = ?", (idx,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def get_samples_by_image_path(
    image_path: str, db_path: Path = DB_PATH
) -> List[Dict[str, Any]]:
    """同一 image_path のサンプルを idx 順で返す。"""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM samples WHERE image_path = ? ORDER BY idx",
            (image_path,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def count_samples(db_path: Path = DB_PATH) -> Tuple[int, int]:
    """(total, annotated) を返す。"""
    with _connect(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        annotated = conn.execute(
            "SELECT COUNT(*) FROM samples WHERE t IS NOT NULL"
        ).fetchone()[0]
    return total, annotated


def update_annotation(idx: int, t: float, db_path: Path = DB_PATH) -> None:
    """idx のサンプルの t を更新する。"""
    with _connect(db_path) as conn:
        conn.execute("UPDATE samples SET t = ? WHERE idx = ?", (t, idx))


def update_skip_flag(
    image_path: str, skip_flag: bool, db_path: Path = DB_PATH
) -> int:
    """同一 image_path の全サンプルの skip_flag を更新する。更新件数を返す。"""
    with _connect(db_path) as conn:
        cur = conn.execute(
            "UPDATE samples SET skip_flag = ? WHERE image_path = ?",
            (int(skip_flag), image_path),
        )
        return cur.rowcount


def batch_update_t(
    updates: List[Tuple[int, float]],
    db_path: Path = DB_PATH,
) -> None:
    """(idx, t) のリストを一括 UPDATE する。"""
    with _connect(db_path) as conn:
        conn.executemany(
            "UPDATE samples SET t = ? WHERE idx = ?",
            [(t, idx) for idx, t in updates],
        )


def get_unannotated_samples_for_eye(
    image_path: str,
    face_index: Optional[int],
    part_name: str,
    db_path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    """同一眼（image_path + face_index + part_name）の未ラベルサンプルを返す。"""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """SELECT * FROM samples
               WHERE image_path = ? AND face_index = ? AND part_name = ? AND t IS NULL
               ORDER BY idx""",
            (image_path, face_index if face_index is not None else 0, part_name),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]



def delete_samples_by_image_paths(
    image_paths: List[str], db_path: Path = DB_PATH
) -> int:
    """指定 image_path のサンプルを削除する。削除件数を返す。"""
    if not image_paths:
        return 0
    placeholders = ",".join("?" * len(image_paths))
    with _connect(db_path) as conn:
        cur = conn.execute(
            f"DELETE FROM samples WHERE image_path IN ({placeholders})",
            image_paths,
        )
        return cur.rowcount


def delete_orphan_samples(
    valid_image_paths: List[str], db_path: Path = DB_PATH
) -> int:
    """valid_image_paths に含まれない画像を参照するサンプルを削除する。削除件数を返す。"""
    if not valid_image_paths:
        return 0
    placeholders = ",".join("?" * len(valid_image_paths))
    with _connect(db_path) as conn:
        cur = conn.execute(
            f"DELETE FROM samples WHERE image_path NOT IN ({placeholders})",
            valid_image_paths,
        )
        return cur.rowcount


def replace_all_samples(
    samples: List[Dict[str, Any]], db_path: Path = DB_PATH
) -> None:
    """全サンプルを削除してから挿入する（preprocessor 再実行用）。"""
    rows = [
        (
            s["idx"],
            s["image_path"],
            s.get("face_index", 0),
            s["part_name"],
            s["sub_key"],
            s["point_index"],
            float(s["px"]),
            float(s["py"]),
            float(s["qx"]),
            float(s["qy"]),
            s.get("t"),
            int(bool(s.get("skip_flag", False))),
        )
        for s in samples
    ]
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM samples")
        conn.executemany(
            """INSERT INTO samples
               (idx, image_path, face_index, part_name, sub_key, point_index,
                px, py, qx, qy, t, skip_flag)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )


def upsert_samples(
    samples: List[Dict[str, Any]], db_path: Path = DB_PATH
) -> None:
    """サンプルを INSERT OR REPLACE する（バッチ）。"""
    rows = [
        (
            s["idx"],
            s["image_path"],
            s.get("face_index", 0),
            s["part_name"],
            s["sub_key"],
            s["point_index"],
            float(s["px"]),
            float(s["py"]),
            float(s["qx"]),
            float(s["qy"]),
            s.get("t"),
            int(bool(s.get("skip_flag", False))),
        )
        for s in samples
    ]
    with _connect(db_path) as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO samples
               (idx, image_path, face_index, part_name, sub_key, point_index,
                px, py, qx, qy, t, skip_flag)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )


def get_image_index(
    db_path: Path = DB_PATH,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """ギャラリー用インメモリインデックスを DB から構築する。
    Returns: (index_dict, order_list)
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            """SELECT image_path,
                      COUNT(*)                                        AS total,
                      SUM(CASE WHEN t IS NOT NULL THEN 1 ELSE 0 END) AS annotated,
                      MAX(CASE WHEN skip_flag THEN 1 ELSE 0 END)     AS skipped,
                      MIN(idx)                                        AS first_pos
               FROM samples
               GROUP BY image_path
               ORDER BY MIN(idx)""",
        ).fetchall()
    index: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for row in rows:
        ip = row["image_path"]
        index[ip] = {
            "first_pos": row["first_pos"],
            "total": row["total"],
            "annotated": row["annotated"],
            "skipped": bool(row["skipped"]),
        }
        order.append(ip)
    return index, order


def get_eye_scale_per_eye(db_path: Path = DB_PATH) -> Dict[Tuple[str, int, str], float]:
    """(image_path, face_index, part_name) ごとの eye_scale を返す。

    eye_scale = 同一 (image_path, face_index, part_name) の全 PQ 点群バウンディングボックス対角長。
    サンプルナビゲーションのソート順計算に使用する。
    """
    import math
    from collections import defaultdict

    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT image_path, face_index, part_name, px, py, qx, qy FROM samples"
        ).fetchall()

    eye_points: dict = defaultdict(list)
    for row in rows:
        key = (row["image_path"], row["face_index"], row["part_name"])
        eye_points[key].append((float(row["px"]), float(row["py"])))
        eye_points[key].append((float(row["qx"]), float(row["qy"])))

    result: Dict[Tuple[str, int, str], float] = {}
    for key, pts in eye_points.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        scale = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)
        result[key] = max(scale, 1.0)

    return result


def get_eye_scale_per_image(db_path: Path = DB_PATH) -> Dict[str, float]:
    """image_path ごとの eye_scale（全 (face_index, part_name) の最小値）を返す。

    eye_scale = 同一 (image_path, face_index, part_name) の全 PQ 点群バウンディングボックス対角長。
    各画像の代表値として全 (face_index, part_name) の最小値を使用する（最小目スケールを基準にソート可能）。
    """
    import math
    from collections import defaultdict

    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT image_path, face_index, part_name, px, py, qx, qy FROM samples"
        ).fetchall()

    eye_points: dict = defaultdict(list)
    for row in rows:
        key = (row["image_path"], row["face_index"], row["part_name"])
        eye_points[key].append((float(row["px"]), float(row["py"])))
        eye_points[key].append((float(row["qx"]), float(row["qy"])))

    eye_scale_per_image: Dict[str, float] = {}
    for (image_path, face_index, part_name), pts in eye_points.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        scale = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)
        scale = max(scale, 1.0)
        if image_path not in eye_scale_per_image or scale < eye_scale_per_image[image_path]:
            eye_scale_per_image[image_path] = scale

    return eye_scale_per_image


def get_annotated_samples(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """t IS NOT NULL のサンプルを idx 順で返す。"""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM samples WHERE t IS NOT NULL ORDER BY idx"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def image_path_exists(image_path: str, db_path: Path = DB_PATH) -> bool:
    """指定 image_path がサンプルとして存在するか確認する。"""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM samples WHERE image_path = ? LIMIT 1", (image_path,)
        ).fetchone()
    return row is not None
