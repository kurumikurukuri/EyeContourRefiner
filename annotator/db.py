"""SQLite データアクセス層。samples テーブルの CRUD を提供する。"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    skip_flag   INTEGER NOT NULL DEFAULT 0,
    face_yaw    REAL,
    face_pitch  REAL,
    face_roll   REAL
)
"""

_CREATE_IDX_IMAGE_PATH = (
    "CREATE INDEX IF NOT EXISTS idx_image_path ON samples(image_path)"
)
_CREATE_IDX_T = (
    "CREATE INDEX IF NOT EXISTS idx_t_annotated ON samples(t)"
)
_CREATE_IDX_EYE_GROUP = (
    "CREATE INDEX IF NOT EXISTS idx_eye_group "
    "ON samples(image_path, face_index, part_name)"
)
_CREATE_IDX_SKIP_FLAG = (
    "CREATE INDEX IF NOT EXISTS idx_skip_flag ON samples(skip_flag, idx)"
)

_CREATE_IMAGE_LABELS = """
CREATE TABLE IF NOT EXISTS image_labels (
    image_id   TEXT PRIMARY KEY,
    labels     TEXT NOT NULL,
    face_flags TEXT NOT NULL DEFAULT '{}'
)
"""

# 追加カラム（後方互換のため ALTER TABLE ADD COLUMN で冪等に付与）。
_ADDITIONAL_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("face_yaw",      "REAL"),
    ("face_pitch",    "REAL"),
    ("face_roll",     "REAL"),
    ("eye_scale",     "REAL"),  # (image_path, face_index, part_name) のPQバウンディングボックス対角長
    ("norm_eye_scale","REAL"),  # eye_scale / sqrt(img_w^2 + img_h^2)
    ("lap_var",       "REAL"),  # 目領域のラプラシアン分散（ブレ指標）
    ("nose_x",        "REAL"),  # MediaPipe face mesh landmark #4 (鼻先) x 座標 [px]
    ("nose_y",        "REAL"),  # MediaPipe face mesh landmark #4 (鼻先) y 座標 [px]
)


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """既存 DB に不足している追加カラムを補充する（冪等）。"""
    existing = {
        row["name"] for row in conn.execute("PRAGMA table_info(samples)").fetchall()
    }
    for col_name, col_type in _ADDITIONAL_COLUMNS:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE samples ADD COLUMN {col_name} {col_type}")


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
    """テーブルとインデックスを作成する（冪等）。既存 DB にカラム不足があれば補充。"""
    with _connect(db_path) as conn:
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_IDX_IMAGE_PATH)
        conn.execute(_CREATE_IDX_T)
        conn.execute(_CREATE_IDX_EYE_GROUP)
        conn.execute(_CREATE_IDX_SKIP_FLAG)
        conn.execute(_CREATE_IMAGE_LABELS)
        _ensure_columns(conn)


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


def reset_t_for_eye(
    image_path: str,
    face_index: Optional[int],
    part_name: str,
    db_path: Path = DB_PATH,
) -> int:
    """同一眼（image_path + face_index + part_name）の全サンプルの t を NULL にリセットする。更新件数を返す。"""
    with _connect(db_path) as conn:
        cur = conn.execute(
            """UPDATE samples SET t = NULL
               WHERE image_path = ? AND face_index = ? AND part_name = ?""",
            (image_path, face_index if face_index is not None else 0, part_name),
        )
        return cur.rowcount


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


def get_mean_pq_dist_for_eye(
    image_path: str,
    face_index: Optional[int],
    part_name: str,
    db_path: Path = DB_PATH,
) -> Optional[float]:
    """同一眼グループ全点の pq_dist 平均値を返す。pq_rescale_mode=mean_pq_dist の逆変換に使用。"""
    import math as _math
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT px, py, qx, qy FROM samples WHERE image_path = ? AND face_index = ? AND part_name = ?",
            (image_path, face_index if face_index is not None else 0, part_name),
        ).fetchall()
    if not rows:
        return None
    dists = [_math.sqrt((r["qx"] - r["px"]) ** 2 + (r["qy"] - r["py"]) ** 2) for r in rows]
    return sum(dists) / len(dists)



def get_eye_height_for_eye(
    image_path: str,
    face_index: Optional[int],
    part_name: str,
    db_path: Path = DB_PATH,
) -> Optional[float]:
    """同一眼グループ全点の P/Q y 座標から eye_height を算出して返す。

    eye_height = max(all_ys) - min(all_ys)（weights.py の compute_eye_meta と同定義）。
    pq_rescale_mode='eye_height' の逆変換 (WebUI) に使用。
    該当行がない場合は None を返す。最小値は 1.0。
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT py, qy FROM samples WHERE image_path = ? AND face_index = ? AND part_name = ?",
            (image_path, face_index if face_index is not None else 0, part_name),
        ).fetchall()
    if not rows:
        return None
    ys = [r["py"] for r in rows] + [r["qy"] for r in rows]
    return max(max(ys) - min(ys), 1.0)


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


def _optional_float(value: Any) -> Optional[float]:
    """None を保持して float 変換。"""
    if value is None:
        return None
    return float(value)


def _sample_to_row(s: Dict[str, Any]) -> Tuple[Any, ...]:
    """サンプル dict を INSERT 用タプルへ変換する。"""
    return (
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
        _optional_float(s.get("face_yaw")),
        _optional_float(s.get("face_pitch")),
        _optional_float(s.get("face_roll")),
        _optional_float(s.get("eye_scale")),
        _optional_float(s.get("norm_eye_scale")),
        _optional_float(s.get("lap_var")),
        _optional_float(s.get("nose_x")),
        _optional_float(s.get("nose_y")),
    )


_INSERT_COLUMNS = (
    "idx, image_path, face_index, part_name, sub_key, point_index, "
    "px, py, qx, qy, t, skip_flag, face_yaw, face_pitch, face_roll, "
    "eye_scale, norm_eye_scale, lap_var, nose_x, nose_y"
)
_INSERT_PLACEHOLDERS = ", ".join(["?"] * 20)


def replace_all_samples(
    samples: List[Dict[str, Any]], db_path: Path = DB_PATH
) -> None:
    """全サンプルを削除してから挿入する（preprocessor 再実行用）。"""
    rows = [_sample_to_row(s) for s in samples]
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM samples")
        conn.executemany(
            f"INSERT INTO samples ({_INSERT_COLUMNS}) VALUES ({_INSERT_PLACEHOLDERS})",
            rows,
        )


def upsert_samples(
    samples: List[Dict[str, Any]], db_path: Path = DB_PATH
) -> None:
    """サンプルを INSERT OR REPLACE する（バッチ）。"""
    rows = [_sample_to_row(s) for s in samples]
    with _connect(db_path) as conn:
        conn.executemany(
            f"INSERT OR REPLACE INTO samples ({_INSERT_COLUMNS}) "
            f"VALUES ({_INSERT_PLACEHOLDERS})",
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


def get_max_eye_scale_per_image(db_path: Path = DB_PATH) -> Dict[str, float]:
    """image_path ごとの eye_scale（全 (face_index, part_name) の最大値）を返す。

    最大値が閾値未満 = その画像の全ての目が小さい（解像度不足）を判定するために使用する。
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
    for (image_path, _face_index, _part_name), pts in eye_points.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        scale = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)
        scale = max(scale, 1.0)
        if image_path not in eye_scale_per_image or scale > eye_scale_per_image[image_path]:
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


# ---------------------------------------------------------------------------
# image_labels (Open Images metadata)
# ---------------------------------------------------------------------------

def get_image_labels(db_path: Path, image_id: str) -> Optional[Dict[str, Any]]:
    """image_id のメタデータを返す。なければ None。"""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT labels, face_flags FROM image_labels WHERE image_id = ?",
            (image_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "labels": row["labels"].split(",") if row["labels"] else [],
        "face_flags": json.loads(row["face_flags"]),
    }


def upsert_image_labels(
    db_path: Path,
    image_id: str,
    labels: str,
    face_flags: str,
) -> None:
    """image_labels を upsert する。"""
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO image_labels (image_id, labels, face_flags) "
            "VALUES (?, ?, ?)",
            (image_id, labels, face_flags),
        )


def get_sample_order_by_idx(db_path: Path = DB_PATH) -> List[int]:
    """idx順のサンプル idx リストを返す（全カラムロードなし）。"""
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT idx FROM samples ORDER BY idx").fetchall()
    return [r["idx"] for r in rows]


def get_sample_order_by_col(
    col: str,
    direction: str = "ASC",
    db_path: Path = DB_PATH,
) -> List[int]:
    """eye_scale / norm_eye_scale / lap_var カラムでソートした idx リストを返す。
    NULL は末尾。カラムが存在しない場合は idx 順にフォールバック。
    """
    _ALLOWED = {"eye_scale", "norm_eye_scale", "lap_var"}
    if col not in _ALLOWED:
        raise ValueError(f"sort column must be one of {_ALLOWED}")
    if direction not in ("ASC", "DESC"):
        raise ValueError("direction must be ASC or DESC")
    with _connect(db_path) as conn:
        # カラムの存在確認（後方互換）
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(samples)").fetchall()}
        if col not in existing:
            return get_sample_order_by_idx(db_path)
        rows = conn.execute(
            f"SELECT idx FROM samples "
            f"ORDER BY {col} IS NULL, {col} {direction}, idx ASC"
        ).fetchall()
    return [r["idx"] for r in rows]


def get_col_min_per_image(
    col: str,
    db_path: Path = DB_PATH,
) -> Dict[str, float]:
    """指定カラムの per-image MIN 値を返す（ギャラリーソート用）。"""
    _ALLOWED = {"eye_scale", "norm_eye_scale", "lap_var"}
    if col not in _ALLOWED:
        raise ValueError(f"col must be one of {_ALLOWED}")
    with _connect(db_path) as conn:
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(samples)").fetchall()}
        if col not in existing:
            return {}
        rows = conn.execute(
            f"SELECT image_path, MIN({col}) AS v FROM samples "
            f"WHERE {col} IS NOT NULL GROUP BY image_path"
        ).fetchall()
    return {row["image_path"]: float(row["v"]) for row in rows}



# ---------------------------------------------------------------------------
# Server-side navigation (large-DB対応：クライアントに全idxを送らない)
# ---------------------------------------------------------------------------

def get_adjacent_idx(
    current_idx: int,
    delta: int,
    db_path: Path = DB_PATH,
) -> Optional[int]:
    """current_idx から delta 方向の次の非スキップサンプルの idx を返す。"""
    with _connect(db_path) as conn:
        if delta > 0:
            row = conn.execute(
                "SELECT MIN(idx) AS adj FROM samples WHERE idx > ? AND skip_flag = 0",
                (current_idx,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT MAX(idx) AS adj FROM samples WHERE idx < ? AND skip_flag = 0",
                (current_idx,),
            ).fetchone()
    return int(row["adj"]) if row and row["adj"] is not None else None


def get_next_eye_start_idx(
    current_idx: int,
    db_path: Path = DB_PATH,
) -> Optional[int]:
    """current_idx が属する目グループの次の非スキップ目グループ先頭 idx を返す。"""
    with _connect(db_path) as conn:
        cur = conn.execute(
            "SELECT image_path, face_index, part_name FROM samples WHERE idx = ?",
            (current_idx,),
        ).fetchone()
        if not cur:
            return None
        row = conn.execute(
            """SELECT MIN(idx) AS next_idx FROM samples
               WHERE idx > ?
                 AND NOT (image_path = ? AND face_index = ? AND part_name = ?)
                 AND skip_flag = 0""",
            (current_idx, cur["image_path"], cur["face_index"], cur["part_name"]),
        ).fetchone()
    return int(row["next_idx"]) if row and row["next_idx"] is not None else None


def get_prev_eye_start_idx(
    current_idx: int,
    db_path: Path = DB_PATH,
) -> Optional[int]:
    """current_idx が属する目グループの前の非スキップ目グループ先頭 idx を返す。"""
    with _connect(db_path) as conn:
        cur = conn.execute(
            "SELECT image_path, face_index, part_name FROM samples WHERE idx = ?",
            (current_idx,),
        ).fetchone()
        if not cur:
            return None
        # 前の非スキップ目グループの末尾 idx を取得
        row = conn.execute(
            """SELECT MAX(idx) AS prev_last FROM samples
               WHERE idx < ?
                 AND NOT (image_path = ? AND face_index = ? AND part_name = ?)
                 AND skip_flag = 0""",
            (current_idx, cur["image_path"], cur["face_index"], cur["part_name"]),
        ).fetchone()
        if not row or row["prev_last"] is None:
            return None
        prev_last = int(row["prev_last"])
        # その目グループの先頭を取得
        prev_grp = conn.execute(
            "SELECT image_path, face_index, part_name FROM samples WHERE idx = ?",
            (prev_last,),
        ).fetchone()
        if not prev_grp:
            return None
        row2 = conn.execute(
            """SELECT MIN(idx) AS start_idx FROM samples
               WHERE image_path = ? AND face_index = ? AND part_name = ?""",
            (prev_grp["image_path"], prev_grp["face_index"], prev_grp["part_name"]),
        ).fetchone()
    return int(row2["start_idx"]) if row2 and row2["start_idx"] is not None else None


def get_next_unannotated_idx(
    current_idx: int,
    db_path: Path = DB_PATH,
) -> Optional[int]:
    """current_idx より後の未アノテーション・非スキップの最初の idx を返す（末尾で折り返す）。"""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT MIN(idx) AS nxt FROM samples WHERE idx > ? AND t IS NULL AND skip_flag = 0",
            (current_idx,),
        ).fetchone()
        if row and row["nxt"] is not None:
            return int(row["nxt"])
        row2 = conn.execute(
            "SELECT MIN(idx) AS nxt FROM samples WHERE t IS NULL AND skip_flag = 0",
        ).fetchone()
    return int(row2["nxt"]) if row2 and row2["nxt"] is not None else None


def get_random_unannotated_idx(db_path: Path = DB_PATH) -> Optional[int]:
    """未アノテーション・非スキップのランダムなサンプル idx を返す。

    ORDER BY RANDOM() は全表スキャンになるため、代わりに:
    1. PRIMARY KEY の min/max を使ってランダム pivot を選ぶ (O(log n))
    2. pivot 以降の最初の未アノテーション行を探す (PRIMARY KEY スキャン)
    3. 見つからなければ先頭から探す
    99.8% の行が未アノテーションなので pivot の直後に見つかる。
    """
    import random as _random
    with _connect(db_path) as conn:
        # ORDER BY idx LIMIT 1 でPRIMARY KEY B-treeの端だけ読む (MIN/MAX より高速)
        lo_row = conn.execute("SELECT idx FROM samples ORDER BY idx ASC LIMIT 1").fetchone()
        hi_row = conn.execute("SELECT idx FROM samples ORDER BY idx DESC LIMIT 1").fetchone()
        if not lo_row:
            return None
        pivot = _random.randint(int(lo_row["idx"]), int(hi_row["idx"]))
        row = conn.execute(
            "SELECT idx FROM samples WHERE idx >= ? AND t IS NULL AND skip_flag = 0 LIMIT 1",
            (pivot,)
        ).fetchone()
        if row:
            return int(row["idx"])
        # wrap: try from beginning
        row = conn.execute(
            "SELECT idx FROM samples WHERE idx < ? AND t IS NULL AND skip_flag = 0 LIMIT 1",
            (pivot,)
        ).fetchone()
    return int(row["idx"]) if row else None


def get_first_idx(db_path: Path = DB_PATH) -> Optional[int]:
    """最初の非スキップサンプルの idx を返す。"""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT MIN(idx) AS first_idx FROM samples WHERE skip_flag = 0"
        ).fetchone()
    return int(row["first_idx"]) if row and row["first_idx"] is not None else None


def get_sample_groups(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """(image_path, face_index, part_name) グループの idx 範囲を返す。

    フロントエンドが sampleMeta を構築するために使用する。
    5.1M 個別サンプルの代わりに ~370K グループを返す。
    """
    with _connect(db_path) as conn:
        rows = conn.execute("""
            SELECT image_path, face_index, part_name,
                   MIN(idx) AS start_idx, COUNT(*) AS cnt
            FROM samples
            GROUP BY image_path, face_index, part_name
            ORDER BY start_idx
        """).fetchall()
    return [
        {
            "image_path": r["image_path"],
            "face_index": int(r["face_index"]),
            "part_name": r["part_name"],
            "start_idx": r["start_idx"],
            "count": r["cnt"],
        }
        for r in rows
    ]
