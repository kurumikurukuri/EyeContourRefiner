"""samples.db からの眼サンプル取得ロジック（ブラウズ UI 用）。

t_pred_scan._get_eye_samples のロジックを移植するが、
ブラウズ時はアノテーション済みの眼も描画したいため
`t IS NULL` 条件を外し `skip_flag = 0` のみで絞る。
"""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


def fetch_eye_samples(
    samples_db: Path, image_path: str, part_name: str, repr_idx: int
) -> List[Dict[str, Any]]:
    """指定眼の P/Q サンプル一覧を取得する。

    repr_idx の PQ 距離の10倍以内の点のみを対象とする
    （同一画像内の別の目など無関係な点を除外するため）。
    repr_idx に対応する行が存在しない場合はフィルタなしで全件返す。
    """
    if not Path(samples_db).exists():
        return []

    with sqlite3.connect(str(samples_db)) as conn:
        conn.row_factory = sqlite3.Row
        ref = conn.execute(
            "SELECT px, py, qx, qy FROM samples WHERE idx = ?", (repr_idx,)
        ).fetchone()
        rows = conn.execute(
            """
            SELECT sub_key, point_index, px, py, qx, qy, eye_scale
            FROM samples
            WHERE image_path = ? AND part_name = ? AND skip_flag = 0
            ORDER BY sub_key, point_index
            """,
            (image_path, part_name),
        ).fetchall()

    if ref is None:
        return [dict(r) for r in rows]

    ref_mx = (float(ref["px"]) + float(ref["qx"])) / 2
    ref_my = (float(ref["py"]) + float(ref["qy"])) / 2
    ref_pq = max(
        math.hypot(float(ref["qx"]) - float(ref["px"]),
                   float(ref["qy"]) - float(ref["py"])),
        1.0,
    )
    threshold = ref_pq * 10.0

    result = []
    for r in rows:
        mx = (float(r["px"]) + float(r["qx"])) / 2
        my = (float(r["py"]) + float(r["qy"])) / 2
        if math.hypot(mx - ref_mx, my - ref_my) <= threshold:
            result.append(dict(r))
    return result
