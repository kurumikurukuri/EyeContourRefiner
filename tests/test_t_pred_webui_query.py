"""tools/annotation_support/t_pred_webui/query.py の fetch_eye_samples テスト。"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tools.annotation_support.t_pred_webui.query import fetch_eye_samples


def _create_samples_db(path: Path) -> None:
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """
            CREATE TABLE samples (
                idx INTEGER PRIMARY KEY,
                image_path TEXT NOT NULL,
                part_name TEXT NOT NULL,
                sub_key TEXT NOT NULL,
                point_index INTEGER NOT NULL,
                px REAL NOT NULL, py REAL NOT NULL,
                qx REAL NOT NULL, qy REAL NOT NULL,
                eye_scale REAL,
                t REAL,
                skip_flag INTEGER NOT NULL DEFAULT 0
            )
            """
        )


def _insert(conn, idx, image_path="a.jpg", part_name="eyeliner_left", sub_key="upper",
            point_index=0, px=100.0, py=100.0, qx=110.0, qy=100.0,
            eye_scale=60.0, t=None, skip_flag=0):
    conn.execute(
        """INSERT INTO samples
           (idx, image_path, part_name, sub_key, point_index, px, py, qx, qy,
            eye_scale, t, skip_flag)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (idx, image_path, part_name, sub_key, point_index, px, py, qx, qy,
         eye_scale, t, skip_flag),
    )


@pytest.fixture()
def samples_db(tmp_path) -> Path:
    path = tmp_path / "samples.db"
    _create_samples_db(path)
    return path


class TestFetchEyeSamples:
    def test_returns_samples_for_image_and_part(self, samples_db):
        with sqlite3.connect(str(samples_db)) as conn:
            _insert(conn, 0, point_index=0)
            _insert(conn, 1, point_index=1)
            conn.commit()
        rows = fetch_eye_samples(samples_db, "a.jpg", "eyeliner_left", repr_idx=0)
        assert len(rows) == 2

    def test_excludes_other_image_path(self, samples_db):
        with sqlite3.connect(str(samples_db)) as conn:
            _insert(conn, 0, image_path="a.jpg")
            _insert(conn, 1, image_path="b.jpg")
            conn.commit()
        rows = fetch_eye_samples(samples_db, "a.jpg", "eyeliner_left", repr_idx=0)
        assert len(rows) == 1

    def test_excludes_other_part_name(self, samples_db):
        with sqlite3.connect(str(samples_db)) as conn:
            _insert(conn, 0, part_name="eyeliner_left")
            _insert(conn, 1, part_name="eyeliner_right")
            conn.commit()
        rows = fetch_eye_samples(samples_db, "a.jpg", "eyeliner_left", repr_idx=0)
        assert len(rows) == 1

    def test_excludes_skip_flag(self, samples_db):
        with sqlite3.connect(str(samples_db)) as conn:
            _insert(conn, 0, skip_flag=0)
            _insert(conn, 1, skip_flag=1)
            conn.commit()
        rows = fetch_eye_samples(samples_db, "a.jpg", "eyeliner_left", repr_idx=0)
        assert len(rows) == 1

    def test_includes_annotated_eyes_unlike_scan(self, samples_db):
        """t_pred_scan の t IS NULL 条件を緩め、アノテーション済みも含む。"""
        with sqlite3.connect(str(samples_db)) as conn:
            _insert(conn, 0, t=0.5, skip_flag=0)
            conn.commit()
        rows = fetch_eye_samples(samples_db, "a.jpg", "eyeliner_left", repr_idx=0)
        assert len(rows) == 1

    def test_filters_by_pq_distance_from_repr(self, samples_db):
        """repr_idx の PQ 距離の10倍を超える点は除外される。"""
        with sqlite3.connect(str(samples_db)) as conn:
            # repr: px=0,py=0,qx=10,qy=0 → pq=10, threshold=100
            _insert(conn, 0, point_index=0, px=0.0, py=0.0, qx=10.0, qy=0.0)
            # 近い点 (mid差 5) は含まれる
            _insert(conn, 1, point_index=1, px=5.0, py=0.0, qx=15.0, qy=0.0)
            # 遠い点 (mid差 1000) は除外される
            _insert(conn, 2, point_index=2, px=1000.0, py=0.0, qx=1010.0, qy=0.0)
            conn.commit()
        rows = fetch_eye_samples(samples_db, "a.jpg", "eyeliner_left", repr_idx=0)
        assert {r["point_index"] for r in rows} == {0, 1}

    def test_returns_expected_columns(self, samples_db):
        with sqlite3.connect(str(samples_db)) as conn:
            _insert(conn, 0)
            conn.commit()
        rows = fetch_eye_samples(samples_db, "a.jpg", "eyeliner_left", repr_idx=0)
        expected_keys = {"sub_key", "point_index", "px", "py", "qx", "qy", "eye_scale"}
        assert expected_keys.issubset(rows[0].keys())

    def test_ordered_by_sub_key_then_point_index(self, samples_db):
        with sqlite3.connect(str(samples_db)) as conn:
            _insert(conn, 0, sub_key="lower", point_index=1)
            _insert(conn, 1, sub_key="upper", point_index=2)
            _insert(conn, 2, sub_key="upper", point_index=0)
            conn.commit()
        rows = fetch_eye_samples(samples_db, "a.jpg", "eyeliner_left", repr_idx=0)
        ordering = [(r["sub_key"], r["point_index"]) for r in rows]
        assert ordering == sorted(ordering)

    def test_no_matching_rows_returns_empty(self, samples_db):
        rows = fetch_eye_samples(samples_db, "nonexistent.jpg", "eyeliner_left", repr_idx=0)
        assert rows == []

    def test_repr_idx_not_found_returns_unfiltered_rows(self, samples_db):
        """repr_idx に対応する行が無い場合は PQ 距離フィルタを適用せず全件返す。"""
        with sqlite3.connect(str(samples_db)) as conn:
            _insert(conn, 0)
            conn.commit()
        rows = fetch_eye_samples(samples_db, "a.jpg", "eyeliner_left", repr_idx=999)
        assert len(rows) == 1

    def test_db_not_exists_returns_empty(self, tmp_path):
        missing = tmp_path / "missing.db"
        assert fetch_eye_samples(missing, "a.jpg", "eyeliner_left", repr_idx=0) == []
