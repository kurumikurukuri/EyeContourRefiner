"""annotator/db.py のテスト。"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from annotator.db import (
    init_db,
    load_all_samples,
    get_sample_by_idx,
    get_samples_by_image_path,
    count_samples,
    update_annotation,
    update_skip_flag,
    get_unannotated_samples_for_eye,
    batch_update_t,
    delete_samples_by_image_paths,
    delete_orphan_samples,
    replace_all_samples,
    upsert_samples,
    get_image_index,
    get_annotated_samples,
    image_path_exists,
)


def _make_sample(idx: int, image_path: str = "data/images/face.jpg", **kwargs) -> dict:
    base = {
        "idx": idx,
        "image_path": image_path,
        "face_index": 0,
        "part_name": "eyeliner_left",
        "sub_key": "upper",
        "point_index": idx % 7,
        "px": 100.0 + idx,
        "py": 200.0,
        "qx": 150.0 + idx,
        "qy": 195.0,
        "t": None,
        "skip_flag": False,
    }
    return {**base, **kwargs}


@pytest.fixture()
def db_path(tmp_path) -> Path:
    path = tmp_path / "test_samples.db"
    init_db(path)
    return path


@pytest.fixture()
def populated_db(db_path) -> Path:
    """5 件挿入済み DB。"""
    samples = [_make_sample(i) for i in range(5)]
    upsert_samples(samples, db_path)
    return db_path


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

class TestInitDb:
    def test_creates_db_file(self, tmp_path):
        path = tmp_path / "new.db"
        assert not path.exists()
        init_db(path)
        assert path.exists()

    def test_idempotent(self, db_path):
        """2 回呼んでもエラーにならない。"""
        init_db(db_path)
        init_db(db_path)


# ---------------------------------------------------------------------------
# upsert / load_all_samples
# ---------------------------------------------------------------------------

class TestLoadAll:
    def test_empty(self, db_path):
        assert load_all_samples(db_path) == []

    def test_basic(self, db_path):
        upsert_samples([_make_sample(0), _make_sample(1)], db_path)
        rows = load_all_samples(db_path)
        assert len(rows) == 2

    def test_ordered_by_idx(self, db_path):
        upsert_samples([_make_sample(2), _make_sample(0), _make_sample(1)], db_path)
        rows = load_all_samples(db_path)
        assert [r["idx"] for r in rows] == [0, 1, 2]

    def test_skip_flag_is_bool(self, db_path):
        upsert_samples([_make_sample(0, skip_flag=True)], db_path)
        row = load_all_samples(db_path)[0]
        assert row["skip_flag"] is True

    def test_t_none_preserved(self, db_path):
        upsert_samples([_make_sample(0, t=None)], db_path)
        row = load_all_samples(db_path)[0]
        assert row["t"] is None


# ---------------------------------------------------------------------------
# get_sample_by_idx
# ---------------------------------------------------------------------------

class TestGetSampleByIdx:
    def test_found(self, populated_db):
        row = get_sample_by_idx(2, populated_db)
        assert row is not None
        assert row["idx"] == 2

    def test_not_found(self, populated_db):
        assert get_sample_by_idx(999, populated_db) is None

    def test_negative(self, populated_db):
        assert get_sample_by_idx(-1, populated_db) is None


# ---------------------------------------------------------------------------
# get_samples_by_image_path
# ---------------------------------------------------------------------------

class TestGetSamplesByImagePath:
    def test_returns_matching(self, db_path):
        samples = [
            _make_sample(0, image_path="a.jpg"),
            _make_sample(1, image_path="b.jpg"),
            _make_sample(2, image_path="a.jpg"),
        ]
        upsert_samples(samples, db_path)
        rows = get_samples_by_image_path("a.jpg", db_path)
        assert len(rows) == 2
        assert all(r["image_path"] == "a.jpg" for r in rows)

    def test_not_found(self, populated_db):
        assert get_samples_by_image_path("nonexistent.jpg", populated_db) == []


# ---------------------------------------------------------------------------
# count_samples
# ---------------------------------------------------------------------------

class TestCountSamples:
    def test_empty(self, db_path):
        assert count_samples(db_path) == (0, 0)

    def test_partial_annotated(self, db_path):
        samples = [
            _make_sample(0, t=0.5),
            _make_sample(1, t=None),
            _make_sample(2, t=0.3),
        ]
        upsert_samples(samples, db_path)
        total, annotated = count_samples(db_path)
        assert total == 3
        assert annotated == 2


# ---------------------------------------------------------------------------
# update_annotation
# ---------------------------------------------------------------------------

class TestUpdateAnnotation:
    def test_sets_t(self, populated_db):
        update_annotation(0, 0.75, populated_db)
        row = get_sample_by_idx(0, populated_db)
        assert abs(row["t"] - 0.75) < 1e-9

    def test_sets_t_another(self, populated_db):
        update_annotation(1, 0.5, populated_db)
        row = get_sample_by_idx(1, populated_db)
        assert abs(row["t"] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# update_skip_flag
# ---------------------------------------------------------------------------

class TestUpdateSkipFlag:
    def test_updates_all_for_image(self, db_path):
        samples = [
            _make_sample(0, image_path="a.jpg"),
            _make_sample(1, image_path="a.jpg"),
            _make_sample(2, image_path="b.jpg"),
        ]
        upsert_samples(samples, db_path)
        count = update_skip_flag("a.jpg", True, db_path)
        assert count == 2
        rows = get_samples_by_image_path("a.jpg", db_path)
        assert all(r["skip_flag"] for r in rows)
        rows_b = get_samples_by_image_path("b.jpg", db_path)
        assert not rows_b[0]["skip_flag"]

    def test_returns_zero_for_missing(self, populated_db):
        assert update_skip_flag("nonexistent.jpg", True, populated_db) == 0


# ---------------------------------------------------------------------------
# get_unannotated_samples_for_eye
# ---------------------------------------------------------------------------

class TestGetUnannotatedForEye:
    def test_returns_unannotated(self, db_path):
        samples = [
            _make_sample(0, t=None, face_index=0, part_name="eyeliner_left"),
            _make_sample(1, t=0.5, face_index=0, part_name="eyeliner_left"),
            _make_sample(2, t=None, face_index=0, part_name="eyeliner_left"),
        ]
        upsert_samples(samples, db_path)
        rows = get_unannotated_samples_for_eye("data/images/face.jpg", 0, "eyeliner_left", db_path)
        assert len(rows) == 2
        assert all(r["t"] is None for r in rows)


# ---------------------------------------------------------------------------
# batch_update_t
# ---------------------------------------------------------------------------

class TestBatchUpdateT:
    def test_updates_multiple(self, populated_db):
        batch_update_t([(0, 0.1), (2, 0.3)], populated_db)
        assert abs(get_sample_by_idx(0, populated_db)["t"] - 0.1) < 1e-9
        assert abs(get_sample_by_idx(2, populated_db)["t"] - 0.3) < 1e-9

    def test_empty_updates(self, populated_db):
        batch_update_t([], populated_db)  # エラーにならない


# ---------------------------------------------------------------------------
# delete_samples_by_image_paths
# ---------------------------------------------------------------------------

class TestDeleteByImagePaths:
    def test_deletes(self, db_path):
        samples = [_make_sample(0, image_path="a.jpg"), _make_sample(1, image_path="b.jpg")]
        upsert_samples(samples, db_path)
        n = delete_samples_by_image_paths(["a.jpg"], db_path)
        assert n == 1
        assert load_all_samples(db_path)[0]["image_path"] == "b.jpg"

    def test_empty_list(self, populated_db):
        assert delete_samples_by_image_paths([], populated_db) == 0


# ---------------------------------------------------------------------------
# delete_orphan_samples
# ---------------------------------------------------------------------------

class TestDeleteOrphanSamples:
    def test_keeps_valid_deletes_orphans(self, db_path):
        samples = [_make_sample(0, image_path="a.jpg"), _make_sample(1, image_path="b.jpg")]
        upsert_samples(samples, db_path)
        n = delete_orphan_samples(["a.jpg"], db_path)
        assert n == 1
        rows = load_all_samples(db_path)
        assert len(rows) == 1
        assert rows[0]["image_path"] == "a.jpg"


# ---------------------------------------------------------------------------
# replace_all_samples
# ---------------------------------------------------------------------------

class TestReplaceAllSamples:
    def test_replaces(self, populated_db):
        new_samples = [_make_sample(0, image_path="new.jpg")]
        replace_all_samples(new_samples, populated_db)
        rows = load_all_samples(populated_db)
        assert len(rows) == 1
        assert rows[0]["image_path"] == "new.jpg"

    def test_preserves_annotations_via_caller(self, db_path):
        """replace_all_samples 自体は保持しないが、呼び出し前にマージすれば保持できる。"""
        upsert_samples([_make_sample(0, t=0.5)], db_path)
        new = [_make_sample(0, t=0.5)]  # 呼び出し元がマージ済みデータを渡す
        replace_all_samples(new, db_path)
        assert abs(get_sample_by_idx(0, db_path)["t"] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# get_image_index
# ---------------------------------------------------------------------------

class TestGetImageIndex:
    def test_structure(self, db_path):
        samples = [
            _make_sample(0, image_path="a.jpg", t=0.5),
            _make_sample(1, image_path="a.jpg", t=None),
            _make_sample(2, image_path="b.jpg", t=0.3),
        ]
        upsert_samples(samples, db_path)
        index, order = get_image_index(db_path)
        assert order == ["a.jpg", "b.jpg"]
        assert index["a.jpg"]["total"] == 2
        assert index["a.jpg"]["annotated"] == 1
        assert index["b.jpg"]["annotated"] == 1

    def test_skip_flag_in_index(self, db_path):
        samples = [
            _make_sample(0, image_path="a.jpg", skip_flag=True),
            _make_sample(1, image_path="a.jpg", skip_flag=False),
        ]
        upsert_samples(samples, db_path)
        index, _ = get_image_index(db_path)
        assert index["a.jpg"]["skipped"] is True


# ---------------------------------------------------------------------------
# get_annotated_samples
# ---------------------------------------------------------------------------

class TestGetAnnotatedSamples:
    def test_filters_null_t(self, db_path):
        samples = [_make_sample(0, t=0.5), _make_sample(1, t=None), _make_sample(2, t=0.3)]
        upsert_samples(samples, db_path)
        rows = get_annotated_samples(db_path)
        assert len(rows) == 2
        assert all(r["t"] is not None for r in rows)


# ---------------------------------------------------------------------------
# image_path_exists
# ---------------------------------------------------------------------------

class TestImagePathExists:
    def test_exists(self, populated_db):
        assert image_path_exists("data/images/face.jpg", populated_db) is True

    def test_not_exists(self, populated_db):
        assert image_path_exists("nonexistent.jpg", populated_db) is False


# ---------------------------------------------------------------------------
# Concurrent access (WAL mode)
# ---------------------------------------------------------------------------

class TestFaceOrientationColumns:
    def test_roundtrip_face_pose_values(self, db_path):
        upsert_samples(
            [_make_sample(0, face_yaw=12.3, face_pitch=-4.5, face_roll=1.0)],
            db_path,
        )
        row = load_all_samples(db_path)[0]
        assert row["face_yaw"] == pytest.approx(12.3)
        assert row["face_pitch"] == pytest.approx(-4.5)
        assert row["face_roll"] == pytest.approx(1.0)

    def test_missing_face_pose_stored_as_null(self, db_path):
        upsert_samples([_make_sample(0)], db_path)  # face_* 未指定
        row = load_all_samples(db_path)[0]
        assert row["face_yaw"] is None
        assert row["face_pitch"] is None
        assert row["face_roll"] is None

    def test_migration_adds_missing_columns(self, tmp_path):
        """旧スキーマ DB → init_db で face_* カラムが自動付与される。"""
        path = tmp_path / "old.db"
        with sqlite3.connect(str(path)) as conn:
            conn.execute(
                """CREATE TABLE samples (
                    idx INTEGER PRIMARY KEY,
                    image_path TEXT NOT NULL,
                    face_index INTEGER NOT NULL DEFAULT 0,
                    part_name TEXT NOT NULL,
                    sub_key TEXT NOT NULL,
                    point_index INTEGER NOT NULL,
                    px REAL NOT NULL, py REAL NOT NULL,
                    qx REAL NOT NULL, qy REAL NOT NULL,
                    t REAL, skip_flag INTEGER NOT NULL DEFAULT 0
                )"""
            )
            conn.execute(
                """INSERT INTO samples VALUES
                   (0, 'img.jpg', 0, 'eyeliner_left', 'upper', 0,
                    1.0, 2.0, 3.0, 4.0, NULL, 0)"""
            )

        init_db(path)  # マイグレーション発火

        with sqlite3.connect(str(path)) as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(samples)").fetchall()}
        assert "face_yaw" in cols
        assert "face_pitch" in cols
        assert "face_roll" in cols

        row = load_all_samples(path)[0]
        assert row["face_yaw"] is None


class TestConcurrentAccess:
    def test_parallel_reads(self, populated_db):
        """複数スレッドからの同時読み取りが正常終了する。"""
        errors: list[Exception] = []

        def reader():
            try:
                load_all_samples(populated_db)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

    def test_parallel_writes(self, db_path):
        """複数スレッドからの同時書き込みが正常終了する。"""
        samples = [_make_sample(i) for i in range(20)]
        upsert_samples(samples, db_path)

        errors: list[Exception] = []

        def writer(idx: int):
            try:
                update_annotation(idx, 0.5, db_path)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        total, annotated = count_samples(db_path)
        assert annotated == 20
