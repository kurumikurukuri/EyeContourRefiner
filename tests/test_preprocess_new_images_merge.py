"""tools/imagesdb_maintenance/preprocess_new_images.py の merge_to_samples_db テスト。

再実行（--merge-only のやり直し等）で samples.db に行が重複しないこと、dry-run で書き込まないこと、
samples.db が無いときに空 DB を作らないことを確認する。
"""
from __future__ import annotations

import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from annotator import db as _db  # noqa: E402

pni = importlib.import_module("tools.imagesdb_maintenance.preprocess_new_images")


def _write_shard(path: Path, image_path: str = "data/images/new.jpg") -> None:
    samples = [
        {"part_name": "eyeliner_right", "sub_key": "upper", "point_index": i,
         "px": 1.0, "py": 2.0, "qx": 3.0, "qy": 4.0}
        for i in range(2)
    ]
    obj = {"image_path": image_path, "faces": [
        {"face_index": 0, "face_yaw": 0.0, "face_pitch": 0.0, "face_roll": 0.0, "samples": samples},
    ]}
    path.write_text(json.dumps(obj) + "\n", encoding="utf-8")


def _count(db_path: Path) -> int:
    with sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True) as conn:
        return conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]


@pytest.fixture()
def samples_db(tmp_path, monkeypatch) -> Path:
    db_path = tmp_path / "samples.db"
    _db.init_db(db_path)
    monkeypatch.setattr(pni, "SAMPLES_DB", db_path)
    return db_path


def test_merge_twice_does_not_duplicate(samples_db, tmp_path):
    shard = tmp_path / "preprocess_shard_0.jsonl"
    _write_shard(shard)

    pni.merge_to_samples_db([shard])
    pni.merge_to_samples_db([shard])

    assert _count(samples_db) == 2


def test_merge_skips_duplicates_across_shards(samples_db, tmp_path):
    shard0 = tmp_path / "preprocess_shard_0.jsonl"
    shard1 = tmp_path / "preprocess_shard_1.jsonl"
    _write_shard(shard0)
    _write_shard(shard1)

    pni.merge_to_samples_db([shard0, shard1])

    assert _count(samples_db) == 2


def test_merge_dry_run_does_not_write(samples_db, tmp_path):
    shard = tmp_path / "preprocess_shard_0.jsonl"
    _write_shard(shard)

    pni.merge_to_samples_db([shard], dry_run=True)

    assert _count(samples_db) == 0


def _make_old_schema_db(db_path: Path) -> None:
    """追加カラム（face_yaw 等）が付与される前の旧スキーマの samples.db を作る。"""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE samples ("
            "idx INTEGER PRIMARY KEY, image_path TEXT NOT NULL, face_index INTEGER NOT NULL DEFAULT 0, "
            "part_name TEXT NOT NULL, sub_key TEXT NOT NULL, point_index INTEGER NOT NULL, "
            "px REAL NOT NULL, py REAL NOT NULL, qx REAL NOT NULL, qy REAL NOT NULL, "
            "t REAL, skip_flag INTEGER NOT NULL DEFAULT 0)"
        )
        conn.commit()
    finally:
        conn.close()


def _columns(db_path: Path) -> set[str]:
    with sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True) as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(samples)")}


def test_merge_migrates_old_schema_db(tmp_path, monkeypatch):
    """旧スキーマの samples.db でも、init_db が追加カラムを補充してからマージできる。"""
    db_path = tmp_path / "samples.db"
    _make_old_schema_db(db_path)
    monkeypatch.setattr(pni, "SAMPLES_DB", db_path)
    shard = tmp_path / "preprocess_shard_0.jsonl"
    _write_shard(shard)

    pni.merge_to_samples_db([shard])

    assert _count(db_path) == 2
    assert "face_yaw" in _columns(db_path)


def test_merge_dry_run_leaves_old_schema_untouched(tmp_path, monkeypatch):
    """dry-run は旧スキーマの DB でも動き、カラム追加もしない。"""
    db_path = tmp_path / "samples.db"
    _make_old_schema_db(db_path)
    monkeypatch.setattr(pni, "SAMPLES_DB", db_path)
    shard = tmp_path / "preprocess_shard_0.jsonl"
    _write_shard(shard)

    pni.merge_to_samples_db([shard], dry_run=True)

    assert _count(db_path) == 0
    assert "face_yaw" not in _columns(db_path)


def test_merge_does_not_create_missing_db(tmp_path, monkeypatch):
    db_path = tmp_path / "samples.db"
    monkeypatch.setattr(pni, "SAMPLES_DB", db_path)
    shard = tmp_path / "preprocess_shard_0.jsonl"
    _write_shard(shard)

    with pytest.raises(FileNotFoundError):
        pni.merge_to_samples_db([shard])
    assert not db_path.exists()
