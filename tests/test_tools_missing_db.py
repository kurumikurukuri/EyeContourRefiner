"""samples.db を開く tools が、DB 不在時に空 DB を作らず FileNotFoundError で止まることのテスト。

sqlite3.connect は存在しないパスに 0 バイトのファイルを作り、以降の exists() ガードを欺く。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _load(rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_not_created(call, db_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        call()
    assert not db_path.exists()


def test_compare_dbs(tmp_path):
    mod = _load("tools/sampledb_maintenance/compare_dbs.py")
    db_path = tmp_path / "samples.db"
    _assert_not_created(lambda: mod.load_keys_with_t(db_path), db_path)
    _assert_not_created(lambda: mod.load_all_keys(db_path), db_path)


def test_fetch_oi_image_labels(tmp_path):
    pytest.importorskip("requests")
    mod = _load("tools/sampledb_maintenance/fetch_oi_image_labels.py")
    db_path = tmp_path / "samples.db"
    _assert_not_created(lambda: mod.load_db_image_ids(str(db_path)), db_path)


def test_backfill_db_features(tmp_path):
    mod = _load("tools/imagesdb_maintenance/backfill_db_features.py")
    db_path = tmp_path / "samples.db"
    _assert_not_created(
        lambda: mod.backfill(samples_db=db_path, images_db=tmp_path / "images.db"), db_path
    )


def test_export_db_to_csv(tmp_path, monkeypatch):
    pytest.importorskip("pyiqa")
    mod = _load("tools/csv_export/export_db_to_csv.py")
    db_path = tmp_path / "samples.db"
    monkeypatch.setattr(mod, "DB_PATH", str(db_path))
    _assert_not_created(mod.load_records, db_path)
