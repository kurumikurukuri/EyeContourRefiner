"""tools/sampledb_maintenance/rebuild_samples_db.py のテスト。

manifest から samples.db を再構築する処理と、既存 DB を壊さないためのガード
（一致なら何もしない / 不一致は拒否 / --overwrite-db 時のみバックアップして置換）。
"""
from __future__ import annotations

import gc
import logging
import math
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from annotator import db as sample_db  # noqa: E402
from tools.data_management import manifest_source  # noqa: E402
from tools.data_management.manifest_source import ManifestRow  # noqa: E402
from tools.sampledb_maintenance import rebuild_samples_db as rb  # noqa: E402

FIXED_NOW = datetime(2026, 9, 21, 12, 34, 56)


def _row(idx: int, image_id="img_a", face_index=0, part="eyeliner_left", sub="upper",
         point=0, px=10.0, py=20.0, qx=13.0, qy=16.0, t=0.5, nose=(30.0, 40.0)) -> ManifestRow:
    return ManifestRow(
        idx=idx, image_id=image_id, image_path=f"data/images/{image_id}.jpg",
        face_index=face_index, part_name=part, sub_key=sub, point_index=point,
        px=px, py=py, qx=qx, qy=qy, t=t,
        nose_x=None if nose is None else nose[0], nose_y=None if nose is None else nose[1],
    )


def _rows() -> tuple[ManifestRow, ...]:
    """1 画像・1 顔・左目 upper 3 点 + 右目 upper 2 点 + 別画像 1 点。"""
    return (
        _row(10, point=0, px=10, py=20, qx=13, qy=16, t=0.25),
        _row(11, point=1, px=20, py=22, qx=22, qy=18, t=0.5),
        _row(12, point=2, px=30, py=24, qx=31, qy=17, t=1.25),
        _row(20, part="eyeliner_right", point=0, px=60, py=20, qx=64, qy=15, t=0.75),
        _row(21, part="eyeliner_right", point=1, px=70, py=21, qx=71, qy=14, t=-0.1),
        _row(30, image_id="img_b", point=0, px=1, py=1, qx=1, qy=1, t=0.9, nose=None),
    )


def _write_jpeg(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((height, width, 3), 128, dtype=np.uint8))


def _load(db_path: Path) -> list[dict]:
    return sample_db.load_all_samples(db_path)


def _core(rows) -> list[tuple]:
    return [(r.idx, r.image_path, r.face_index, r.part_name, r.sub_key, r.point_index,
             r.px, r.py, r.qx, r.qy, r.t) for r in rows]


def _db_core(samples: list[dict]) -> list[tuple]:
    return [(s["idx"], s["image_path"], s["face_index"], s["part_name"], s["sub_key"],
             s["point_index"], s["px"], s["py"], s["qx"], s["qy"], s["t"]) for s in samples]


class TestBuildSamples:

    def test_maps_manifest_rows_to_db_samples(self):
        samples = rb.build_samples(_rows(), images_dir=None)

        assert [s["idx"] for s in samples] == [10, 11, 12, 20, 21, 30]
        first = samples[0]
        assert first["image_path"] == "data/images/img_a.jpg"
        assert (first["face_index"], first["part_name"], first["sub_key"], first["point_index"]) == (
            0, "eyeliner_left", "upper", 0)
        assert (first["px"], first["py"], first["qx"], first["qy"], first["t"]) == (10, 20, 13, 16, 0.25)
        assert first["skip_flag"] is False
        assert (first["nose_x"], first["nose_y"]) == (30.0, 40.0)
        assert samples[-1]["nose_x"] is None
        for key in ("face_yaw", "face_pitch", "face_roll", "lap_var"):
            assert first[key] is None

    def test_eye_scale_is_pq_bounding_box_diagonal_per_eye(self):
        samples = {s["idx"]: s for s in rb.build_samples(_rows(), images_dir=None)}

        # 左目 3 点: x ∈ [10, 31], y ∈ [16, 24] → hypot(21, 8)
        expected_left = math.hypot(31 - 10, 24 - 16)
        assert samples[10]["eye_scale"] == pytest.approx(expected_left)
        assert samples[11]["eye_scale"] == samples[12]["eye_scale"] == samples[10]["eye_scale"]
        # 右目 2 点: x ∈ [60, 71], y ∈ [14, 21]
        assert samples[20]["eye_scale"] == pytest.approx(math.hypot(11, 7))

    def test_eye_scale_has_a_floor_of_one_pixel(self):
        samples = rb.build_samples(_rows(), images_dir=None)

        assert samples[-1]["eye_scale"] == 1.0  # img_b は P=Q の 1 点のみ

    def test_eye_scale_groups_by_face_and_image(self):
        rows = (
            _row(0, face_index=0, px=0, py=0, qx=3, qy=4),
            _row(1, face_index=1, px=100, py=100, qx=100, qy=110),
            _row(2, image_id="other", px=0, py=0, qx=30, qy=40),
        )

        by_idx = {s["idx"]: s["eye_scale"] for s in rb.build_samples(rows, images_dir=None)}

        assert by_idx == {0: pytest.approx(5.0), 1: pytest.approx(10.0), 2: pytest.approx(50.0)}

    def test_norm_eye_scale_uses_the_image_diagonal_when_the_image_exists(self, tmp_path):
        _write_jpeg(tmp_path / "img_a.jpg", width=300, height=400)

        samples = {s["idx"]: s for s in rb.build_samples(_rows(), images_dir=tmp_path)}

        assert samples[10]["norm_eye_scale"] == pytest.approx(samples[10]["eye_scale"] / 500.0)
        assert samples[30]["norm_eye_scale"] is None  # img_b の画像は無い

    def test_norm_eye_scale_is_null_without_images_dir(self):
        assert all(s["norm_eye_scale"] is None for s in rb.build_samples(_rows(), images_dir=None))

    def test_corrupt_image_leaves_norm_eye_scale_null(self, tmp_path):
        (tmp_path / "img_a.jpg").write_bytes(b"not an image")

        samples = rb.build_samples(_rows(), images_dir=tmp_path)

        assert samples[0]["norm_eye_scale"] is None

    def test_inputs_are_not_mutated(self):
        rows = _rows()

        rb.build_samples(rows, images_dir=None)

        assert rows == _rows()


class TestCreate:

    def test_creates_db_from_manifest(self, tmp_path):
        db_path = tmp_path / "data" / "samples.db"

        result = rb.rebuild_samples_db(_rows(), db_path, images_dir=None, now=lambda: FIXED_NOW)

        # 単一ファイルとして完結していること（-wal/-shm を残さない）。読み込みで WAL が
        # 作られる前に確認する（annotator.db は WAL モードで接続を GC まで閉じない）。
        assert sorted(p.name for p in db_path.parent.iterdir()) == ["samples.db"]
        assert result.action == "created"
        assert (result.n_rows, result.n_images) == (6, 2)
        assert result.backup_path is None
        assert _db_core(_load(db_path)) == _core(_rows())

    def test_created_db_has_annotator_schema_and_annotation_counts(self, tmp_path):
        db_path = tmp_path / "samples.db"

        rb.rebuild_samples_db(_rows(), db_path, images_dir=None, now=lambda: FIXED_NOW)

        assert sample_db.count_samples(db_path) == (6, 6)
        with sqlite3.connect(db_path) as conn:
            columns = {r[1] for r in conn.execute("PRAGMA table_info(samples)")}
        assert {"face_yaw", "lap_var", "eye_scale", "norm_eye_scale", "nose_x"} <= columns

    def test_negative_and_extreme_t_are_preserved(self, tmp_path):
        db_path = tmp_path / "samples.db"

        rb.rebuild_samples_db(_rows(), db_path, images_dir=None, now=lambda: FIXED_NOW)

        assert sorted(s["t"] for s in _load(db_path))[0] == -0.1

    def test_empty_manifest_is_rejected(self, tmp_path):
        with pytest.raises(rb.RebuildError, match="空"):
            rb.rebuild_samples_db((), tmp_path / "samples.db", images_dir=None)


class TestExistingDbGuard:

    def _existing_with_other_rows(self, db_path: Path) -> None:
        sample_db.init_db(db_path)
        sample_db.replace_all_samples(rb.build_samples(
            (_row(1, image_id="someone_else", t=0.3), _row(2, image_id="someone_else", point=1, t=0.4)),
            images_dir=None), db_path)

    def test_matching_db_is_left_untouched(self, tmp_path):
        db_path = tmp_path / "samples.db"
        rb.rebuild_samples_db(_rows(), db_path, images_dir=None, now=lambda: FIXED_NOW)
        before = db_path.read_bytes()

        result = rb.rebuild_samples_db(_rows(), db_path, images_dir=None, now=lambda: FIXED_NOW)

        assert result.action == "unchanged"
        assert db_path.read_bytes() == before
        assert not list(tmp_path.glob("*.bak-*"))

    def test_matching_db_keeps_restored_features(self, tmp_path):
        """features 復元後に db ステップを再実行しても、復元済みの値を消さない。"""
        db_path = tmp_path / "samples.db"
        rb.rebuild_samples_db(_rows(), db_path, images_dir=None, now=lambda: FIXED_NOW)
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE samples SET face_yaw = 12.5, lap_var = 99.0, skip_flag = 1")

        result = rb.rebuild_samples_db(_rows(), db_path, images_dir=None, now=lambda: FIXED_NOW)

        assert result.action == "unchanged"
        assert {s["face_yaw"] for s in _load(db_path)} == {12.5}

    def test_different_db_is_refused_and_not_modified(self, tmp_path):
        db_path = tmp_path / "samples.db"
        self._existing_with_other_rows(db_path)
        before = _load(db_path)

        with pytest.raises(rb.DbGuardError, match="overwrite-db"):
            rb.rebuild_samples_db(_rows(), db_path, images_dir=None, now=lambda: FIXED_NOW)

        assert _load(db_path) == before
        assert not list(tmp_path.glob("*.bak-*"))

    def test_db_with_extra_unannotated_rows_counts_as_different(self, tmp_path):
        db_path = tmp_path / "samples.db"
        rb.rebuild_samples_db(_rows(), db_path, images_dir=None, now=lambda: FIXED_NOW)
        extra = rb.build_samples((_row(999, image_id="unlabeled", t=0.0),), images_dir=None)
        sample_db.upsert_samples([{**extra[0], "t": None}], db_path)

        with pytest.raises(rb.DbGuardError):
            rb.rebuild_samples_db(_rows(), db_path, images_dir=None, now=lambda: FIXED_NOW)

    def test_modified_annotation_counts_as_different(self, tmp_path):
        db_path = tmp_path / "samples.db"
        rb.rebuild_samples_db(_rows(), db_path, images_dir=None, now=lambda: FIXED_NOW)
        sample_db.update_annotation(11, 0.123, db_path)

        with pytest.raises(rb.DbGuardError):
            rb.rebuild_samples_db(_rows(), db_path, images_dir=None, now=lambda: FIXED_NOW)

    def test_overwrite_backs_up_then_replaces(self, tmp_path):
        db_path = tmp_path / "samples.db"
        self._existing_with_other_rows(db_path)
        old_rows = _load(db_path)

        result = rb.rebuild_samples_db(
            _rows(), db_path, images_dir=None, overwrite=True, now=lambda: FIXED_NOW)

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "samples.db", "samples.db.bak-20260921-123456"]
        assert result.action == "replaced"
        assert result.backup_path == tmp_path / "samples.db.bak-20260921-123456"
        assert _load(result.backup_path) == old_rows
        assert _db_core(_load(db_path)) == _core(_rows())

    def test_overwrite_never_reuses_an_existing_backup_name(self, tmp_path):
        db_path = tmp_path / "samples.db"
        self._existing_with_other_rows(db_path)
        taken = tmp_path / "samples.db.bak-20260921-123456"
        taken.write_bytes(b"precious")

        result = rb.rebuild_samples_db(
            _rows(), db_path, images_dir=None, overwrite=True, now=lambda: FIXED_NOW)

        assert taken.read_bytes() == b"precious"
        assert result.backup_path is not None and result.backup_path != taken
        assert result.backup_path.name.startswith("samples.db.bak-20260921-123456")

    def test_replace_does_not_resurrect_rows_from_a_stale_wal(self, tmp_path):
        db_path = tmp_path / "samples.db"
        self._existing_with_other_rows(db_path)
        keeper = sqlite3.connect(db_path)  # 他プロセス相当: WAL を保持したまま開いている
        keeper.execute("SELECT COUNT(*) FROM samples").fetchone()
        try:
            rb.rebuild_samples_db(
                _rows(), db_path, images_dir=None, overwrite=True, now=lambda: FIXED_NOW)
        except (PermissionError, rb.RebuildError):
            pytest.skip("Windows では開いたままの DB は置換できない（エラーで安全に中止する）")
        finally:
            keeper.close()

        assert _db_core(_load(db_path)) == _core(_rows())

    def test_wal_left_behind_by_a_crashed_process_is_not_replayed_into_the_new_db(self, tmp_path):
        """クラッシュ後に残った旧 DB の -wal が、置換後の新 DB に再生されて行が混ざらないこと。"""
        live = tmp_path / "live"
        crashed = tmp_path / "crashed"
        live.mkdir()
        crashed.mkdir()
        old = live / "samples.db"
        sample_db.init_db(old)
        gc.collect()
        writer = sqlite3.connect(old)
        try:
            writer.execute("PRAGMA wal_autocheckpoint=0")
            writer.execute(
                "INSERT INTO samples (idx, image_path, part_name, sub_key, point_index, px, py, qx, qy, t)"
                " VALUES (777, 'data/images/ghost.jpg', 'eyeliner_left', 'upper', 0, 1, 1, 2, 2, 0.5)")
            writer.commit()
            # 書き込み中のプロセスが落ちた状態を再現: 本体と未チェックポイントの -wal だけをコピーする
            shutil.copy2(old, crashed / "samples.db")
            shutil.copy2(f"{old}-wal", crashed / "samples.db-wal")
        finally:
            writer.close()
        assert (crashed / "samples.db-wal").stat().st_size > 0

        rb.rebuild_samples_db(
            _rows(), crashed / "samples.db", images_dir=None, overwrite=True, now=lambda: FIXED_NOW)

        gc.collect()
        assert {s["idx"] for s in _load(crashed / "samples.db")} == {r.idx for r in _rows()}

    def test_orphaned_wal_without_a_main_db_is_not_replayed_when_creating(self, tmp_path):
        """samples.db だけ消して -wal が残った状態で新規作成しても、古い WAL が新 DB に混ざらない。"""
        live = tmp_path / "live"
        orphan_dir = tmp_path / "orphan"
        live.mkdir()
        orphan_dir.mkdir()
        old = live / "samples.db"
        sample_db.init_db(old)
        gc.collect()
        writer = sqlite3.connect(old)
        try:
            writer.execute("PRAGMA wal_autocheckpoint=0")
            writer.execute(
                "INSERT INTO samples (idx, image_path, part_name, sub_key, point_index, px, py, qx, qy, t)"
                " VALUES (777, 'data/images/ghost.jpg', 'eyeliner_left', 'upper', 0, 1, 1, 2, 2, 0.5)")
            writer.commit()
            shutil.copy2(f"{old}-wal", orphan_dir / "samples.db-wal")  # 本体は無く、WAL だけが残る
        finally:
            writer.close()
        (orphan_dir / "samples.db-journal").write_bytes(b"stale rollback journal")
        assert not (orphan_dir / "samples.db").exists()

        result = rb.rebuild_samples_db(
            _rows(), orphan_dir / "samples.db", images_dir=None, now=lambda: FIXED_NOW)

        gc.collect()
        assert result.action == "created"
        assert {s["idx"] for s in _load(orphan_dir / "samples.db")} == {r.idx for r in _rows()}

    @pytest.mark.parametrize("message", ["database is locked", "unable to open database file"])
    def test_an_unreadable_but_healthy_db_is_never_treated_as_garbage(self, tmp_path, monkeypatch, message):
        """ロック中など『読めない理由が壊れているからではない』DB は、生コピーや置換をせずエラーで止める。"""
        db_path = tmp_path / "samples.db"
        self._existing_with_other_rows(db_path)
        gc.collect()
        before = db_path.read_bytes()

        def locked(path):
            raise sqlite3.OperationalError(message)

        monkeypatch.setattr(rb, "read_db_core", locked)

        with pytest.raises(rb.RebuildError) as excinfo:
            rb.rebuild_samples_db(
                _rows(), db_path, images_dir=None, overwrite=True, now=lambda: FIXED_NOW)

        assert not isinstance(excinfo.value, rb.DbGuardError)
        assert message in str(excinfo.value)
        assert db_path.read_bytes() == before
        assert not list(tmp_path.glob("*.bak-*"))

    def test_a_failed_backup_aborts_before_anything_is_replaced(self, tmp_path, monkeypatch):
        db_path = tmp_path / "samples.db"
        self._existing_with_other_rows(db_path)
        gc.collect()
        before = _load(db_path)
        gc.collect()

        def broken_backup(src, dest):
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(rb, "_backup", broken_backup)

        with pytest.raises(rb.RebuildError, match="バックアップ"):
            rb.rebuild_samples_db(
                _rows(), db_path, images_dir=None, overwrite=True, now=lambda: FIXED_NOW)

        assert _load(db_path) == before

    def test_backup_releases_the_source_when_the_target_cannot_be_opened(self, tmp_path, monkeypatch):
        """バックアップ先を開けなくても、読み取り元の接続を閉じる（Windows では開いたままだと DB を消せない）。"""
        db_path = tmp_path / "samples.db"
        self._existing_with_other_rows(db_path)
        gc.collect()
        real_connect = sqlite3.connect

        def connect(target, *args, **kwargs):
            if str(target).endswith(".bak-target"):
                raise sqlite3.OperationalError("unable to open database file")
            return real_connect(target, *args, **kwargs)

        monkeypatch.setattr(rb.sqlite3, "connect", connect)

        with pytest.raises(rb.RebuildError):
            rb._backup(db_path, tmp_path / "samples.db.bak-target")

        db_path.unlink()  # 読み取り元が閉じられていなければ Windows では PermissionError

    def test_a_db_created_by_someone_else_during_the_rebuild_is_not_clobbered(self, tmp_path, monkeypatch):
        db_path = tmp_path / "samples.db"
        real_write = rb._write_new_db

        def write_and_race(samples, path):
            real_write(samples, path)
            db_path.write_bytes(b"annotator created this meanwhile")  # 計画後に別プロセスが作成

        monkeypatch.setattr(rb, "_write_new_db", write_and_race)

        with pytest.raises(rb.RebuildError, match="別のプロセス"):
            rb.rebuild_samples_db(_rows(), db_path, images_dir=None, now=lambda: FIXED_NOW)

        assert db_path.read_bytes() == b"annotator created this meanwhile"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["samples.db"]

    def test_garbage_file_is_refused_then_backed_up_byte_for_byte_on_overwrite(self, tmp_path):
        db_path = tmp_path / "samples.db"
        db_path.write_bytes(b"this is not sqlite" * 50)

        with pytest.raises(rb.DbGuardError):
            rb.rebuild_samples_db(_rows(), db_path, images_dir=None, now=lambda: FIXED_NOW)
        assert db_path.read_bytes() == b"this is not sqlite" * 50

        result = rb.rebuild_samples_db(
            _rows(), db_path, images_dir=None, overwrite=True, now=lambda: FIXED_NOW)

        assert result.backup_path.read_bytes() == b"this is not sqlite" * 50
        assert _db_core(_load(db_path)) == _core(_rows())

    def test_empty_placeholder_files_are_replaced_without_overwrite_flag(self, tmp_path):
        zero = tmp_path / "zero.db"
        zero.write_bytes(b"")
        empty = tmp_path / "empty.db"
        sample_db.init_db(empty)

        for path in (zero, empty):
            result = rb.rebuild_samples_db(_rows(), path, images_dir=None, now=lambda: FIXED_NOW)
            assert result.action == "replaced"
            assert result.backup_path is None
            assert _db_core(_load(path)) == _core(_rows())


class TestNormEyeScaleBackfill:
    """fetch が一部失敗して db ステップが NULL で作った norm_eye_scale を、再実行で補完する。"""

    def test_unchanged_db_gets_norm_eye_scale_once_the_image_is_available(self, tmp_path):
        db_path = tmp_path / "samples.db"
        images = tmp_path / "images"
        rb.rebuild_samples_db(_rows(), db_path, images_dir=images, now=lambda: FIXED_NOW)
        assert all(s["norm_eye_scale"] is None for s in _load(db_path))
        gc.collect()
        _write_jpeg(images / "img_a.jpg", width=300, height=400)

        result = rb.rebuild_samples_db(_rows(), db_path, images_dir=images, now=lambda: FIXED_NOW)

        by_image = {}
        for s in _load(db_path):
            by_image.setdefault(s["image_path"], []).append(s["norm_eye_scale"])
        assert result.action == "unchanged"
        assert result.n_backfilled == 5  # img_a の 5 行（img_b の画像はまだ無い）
        assert all(v is not None for v in by_image["data/images/img_a.jpg"])
        assert all(v is None for v in by_image["data/images/img_b.jpg"])

    def test_backfill_uses_the_same_definition_as_the_initial_build(self, tmp_path):
        db_path = tmp_path / "samples.db"
        images = tmp_path / "images"
        rb.rebuild_samples_db(_rows(), db_path, images_dir=images)
        gc.collect()
        _write_jpeg(images / "img_a.jpg", width=300, height=400)

        rb.rebuild_samples_db(_rows(), db_path, images_dir=images)

        first = next(s for s in _load(db_path) if s["idx"] == 10)
        assert first["norm_eye_scale"] == pytest.approx(first["eye_scale"] / 500.0)

    def test_nothing_is_backfilled_without_images_or_in_a_dry_run(self, tmp_path):
        db_path = tmp_path / "samples.db"
        images = tmp_path / "images"
        rb.rebuild_samples_db(_rows(), db_path, images_dir=images)
        gc.collect()
        _write_jpeg(images / "img_a.jpg", width=300, height=400)

        dry = rb.rebuild_samples_db(_rows(), db_path, images_dir=images, dry_run=True)
        none = rb.rebuild_samples_db(_rows(), db_path, images_dir=None)

        assert dry.n_backfilled == 0 and none.n_backfilled == 0
        assert all(s["norm_eye_scale"] is None for s in _load(db_path))

    def test_unreadable_images_are_reported_not_silently_ignored(self, tmp_path, caplog):
        (tmp_path / "img_a.jpg").write_bytes(b"not an image")

        with caplog.at_level(logging.WARNING):
            samples = rb.build_samples(_rows(), images_dir=tmp_path)

        assert samples[0]["norm_eye_scale"] is None
        assert "img_a" in caplog.text and "norm_eye_scale" in caplog.text


class TestPlanAndDryRun:

    def test_plan_reports_each_situation(self, tmp_path):
        missing = rb.plan_rebuild(_rows(), tmp_path / "none.db")
        assert missing.action == "create"

        db_path = tmp_path / "samples.db"
        rb.rebuild_samples_db(_rows(), db_path, images_dir=None, now=lambda: FIXED_NOW)
        assert rb.plan_rebuild(_rows(), db_path).action == "unchanged"

        other = rb.plan_rebuild(_rows()[:3], db_path)
        assert other.action == "refuse"
        assert "3" in other.detail and "6" in other.detail

        assert rb.plan_rebuild(_rows()[:3], db_path, overwrite=True).action == "replace"

    def test_comparison_is_read_only(self, tmp_path):
        db_path = tmp_path / "samples.db"
        rb.rebuild_samples_db(_rows(), db_path, images_dir=None, now=lambda: FIXED_NOW)
        before = db_path.read_bytes()

        rb.plan_rebuild(_rows()[:2], db_path)

        assert db_path.read_bytes() == before

    def test_dry_run_writes_nothing(self, tmp_path):
        db_path = tmp_path / "sub" / "samples.db"

        result = rb.rebuild_samples_db(_rows(), db_path, images_dir=None, dry_run=True)

        assert result.action == "create" and result.dry_run
        assert not db_path.parent.exists()

    def test_dry_run_reports_refusal_instead_of_raising(self, tmp_path):
        db_path = tmp_path / "samples.db"
        rb.rebuild_samples_db(_rows(), db_path, images_dir=None, now=lambda: FIXED_NOW)

        result = rb.rebuild_samples_db(_rows()[:2], db_path, images_dir=None, dry_run=True)

        assert result.action == "refuse"
        assert _db_core(_load(db_path)) == _core(_rows())


class TestRealManifest:

    def test_bundled_manifest_round_trips_through_a_db(self, tmp_path):
        source = manifest_source.find_latest_manifest(ROOT / "experiments")
        rows = manifest_source.load_manifest(source)
        db_path = tmp_path / "samples.db"

        result = rb.rebuild_samples_db(rows, db_path, images_dir=None)

        assert result.n_rows == len(rows)
        total, annotated = sample_db.count_samples(db_path)
        assert total == annotated == len(rows)
        assert rb.rebuild_samples_db(rows, db_path, images_dir=None).action == "unchanged"


class TestAllowExtraRows:
    """`allow_extra_rows=True`（restore_dataset --source all）: DB が manifest の行をすべて含めばよい。

    DB のみの行（追記した未アノテーション行・第三者のアノテーション）は許し、既存の行の値の違いは見ない。
    idx が別のサンプルに使われている・manifest の行が欠けている、は従来どおり拒否する。
    """

    def _built(self, tmp_path: Path) -> Path:
        db_path = tmp_path / "samples.db"
        rb.rebuild_samples_db(_rows(), db_path, images_dir=None, now=lambda: FIXED_NOW)
        return db_path

    def _add_unannotated_row(self, db_path: Path, idx: int = 999) -> None:
        extra = rb.build_samples((_row(idx, image_id="unlabeled", t=0.0),), images_dir=None)
        sample_db.upsert_samples([{**extra[0], "t": None}], db_path)

    def test_a_db_with_extra_rows_is_left_untouched(self, tmp_path):
        db_path = self._built(tmp_path)
        self._add_unannotated_row(db_path)
        gc.collect()
        before = _load(db_path)

        result = rb.rebuild_samples_db(
            _rows(), db_path, images_dir=None, allow_extra_rows=True, now=lambda: FIXED_NOW)

        assert result.action == "unchanged"
        assert result.detail == "既存 DB は manifest の行をすべて含んでいます（DB 7 行のうち manifest 6 行）"
        assert _load(db_path) == before
        assert not list(tmp_path.glob("*.bak-*"))

    def test_a_changed_annotation_does_not_make_the_db_different(self, tmp_path):
        """既存の行は変更しない方針なので、値（t）の違いは見ない。"""
        db_path = self._built(tmp_path)
        sample_db.update_annotation(11, 0.123, db_path)

        result = rb.rebuild_samples_db(
            _rows(), db_path, images_dir=None, allow_extra_rows=True, now=lambda: FIXED_NOW)

        assert result.action == "unchanged"
        assert sample_db.get_sample_by_idx(11, db_path)["t"] == 0.123

    def test_value_differences_beyond_t_are_ignored_too(self, tmp_path):
        """座標・skip_flag が変わっていても、同じサンプル（自然キー）であれば一致とみなす。"""
        db_path = self._built(tmp_path)
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE samples SET px = px + 5, qy = qy - 3, skip_flag = 1 WHERE idx = 12")

        result = rb.rebuild_samples_db(
            _rows(), db_path, images_dir=None, allow_extra_rows=True, now=lambda: FIXED_NOW)

        assert result.action == "unchanged"
        assert sample_db.get_sample_by_idx(12, db_path)["skip_flag"] is True

    def test_strict_mode_still_refuses_the_same_db(self, tmp_path):
        db_path = self._built(tmp_path)
        self._add_unannotated_row(db_path)

        with pytest.raises(rb.DbGuardError):
            rb.rebuild_samples_db(_rows(), db_path, images_dir=None, now=lambda: FIXED_NOW)

    def test_a_manifest_row_missing_from_the_db_is_refused(self, tmp_path):
        db_path = tmp_path / "samples.db"
        rb.rebuild_samples_db(_rows()[:3], db_path, images_dir=None, now=lambda: FIXED_NOW)
        self._add_unannotated_row(db_path)
        before = _load(db_path)

        with pytest.raises(rb.DbGuardError, match="manifest のみ 3"):
            rb.rebuild_samples_db(_rows(), db_path, images_dir=None, allow_extra_rows=True)

        assert _load(db_path) == before

    def test_an_idx_used_for_a_different_sample_is_refused(self, tmp_path):
        db_path = self._built(tmp_path)
        with sqlite3.connect(db_path) as conn:  # idx 11 が別の点（point_index 5）になっている DB
            conn.execute("UPDATE samples SET point_index = 5 WHERE idx = 11")

        with pytest.raises(rb.DbGuardError, match="識別子の不一致 1"):
            rb.rebuild_samples_db(_rows(), db_path, images_dir=None, allow_extra_rows=True)

    def test_create_and_empty_db_are_unaffected(self, tmp_path):
        assert rb.plan_rebuild(_rows(), tmp_path / "none.db", allow_extra_rows=True).action == "create"
        empty = tmp_path / "empty.db"
        sample_db.init_db(empty)

        assert rb.plan_rebuild(_rows(), empty, allow_extra_rows=True).action == "replace"

    def test_overwrite_still_backs_up_and_replaces(self, tmp_path):
        db_path = tmp_path / "samples.db"
        rb.rebuild_samples_db(_rows()[:3], db_path, images_dir=None, now=lambda: FIXED_NOW)
        self._add_unannotated_row(db_path)

        result = rb.rebuild_samples_db(
            _rows(), db_path, images_dir=None, allow_extra_rows=True, overwrite=True, now=lambda: FIXED_NOW)

        assert result.action == "replaced" and result.backup_path is not None
        assert _db_core(_load(db_path)) == _core(_rows())

    def test_dry_run_reports_the_plan_without_writing(self, tmp_path):
        db_path = self._built(tmp_path)
        self._add_unannotated_row(db_path)
        before = db_path.read_bytes()

        result = rb.rebuild_samples_db(
            _rows(), db_path, images_dir=None, allow_extra_rows=True, dry_run=True)

        assert result.action == "unchanged" and result.dry_run
        assert db_path.read_bytes() == before

    def test_norm_eye_scale_is_still_backfilled_for_an_unchanged_db(self, tmp_path):
        db_path = self._built(tmp_path)
        self._add_unannotated_row(db_path)
        images_dir = tmp_path / "images"
        _write_jpeg(images_dir / "img_a.jpg", 200, 100)

        result = rb.rebuild_samples_db(
            _rows(), db_path, images_dir=images_dir, allow_extra_rows=True, now=lambda: FIXED_NOW)

        assert result.action == "unchanged" and result.n_backfilled > 0
        assert sample_db.get_sample_by_idx(10, db_path)["norm_eye_scale"] is not None
