"""tools/sampledb_maintenance/add_new_images.py のテスト。

手元にあって samples.db に行が無い画像に、FaceMesh で P/Q 候補の行（未アノテーション）を追記する処理。
既存の行には触れないこと・何度実行しても同じ結果になること・画像単位で全部入るか全く入らないことを確認する
（FaceMesh は合成ランドマークのフェイク）。
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from restore_fixtures import (  # noqa: E402
    BASE_WIDTH,
    IMG_H,
    FakeFaceMesh,
    build_db,
    db_rows,
    expected_pose,
    make_face,
    make_factory,
    manifest_rows,
    write_image,
)
from tools.sampledb_maintenance import add_new_images as ani  # noqa: E402
from tools.sampledb_maintenance import restore_face_pose as rfp  # noqa: E402

ROWS_PER_FACE = 28  # 2 眼 × (upper 7 + lower 7)


def _setup(tmp_path: Path, *, manifest_ids=("m1",)):
    """manifest の画像（幅 BASE_WIDTH + i）から DB を作り、その画像も置く。"""
    rows = []
    images_dir = tmp_path / "images"
    for i, image_id in enumerate(manifest_ids):
        rows += manifest_rows(image_id, BASE_WIDTH + i, start_idx=len(rows) + 100)
        write_image(images_dir, image_id, BASE_WIDTH + i)
    db_path = build_db(tmp_path, rows)
    return db_path, images_dir, rows


def _new_image(images_dir: Path, image_id: str, width: int) -> None:
    write_image(images_dir, image_id, width)


def _digest(db_path: Path) -> str:
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute("SELECT * FROM samples ORDER BY idx").fetchall()
    return hashlib.sha256(repr(rows).encode()).hexdigest()


def _forbidden_factory():
    def factory():
        raise AssertionError("FaceMesh を作るべきではない")
    return factory


class TestAddNewImages:

    def test_adds_unannotated_rows_for_an_image_that_has_none(self, tmp_path):
        db_path, images_dir, _ = _setup(tmp_path)
        _new_image(images_dir, "new1", 330)
        face = make_face()

        result = ani.add_new_images(db_path, images_dir, face_mesh_factory=make_factory({330: [face]}))

        rows = [r for r in db_rows(db_path).values() if r["image_path"] == "data/images/new1.jpg"]
        assert (result.n_added_images, result.n_added_rows) == (1, ROWS_PER_FACE)
        assert len(rows) == ROWS_PER_FACE
        assert {(r["part_name"], r["sub_key"], r["point_index"]) for r in rows} == {
            (part, sub, k) for part in ("eyeliner_left", "eyeliner_right")
            for sub in ("upper", "lower") for k in range(7)}
        assert all(r["t"] is None and r["skip_flag"] is False for r in rows)
        assert all(r["face_index"] == 0 for r in rows)
        yaw, pitch, roll = expected_pose(face, 330)
        assert rows[0]["face_yaw"] == pytest.approx(yaw)
        assert rows[0]["face_pitch"] == pytest.approx(pitch)
        assert rows[0]["face_roll"] == pytest.approx(roll)

    def test_fills_the_derived_columns_so_no_separate_backfill_is_needed(self, tmp_path):
        db_path, images_dir, _ = _setup(tmp_path)
        _new_image(images_dir, "new1", 330)
        mesh_factory = make_factory({330: [make_face()]})

        ani.add_new_images(db_path, images_dir, face_mesh_factory=mesh_factory)

        rows = [r for r in db_rows(db_path).values() if r["image_path"] == "data/images/new1.jpg"]
        # eye_scale: 眼ごとの P/Q バウンディングボックス対角長（下限 1px）
        for part in ("eyeliner_left", "eyeliner_right"):
            eye = [r for r in rows if r["part_name"] == part]
            xs = [v for r in eye for v in (r["px"], r["qx"])]
            ys = [v for r in eye for v in (r["py"], r["qy"])]
            expected = max(math.hypot(max(xs) - min(xs), max(ys) - min(ys)), 1.0)
            assert all(r["eye_scale"] == pytest.approx(expected) for r in eye)
            assert all(r["norm_eye_scale"] == pytest.approx(expected / math.hypot(330, IMG_H)) for r in eye)
        # lap_var は復元（restore_face_pose）と同じ定義の値
        rgb = rfp._read_rgb(images_dir / "new1.jpg")
        assert rows[0]["lap_var"] == pytest.approx(rfp.analyze_image(rgb, FakeFaceMesh({330: [make_face()]})).lap_var)
        assert all(r["nose_x"] is not None and r["nose_y"] is not None for r in rows)

    def test_new_rows_continue_after_the_current_maximum_idx_in_image_id_order(self, tmp_path):
        db_path, images_dir, rows = _setup(tmp_path)
        max_idx = max(r.idx for r in rows)
        _new_image(images_dir, "b2", 331)
        _new_image(images_dir, "a1", 330)

        result = ani.add_new_images(
            db_path, images_dir, face_mesh_factory=make_factory({330: [make_face()], 331: [make_face()]}))

        db = db_rows(db_path)
        block = {p: sorted(i for i, r in db.items() if r["image_path"] == p) for p in
                 ("data/images/a1.jpg", "data/images/b2.jpg")}
        assert block["data/images/a1.jpg"] == list(range(max_idx + 1, max_idx + 1 + ROWS_PER_FACE))
        assert block["data/images/b2.jpg"] == list(range(max_idx + 1 + ROWS_PER_FACE, max_idx + 1 + 2 * ROWS_PER_FACE))
        assert (result.first_idx, result.last_idx) == (max_idx + 1, max_idx + 2 * ROWS_PER_FACE)

    def test_face_index_follows_x_order_of_the_detected_faces(self, tmp_path):
        db_path, images_dir, _ = _setup(tmp_path)
        _new_image(images_dir, "two", 330)
        left, right = make_face(dx=-0.2), make_face(dx=0.2)

        ani.add_new_images(db_path, images_dir, face_mesh_factory=make_factory({330: [right, left]}))

        rows = [r for r in db_rows(db_path).values() if r["image_path"] == "data/images/two.jpg"]
        assert len(rows) == 2 * ROWS_PER_FACE
        mean_px = {fi: sum(r["px"] for r in rows if r["face_index"] == fi) / ROWS_PER_FACE for fi in (0, 1)}
        assert mean_px[0] < mean_px[1]

    def test_existing_rows_are_never_modified(self, tmp_path):
        db_path, images_dir, rows = _setup(tmp_path)
        with closing(sqlite3.connect(db_path)) as conn:  # 誰かが復元済みの行のアノテーションを直した状況
            conn.execute("UPDATE samples SET t = 0.9, skip_flag = 1 WHERE idx = ?", (rows[0].idx,))
            conn.commit()
        before = db_rows(db_path)
        _new_image(images_dir, "new1", 330)

        ani.add_new_images(db_path, images_dir, face_mesh_factory=make_factory({330: [make_face()]}))

        after = db_rows(db_path)
        assert all(after[i] == before[i] for i in before)
        assert after[rows[0].idx]["t"] == 0.9 and after[rows[0].idx]["skip_flag"] is True

    def test_is_idempotent(self, tmp_path):
        db_path, images_dir, _ = _setup(tmp_path)
        _new_image(images_dir, "new1", 330)
        factory = make_factory({330: [make_face()]})
        ani.add_new_images(db_path, images_dir, face_mesh_factory=factory)
        digest = _digest(db_path)
        processed = list(factory.mesh.processed_widths)

        second = ani.add_new_images(db_path, images_dir, face_mesh_factory=factory)

        assert (second.n_candidates, second.n_added_images, second.n_added_rows) == (0, 0, 0)
        assert _digest(db_path) == digest
        assert factory.mesh.processed_widths == processed  # 行がある画像は解析し直さない

    def test_images_that_already_have_rows_are_not_analysed(self, tmp_path):
        db_path, images_dir, _ = _setup(tmp_path, manifest_ids=("m1", "m2"))

        result = ani.add_new_images(db_path, images_dir, face_mesh_factory=_forbidden_factory())

        assert (result.n_candidates, result.n_added_rows) == (0, 0)

    def test_an_image_without_a_face_adds_nothing_and_is_reported(self, tmp_path):
        db_path, images_dir, _ = _setup(tmp_path)
        _new_image(images_dir, "noface", 330)
        digest = _digest(db_path)

        result = ani.add_new_images(db_path, images_dir, face_mesh_factory=make_factory({}))

        assert (result.n_candidates, result.n_added_images, result.n_no_face) == (1, 0, 1)
        assert _digest(db_path) == digest

    def test_an_unreadable_image_is_skipped_and_the_others_continue(self, tmp_path):
        db_path, images_dir, _ = _setup(tmp_path)
        (images_dir / "bad1.jpg").write_bytes(b"not a jpeg")
        _new_image(images_dir, "good1", 330)

        result = ani.add_new_images(db_path, images_dir, face_mesh_factory=make_factory({330: [make_face()]}))

        assert (result.n_unreadable, result.n_added_images) == (1, 1)
        paths = {r["image_path"] for r in db_rows(db_path).values()}
        assert "data/images/good1.jpg" in paths and "data/images/bad1.jpg" not in paths

    def test_only_jpg_files_with_a_valid_image_id_are_candidates(self, tmp_path):
        db_path, images_dir, _ = _setup(tmp_path)
        _new_image(images_dir, "ok1", 330)
        write_image(images_dir, "bad name", 331)   # 空白入りは ID として不正
        (images_dir / "x.png").write_bytes(b"png")  # jpg 以外は対象外

        assert ani.list_images_without_rows(db_path, images_dir) == ["ok1"]

    def test_each_image_is_added_atomically_so_a_crash_leaves_no_partial_image(self, tmp_path):
        db_path, images_dir, _ = _setup(tmp_path)
        _new_image(images_dir, "a1", 330)
        _new_image(images_dir, "b2", 331)

        # b2 の最後の 1 行（28 行目）の挿入だけを失敗させる → 27 行を入れたあとで失敗する状況
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute(
                "CREATE TRIGGER fail_last_row BEFORE INSERT ON samples "
                "WHEN NEW.image_path = 'data/images/b2.jpg' AND NEW.part_name = 'eyeliner_right' "
                "AND NEW.sub_key = 'lower' AND NEW.point_index = 6 "
                "BEGIN SELECT RAISE(ABORT, 'boom'); END")
            conn.commit()
        factory = make_factory({330: [make_face()], 331: [make_face()]})

        with pytest.raises(sqlite3.IntegrityError, match="boom"):
            ani.add_new_images(db_path, images_dir, face_mesh_factory=factory)

        paths = [r["image_path"] for r in db_rows(db_path).values()]
        assert paths.count("data/images/a1.jpg") == ROWS_PER_FACE  # 完了した画像は残る
        assert paths.count("data/images/b2.jpg") == 0               # 途中まで入れた画像も 1 行も残らない

        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("DROP TRIGGER fail_last_row")
            conn.commit()
        retry = ani.add_new_images(db_path, images_dir, face_mesh_factory=factory)
        assert (retry.n_added_images, retry.n_added_rows) == (1, ROWS_PER_FACE)  # 残りの画像だけを追記

    def test_dry_run_reports_candidates_and_touches_nothing(self, tmp_path):
        db_path, images_dir, _ = _setup(tmp_path)
        _new_image(images_dir, "a1", 330)
        _new_image(images_dir, "b2", 331)
        digest = _digest(db_path)

        result = ani.add_new_images(db_path, images_dir, dry_run=True, face_mesh_factory=_forbidden_factory())

        assert result.dry_run is True and result.n_candidates == 2
        assert (result.n_added_images, result.n_added_rows) == (0, 0)
        assert _digest(db_path) == digest

    def test_dry_run_before_the_db_exists_treats_known_ids_as_already_present(self, tmp_path):
        images_dir = tmp_path / "images"
        for i, image_id in enumerate(("m1", "new1")):
            write_image(images_dir, image_id, BASE_WIDTH + i)

        result = ani.add_new_images(
            tmp_path / "missing.db", images_dir, dry_run=True, known_image_ids={"m1"})

        assert result.n_candidates == 1

    def test_dry_run_with_a_pending_rebuild_ignores_the_current_db(self, tmp_path):
        """直前の手順が DB を作り直す予定なら、いまの DB の行ではなく known_image_ids だけを「行あり」とみなす。"""
        db_path, images_dir, _ = _setup(tmp_path)  # 現在の DB には m1 の行がある
        _new_image(images_dir, "new1", 330)

        result = ani.add_new_images(
            db_path, images_dir, dry_run=True, db_pending_rebuild=True, known_image_ids={"other"})

        assert result.n_candidates == 2  # m1 も new1 も、作り直した DB には行が無い扱い

    def test_pending_rebuild_is_only_meaningful_for_dry_runs(self, tmp_path):
        db_path, images_dir, _ = _setup(tmp_path)

        with pytest.raises(ValueError, match="dry_run"):
            ani.add_new_images(db_path, images_dir, db_pending_rebuild=True)

    def test_a_concurrent_writer_does_not_break_the_run(self, tmp_path):
        """別のプロセスが途中で行を足しても、idx は書き込むトランザクションの中で採番するので衝突しない。"""
        db_path, images_dir, _ = _setup(tmp_path)
        _new_image(images_dir, "a1", 330)
        _new_image(images_dir, "b2", 331)

        class Intruder(FakeFaceMesh):
            """最初の画像を解析している間に、別プロセスが次の idx を使ってしまう。"""
            done = False

            def process(self, rgb):
                if not self.done:
                    self.done = True
                    with closing(sqlite3.connect(db_path)) as other:
                        next_idx = other.execute("SELECT MAX(idx) FROM samples").fetchone()[0] + 1
                        other.execute(
                            "INSERT INTO samples (idx, image_path, part_name, sub_key, point_index, px, py, qx, qy) "
                            "VALUES (?, 'data/images/other.jpg', 'eyeliner_left', 'upper', 0, 1, 1, 1, 1)", (next_idx,))
                        other.commit()
                return super().process(rgb)

        mesh = Intruder({330: [make_face()], 331: [make_face()]})

        result = ani.add_new_images(db_path, images_dir, face_mesh_factory=lambda: mesh)

        assert (result.n_added_images, result.n_added_rows) == (2, 2 * ROWS_PER_FACE)
        paths = [r["image_path"] for r in db_rows(db_path).values()]
        assert paths.count("data/images/a1.jpg") == paths.count("data/images/b2.jpg") == ROWS_PER_FACE
        assert paths.count("data/images/other.jpg") == 1

    def test_a_file_that_is_not_a_samples_db_is_reported(self, tmp_path):
        images_dir = tmp_path / "images"
        _new_image(images_dir, "a1", 330)
        garbage = tmp_path / "samples.db"
        garbage.write_bytes(b"this is not a sqlite database" * 20)

        with pytest.raises(ani.AddImagesError, match="samples.db"):
            ani.add_new_images(garbage, images_dir, face_mesh_factory=_forbidden_factory())

    def test_the_dry_run_message_says_it_only_counts_images_already_on_disk(self):
        lines = ani.format_result(ani.AddImagesResult(n_candidates=3, dry_run=True))

        assert "候補: 3 枚" in lines[0] and "手元にある画像" in lines[0]

    def test_a_missing_db_is_reported_instead_of_created(self, tmp_path):
        with pytest.raises(ani.AddImagesError, match="samples.db"):
            ani.add_new_images(tmp_path / "missing.db", tmp_path / "images")
        assert not (tmp_path / "missing.db").exists()

    def test_a_missing_images_dir_means_nothing_to_add(self, tmp_path):
        db_path, _, _ = _setup(tmp_path)

        result = ani.add_new_images(db_path, tmp_path / "nowhere", face_mesh_factory=_forbidden_factory())

        assert result.n_candidates == 0


class TestSamplesMatchTheAnnotatorPreprocessor:
    """新規行の P/Q・顔番号・姿勢は、既存の annotator/preprocessor.py が作るものと一致する。"""

    def test_same_samples_as_extract_samples_from_image(self, tmp_path, monkeypatch):
        db_path, images_dir, _ = _setup(tmp_path)
        _new_image(images_dir, "two", 330)
        faces = [make_face(dx=0.2, yaw_z=0.03), make_face(dx=-0.2, chin_z=0.02)]
        ani.add_new_images(db_path, images_dir, face_mesh_factory=make_factory({330: faces}))
        mine = sorted(
            (r for r in db_rows(db_path).values() if r["image_path"] == "data/images/two.jpg"),
            key=lambda r: (r["face_index"], r["part_name"], r["sub_key"], r["point_index"]))

        from annotator import preprocessor  # noqa: PLC0415
        fake_mesh = FakeFaceMesh({330: faces})
        monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace(
            solutions=SimpleNamespace(face_mesh=SimpleNamespace(FaceMesh=lambda **kw: fake_mesh))))
        canonical = preprocessor._extract_samples_from_image(images_dir / "two.jpg", {})
        canonical = sorted(canonical, key=lambda r: (r["face_index"], r["part_name"], r["sub_key"], r["point_index"]))

        assert len(mine) == len(canonical) == 2 * ROWS_PER_FACE
        for got, want in zip(mine, canonical, strict=True):
            for key in ("face_index", "part_name", "sub_key", "point_index", "t", "skip_flag"):
                assert got[key] == want[key]
            for key in ("px", "py", "qx", "qy", "face_yaw", "face_pitch", "face_roll", "nose_x", "nose_y"):
                assert got[key] == pytest.approx(want[key], rel=1e-6, abs=1e-6)
