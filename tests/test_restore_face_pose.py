"""tools/sampledb_maintenance/restore_face_pose.py のテスト。

FaceMesh を再実行して face_yaw/pitch/roll・lap_var を復元する処理と、
「復元対象画像がアノテーション時と同じ画像か」の P/Q 重心照合、
大規模 DB を誤って上書きしないためのガードを確認する（FaceMesh は合成ランドマークのフェイク）。
"""
from __future__ import annotations

import dataclasses
import gc
import importlib
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from annotator import db as sample_db  # noqa: E402
from eyeliner.data.eyeliner_landmarks import EYELINER_SAMPLE_PQ  # noqa: E402
from restore_fixtures import (  # noqa: E402
    BASE_WIDTH,
    FakeFaceMesh,
    build_db,
    db_rows,
    expected_pose,
    make_face,
    make_factory,
    manifest_rows,
    write_image,
)
from tools.sampledb_maintenance import rebuild_samples_db as rb  # noqa: E402
from tools.sampledb_maintenance import restore_face_pose as rfp  # noqa: E402

# ---------------------------------------------------------------------------
# 復元
# ---------------------------------------------------------------------------

class TestRestore:

    def test_restores_pose_and_lap_var_for_a_matching_image(self, tmp_path):
        width = BASE_WIDTH
        face = make_face(yaw_z=0.1, chin_z=0.05)
        rows = manifest_rows("img_a", width)
        db_path = build_db(tmp_path, rows)
        write_image(tmp_path / "images", "img_a", width)

        result = rfp.restore_face_features(
            rows, db_path, tmp_path / "images", face_mesh_factory=make_factory({width: [face]}))

        yaw, pitch, roll = expected_pose(face, width)
        assert (result.n_images, result.n_verified, result.n_unverified) == (1, 1, 0)
        assert result.n_rows_updated == len(rows)
        stored = db_rows(db_path)
        assert len(stored) == len(rows)
        for row in stored.values():
            assert row["face_yaw"] == pytest.approx(yaw)
            assert row["face_pitch"] == pytest.approx(pitch)
            assert row["face_roll"] == pytest.approx(roll)
            assert row["lap_var"] is not None and row["lap_var"] > 0
        assert yaw > 5.0  # 合成した奥行き差が実際に yaw に反映されている

    def test_lap_var_matches_the_existing_image_registry_implementation(self, tmp_path):
        """DB の lap_var は images.db の max_lap_var（build_image_registry）と同じ定義。"""
        registry = importlib.import_module("tools.imagesdb_maintenance.build_image_registry")
        width = BASE_WIDTH
        face_a, face_b = make_face(), make_face(dy=0.25)
        rows = manifest_rows("img_a", width, face_index=0) + manifest_rows(
            "img_a", width, face_index=1, dy=0.25, start_idx=100)
        db_path = build_db(tmp_path, rows)
        write_image(tmp_path / "images", "img_a", width, seed=3)
        mesh = FakeFaceMesh({width: [face_a, face_b]})

        rfp.restore_face_features(rows, db_path, tmp_path / "images", face_mesh_factory=lambda: mesh)

        rgb = cv2.cvtColor(cv2.imread(str(tmp_path / "images" / "img_a.jpg")), cv2.COLOR_BGR2RGB)
        oracle = registry._extract_features_from_rgb(rgb, mesh)["max_lap_var"]
        assert oracle > 0
        stored = {r["lap_var"] for r in db_rows(db_path).values()}
        assert len(stored) == 1  # 画像単位の値を全行へ
        assert stored.pop() == pytest.approx(oracle)

    def test_matches_each_face_index_to_the_nearest_detected_face(self, tmp_path):
        width = BASE_WIDTH
        upper, lower = make_face(yaw_z=0.02), make_face(dy=0.25, yaw_z=0.12)
        rows = manifest_rows("img_a", width, face_index=0) + manifest_rows(
            "img_a", width, face_index=1, dy=0.25, start_idx=100)
        db_path = build_db(tmp_path, rows)
        write_image(tmp_path / "images", "img_a", width)
        # 検出順は manifest の face_index と逆（下の顔が先）でも、位置で正しく対応付く
        factory = make_factory({width: [lower, upper]})

        result = rfp.restore_face_features(rows, db_path, tmp_path / "images", face_mesh_factory=factory)

        stored = db_rows(db_path)
        assert result.n_verified == 1
        assert stored[0]["face_yaw"] == pytest.approx(expected_pose(upper, width)[0])
        assert stored[100]["face_yaw"] == pytest.approx(expected_pose(lower, width)[0])
        assert stored[0]["face_yaw"] != pytest.approx(stored[100]["face_yaw"])

    def test_extra_unannotated_faces_do_not_prevent_verification(self, tmp_path):
        width = BASE_WIDTH
        rows = manifest_rows("img_a", width, face_index=0)
        db_path = build_db(tmp_path, rows)
        write_image(tmp_path / "images", "img_a", width)
        faces = [make_face(dy=0.25), make_face()]

        result = rfp.restore_face_features(
            rows, db_path, tmp_path / "images", face_mesh_factory=make_factory({width: faces}))

        assert result.n_verified == 1

    def test_processes_each_image_once_and_only_images_in_the_manifest(self, tmp_path):
        rows = manifest_rows("img_a", 320) + manifest_rows("img_b", 330, start_idx=100)
        db_path = build_db(tmp_path, rows)
        write_image(tmp_path / "images", "img_a", 320)
        write_image(tmp_path / "images", "img_b", 330)
        write_image(tmp_path / "images", "unrelated", 340)
        factory = make_factory({320: [make_face()], 330: [make_face()], 340: [make_face()]})

        rfp.restore_face_features(rows, db_path, tmp_path / "images", face_mesh_factory=factory)

        assert sorted(factory.mesh.processed_widths) == [320, 330]

    def test_is_idempotent(self, tmp_path):
        width = BASE_WIDTH
        rows = manifest_rows("img_a", width)
        db_path = build_db(tmp_path, rows)
        write_image(tmp_path / "images", "img_a", width)
        factory = make_factory({width: [make_face(yaw_z=0.1)]})

        rfp.restore_face_features(rows, db_path, tmp_path / "images", face_mesh_factory=factory)
        first = db_rows(db_path)
        rfp.restore_face_features(rows, db_path, tmp_path / "images", face_mesh_factory=factory)

        assert db_rows(db_path) == first


# ---------------------------------------------------------------------------
# 画像同一性の検証
# ---------------------------------------------------------------------------

class TestImageIdentityCheck:

    def _run(self, tmp_path, *, face, offset_px=(0.0, 0.0), tol=None, preset=None):
        width = BASE_WIDTH
        rows = manifest_rows("img_a", width, offset_px=offset_px)
        db_path = build_db(tmp_path, rows)
        if preset:
            with sqlite3.connect(db_path) as conn:
                conn.execute("UPDATE samples SET face_yaw = ?, lap_var = ?", preset)
        write_image(tmp_path / "images", "img_a", width)
        kwargs = {} if tol is None else {"tol": tol}
        result = rfp.restore_face_features(
            rows, db_path, tmp_path / "images",
            face_mesh_factory=make_factory({width: [face]}), **kwargs)
        return result, db_rows(db_path)

    def test_a_different_image_is_not_written(self, tmp_path):
        """別の画像（顔の位置が大きくずれている）では face_pose・lap_var とも書かない。"""
        result, stored = self._run(tmp_path, face=make_face(dx=0.15, dy=0.2))

        assert (result.n_verified, result.n_unverified) == (0, 1)
        assert result.n_rows_updated == 0
        assert all(r["face_yaw"] is None and r["lap_var"] is None for r in stored.values())
        assert result.outcomes[0].status == "unverified"
        assert result.outcomes[0].score > rfp.DEFAULT_TOLERANCE

    def test_unverified_image_keeps_previous_values(self, tmp_path):
        _, stored = self._run(tmp_path, face=make_face(dx=0.15, dy=0.2), preset=(9.5, 123.0))

        assert {r["face_yaw"] for r in stored.values()} == {9.5}
        assert {r["lap_var"] for r in stored.values()} == {123.0}

    def test_score_is_offset_normalized_by_eye_diagonal(self, tmp_path):
        result, _ = self._run(tmp_path, face=make_face(), offset_px=(6.0, 0.0))

        (outcome,) = result.outcomes
        assert outcome.status == "restored"
        # PQ バウンディングボックス対角 ≈ 38px、ずれ 6px → 約 0.16
        assert 0.10 < outcome.score < 0.22

    def test_tolerance_is_configurable(self, tmp_path):
        offset = (7.0, 0.0)  # スコア ≈ 0.18

        loose, _ = self._run(tmp_path / "loose", face=make_face(), offset_px=offset, tol=0.25)
        strict, _ = self._run(tmp_path / "strict", face=make_face(), offset_px=offset, tol=0.10)

        assert loose.n_verified == 1 and strict.n_verified == 0

    def test_worst_of_both_eyes_decides(self, tmp_path):
        """片目だけが大きくずれていても、両目の最悪値で判定して不一致にする。"""
        width = BASE_WIDTH
        face = make_face()
        rows = manifest_rows("img_a", width, parts=("eyeliner_left",)) + manifest_rows(
            "img_a", width, parts=("eyeliner_right",), offset_px=(30.0, 0.0), start_idx=100)
        (tmp_path / "s").mkdir()
        db_path = build_db(tmp_path / "s", rows)
        write_image(tmp_path / "images", "img_a", width)

        result = rfp.restore_face_features(
            rows, db_path, tmp_path / "images", face_mesh_factory=make_factory({width: [face]}))

        assert result.n_verified == 0
        assert result.outcomes[0].score > 0.5

    def test_fewer_detected_faces_than_annotated_faces_is_unverified(self, tmp_path):
        width = BASE_WIDTH
        rows = manifest_rows("img_a", width, face_index=0) + manifest_rows(
            "img_a", width, face_index=1, dy=0.25, start_idx=100)
        db_path = build_db(tmp_path, rows)
        write_image(tmp_path / "images", "img_a", width)

        result = rfp.restore_face_features(
            rows, db_path, tmp_path / "images", face_mesh_factory=make_factory({width: [make_face()]}))

        assert result.n_verified == 0 and result.n_unverified == 1
        assert all(r["face_yaw"] is None for r in db_rows(db_path).values())

    def test_one_face_is_never_matched_to_two_annotated_faces(self, tmp_path):
        width = BASE_WIDTH
        rows = manifest_rows("img_a", width, face_index=0) + manifest_rows(
            "img_a", width, face_index=1, start_idx=100)  # 同じ位置に 2 顔分の注釈
        db_path = build_db(tmp_path, rows)
        write_image(tmp_path / "images", "img_a", width)

        result = rfp.restore_face_features(
            rows, db_path, tmp_path / "images", face_mesh_factory=make_factory({width: [make_face()]}))

        assert result.n_verified == 0


class TestUnrestorableImages:

    def test_image_without_a_detected_face(self, tmp_path):
        rows = manifest_rows("img_a", BASE_WIDTH)
        db_path = build_db(tmp_path, rows)
        write_image(tmp_path / "images", "img_a", BASE_WIDTH)

        result = rfp.restore_face_features(
            rows, db_path, tmp_path / "images", face_mesh_factory=make_factory({}))

        assert (result.n_no_face, result.n_verified) == (1, 0)
        assert all(r["lap_var"] is None for r in db_rows(db_path).values())

    def test_missing_image_file(self, tmp_path):
        rows = manifest_rows("img_a", BASE_WIDTH)
        db_path = build_db(tmp_path, rows)
        factory = make_factory({BASE_WIDTH: [make_face()]})

        result = rfp.restore_face_features(rows, db_path, tmp_path / "images", face_mesh_factory=factory)

        assert (result.n_missing_image, result.n_verified) == (1, 0)
        assert factory.mesh.processed_widths == []

    def test_unreadable_image_file(self, tmp_path):
        rows = manifest_rows("img_a", BASE_WIDTH)
        db_path = build_db(tmp_path, rows)
        (tmp_path / "images").mkdir()
        (tmp_path / "images" / "img_a.jpg").write_bytes(b"not a jpeg")

        result = rfp.restore_face_features(
            rows, db_path, tmp_path / "images",
            face_mesh_factory=make_factory({BASE_WIDTH: [make_face()]}))

        assert (result.n_unreadable, result.n_verified) == (1, 0)

    def test_unicode_images_dir_is_readable(self, tmp_path):
        images = tmp_path / "画像"
        rows = manifest_rows("img_a", BASE_WIDTH)
        db_path = build_db(tmp_path, rows)
        write_image(tmp_path / "ascii", "img_a", BASE_WIDTH)
        images.mkdir()
        (tmp_path / "ascii" / "img_a.jpg").replace(images / "img_a.jpg")

        result = rfp.restore_face_features(
            rows, db_path, images, face_mesh_factory=make_factory({BASE_WIDTH: [make_face()]}))

        assert result.n_verified == 1

    def test_an_image_declaring_an_enormous_canvas_is_refused_before_decoding(self, tmp_path, monkeypatch):
        """小さな JPEG が巨大なキャンバスを宣言して RAM を使い切るのを、デコード前のヘッダ検査で防ぐ。"""
        monkeypatch.setattr(rfp, "MAX_IMAGE_PIXELS", 100)  # 320x240 = 76,800 px は上限超過
        rows = manifest_rows("img_a", BASE_WIDTH)
        db_path = build_db(tmp_path, rows)
        write_image(tmp_path / "images", "img_a", BASE_WIDTH)
        factory = make_factory({BASE_WIDTH: [make_face()]})

        result = rfp.restore_face_features(rows, db_path, tmp_path / "images", face_mesh_factory=factory)

        assert (result.n_unreadable, result.n_verified) == (1, 0)
        assert factory.mesh.processed_widths == []

    def test_one_bad_image_does_not_stop_the_others(self, tmp_path):
        rows = manifest_rows("img_a", 320) + manifest_rows("img_b", 330, start_idx=100)
        db_path = build_db(tmp_path, rows)
        write_image(tmp_path / "images", "img_b", 330)

        result = rfp.restore_face_features(
            rows, db_path, tmp_path / "images", face_mesh_factory=make_factory({330: [make_face()]}))

        assert (result.n_images, result.n_missing_image, result.n_verified) == (2, 1, 1)


# ---------------------------------------------------------------------------
# 誤上書きガード
# ---------------------------------------------------------------------------

class TestGuard:

    def test_refuses_a_db_with_rows_outside_the_manifest(self, tmp_path):
        """本来の大規模 samples.db（未アノテーション行を含む）に復元用の小さな manifest を適用しない。"""
        rows = manifest_rows("img_a", BASE_WIDTH)
        db_path = build_db(tmp_path, rows)
        extra = rb.build_samples(manifest_rows("other", 300, start_idx=500)[:1], images_dir=None)
        sample_db.upsert_samples([{**extra[0], "t": None}], db_path)
        write_image(tmp_path / "images", "img_a", BASE_WIDTH)
        factory = make_factory({BASE_WIDTH: [make_face(yaw_z=0.1)]})
        before = db_rows(db_path)

        with pytest.raises(rfp.FeatureGuardError, match="manifest"):
            rfp.restore_face_features(rows, db_path, tmp_path / "images", face_mesh_factory=factory)

        assert db_rows(db_path) == before
        assert factory.mesh.processed_widths == []

    def test_refuses_a_row_whose_identity_differs_from_the_manifest(self, tmp_path):
        rows = manifest_rows("img_a", BASE_WIDTH)
        db_path = build_db(tmp_path, rows)
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE samples SET image_path = 'data/images/swapped.jpg' WHERE idx = 0")

        with pytest.raises(rfp.FeatureGuardError):
            rfp.restore_face_features(
                rows, db_path, tmp_path / "images", face_mesh_factory=make_factory({}))

    def test_a_db_holding_only_part_of_the_manifest_is_allowed(self, tmp_path):
        rows = manifest_rows("img_a", BASE_WIDTH) + manifest_rows("img_b", 330, start_idx=100)
        db_path = build_db(tmp_path, rows[:28])  # img_a のみ
        write_image(tmp_path / "images", "img_a", BASE_WIDTH)

        result = rfp.restore_face_features(
            rows, db_path, tmp_path / "images",
            face_mesh_factory=make_factory({BASE_WIDTH: [make_face()]}))

        assert result.n_verified == 1

    def test_missing_db_is_reported_with_the_step_to_run_first(self, tmp_path):
        with pytest.raises(rfp.FeatureGuardError, match="db"):
            rfp.restore_face_features(
                manifest_rows("img_a", BASE_WIDTH), tmp_path / "none.db", tmp_path,
                face_mesh_factory=make_factory({}))

    def test_empty_db_is_refused(self, tmp_path):
        db_path = tmp_path / "samples.db"
        sample_db.init_db(db_path)
        gc.collect()

        with pytest.raises(rfp.FeatureGuardError):
            rfp.restore_face_features(
                manifest_rows("img_a", BASE_WIDTH), db_path, tmp_path,
                face_mesh_factory=make_factory({}))


class TestDataTypes:

    def test_result_counts_default_to_zero(self):
        result = rfp.FeatureRestoreResult(n_images=2, dry_run=True)

        assert (result.n_verified, result.n_rows_updated, result.outcomes) == (0, 0, ())

    def test_outcome_statuses_are_a_closed_set_that_still_compares_to_plain_strings(self):
        assert rfp.ImageStatus.RESTORED == "restored"
        assert {s.value for s in rfp.ImageStatus} == {
            "restored", "unverified", "no_face", "missing_image", "unreadable"}

    def test_eye_contour_table_is_read_only(self):
        with pytest.raises(TypeError):
            rfp.EYE_CONTOUR_LANDMARKS["eyeliner_middle"] = (1, 2, 3)  # type: ignore[index]

    def test_guard_returns_the_idx_set_of_the_database(self, tmp_path):
        rows = manifest_rows("img_a", BASE_WIDTH)
        db_path = build_db(tmp_path, rows)

        assert rfp.assert_db_within_manifest(rows, db_path) == frozenset(r.idx for r in rows)


class TestDryRun:

    def test_dry_run_runs_no_facemesh_and_writes_nothing(self, tmp_path):
        rows = manifest_rows("img_a", BASE_WIDTH) + manifest_rows("img_b", 330, start_idx=100)
        db_path = build_db(tmp_path, rows)
        factory = make_factory({BASE_WIDTH: [make_face()]})

        result = rfp.restore_face_features(
            rows, db_path, tmp_path / "images", dry_run=True, face_mesh_factory=factory)

        assert result.dry_run and result.n_images == 2
        assert result.n_rows_updated == 0
        assert factory.mesh.processed_widths == []
        assert all(r["face_yaw"] is None for r in db_rows(db_path).values())

    def test_dry_run_before_the_db_exists_reports_all_manifest_images(self, tmp_path):
        rows = manifest_rows("img_a", 320) + manifest_rows("img_b", 330, start_idx=100)

        result = rfp.restore_face_features(
            rows, tmp_path / "not_yet.db", tmp_path, dry_run=True, db_pending_rebuild=True)

        assert result.dry_run and result.n_images == 2

    def test_pending_rebuild_is_only_meaningful_for_dry_runs(self, tmp_path):
        with pytest.raises(ValueError, match="dry_run"):
            rfp.restore_face_features(
                manifest_rows("img_a", 320), tmp_path / "x.db", tmp_path, db_pending_rebuild=True)

    def test_dry_run_still_enforces_the_guard(self, tmp_path):
        with pytest.raises(rfp.FeatureGuardError):
            rfp.restore_face_features(
                manifest_rows("img_a", BASE_WIDTH), tmp_path / "none.db", tmp_path, dry_run=True)


# ---------------------------------------------------------------------------
# mediapipe の有無・バージョン
# ---------------------------------------------------------------------------

class TestFaceMeshAvailability:

    def test_missing_mediapipe_gives_an_install_hint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "mediapipe", None)

        with pytest.raises(rfp.FaceMeshUnavailableError, match=r"mediapipe.*0\.10\.30"):
            rfp.create_face_mesh()

    def test_mediapipe_without_the_legacy_solutions_api_is_rejected(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace(__version__="0.10.32"))

        with pytest.raises(rfp.FaceMeshUnavailableError, match=r"0\.10\.32"):
            rfp.create_face_mesh()

    def test_builds_facemesh_with_the_annotation_time_settings(self, monkeypatch):
        captured = {}

        class FaceMesh:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake = SimpleNamespace(
            __version__="0.10.21", solutions=SimpleNamespace(face_mesh=SimpleNamespace(FaceMesh=FaceMesh)))
        monkeypatch.setitem(sys.modules, "mediapipe", fake)

        mesh = rfp.create_face_mesh()

        assert isinstance(mesh, FaceMesh)
        assert captured == {"static_image_mode": True, "max_num_faces": 10, "refine_landmarks": True}

    def test_eye_contours_match_the_image_registry_definition(self):
        registry = importlib.import_module("tools.imagesdb_maintenance.build_image_registry")

        assert list(rfp.EYE_CONTOUR_LANDMARKS["eyeliner_left"]) == registry._LEFT_EYE_CONTOUR
        assert list(rfp.EYE_CONTOUR_LANDMARKS["eyeliner_right"]) == registry._RIGHT_EYE_CONTOUR

    def test_eyeliner_left_contour_is_the_eye_the_pq_table_annotates(self):
        """eyeliner_left の P/Q ランドマークは、eyeliner_left の輪郭ランドマーク集合に近い側の目。"""
        for part, contour in rfp.EYE_CONTOUR_LANDMARKS.items():
            pq = {p for sub in EYELINER_SAMPLE_PQ[part].values() for _, p, _ in sub}
            assert len(pq & set(contour)) >= 10


# ---------------------------------------------------------------------------
# fill_missing_only（restore_dataset --source all）
# ---------------------------------------------------------------------------

class TestFillMissingOnly:
    """DB に manifest 以外の行があってよく、特徴量は NULL の列だけを埋める（既存の値は上書きしない）。"""

    def _db_with_extra_row(self, tmp_path, rows):
        db_path = build_db(tmp_path, rows)
        extra = rb.build_samples(manifest_rows("other", 300, start_idx=500)[:1], images_dir=None)
        sample_db.upsert_samples([{**extra[0], "t": None}], db_path)
        gc.collect()
        return db_path

    def _run(self, rows, db_path, images_dir, factory, **kwargs):
        return rfp.restore_face_features(
            rows, db_path, images_dir, face_mesh_factory=factory, fill_missing_only=True, **kwargs)

    def test_accepts_a_db_with_rows_outside_the_manifest_and_leaves_them_untouched(self, tmp_path):
        rows = manifest_rows("img_a", BASE_WIDTH)
        db_path = self._db_with_extra_row(tmp_path, rows)
        write_image(tmp_path / "images", "img_a", BASE_WIDTH)
        face = make_face(yaw_z=0.1)

        result = self._run(rows, db_path, tmp_path / "images", make_factory({BASE_WIDTH: [face]}))

        stored = db_rows(db_path)
        assert result.n_verified == 1
        assert stored[500]["face_yaw"] is None and stored[500]["lap_var"] is None  # manifest 外の行は触らない
        assert stored[0]["face_yaw"] == pytest.approx(expected_pose(face, BASE_WIDTH)[0])

    def test_existing_values_are_never_overwritten_and_complete_images_skip_facemesh(self, tmp_path):
        rows = manifest_rows("img_a", BASE_WIDTH)
        db_path = build_db(tmp_path, rows)
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE samples SET face_yaw = 11.0, face_pitch = 12.0, face_roll = 13.0, lap_var = 14.0")
        write_image(tmp_path / "images", "img_a", BASE_WIDTH)
        def forbidden():
            raise AssertionError("FaceMesh を作るべきではない（顔姿勢が揃っている）")

        result = self._run(rows, db_path, tmp_path / "images", forbidden)

        assert (result.n_images, result.n_skipped_complete, result.n_rows_updated) == (0, 1, 0)
        assert {(r["face_yaw"], r["face_pitch"], r["face_roll"], r["lap_var"])
                for r in db_rows(db_path).values()} == {(11.0, 12.0, 13.0, 14.0)}

    def test_only_null_columns_are_filled(self, tmp_path):
        """顔姿勢の一部だけが残っている行は、残っている値を変えず、NULL の列だけを埋める。"""
        rows = manifest_rows("img_a", BASE_WIDTH)
        db_path = build_db(tmp_path, rows)
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE samples SET face_yaw = 11.0")
        write_image(tmp_path / "images", "img_a", BASE_WIDTH)

        result = self._run(rows, db_path, tmp_path / "images", make_factory({BASE_WIDTH: [make_face(yaw_z=0.1)]}))

        assert result.n_verified == 1
        for row in db_rows(db_path).values():
            assert row["face_yaw"] == 11.0  # 既存の値のまま
            assert row["face_pitch"] is not None and row["face_roll"] is not None
            assert row["lap_var"] is not None and row["lap_var"] > 0  # 処理する画像なので、lap_var もついでに埋まる

    def test_an_image_lacking_only_lap_var_is_left_alone(self, tmp_path):
        """lap_var は分析用の列で NULL でもよい。顔姿勢が揃っていれば、lap_var が NULL でも FaceMesh に通さない。"""
        rows = manifest_rows("img_a", BASE_WIDTH)
        db_path = build_db(tmp_path, rows)
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE samples SET face_yaw = 1.0, face_pitch = 2.0, face_roll = 3.0")
        write_image(tmp_path / "images", "img_a", BASE_WIDTH)

        def forbidden():
            raise AssertionError("FaceMesh を作るべきではない（lap_var だけが NULL）")

        result = self._run(rows, db_path, tmp_path / "images", forbidden)

        assert (result.n_images, result.n_skipped_complete, result.n_rows_updated) == (0, 1, 0)
        assert {r["lap_var"] for r in db_rows(db_path).values()} == {None}

    def test_only_images_that_still_lack_features_are_processed(self, tmp_path):
        rows = manifest_rows("img_a", BASE_WIDTH) + manifest_rows("img_b", 330, start_idx=100)
        db_path = build_db(tmp_path, rows)
        with sqlite3.connect(db_path) as conn:  # img_a だけ揃っている
            conn.execute("UPDATE samples SET face_yaw = 1.0, face_pitch = 2.0, face_roll = 3.0, lap_var = 4.0 "
                         "WHERE image_path = 'data/images/img_a.jpg'")
        for image_id, width in (("img_a", BASE_WIDTH), ("img_b", 330)):
            write_image(tmp_path / "images", image_id, width)
        factory = make_factory({BASE_WIDTH: [make_face()], 330: [make_face()]})

        result = self._run(rows, db_path, tmp_path / "images", factory)

        assert factory.mesh.processed_widths == [330]
        assert (result.n_images, result.n_skipped_complete, result.n_verified) == (1, 1, 1)

    def test_rows_outside_the_manifest_are_untouched_even_when_they_share_a_face(self, tmp_path):
        """同じ顔（image_path・face_index）の行でも、manifest に無い行には書かない。"""
        rows = manifest_rows("img_a", BASE_WIDTH)
        db_path = build_db(tmp_path, rows)
        same_face = dataclasses.replace(rows[0], idx=700, point_index=9)  # 自然キーが重ならない別の点
        sample_db.upsert_samples([{**rb.build_samples((same_face,), images_dir=None)[0], "t": None}], db_path)
        gc.collect()
        write_image(tmp_path / "images", "img_a", BASE_WIDTH)

        result = self._run(rows, db_path, tmp_path / "images", make_factory({BASE_WIDTH: [make_face(yaw_z=0.1)]}))

        stored = db_rows(db_path)
        assert stored[700]["face_yaw"] is None and stored[700]["lap_var"] is None
        assert stored[0]["face_yaw"] is not None
        assert result.n_rows_updated == len(rows)  # manifest の行だけを数える

    def test_updated_row_count_only_counts_rows_that_were_missing_something(self, tmp_path):
        rows = manifest_rows("img_a", BASE_WIDTH)
        db_path = build_db(tmp_path, rows)
        with sqlite3.connect(db_path) as conn:  # 前半の 14 行は揃っている。後半の 14 行は何も無い
            conn.execute("UPDATE samples SET face_yaw = 1.0, face_pitch = 2.0, face_roll = 3.0, lap_var = 4.0 "
                         "WHERE idx < 14")
        write_image(tmp_path / "images", "img_a", BASE_WIDTH)

        result = self._run(rows, db_path, tmp_path / "images", make_factory({BASE_WIDTH: [make_face()]}))

        assert result.n_rows_updated == 14

    def test_a_manifest_row_swapped_for_another_sample_is_still_refused(self, tmp_path):
        rows = manifest_rows("img_a", BASE_WIDTH)
        db_path = self._db_with_extra_row(tmp_path, rows)
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE samples SET image_path = 'data/images/swapped.jpg' WHERE idx = 0")

        with pytest.raises(rfp.FeatureGuardError, match="食い違う"):
            self._run(rows, db_path, tmp_path / "images", make_factory({}))

    def test_manifest_rows_missing_from_the_db_are_simply_skipped(self, tmp_path):
        rows = manifest_rows("img_a", BASE_WIDTH) + manifest_rows("img_b", 330, start_idx=100)
        db_path = build_db(tmp_path, rows[:28])  # img_a のみ
        write_image(tmp_path / "images", "img_a", BASE_WIDTH)

        result = self._run(rows, db_path, tmp_path / "images", make_factory({BASE_WIDTH: [make_face()]}))

        assert result.n_verified == 1

    def test_dry_run_counts_only_the_images_that_need_features(self, tmp_path):
        rows = manifest_rows("img_a", BASE_WIDTH) + manifest_rows("img_b", 330, start_idx=100)
        db_path = build_db(tmp_path, rows)
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE samples SET face_yaw = 1.0, face_pitch = 2.0, face_roll = 3.0, lap_var = 4.0 "
                         "WHERE image_path = 'data/images/img_a.jpg'")
        factory = make_factory({})

        result = self._run(rows, db_path, tmp_path / "images", factory, dry_run=True)

        assert result.dry_run and (result.n_images, result.n_skipped_complete) == (1, 1)
        assert factory.mesh.processed_widths == []

    def test_the_default_mode_still_overwrites_existing_values(self, tmp_path):
        """fill_missing_only を付けない従来の復元は、manifest から再構築した DB を毎回再計算して上書きする。"""
        rows = manifest_rows("img_a", BASE_WIDTH)
        db_path = build_db(tmp_path, rows)
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE samples SET face_yaw = 999.0")
        write_image(tmp_path / "images", "img_a", BASE_WIDTH)
        face = make_face(yaw_z=0.1)

        rfp.restore_face_features(
            rows, db_path, tmp_path / "images", face_mesh_factory=make_factory({BASE_WIDTH: [face]}))

        yaws = {r["face_yaw"] for r in db_rows(db_path).values()}
        assert len(yaws) == 1
        assert yaws.pop() == pytest.approx(expected_pose(face, BASE_WIDTH)[0])

    def test_the_summary_mentions_skipped_images(self):
        result = rfp.FeatureRestoreResult(n_images=2, n_verified=2, n_skipped_complete=5)

        assert any("5" in line and "揃って" in line for line in rfp.format_result(result))
