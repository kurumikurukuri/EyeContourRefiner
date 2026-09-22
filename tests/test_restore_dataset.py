"""tools/data_management/restore_dataset.py（fetch → db → features の復元パイプライン）のテスト。

前半は下位処理をフェイクに差し替えて制御（順序・fail-fast・オプション伝播）を、
後半は実モジュールを通し、ネットワークと FaceMesh だけフェイクにした結合を確認する。
"""
from __future__ import annotations

import csv
import gc
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from annotator import db as sample_db  # noqa: E402
from restore_fixtures import (  # noqa: E402
    BASE_WIDTH,
    IMG_H,
    FakeFaceMesh,
    db_rows,
    expected_pose,
    make_face,
    manifest_rows,
)
from tools.data_management import fetch_images as fi  # noqa: E402
from tools.data_management import oi_filter as oif  # noqa: E402
from tools.data_management import oi_metadata as oim  # noqa: E402
from tools.data_management import restore_dataset as rd  # noqa: E402
from tools.sampledb_maintenance import add_new_images as ani  # noqa: E402
from tools.sampledb_maintenance import rebuild_samples_db as rb  # noqa: E402
from tools.sampledb_maintenance import restore_face_pose as rfp  # noqa: E402

MANIFEST_HEADER = [
    "idx", "image_path", "face_index", "part_name", "sub_key", "point_index",
    "px", "py", "qx", "qy", "t", "nose_x", "nose_y",
]


def write_manifest(experiments_dir: Path, exp_name: str, rows, header=MANIFEST_HEADER) -> Path:
    path = experiments_dir / exp_name / "manifests" / "manifest_train.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "idx": r.idx, "image_path": r.image_path, "face_index": r.face_index,
                "part_name": r.part_name, "sub_key": r.sub_key, "point_index": r.point_index,
                "px": r.px, "py": r.py, "qx": r.qx, "qy": r.qy, "t": r.t,
                "nose_x": "", "nose_y": "",
            })
    return path


def argv_for(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--experiments-dir", str(tmp_path / "experiments"),
        "--db", str(tmp_path / "data" / "samples.db"),
        "--images-dir", str(tmp_path / "data" / "images"),
        "--attribution-csv", str(tmp_path / "data" / "attribution.csv"),
        *extra,
    ]


# ---------------------------------------------------------------------------
# ステップ選択
# ---------------------------------------------------------------------------

class TestStepSelection:

    def test_default_runs_all_steps_in_order(self, tmp_path):
        assert rd.parse_args(argv_for(tmp_path)).steps == ("fetch", "db", "features")

    @pytest.mark.parametrize("given, expected", [
        (["features", "db"], ("db", "features")),
        (["db", "features"], ("db", "features")),
        (["features", "fetch"], ("fetch", "features")),
        (["db", "db", "fetch"], ("fetch", "db")),
        (["features"], ("features",)),
    ])
    def test_steps_run_in_canonical_order_without_duplicates(self, tmp_path, given, expected):
        options = rd.parse_args(argv_for(tmp_path, "--steps", *given))

        assert options.steps == expected

    def test_unknown_step_is_a_usage_error(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            rd.parse_args(argv_for(tmp_path, "--steps", "download"))

        assert exc.value.code == 2

    def test_options_are_mapped(self, tmp_path):
        options = rd.parse_args(argv_for(
            tmp_path, "--exp", "32", "--overwrite-db", "--dry-run", "--tol", "0.3",
            "--workers", "2", "--retries", "1"))

        assert options.exp == "32"
        assert options.overwrite_db and options.dry_run
        assert (options.tol, options.workers, options.retries) == (0.3, 2, 1)
        assert options.db_path == tmp_path / "data" / "samples.db"

    @pytest.mark.parametrize("bad", [["--tol", "0"], ["--workers", "0"], ["--retries", "-1"]])
    def test_invalid_numbers_are_usage_errors(self, tmp_path, bad):
        with pytest.raises(SystemExit) as exc:
            rd.parse_args(argv_for(tmp_path, *bad))

        assert exc.value.code == 2


# ---------------------------------------------------------------------------
# 制御（下位処理はフェイク）
# ---------------------------------------------------------------------------

@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """3 つの下位処理をフェイクにして呼び出しを記録する。"""
    rows = manifest_rows("img_a", BASE_WIDTH)
    write_manifest(tmp_path / "experiments", "EXP032_x", rows)
    state = {"calls": [], "fetch_ok": True, "db_error": None, "feature_error": None}

    def fake_fetch(options, **kwargs):
        state["calls"].append(("fetch", options))
        summary = fi.FetchSummary(downloaded=1) if state["fetch_ok"] else fi.FetchSummary(failed=1)
        return fi.FetchReport(fi.FetchPlan((), {}, 1, 0), (), summary, options.dry_run, 0)

    def fake_rebuild(manifest_rows_, db_path, **kwargs):
        state["calls"].append(("db", manifest_rows_, db_path, kwargs))
        if state["db_error"]:
            raise state["db_error"]
        return rb.RebuildResult("created", "DB を新規作成します", len(manifest_rows_), 1,
                                dry_run=kwargs.get("dry_run", False))

    def fake_features(manifest_rows_, db_path, images_dir, **kwargs):
        state["calls"].append(("features", manifest_rows_, db_path, images_dir, kwargs))
        if state["feature_error"]:
            raise state["feature_error"]
        return rfp.FeatureRestoreResult(1, 1, 0, 0, 0, 0, 28, kwargs.get("dry_run", False), ())

    monkeypatch.setattr(fi, "run_fetch", fake_fetch)
    monkeypatch.setattr(rb, "rebuild_samples_db", fake_rebuild)
    monkeypatch.setattr(rfp, "restore_face_features", fake_features)
    state["rows"] = rows
    return state


class TestOrchestration:

    def test_runs_fetch_then_db_then_features(self, tmp_path, stubbed):
        code = rd.main(argv_for(tmp_path))

        assert code == 0
        assert [c[0] for c in stubbed["calls"]] == ["fetch", "db", "features"]

    def test_only_selected_steps_run(self, tmp_path, stubbed):
        rd.main(argv_for(tmp_path, "--steps", "db"))

        assert [c[0] for c in stubbed["calls"]] == ["db"]

    def test_all_steps_share_the_same_manifest_rows_and_paths(self, tmp_path, stubbed):
        rd.main(argv_for(tmp_path, "--steps", "db", "features"))

        (_, db_rows_, db_path, db_kwargs), (_, feat_rows, feat_db, images_dir, feat_kwargs) = stubbed["calls"]
        assert db_rows_ == feat_rows == tuple(sorted(stubbed["rows"], key=lambda r: r.idx))
        assert db_path == feat_db == tmp_path / "data" / "samples.db"
        assert images_dir == tmp_path / "data" / "images"
        assert db_kwargs["images_dir"] == images_dir

    def test_fetch_options_target_the_detected_experiment(self, tmp_path, stubbed):
        rd.main(argv_for(tmp_path, "--steps", "fetch", "--workers", "3"))

        (_, options), = stubbed["calls"]
        assert options.source == "manifest"
        assert options.exp == 32
        # 読み込み済みの manifest から ID を渡す（fetch が manifest を読み直さない）
        assert options.image_ids == ("img_a",)
        assert options.experiments_dir == tmp_path / "experiments"
        assert options.output_dir == tmp_path / "data" / "images"
        assert options.attribution_csv == tmp_path / "data" / "attribution.csv"
        assert options.workers == 3

    def test_dry_run_and_flags_are_propagated(self, tmp_path, stubbed):
        rd.main(argv_for(tmp_path, "--dry-run", "--overwrite-db", "--tol", "0.4"))

        fetch, db, features = stubbed["calls"]
        assert fetch[1].dry_run is True
        assert db[3]["dry_run"] is True and db[3]["overwrite"] is True
        assert features[4]["dry_run"] is True and features[4]["tol"] == 0.4

    def test_a_refused_db_stops_the_pipeline_before_features(self, tmp_path, stubbed):
        stubbed["db_error"] = rb.DbGuardError("既存 DB が manifest と一致しません")

        code = rd.main(argv_for(tmp_path))

        assert code == 1
        assert [c[0] for c in stubbed["calls"]] == ["fetch", "db"]

    def test_incomplete_fetch_is_reported_but_later_steps_still_run(self, tmp_path, stubbed):
        stubbed["fetch_ok"] = False

        code = rd.main(argv_for(tmp_path))

        assert code == 1
        assert [c[0] for c in stubbed["calls"]] == ["fetch", "db", "features"]

    def test_feature_guard_failure_is_reported(self, tmp_path, stubbed, capsys):
        stubbed["feature_error"] = rfp.FeatureGuardError("manifest に無い行があります")

        code = rd.main(argv_for(tmp_path, "--steps", "features"))

        assert code == 1
        assert "manifest に無い行があります" in capsys.readouterr().out

    def test_missing_mediapipe_is_reported(self, tmp_path, stubbed, capsys):
        stubbed["feature_error"] = rfp.FaceMeshUnavailableError("mediapipe が見つかりません")

        code = rd.main(argv_for(tmp_path, "--steps", "features"))

        assert code == 1
        assert "mediapipe" in capsys.readouterr().out

    @pytest.mark.parametrize("error", [
        sqlite3.OperationalError("database is locked"),
        csv.Error("field larger than field limit (131072)"),
        PermissionError("in use"),
    ])
    def test_unexpected_step_errors_are_reported_as_failures_not_tracebacks(
            self, tmp_path, stubbed, capsys, error):
        """annotator が DB を掴んでいる等の実行時エラーでも、それまでのステップの報告を失わない。"""
        stubbed["feature_error"] = error

        code = rd.main(argv_for(tmp_path))

        out = capsys.readouterr().out
        assert code == 1
        assert "[fetch] OK" in out and "[db] OK" in out and "[features] 失敗" in out
        assert str(error) in out

    def test_unusable_manifest_aborts_before_any_step(self, tmp_path, stubbed):
        legacy = [h for h in MANIFEST_HEADER if h not in ("face_index", "nose_x", "nose_y")]
        write_manifest(tmp_path / "experiments", "EXP099_legacy", stubbed["rows"], header=legacy)

        code = rd.main(argv_for(tmp_path))

        assert code == 1
        assert stubbed["calls"] == []

    def test_unverified_images_are_a_warning_not_a_failure(self, tmp_path, stubbed, monkeypatch):
        monkeypatch.setattr(
            rfp, "restore_face_features",
            lambda *a, **k: rfp.FeatureRestoreResult(
                2, 1, 1, 0, 0, 0, 28, False, (rfp.ImageOutcome("img_b", "unverified", 0.9),)))

        assert rd.main(argv_for(tmp_path, "--steps", "features")) == 0

    def test_report_names_the_manifest_and_each_step(self, tmp_path, stubbed, capsys):
        rd.main(argv_for(tmp_path))

        out = capsys.readouterr().out
        assert "EXP032" in out
        for step in ("fetch", "db", "features"):
            assert step in out


# ---------------------------------------------------------------------------
# 結合（ネットワークと FaceMesh のみフェイク）
# ---------------------------------------------------------------------------

def jpeg_bytes(width: int, seed: int = 0) -> bytes:
    pixels = np.random.default_rng(seed).integers(0, 256, size=(IMG_H, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", pixels)
    assert ok
    return encoded.tobytes()


class FakeS3:
    def __init__(self, images: dict[str, bytes]):
        self.images = images
        self.urls: list[str] = []

    class _Response:
        def __init__(self, status, body=b""):
            self.status_code = status
            self.headers = {"Content-Length": str(len(body))}
            self._body = body

        def iter_content(self, chunk_size=1):
            yield self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def get(self, url, stream=True, timeout=None, **kwargs):
        self.urls.append(url)
        split, name = url.rsplit("/", 2)[-2:]
        image_id = name.removesuffix(".jpg")
        if split == "validation" and image_id in self.images:
            return self._Response(200, self.images[image_id])
        return self._Response(404)


@pytest.fixture
def world(monkeypatch, tmp_path):
    """img_a（1 顔）と img_b（1 顔）の manifest・偽 S3・偽 FaceMesh。"""
    rows = manifest_rows("img_a", 320) + manifest_rows("img_b", 330, start_idx=100)
    write_manifest(tmp_path / "experiments", "EXP032_x", rows)
    s3 = FakeS3({"img_a": jpeg_bytes(320, 1), "img_b": jpeg_bytes(330, 2)})
    face_a, face_b = make_face(yaw_z=0.05), make_face(yaw_z=0.15)
    mesh = FakeFaceMesh({320: [face_a], 330: [face_b]})
    monkeypatch.setattr(fi, "_new_session", lambda workers: s3)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(rfp, "create_face_mesh", lambda: mesh)
    return {"rows": rows, "s3": s3, "mesh": mesh, "faces": {"img_a": (face_a, 320), "img_b": (face_b, 330)}}


class TestEndToEnd:

    def test_restores_images_db_and_features(self, tmp_path, world):
        code = rd.main(argv_for(tmp_path))

        assert code == 0
        images = tmp_path / "data" / "images"
        assert sorted(p.name for p in images.iterdir()) == ["img_a.jpg", "img_b.jpg"]
        stored = db_rows(tmp_path / "data" / "samples.db")
        assert len(stored) == len(world["rows"])
        for row in stored.values():
            face, width = world["faces"][row["image_path"].split("/")[-1].removesuffix(".jpg")]
            assert row["face_yaw"] == pytest.approx(expected_pose(face, width)[0])
            assert row["lap_var"] > 0
            assert row["norm_eye_scale"] is not None and row["eye_scale"] > 1.0

    def test_second_run_is_a_no_op(self, tmp_path, world):
        rd.main(argv_for(tmp_path))
        first = db_rows(tmp_path / "data" / "samples.db")
        world["s3"].urls.clear()

        code = rd.main(argv_for(tmp_path))

        assert code == 0
        assert world["s3"].urls == []  # 取得済み画像は再取得しない
        assert db_rows(tmp_path / "data" / "samples.db") == first

    def test_dry_run_writes_and_downloads_nothing(self, tmp_path, world):
        code = rd.main(argv_for(tmp_path, "--dry-run"))

        assert code == 0
        assert not (tmp_path / "data").exists()
        assert world["s3"].urls == []
        assert world["mesh"].processed_widths == []

    def test_dry_run_reports_the_feature_step_that_will_follow_the_db_step(self, tmp_path, world, capsys):
        """DB がまだ無い dry-run でも features 段でエラーにならず、実行予定として表示される。"""
        code = rd.main(argv_for(tmp_path, "--dry-run"))

        assert code == 0
        assert "2" in capsys.readouterr().out

    def test_features_alone_without_a_db_asks_for_the_db_step(self, tmp_path, world, capsys):
        code = rd.main(argv_for(tmp_path, "--steps", "features"))

        assert code == 1
        assert "db" in capsys.readouterr().out

    def test_features_never_touch_a_larger_existing_database(self, tmp_path, world):
        """本来の大規模 DB（manifest に無い行を含む）には、復元用 manifest 由来の値を書き込まない。"""
        db_path = tmp_path / "data" / "samples.db"
        rd.main(argv_for(tmp_path, "--steps", "fetch", "db"))
        extra = rb.build_samples(manifest_rows("someone", 300, start_idx=900)[:1], images_dir=None)
        sample_db.upsert_samples([{**extra[0], "t": None}], db_path)
        gc.collect()
        before = db_rows(db_path)

        code = rd.main(argv_for(tmp_path, "--steps", "features"))

        assert code == 1
        assert db_rows(db_path) == before
        assert world["mesh"].processed_widths == []

    def test_a_different_existing_db_is_refused_unless_overwrite_is_given(self, tmp_path, world):
        db_path = tmp_path / "data" / "samples.db"
        db_path.parent.mkdir(parents=True)
        rb.rebuild_samples_db(manifest_rows("other", 300, start_idx=500)[:3], db_path, images_dir=None)
        before = db_rows(db_path)

        assert rd.main(argv_for(tmp_path, "--steps", "db")) == 1
        assert db_rows(db_path) == before

        assert rd.main(argv_for(tmp_path, "--steps", "db", "--overwrite-db")) == 0
        assert len(db_rows(db_path)) == len(world["rows"])
        assert len(list(db_path.parent.glob("samples.db.bak-*"))) == 1

    def test_missing_s3_image_fails_the_run_but_restores_the_rest(self, tmp_path, world):
        del world["s3"].images["img_b"]

        code = rd.main(argv_for(tmp_path))

        assert code == 1
        stored = db_rows(tmp_path / "data" / "samples.db")
        assert len(stored) == len(world["rows"])
        by_image = {}
        for row in stored.values():
            by_image.setdefault(row["image_path"], []).append(row["face_yaw"])
        assert all(v is not None for v in by_image["data/images/img_a.jpg"])
        assert all(v is None for v in by_image["data/images/img_b.jpg"])

    def test_wrong_image_content_is_not_written_to_the_db(self, tmp_path, world):
        """同じ ID でも、アノテーション時と別の画像（顔の位置が違う）なら特徴量を書かない。"""
        world["mesh"]._faces_by_width[320] = [make_face(dx=0.2, dy=0.2)]

        code = rd.main(argv_for(tmp_path))

        assert code == 0  # データ不一致は警告（実行の失敗ではない）
        stored = db_rows(tmp_path / "data" / "samples.db")
        img_a = [r for r in stored.values() if r["image_path"].endswith("img_a.jpg")]
        img_b = [r for r in stored.values() if r["image_path"].endswith("img_b.jpg")]
        assert all(r["face_yaw"] is None and r["lap_var"] is None for r in img_a)
        assert all(r["face_yaw"] is not None for r in img_b)

    def test_explicit_exp_selects_an_older_experiment(self, tmp_path, world):
        write_manifest(tmp_path / "experiments", "EXP040_newer", manifest_rows("img_a", 320)[:2])

        rd.main(argv_for(tmp_path, "--steps", "db", "--exp", "32"))

        assert len(db_rows(tmp_path / "data" / "samples.db")) == len(world["rows"])


class TestBundledManifest:

    def test_dry_run_against_the_repository_manifest(self, tmp_path, capsys):
        code = rd.main([
            "--dry-run",
            "--experiments-dir", str(ROOT / "experiments"),
            "--db", str(tmp_path / "samples.db"),
            "--images-dir", str(tmp_path / "images"),
            "--attribution-csv", str(tmp_path / "attribution.csv"),
        ])

        out = capsys.readouterr().out
        assert code == 0
        assert "EXP0" in out
        assert not (tmp_path / "samples.db").exists()
        assert not (tmp_path / "images").exists()


# ---------------------------------------------------------------------------
# --source all（manifest + Open Images。DB は追記のみ）
# ---------------------------------------------------------------------------

class TestSourceOption:

    def test_source_defaults_to_manifest(self, tmp_path):
        options = rd.parse_args(argv_for(tmp_path))

        assert (options.source, options.splits, options.max_images) == ("manifest", ("validation",), None)

    def test_all_options_are_mapped(self, tmp_path):
        options = rd.parse_args(argv_for(
            tmp_path, "--source", "all", "--split", "validation", "test", "--max-images", "7",
            "--cache-dir", str(tmp_path / "cache")))

        assert (options.source, options.splits, options.max_images) == ("all", ("validation", "test"), 7)
        assert options.cache_dir == tmp_path / "cache"

    @pytest.mark.parametrize("bad", [
        ["--source", "csv"],
        ["--source", "all", "--max-images", "-1"],
        ["--source", "all", "--split", "dev"],
    ])
    def test_invalid_values_are_usage_errors(self, tmp_path, bad):
        with pytest.raises(SystemExit) as exc:
            rd.parse_args(argv_for(tmp_path, *bad))

        assert exc.value.code == 2

    @pytest.mark.parametrize("extra", [["--max-images", "5"], ["--split", "test"]])
    def test_all_only_options_are_rejected_with_the_manifest_source(self, tmp_path, extra, capsys):
        """`--source manifest --max-images 5` は 5 枚しか取得せず、DB だけ全行を作る中途半端な復元になる。"""
        with pytest.raises(SystemExit) as exc:
            rd.parse_args(argv_for(tmp_path, *extra))

        assert exc.value.code == 2
        assert "--source all" in capsys.readouterr().err


@pytest.fixture
def stubbed_add(monkeypatch, stubbed):
    """`stubbed` に加えて、行の追記（add_new_images）もフェイクにして呼び出しを記録する。"""
    stubbed["add_error"] = None

    def fake_add(db_path, images_dir, **kwargs):
        stubbed["calls"].append(("add", db_path, images_dir, kwargs))
        if stubbed["add_error"]:
            raise stubbed["add_error"]
        return ani.AddImagesResult(
            n_candidates=3, n_added_images=2, n_added_rows=56, first_idx=200, last_idx=255,
            dry_run=kwargs.get("dry_run", False))

    monkeypatch.setattr(ani, "add_new_images", fake_add)
    return stubbed


class TestAllOrchestration:

    def test_all_hands_the_source_and_limits_to_fetch(self, tmp_path, stubbed_add):
        rd.main(argv_for(tmp_path, "--source", "all", "--split", "validation", "test",
                         "--max-images", "9", "--cache-dir", str(tmp_path / "cache"), "--steps", "fetch"))

        (_, options), = stubbed_add["calls"]
        assert options.source == "all"
        assert options.splits == ("validation", "test")
        assert options.max_images == 9
        assert options.cache_dir == tmp_path / "cache"
        assert options.image_ids == ("img_a",)  # manifest の画像は読み込み済みの ID を渡す

    def test_all_runs_fetch_db_add_features_in_order(self, tmp_path, stubbed_add):
        code = rd.main(argv_for(tmp_path, "--source", "all"))

        assert code == 0
        assert [c[0] for c in stubbed_add["calls"]] == ["fetch", "db", "add", "features"]

    def test_all_lets_the_db_step_accept_extra_rows_then_adds_new_images(self, tmp_path, stubbed_add):
        rd.main(argv_for(tmp_path, "--source", "all", "--steps", "db"))

        (_, _, db_path, db_kwargs), (_, add_db, add_images, add_kwargs) = stubbed_add["calls"]
        assert db_kwargs["allow_extra_rows"] is True
        assert add_db == db_path == tmp_path / "data" / "samples.db"
        assert add_images == tmp_path / "data" / "images"
        assert add_kwargs["dry_run"] is False

    def test_all_features_only_fill_missing_values(self, tmp_path, stubbed_add):
        rd.main(argv_for(tmp_path, "--source", "all", "--steps", "features"))

        (_, _, _, _, features_kwargs), = stubbed_add["calls"]
        assert features_kwargs["fill_missing_only"] is True

    def test_manifest_source_stays_strict_and_adds_nothing(self, tmp_path, stubbed_add):
        rd.main(argv_for(tmp_path))

        names = [c[0] for c in stubbed_add["calls"]]
        assert names == ["fetch", "db", "features"]
        db_kwargs = next(c[3] for c in stubbed_add["calls"] if c[0] == "db")
        features_kwargs = next(c[4] for c in stubbed_add["calls"] if c[0] == "features")
        assert db_kwargs["allow_extra_rows"] is False
        assert features_kwargs["fill_missing_only"] is False

    def test_a_refused_db_adds_nothing_and_stops(self, tmp_path, stubbed_add):
        stubbed_add["db_error"] = rb.DbGuardError("既存 DB が manifest の行を含みません")

        code = rd.main(argv_for(tmp_path, "--source", "all"))

        assert code == 1
        assert [c[0] for c in stubbed_add["calls"]] == ["fetch", "db"]

    def test_an_add_failure_stops_the_pipeline_before_features(self, tmp_path, stubbed_add, capsys):
        stubbed_add["add_error"] = ani.AddImagesError("samples.db を読めません")

        code = rd.main(argv_for(tmp_path, "--source", "all"))

        assert code == 1
        assert [c[0] for c in stubbed_add["calls"]] == ["fetch", "db", "add"]
        assert "samples.db を読めません" in capsys.readouterr().out

    def test_missing_mediapipe_while_adding_is_reported(self, tmp_path, stubbed_add, capsys):
        stubbed_add["add_error"] = rfp.FaceMeshUnavailableError("mediapipe が見つかりません")

        code = rd.main(argv_for(tmp_path, "--source", "all", "--steps", "db"))

        assert code == 1
        assert "mediapipe" in capsys.readouterr().out

    def test_report_shows_what_was_added(self, tmp_path, stubbed_add, capsys):
        rd.main(argv_for(tmp_path, "--source", "all", "--steps", "db"))

        out = capsys.readouterr().out
        assert "追記" in out and "56" in out

    def test_dry_run_is_propagated_to_the_add_step(self, tmp_path, stubbed_add):
        rd.main(argv_for(tmp_path, "--source", "all", "--steps", "db", "--dry-run"))

        add = next(c for c in stubbed_add["calls"] if c[0] == "add")
        assert add[3]["dry_run"] is True


# ---------------------------------------------------------------------------
# 結合: --source all（ネットワークと FaceMesh のみフェイク）
# ---------------------------------------------------------------------------

CC_BY = "https://creativecommons.org/licenses/by/2.0/"


@pytest.fixture
def open_images(monkeypatch, world):
    """Open Images の validation に、manifest に無い新規の画像が 2 枚ある状況。"""
    metas = {
        "new1": oim.ImageMeta("new1", CC_BY, "o1", "author1", "t"),
        "new2": oim.ImageMeta("new2", CC_BY, "o2", "author2", "t"),
    }
    world["s3"].images.update({"new1": jpeg_bytes(340, 3), "new2": jpeg_bytes(350, 4)})
    world["mesh"]._faces_by_width.update({340: [make_face(yaw_z=0.2)], 350: [make_face(yaw_z=0.25)]})
    monkeypatch.setattr(oim, "fetch_permissive_metadata",
                        lambda split, cache_dir, refresh=False: metas if split == "validation" else {})
    monkeypatch.setattr(oim, "resolve_annotations_csv", lambda *a, **k: Path("ann.csv"))
    monkeypatch.setattr(oim, "resolve_classes_csv", lambda *a, **k: Path("classes.csv"))
    monkeypatch.setattr(oif, "filter_image_ids", lambda candidates, ann, classes, config: set(candidates))
    world["metas"] = metas
    return world


def all_argv(tmp_path: Path, *extra: str) -> list[str]:
    return argv_for(tmp_path, "--source", "all", "--cache-dir", str(tmp_path / "cache"), *extra)


class TestEndToEndAll:

    def test_restores_the_manifest_and_adds_unannotated_rows_for_new_images(self, tmp_path, open_images):
        code = rd.main(all_argv(tmp_path))

        assert code == 0
        images = tmp_path / "data" / "images"
        assert sorted(p.name for p in images.iterdir()) == ["img_a.jpg", "img_b.jpg", "new1.jpg", "new2.jpg"]
        stored = db_rows(tmp_path / "data" / "samples.db")
        manifest_idx = {r.idx for r in open_images["rows"]}
        restored = [r for i, r in stored.items() if i in manifest_idx]
        added = [r for i, r in stored.items() if i not in manifest_idx]
        assert len(restored) == len(open_images["rows"]) and len(added) == 56
        assert all(r["t"] is not None and r["face_yaw"] is not None for r in restored)  # features も復元
        assert all(r["t"] is None and r["face_yaw"] is not None and r["lap_var"] > 0 for r in added)
        assert min(i for i in stored if i not in manifest_idx) == max(manifest_idx) + 1
        assert {r["image_path"] for r in added} == {"data/images/new1.jpg", "data/images/new2.jpg"}

    def test_attribution_is_recorded_for_the_new_images_only(self, tmp_path, open_images):
        rd.main(all_argv(tmp_path))

        from tools.data_management import attribution as attr  # noqa: PLC0415
        assert set(attr.read_attribution(tmp_path / "data" / "attribution.csv")) == {"new1", "new2"}

    def test_second_run_is_a_no_op(self, tmp_path, open_images):
        rd.main(all_argv(tmp_path))
        first = db_rows(tmp_path / "data" / "samples.db")
        open_images["s3"].urls.clear()
        processed = list(open_images["mesh"].processed_widths)

        code = rd.main(all_argv(tmp_path))

        assert code == 0
        assert open_images["s3"].urls == []  # 手元にある画像は再取得しない
        assert open_images["mesh"].processed_widths == processed  # 行がある画像・顔姿勢が揃った画像は解析し直さない
        assert db_rows(tmp_path / "data" / "samples.db") == first

    def test_annotations_made_after_the_first_run_survive_a_rerun(self, tmp_path, open_images):
        rd.main(all_argv(tmp_path))
        db_path = tmp_path / "data" / "samples.db"
        stored = db_rows(db_path)
        new_idx = min(i for i, r in stored.items() if r["image_path"].endswith("new1.jpg"))
        sample_db.update_annotation(new_idx, 0.4, db_path)   # 新規行のアノテーション
        sample_db.update_annotation(0, 0.77, db_path)        # 復元済みの行の修正
        gc.collect()

        code = rd.main(all_argv(tmp_path))

        assert code == 0
        after = db_rows(db_path)
        assert after[new_idx]["t"] == 0.4 and after[0]["t"] == 0.77

    def test_a_wiped_lap_var_is_left_alone_and_causes_no_reprocessing(self, tmp_path, open_images):
        """lap_var は分析用で NULL でもよい。NULL になっても FaceMesh に通し直さず、他の値も変えない。"""
        rd.main(all_argv(tmp_path))
        db_path = tmp_path / "data" / "samples.db"
        before = db_rows(db_path)
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE samples SET lap_var = NULL")
        gc.collect()
        processed = list(open_images["mesh"].processed_widths)

        code = rd.main(all_argv(tmp_path))

        assert code == 0
        assert open_images["mesh"].processed_widths == processed
        after = db_rows(db_path)
        assert all(r["lap_var"] is None for r in after.values())
        assert all(after[i]["face_yaw"] == before[i]["face_yaw"] for i in before)

    def test_the_manifest_source_still_refuses_a_db_with_added_rows(self, tmp_path, open_images):
        rd.main(all_argv(tmp_path))
        db_path = tmp_path / "data" / "samples.db"
        before = db_rows(db_path)

        code = rd.main(argv_for(tmp_path, "--steps", "db"))

        assert code == 1
        assert db_rows(db_path) == before

    def test_dry_run_counts_images_that_will_get_rows_before_the_db_exists(self, tmp_path, open_images, capsys):
        images = tmp_path / "data" / "images"
        images.mkdir(parents=True)
        (images / "new1.jpg").write_bytes(jpeg_bytes(340, 3))

        code = rd.main(all_argv(tmp_path, "--steps", "db", "--dry-run"))

        out = capsys.readouterr().out
        assert code == 0
        assert "候補: 1 枚" in out
        assert not (tmp_path / "data" / "samples.db").exists()
        assert open_images["mesh"].processed_widths == []

    def test_dry_run_with_overwrite_does_not_mistake_extra_rows_for_a_rebuild(self, tmp_path, open_images, capsys):
        rd.main(all_argv(tmp_path))
        capsys.readouterr()

        code = rd.main(all_argv(tmp_path, "--steps", "db", "features", "--dry-run", "--overwrite-db"))

        out = capsys.readouterr().out
        assert code == 0
        assert "unchanged" in out                    # db: manifest の行をすべて含むので置換しない
        assert "対象画像: 0 枚" in out                # features: 揃っているので FaceMesh の対象は 0
        assert "飛ばした画像: 2 枚" in out

    def test_dry_run_downloads_and_writes_nothing(self, tmp_path, open_images):
        code = rd.main(all_argv(tmp_path, "--dry-run"))

        assert code == 0
        assert not (tmp_path / "data").exists()
        assert open_images["s3"].urls == []
        assert open_images["mesh"].processed_widths == []

    def test_repeating_with_a_limit_adds_the_next_images_each_time(self, tmp_path, open_images):
        """--max-images は「1 回の実行で追加する枚数」。手元にある画像は数えないので、繰り返すと次の画像が入る。"""
        images = tmp_path / "data" / "images"

        assert rd.main(all_argv(tmp_path, "--max-images", "1")) == 0
        first = sorted(p.stem for p in images.iterdir())
        assert rd.main(all_argv(tmp_path, "--max-images", "1")) == 0
        second = sorted(p.stem for p in images.iterdir())
        assert rd.main(all_argv(tmp_path, "--max-images", "1")) == 0  # 追加候補が尽きれば何も増えない

        assert len(first) == 3 and len(second) == 4
        assert sorted(p.stem for p in images.iterdir()) == second
        stored = db_rows(tmp_path / "data" / "samples.db")
        assert {r["image_path"] for r in stored.values() if r["t"] is None} == {
            "data/images/new1.jpg", "data/images/new2.jpg"}

    def test_max_images_limits_only_the_additional_images(self, tmp_path, open_images):
        code = rd.main(all_argv(tmp_path, "--max-images", "1"))

        assert code == 0
        names = sorted(p.name for p in (tmp_path / "data" / "images").iterdir())
        assert names[:2] == ["img_a.jpg", "img_b.jpg"] and len(names) == 3  # manifest の 2 枚 + 追加 1 枚
