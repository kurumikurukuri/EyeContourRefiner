"""fetch_images.py の取得元切替（--source manifest / all）・シャード・dry-run・attribution のテスト。"""
from __future__ import annotations

import csv
import dataclasses
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tools.data_management import attribution as attr  # noqa: E402
from tools.data_management import fetch_images as fi  # noqa: E402
from tools.data_management import oi_filter as oif  # noqa: E402
from tools.data_management import oi_metadata as oim  # noqa: E402
from tools.data_management.manifest_source import ManifestError  # noqa: E402

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 60 + b"\xff\xd9"
CC_BY = "https://creativecommons.org/licenses/by/2.0/"
CC0 = "https://creativecommons.org/publicdomain/zero/1.0/"
CC_SA = "https://creativecommons.org/licenses/by-sa/2.0/"

MANIFEST_HEADER = [
    "idx", "image_path", "face_index", "part_name", "sub_key", "point_index",
    "px", "py", "qx", "qy", "t", "nose_x", "nose_y",
]


class _Response:
    def __init__(self, status=200, body=JPEG):
        self.status_code = status
        self._body = body
        self.headers = {"Content-Length": str(len(body))}

    def iter_content(self, chunk_size=1):
        yield self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class RecordingSession:
    """`serve` に含まれる (split, id) だけ 200、他は 404。"""

    def __init__(self, serve: set[tuple[str, str]] | None = None, fail: dict | None = None):
        self.serve = serve
        self.fail = fail or {}
        self.calls: list[str] = []

    def get(self, url, stream=True, timeout=None, **kwargs):
        self.calls.append(url)
        _, split, name = url.rsplit("/", 2)
        image_id = name.removesuffix(".jpg")
        if image_id in self.fail:
            return _Response(status=self.fail[image_id], body=b"")
        if self.serve is None or (split, image_id) in self.serve:
            return _Response()
        return _Response(status=404, body=b"")


class ForbiddenSession:
    def get(self, *a, **k):
        raise AssertionError("dry-run 中にネットワークアクセスが発生した")


def _write_manifest_paths(path: Path, image_paths: list[str], start_idx: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(MANIFEST_HEADER)
        for n, image_path in enumerate(image_paths, start=start_idx):
            writer.writerow([n, image_path, 0, "eyeliner_left", "upper",
                             0, 1.0, 2.0, 3.0, 4.0, 0.5, "", ""])
    return path


def _write_manifest(path: Path, image_ids: list[str], start_idx: int = 0) -> Path:
    return _write_manifest_paths(path, [f"data/images/{i}.jpg" for i in image_ids], start_idx)


def _write_experiment(tmp_path: Path, image_ids: list[str], exp: str = "EXP032_y",
                      name: str = "manifest_train.csv", start_idx: int = 0) -> Path:
    """`experiments/<exp>/manifests/<name>` に manifest を置く（取得元 manifest の既定の探索先）。"""
    return _write_manifest(tmp_path / "experiments" / exp / "manifests" / name, image_ids, start_idx)


def _write_attribution(path: Path, image_ids: list[str]) -> Path:
    attr.append_attribution(path, [
        oim.ImageMeta(i, CC_BY, f"https://o/{i}", f"author-{i}", "t") for i in image_ids
    ])
    return path


def _put_images(tmp_path: Path, image_ids: list[str], body: bytes = JPEG) -> Path:
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    for image_id in image_ids:
        (images / f"{image_id}.jpg").write_bytes(body)
    return images


def _options(tmp_path: Path, **overrides) -> fi.FetchOptions:
    base = dict(
        output_dir=tmp_path / "images",
        cache_dir=tmp_path / "cache",
        attribution_csv=tmp_path / "attribution.csv",
        experiments_dir=tmp_path / "experiments",
    )
    base.update(overrides)
    return fi.FetchOptions(**base)


def _all_options(tmp_path: Path, **overrides) -> fi.FetchOptions:
    """`--source all` 用。manifest の画像を明示しない限り、manifest 側は空にして Open Images 側だけを見る。"""
    overrides.setdefault("image_ids", ())
    return _options(tmp_path, source="all", **overrides)


class TestShardSlice:

    def test_contiguous_chunks_cover_everything_exactly_once(self):
        items = list(range(10))

        parts = [fi.shard_slice(items, k, 3) for k in range(3)]

        assert parts == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
        assert sum(parts, []) == items

    def test_no_sharding_returns_everything(self):
        assert fi.shard_slice([1, 2, 3], None, None) == [1, 2, 3]

    def test_more_shards_than_items(self):
        parts = [fi.shard_slice([1, 2], k, 5) for k in range(5)]

        assert sum(parts, []) == [1, 2]

    @pytest.mark.parametrize("shard, count", [(0, None), (None, 3), (3, 3), (-1, 3), (0, 0)])
    def test_invalid_arguments_are_rejected(self, shard, count):
        with pytest.raises(ValueError):
            fi.shard_slice([1, 2, 3], shard, count)


class TestPlanTargets:

    def _targets(self, n=10):
        return [fi.FetchTarget(f"id{i:02d}", "train") for i in range(n)]

    def test_orders_shards_excludes_then_limits(self):
        targets = list(reversed(self._targets(10)))

        planned = fi.plan_targets(
            targets, shard=1, shard_count=2, skip_ids={"id05", "id06"}, max_images=2)

        assert [t.image_id for t in planned] == ["id07", "id08"]

    def test_shard_boundaries_do_not_move_when_images_are_already_downloaded(self):
        targets = self._targets(9)

        fresh = fi.plan_targets(targets, shard=1, shard_count=3, skip_ids=set(), max_images=None)
        resumed = fi.plan_targets(
            targets, shard=1, shard_count=3, skip_ids={"id03"}, max_images=None)

        assert [t.image_id for t in fresh] == ["id03", "id04", "id05"]
        assert [t.image_id for t in resumed] == ["id04", "id05"]

    def test_duplicates_are_collapsed(self):
        targets = self._targets(3) + self._targets(3)

        planned = fi.plan_targets(targets, shard=None, shard_count=None, skip_ids=set(), max_images=None)

        assert len(planned) == 3

    def test_later_duplicate_wins_so_a_known_split_can_replace_an_unknown_one(self):
        targets = [fi.FetchTarget("a", None), fi.FetchTarget("a", "validation")]

        planned = fi.plan_targets(targets, shard=None, shard_count=None, skip_ids=set(), max_images=None)

        assert planned == [fi.FetchTarget("a", "validation")]

    def test_max_images_zero_plans_nothing(self):
        assert fi.plan_targets(self._targets(3), shard=None, shard_count=None,
                               skip_ids=set(), max_images=0) == []

    def test_uncapped_ids_are_kept_and_do_not_consume_the_limit(self):
        planned = fi.plan_targets(
            self._targets(6), shard=None, shard_count=None, skip_ids=set(),
            max_images=2, uncapped_ids={"id00", "id05"})

        assert [t.image_id for t in planned] == ["id00", "id01", "id02", "id05"]

    def test_uncapped_ids_are_still_sharded_and_skipped(self):
        planned = fi.plan_targets(
            self._targets(6), shard=1, shard_count=2, skip_ids={"id04"},
            max_images=0, uncapped_ids={"id03", "id04"})

        assert [t.image_id for t in planned] == ["id03"]


class TestManifestSource:

    def test_explicit_image_ids_are_used_without_reading_any_manifest(self, tmp_path):
        """restore_dataset が読み込み済みの manifest から ID を渡す用途（manifest を二重に読まない）。"""
        plan = fi.build_plan(_options(tmp_path, source="manifest", image_ids=("b2", "a1", "a1")))

        assert [t.image_id for t in plan.targets] == ["a1", "b2"]

    def test_explicit_image_ids_are_validated(self, tmp_path):
        with pytest.raises(ManifestError):
            fi.build_plan(_options(tmp_path, source="manifest", image_ids=("../evil",)))

    def test_source_defaults_to_manifest(self):
        assert fi.FetchOptions().source == "manifest"

    def test_defaults_to_latest_experiment_manifest(self, tmp_path):
        _write_experiment(tmp_path, ["old1"], exp="EXP008_x")
        _write_experiment(tmp_path, ["new1"], exp="EXP032_y")
        _write_experiment(tmp_path, ["new2"], exp="EXP032_y", name="manifest_test.csv", start_idx=100)

        plan = fi.build_plan(_options(tmp_path, source="manifest"))

        assert [t.image_id for t in plan.targets] == ["new1", "new2"]
        assert all(t.split is None for t in plan.targets)  # split は不明（S3 を順に探索する）

    def test_exp_option_selects_that_experiment(self, tmp_path):
        _write_experiment(tmp_path, ["old1"], exp="EXP008_x")
        _write_experiment(tmp_path, ["new1"], exp="EXP032_y")

        plan = fi.build_plan(_options(tmp_path, source="manifest", exp=8))

        assert [t.image_id for t in plan.targets] == ["old1"]

    def test_missing_manifest_is_rejected(self, tmp_path):
        with pytest.raises(ManifestError):
            fi.build_plan(_options(tmp_path, source="manifest"))

    @pytest.mark.parametrize("bad_path", ["../../etc/passwd.jpg", "data/images/../x.jpg", "/abs/x.jpg"])
    def test_path_traversal_in_manifest_is_rejected(self, tmp_path, bad_path):
        _write_manifest_paths(
            tmp_path / "experiments" / "EXP032_y" / "manifests" / "manifest_train.csv", [bad_path])

        with pytest.raises(ManifestError):
            fi.build_plan(_options(tmp_path, source="manifest"))

    def test_existing_files_are_excluded_but_attribution_known_ids_are_not(self, tmp_path):
        _write_experiment(tmp_path, ["have", "known", "new"])
        _write_attribution(tmp_path / "attribution.csv", ["known"])
        _put_images(tmp_path, ["have"])

        plan = fi.build_plan(_options(tmp_path, source="manifest"))

        assert [t.image_id for t in plan.targets] == ["known", "new"]
        assert plan.n_existing == 1

    def test_shard_and_limit_apply_to_manifest_ids(self, tmp_path):
        _write_experiment(tmp_path, [f"id{i}" for i in range(6)])

        plan = fi.build_plan(_options(tmp_path, source="manifest", shard=1, shard_count=2, max_images=2))

        assert [t.image_id for t in plan.targets] == ["id3", "id4"]


@pytest.fixture
def all_source(monkeypatch):
    """--source all のメタデータ・注釈・フィルタをフェイクに差し替える。"""
    state = {
        "metas": {
            "validation": {
                "v1": oim.ImageMeta("v1", CC_BY, "o1", "A", "t"),
                "v2": oim.ImageMeta("v2", CC0, "o2", "B", "t"),
                "v3": oim.ImageMeta("v3", CC_BY, "o3", "C", "t"),
            },
            "test": {"t1": oim.ImageMeta("t1", CC_BY, "o4", "D", "t")},
        },
        "filtered": {"validation": {"v1", "v2"}, "test": {"t1"}},
        "filter_calls": [],
    }

    monkeypatch.setattr(oim, "fetch_permissive_metadata",
                        lambda split, cache_dir, refresh=False: state["metas"][split])
    monkeypatch.setattr(oim, "resolve_annotations_csv", lambda *a, **k: Path("ann.csv"))
    monkeypatch.setattr(oim, "resolve_classes_csv", lambda *a, **k: Path("classes.csv"))

    def fake_filter(candidates, ann, classes, config):
        state["filter_calls"].append(set(candidates))
        return {i for i in candidates if i in state["filtered"]["validation"] | state["filtered"]["test"]}

    monkeypatch.setattr(oif, "filter_image_ids", fake_filter)
    return state


def _many_validation(state, n: int, prefix: str = "v") -> list[str]:
    """validation に n 枚の候補を作り、すべてフィルタを通す。ID の一覧を返す。"""
    ids = [f"{prefix}{i:04d}" for i in range(n)]
    state["metas"]["validation"] = {i: oim.ImageMeta(i, CC_BY, "o", "A", "t") for i in ids}
    state["filtered"]["validation"] = set(ids)
    return ids


class TestAllSource:

    def test_targets_carry_their_split(self, tmp_path, all_source):
        plan = fi.build_plan(_all_options(tmp_path, splits=("validation", "test")))

        assert {(t.image_id, t.split) for t in plan.targets} == {
            ("v1", "validation"), ("v2", "validation"), ("t1", "test"),
        }
        assert set(plan.metas) == {"v1", "v2", "t1"}

    def test_default_split_is_validation_only(self, tmp_path, all_source):
        plan = fi.build_plan(_all_options(tmp_path))

        assert {t.split for t in plan.targets} == {"validation"}

    def test_license_restriction_from_config_is_applied_before_filtering(self, tmp_path, all_source):
        config = oif.FilterConfig(allowed_licenses=("https://creativecommons.org/publicdomain/zero/1.0",))

        plan = fi.build_plan(_all_options(tmp_path, filter_config=config))

        assert [t.image_id for t in plan.targets] == ["v2"]
        assert all_source["filter_calls"] == [{"v2"}]

    def test_attribution_registered_images_that_are_not_on_disk_are_still_fetched(
            self, tmp_path, all_source):
        """クローンした第三者の手元には無い画像でも、リポジトリ同梱の attribution に載っているだけで隠さない。"""
        _write_attribution(tmp_path / "attribution.csv", ["v1", "v2"])

        plan = fi.build_plan(_all_options(tmp_path))

        assert [t.image_id for t in plan.targets] == ["v1", "v2"]

    def test_images_on_disk_and_already_registered_are_skipped(self, tmp_path, all_source):
        _write_attribution(tmp_path / "attribution.csv", ["v1", "v2"])
        _put_images(tmp_path, ["v1", "v2"])

        plan = fi.build_plan(_all_options(tmp_path))

        assert plan.targets == ()
        assert plan.n_existing == 2

    def test_downloaded_but_unregistered_images_stay_targets_so_attribution_is_completed(
            self, tmp_path, all_source):
        """強制終了で attribution の記録が漏れても、次回の実行で補完される（CC BY の表示義務）。"""
        _write_attribution(tmp_path / "attribution.csv", ["v1"])
        _put_images(tmp_path, ["v1", "v2"])

        plan = fi.build_plan(_all_options(tmp_path))

        assert [t.image_id for t in plan.targets] == ["v2"]

    def test_unregistered_existing_image_gets_its_attribution_without_any_download(
            self, tmp_path, all_source):
        images = _put_images(tmp_path, ["v2"], body=b"already-here")
        session = RecordingSession()

        report = fi.run_fetch(_all_options(tmp_path, workers=1), session=session, sleep=lambda s: None)

        registered = attr.read_attribution(tmp_path / "attribution.csv")
        assert registered["v2"].author == "B"
        assert (images / "v2.jpg").read_bytes() == b"already-here"  # 既存ファイルは上書きしない
        assert (report.summary.downloaded, report.summary.skipped) == (1, 1)
        assert all("v2" not in url for url in session.calls)  # v2 は再取得しない

    def test_ids_from_remote_metadata_are_revalidated_before_planning(self, tmp_path, all_source):
        all_source["metas"]["validation"]["../evil"] = oim.ImageMeta("../evil", CC_BY, "o", "X", "t")
        all_source["filtered"]["validation"] = all_source["filtered"]["validation"] | {"../evil"}

        plan = fi.build_plan(_all_options(tmp_path))

        assert "../evil" not in {t.image_id for t in plan.targets}

    def test_filter_result_is_cached_per_condition(self, tmp_path, all_source):
        opts = _all_options(tmp_path)

        fi.build_plan(opts)
        fi.build_plan(opts)
        fi.build_plan(dataclasses.replace(opts, filter_config=oif.FilterConfig(max_faces=3)))

        assert len(all_source["filter_calls"]) == 2  # 2 回目は条件ハッシュ付きキャッシュ

    def test_refresh_cache_recomputes_the_filter(self, tmp_path, all_source):
        opts = _all_options(tmp_path)
        fi.build_plan(opts)

        fi.build_plan(dataclasses.replace(opts, refresh_cache=True))

        assert len(all_source["filter_calls"]) == 2

    def test_shard_boundaries_are_fixed_before_excluding_existing_images(self, tmp_path, all_source):
        """手元にある画像（シャード 0 の範囲）が、シャード 1 の境界を動かさない。

        除外してからシャード分割すると、シャード 1 は v0004 から始まってしまう。
        """
        _many_validation(all_source, 6, prefix="v")
        _write_attribution(tmp_path / "attribution.csv", ["v0001"])
        _put_images(tmp_path, ["v0001"])

        shard0 = fi.build_plan(_all_options(tmp_path, shard=0, shard_count=2))
        shard1 = fi.build_plan(_all_options(tmp_path, shard=1, shard_count=2))

        assert [t.image_id for t in shard0.targets] == ["v0000", "v0002"]
        assert [t.image_id for t in shard1.targets] == ["v0003", "v0004", "v0005"]

    def test_shards_together_cover_every_target_exactly_once(self, tmp_path, all_source):
        ids = _many_validation(all_source, 10)
        manifest = ("m1", "m2", "m3")

        plans = [fi.build_plan(_all_options(tmp_path, image_ids=manifest, shard=k, shard_count=3))
                 for k in range(3)]

        planned = [t.image_id for p in plans for t in p.targets]
        assert sorted(planned) == sorted(list(manifest) + ids)
        assert len(planned) == len(set(planned))

    def test_shards_with_a_limit_still_fetch_every_manifest_image_exactly_once(
            self, tmp_path, all_source):
        _many_validation(all_source, 10)
        manifest = ("m1", "m2", "m3")

        plans = [fi.build_plan(_all_options(tmp_path, image_ids=manifest, shard=k, shard_count=3,
                                            max_images=2)) for k in range(3)]

        planned = [t.image_id for p in plans for t in p.targets]
        assert len(planned) == len(set(planned))
        assert set(manifest) <= set(planned)
        # --max-images はシャードごとに Open Images 側の枚数だけを数える
        assert all(sum(t.image_id.startswith("v") for t in p.targets) <= 2 for p in plans)

    def test_default_limit_protects_against_accidental_full_download(self, tmp_path, all_source):
        assert fi.DEFAULT_ALL_MAX_IMAGES == 500
        _many_validation(all_source, 600)

        default = fi.build_plan(_all_options(tmp_path))
        zero = fi.build_plan(_all_options(tmp_path, max_images=0))
        explicit = fi.build_plan(_all_options(tmp_path, max_images=550))

        assert len(default.targets) == 500
        assert len(zero.targets) == 0
        assert len(explicit.targets) == 550

    def test_images_already_on_disk_do_not_consume_the_limit(self, tmp_path, all_source):
        ids = _many_validation(all_source, 600)
        _put_images(tmp_path, ids[:3])  # 手元にあるが attribution 未登録（記録の補完だけが必要）

        plan = fi.build_plan(_all_options(tmp_path, max_images=5))

        assert [t.image_id for t in plan.targets] == ids[:8]  # 補完 3 件 + 新規取得 5 件

    def test_manifest_images_are_always_included_and_ignore_the_limit(self, tmp_path, all_source):
        _many_validation(all_source, 600)

        plan = fi.build_plan(_all_options(tmp_path, image_ids=("m1", "m2"), max_images=10))

        by_id = {t.image_id: t for t in plan.targets}
        assert len(plan.targets) == 12  # manifest 2 枚 + 追加 10 枚
        assert by_id["m1"].split is None and by_id["m2"].split is None
        assert "m1" not in plan.metas  # Open Images の候補ではないので帰属情報は無い

    def test_manifest_image_that_is_also_a_candidate_uses_its_known_split_and_metadata(
            self, tmp_path, all_source):
        plan = fi.build_plan(_all_options(tmp_path, image_ids=("v1", "m1")))

        splits = {t.image_id: t.split for t in plan.targets}
        assert splits["v1"] == "validation" and splits["m1"] is None
        assert plan.metas["v1"].author == "A"

    def test_manifest_images_on_disk_with_nothing_to_record_are_skipped(self, tmp_path, all_source):
        _put_images(tmp_path, ["m1"])  # 候補ではなく、attribution にも無い → 記録するものが無い

        plan = fi.build_plan(_all_options(tmp_path, image_ids=("m1",)))

        assert "m1" not in {t.image_id for t in plan.targets}

    def test_defaults_to_the_latest_experiment_manifest_plus_open_images(self, tmp_path, all_source):
        _write_experiment(tmp_path, ["mA"])

        plan = fi.build_plan(_options(tmp_path, source="all"))

        assert [t.image_id for t in plan.targets] == ["mA", "v1", "v2"]

    def test_missing_manifest_is_rejected(self, tmp_path, all_source):
        with pytest.raises(ManifestError):
            fi.build_plan(_options(tmp_path, source="all"))

    def test_unknown_split_is_rejected(self, tmp_path, all_source):
        with pytest.raises(ValueError, match="split"):
            fi.build_plan(_all_options(tmp_path, splits=("dev",)))

    @pytest.mark.parametrize("source", ["ftp", "csv"])
    def test_unknown_source_is_rejected(self, tmp_path, source):
        with pytest.raises(ValueError, match="source"):
            fi.build_plan(_options(tmp_path, source=source))


class TestRunFetch:

    def test_dry_run_downloads_nothing_and_estimates_size(self, tmp_path):
        images = _put_images(tmp_path, ["id0"])

        report = fi.run_fetch(
            _options(tmp_path, source="manifest", image_ids=tuple(f"id{i}" for i in range(10)),
                     dry_run=True),
            session=ForbiddenSession())

        assert report.dry_run is True
        assert report.results == ()
        assert len(report.plan.targets) == 9
        assert report.estimated_bytes == 9 * 320 * 1024
        assert sorted(p.name for p in images.iterdir()) == ["id0.jpg"]
        assert not (tmp_path / "attribution.csv").exists()

    def test_manifest_mode_restores_manifest_images(self, tmp_path):
        session = RecordingSession(serve={("validation", "a1"), ("train", "b2")})

        report = fi.run_fetch(
            _options(tmp_path, source="manifest", image_ids=("a1", "b2"), workers=1),
            session=session, sleep=lambda s: None)

        assert report.summary.downloaded == 2 and report.summary.ok
        assert (tmp_path / "images" / "a1.jpg").is_file()
        assert (tmp_path / "images" / "b2.jpg").is_file()

    def test_manifest_source_does_not_touch_attribution(self, tmp_path):
        fi.run_fetch(_options(tmp_path, source="manifest", image_ids=("a1",), workers=1),
                     session=RecordingSession(), sleep=lambda s: None)

        assert not (tmp_path / "attribution.csv").exists()

    def test_all_mode_records_attribution_for_downloaded_images(self, tmp_path, all_source):
        session = RecordingSession()

        report = fi.run_fetch(
            _all_options(tmp_path, splits=("validation", "test"), workers=2),
            session=session, sleep=lambda s: None)

        assert report.summary.downloaded == 3
        known = attr.read_attribution(tmp_path / "attribution.csv")
        assert set(known) == {"v1", "v2", "t1"}
        assert known["v1"].author == "A" and known["t1"].original_url == "o4"
        assert sorted(s for s in session.calls) == sorted(
            f"https://open-images-dataset.s3.amazonaws.com/{s}/{i}.jpg"
            for s, i in [("validation", "v1"), ("validation", "v2"), ("test", "t1")])

    def test_all_mode_downloads_manifest_images_without_recording_attribution_for_them(
            self, tmp_path, all_source):
        session = RecordingSession()

        report = fi.run_fetch(
            _all_options(tmp_path, image_ids=("m1",), workers=1), session=session, sleep=lambda s: None)

        assert report.summary.downloaded == 3  # m1 + v1 + v2
        assert (tmp_path / "images" / "m1.jpg").is_file()
        assert set(attr.read_attribution(tmp_path / "attribution.csv")) == {"v1", "v2"}

    def test_manifest_image_that_is_also_a_candidate_gets_its_attribution_without_a_download(
            self, tmp_path, all_source):
        """復元済み（手元にある）manifest の画像が Open Images の候補でもあれば、再取得せず帰属情報だけ記録する。"""
        images = _put_images(tmp_path, ["v1"], body=b"restored")
        session = RecordingSession()

        fi.run_fetch(_all_options(tmp_path, image_ids=("v1",), workers=1),
                     session=session, sleep=lambda s: None)

        assert attr.read_attribution(tmp_path / "attribution.csv")["v1"].author == "A"
        assert (images / "v1.jpg").read_bytes() == b"restored"
        assert all(not url.endswith("/v1.jpg") for url in session.calls)

    def test_registered_manifest_image_that_is_missing_is_fetched_from_its_known_split(
            self, tmp_path, all_source):
        """クローン直後: 同梱の attribution に載っていても、手元に無ければ取得する。split が分かるので 1 回で取れる。"""
        _write_attribution(tmp_path / "attribution.csv", ["v1"])
        session = RecordingSession()

        fi.run_fetch(_all_options(tmp_path, image_ids=("v1",), workers=1),
                     session=session, sleep=lambda s: None)

        assert (tmp_path / "images" / "v1.jpg").is_file()
        assert [u for u in session.calls if u.endswith("/v1.jpg")] == [
            "https://open-images-dataset.s3.amazonaws.com/validation/v1.jpg"]

    def test_all_mode_does_not_duplicate_attribution_on_rerun(self, tmp_path, all_source):
        opts = _all_options(tmp_path, workers=1)

        fi.run_fetch(opts, session=RecordingSession(), sleep=lambda s: None)
        second = RecordingSession()
        fi.run_fetch(opts, session=second, sleep=lambda s: None)

        ids = list(attr.read_attribution(tmp_path / "attribution.csv"))
        assert sorted(ids) == ["v1", "v2"]
        assert second.calls == []  # 手元にあり登録済みの画像は再取得しない

    def test_failed_images_are_not_recorded_in_attribution(self, tmp_path, all_source):
        session = RecordingSession(fail={"v1": 403})

        report = fi.run_fetch(_all_options(tmp_path, workers=1), session=session, sleep=lambda s: None)

        assert report.summary.failed == 1
        assert set(attr.read_attribution(tmp_path / "attribution.csv")) == {"v2"}

    def test_attribution_is_saved_even_when_interrupted(self, tmp_path, all_source, monkeypatch):
        calls = {"n": 0}

        class Interrupting(RecordingSession):
            def get(self, url, stream=True, timeout=None, **kwargs):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise KeyboardInterrupt
                return super().get(url, stream, timeout)

        with pytest.raises(KeyboardInterrupt):
            fi.run_fetch(_all_options(tmp_path, workers=1),
                         session=Interrupting(), sleep=lambda s: None)

        assert set(attr.read_attribution(tmp_path / "attribution.csv")) == {"v1"}

    def test_registered_ids_are_not_reread_for_every_download(self, tmp_path, all_source, monkeypatch):
        """クローン直後（attribution は登録済み・画像は無い）でも、attribution CSV を繰り返し読み直さない。"""
        ids = _many_validation(all_source, 120, prefix="id")
        path = _write_attribution(tmp_path / "attribution.csv", ids)
        before = path.read_bytes()
        reads = {"n": 0}
        original = attr._rows

        def counting_rows(p):
            reads["n"] += 1
            return original(p)

        monkeypatch.setattr(attr, "_rows", counting_rows)

        report = fi.run_fetch(
            _all_options(tmp_path, max_images=1000, workers=2),
            session=RecordingSession(), sleep=lambda s: None)

        assert report.summary.downloaded == 120
        assert path.read_bytes() == before  # 登録済みなので追記も重複もしない
        assert reads["n"] <= 3  # 計画時の読み込み + 登録済み ID の取得（50 件ごとの追記では読み直さない）

    def test_missing_images_are_reported_as_not_ok(self, tmp_path):
        session = RecordingSession(serve=set())

        report = fi.run_fetch(
            _options(tmp_path, source="manifest", image_ids=("a1",), workers=1),
            session=session, sleep=lambda s: None)

        assert report.summary.not_found == 1
        assert not report.summary.ok


class TestCli:

    def test_parse_maps_arguments_to_options(self, tmp_path):
        options = fi.parse_args([
            "--source", "all", "--split", "train", "validation", "--max-images", "10",
            "--shard", "1", "--shard-count", "3", "--exclude-children",
            "--refresh-cache", "--dry-run", "--workers", "3", "--output", str(tmp_path / "img"),
            "--min-face-area", "0.05", "--max-faces", "2", "--exp", "32",
        ])

        assert options.source == "all"
        assert options.splits == ("train", "validation")
        assert (options.max_images, options.shard, options.shard_count) == (10, 1, 3)
        assert options.refresh_cache and options.dry_run
        assert options.workers == 3
        assert options.exp == "32"
        assert options.filter_config.exclude_children is True
        assert options.filter_config.min_face_area == 0.05
        assert options.filter_config.max_faces == 2
        assert options.output_dir == tmp_path / "img"

    def test_defaults_are_conservative(self):
        options = fi.parse_args([])

        assert options.source == "manifest"
        assert options.splits == ("validation",)
        assert options.max_images is None
        assert options.dry_run is False
        assert options.filter_config == oif.FilterConfig()

    @pytest.mark.parametrize("argv", [
        ["--shard", "0"],
        ["--shard-count", "2"],
        ["--shard", "2", "--shard-count", "2"],
        ["--source", "ftp"],
        ["--source", "csv"],  # 旧 --source csv は manifest に統合した
        ["--csv", "x.csv"],  # 任意 CSV の指定は廃止
        ["--include-known"],  # attribution 登録済みの除外は手元のファイルの有無で判定する
        ["--workers", "0"],
        ["--max-images", "-1"],
    ])
    def test_invalid_arguments_exit_with_usage_error(self, argv, capsys):
        with pytest.raises(SystemExit) as exc:
            fi.parse_args(argv)

        assert exc.value.code == 2

    def test_main_returns_nonzero_when_some_images_are_missing(self, tmp_path, monkeypatch, capsys):
        _write_experiment(tmp_path, ["a1"])
        monkeypatch.setattr(fi, "_new_session", lambda workers: RecordingSession(serve=set()))
        monkeypatch.setattr(time, "sleep", lambda s: None)

        code = fi.main([
            "--source", "manifest", "--experiments-dir", str(tmp_path / "experiments"),
            "--output", str(tmp_path / "images"),
            "--attribution-csv", str(tmp_path / "attr.csv"), "--workers", "1"])

        assert code == 1

    def test_main_reports_a_missing_manifest_instead_of_a_traceback(self, tmp_path, capsys):
        code = fi.main(["--experiments-dir", str(tmp_path / "experiments"),
                        "--output", str(tmp_path / "images"),
                        "--attribution-csv", str(tmp_path / "attr.csv"), "--dry-run"])

        assert code == 2

    def test_main_reports_a_csv_parse_error_instead_of_a_traceback(self, tmp_path, capsys):
        huge = tmp_path / "experiments" / "EXP032_y" / "manifests" / "manifest_train.csv"
        huge.parent.mkdir(parents=True)
        huge.write_text(
            ",".join(MANIFEST_HEADER) + "\n"
            + "0,data/images/" + "a" * 300_000 + ".jpg,0,eyeliner_left,upper,0,1,2,3,4,0.5,,\n",
            encoding="utf-8")

        code = fi.main(["--experiments-dir", str(tmp_path / "experiments"),
                        "--output", str(tmp_path / "images"),
                        "--attribution-csv", str(tmp_path / "attr.csv"), "--dry-run"])

        assert code == 2

    def test_main_dry_run_returns_zero(self, tmp_path, capsys):
        _write_experiment(tmp_path, ["a1", "b2"])

        code = fi.main([
            "--experiments-dir", str(tmp_path / "experiments"), "--output", str(tmp_path / "images"),
            "--attribution-csv", str(tmp_path / "attr.csv"), "--dry-run"])

        out = capsys.readouterr().out
        assert code == 0
        assert "2" in out
        assert not (tmp_path / "images").exists()
