"""tools/data_management/manifest_source.py のテスト。

最新実験の manifest 検出・列検証・image_path の境界検証を確認する。
"""
from __future__ import annotations

import csv
import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tools.data_management import manifest_source as ms  # noqa: E402

FULL_HEADER = [
    "idx", "image_path", "face_index", "part_name", "sub_key", "point_index",
    "px", "py", "qx", "qy", "t", "nose_x", "nose_y",
]
LEGACY_HEADER = [c for c in FULL_HEADER if c not in ("face_index", "nose_x", "nose_y")]


def _row(idx: int, image_id: str = "aaaa1111", **overrides) -> dict:
    base = {
        "idx": idx,
        "image_path": f"data/images/{image_id}.jpg",
        "face_index": 0,
        "part_name": "eyeliner_left",
        "sub_key": "upper",
        "point_index": idx % 7,
        "px": 10.5 + idx, "py": 20.25, "qx": 12.5 + idx, "qy": 18.75,
        "t": 0.8,
        "nose_x": 30.0, "nose_y": 40.0,
    }
    base.update(overrides)
    return base


def _write_manifest(path: Path, rows: list[dict], header: list[str] = FULL_HEADER) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _make_exp(root: Path, name: str, train=None, test=None, header=FULL_HEADER) -> Path:
    exp_dir = root / name
    manifests = exp_dir / "manifests"
    manifests.mkdir(parents=True)
    if train is not None:
        _write_manifest(manifests / "manifest_train.csv", train, header)
    if test is not None:
        _write_manifest(manifests / "manifest_test.csv", test, header)
    return exp_dir


class TestParseImagePath:

    def test_accepts_canonical_path(self):
        assert ms.parse_image_path("data/images/0021520c13029a24.jpg") == "0021520c13029a24"

    def test_normalizes_backslashes(self):
        assert ms.parse_image_path("data\\images\\abc_-1.jpg") == "abc_-1"

    @pytest.mark.parametrize("bad", [
        "../data/images/a.jpg",
        "data/images/../secret.jpg",
        "data/images/sub/a.jpg",
        "/etc/passwd",
        "C:/data/images/a.jpg",
        "data/images/a.png",
        "data/images/.jpg",
        "data/images/a b.jpg",
        "data/images/a.jpg.exe",
        "images/a.jpg",
        "",
    ])
    def test_rejects_paths_outside_the_images_dir(self, bad):
        with pytest.raises(ms.ManifestError):
            ms.parse_image_path(bad)


class TestValidateImageId:

    @pytest.mark.parametrize("ok", ["0021520c13029a24", "A_b-9"])
    def test_accepts_alnum_underscore_hyphen(self, ok):
        assert ms.validate_image_id(ok) == ok

    @pytest.mark.parametrize("bad", ["", "a/b", "a.b", "a b", "..", "a\n", "日本語"])
    def test_rejects_other_characters(self, bad):
        with pytest.raises(ms.ManifestError):
            ms.validate_image_id(bad)


class TestIsValidImageId:

    @pytest.mark.parametrize("value, expected", [
        ("0021520c13029a24", True), ("A_b-9", True), ("", False), ("a/b", False), ("..", False),
        ("a\n", False),
    ])
    def test_predicate_matches_validate_image_id(self, value, expected):
        assert ms.is_valid_image_id(value) is expected


class TestFindLatestManifest:

    def test_picks_highest_experiment_number(self, tmp_path):
        _make_exp(tmp_path, "EXP008_a", [_row(0)], [_row(1)])
        _make_exp(tmp_path, "EXP032_b", [_row(0)], [_row(1)])
        _make_exp(tmp_path, "EXP020_c", [_row(0)], [_row(1)])

        source = ms.find_latest_manifest(tmp_path)

        assert source.exp_id == 32
        assert [p.name for p in source.paths] == ["manifest_train.csv", "manifest_test.csv"]

    def test_skips_experiments_without_manifests(self, tmp_path):
        _make_exp(tmp_path, "EXP010_a", [_row(0)], [_row(1)])
        (tmp_path / "EXP033_no_manifest" / "config").mkdir(parents=True)

        assert ms.find_latest_manifest(tmp_path).exp_id == 10

    def test_explicit_exp_accepts_int_and_padded_string(self, tmp_path):
        _make_exp(tmp_path, "EXP008_a", [_row(0)], [_row(1)])
        _make_exp(tmp_path, "EXP032_b", [_row(0)], [_row(1)])

        assert ms.find_latest_manifest(tmp_path, exp=8).exp_id == 8
        assert ms.find_latest_manifest(tmp_path, exp="008").exp_id == 8
        assert ms.find_latest_manifest(tmp_path, exp="EXP008").exp_id == 8

    def test_explicit_exp_not_found_lists_available(self, tmp_path):
        _make_exp(tmp_path, "EXP008_a", [_row(0)], [_row(1)])

        with pytest.raises(ms.ManifestError, match="EXP099"):
            ms.find_latest_manifest(tmp_path, exp=99)

    def test_no_manifest_anywhere_is_an_error(self, tmp_path):
        with pytest.raises(ms.ManifestError, match="manifest"):
            ms.find_latest_manifest(tmp_path)

    def test_uses_only_existing_split_files(self, tmp_path):
        _make_exp(tmp_path, "EXP040_only_train", [_row(0)], None)

        source = ms.find_latest_manifest(tmp_path)

        assert [p.name for p in source.paths] == ["manifest_train.csv"]

    def test_invalid_explicit_exp_value(self, tmp_path):
        with pytest.raises(ms.ManifestError):
            ms.find_latest_manifest(tmp_path, exp="abc")


class TestLoadManifestRows:

    def test_loads_and_types_rows_sorted_by_idx(self, tmp_path):
        train = _write_manifest(tmp_path / "train.csv", [_row(5), _row(3)])
        test = _write_manifest(tmp_path / "test.csv", [_row(4, image_id="bbbb2222")])

        rows = ms.load_manifest_rows([train, test])

        assert [r.idx for r in rows] == [3, 4, 5]
        first = rows[0]
        assert first.image_id == "aaaa1111"
        assert first.image_path == "data/images/aaaa1111.jpg"
        assert isinstance(first.face_index, int)
        assert first.px == pytest.approx(13.5)
        assert first.nose_x == 30.0

    def test_rows_are_immutable(self, tmp_path):
        path = _write_manifest(tmp_path / "m.csv", [_row(0)])
        (row,) = ms.load_manifest_rows([path])

        with pytest.raises(dataclasses.FrozenInstanceError):
            row.t = 0.1  # type: ignore[misc]

    def test_nose_columns_are_optional(self, tmp_path):
        header = [c for c in FULL_HEADER if c not in ("nose_x", "nose_y")]
        path = _write_manifest(tmp_path / "m.csv", [_row(0)], header)

        (row,) = ms.load_manifest_rows([path])

        assert row.nose_x is None and row.nose_y is None

    def test_blank_nose_values_become_none(self, tmp_path):
        path = _write_manifest(tmp_path / "m.csv", [_row(0, nose_x="", nose_y="")])

        (row,) = ms.load_manifest_rows([path])

        assert row.nose_x is None and row.nose_y is None

    @pytest.mark.parametrize("missing", ["nan", "inf", "NaN"])
    def test_non_finite_nose_values_become_none(self, tmp_path, missing):
        """同梱 manifest には鼻先が未検出で nose_x/nose_y が nan の行がある。"""
        path = _write_manifest(tmp_path / "m.csv", [_row(0, nose_x=missing, nose_y=missing)])

        (row,) = ms.load_manifest_rows([path])

        assert row.nose_x is None and row.nose_y is None

    def test_garbage_nose_value_is_rejected(self, tmp_path):
        path = _write_manifest(tmp_path / "m.csv", [_row(0, nose_x="abc")])

        with pytest.raises(ms.ManifestError, match=r"m\.csv:2"):
            ms.load_manifest_rows([path])

    def test_missing_face_index_column_is_rejected_with_hint(self, tmp_path):
        path = _write_manifest(tmp_path / "old.csv", [_row(0)], LEGACY_HEADER)

        with pytest.raises(ms.ManifestError, match="face_index"):
            ms.load_manifest_rows([path])

    def test_error_names_the_offending_file(self, tmp_path):
        path = _write_manifest(tmp_path / "old.csv", [_row(0)], LEGACY_HEADER)

        with pytest.raises(ms.ManifestError, match="old.csv"):
            ms.load_manifest_rows([path])

    def test_duplicate_idx_across_files_is_rejected(self, tmp_path):
        a = _write_manifest(tmp_path / "a.csv", [_row(1)])
        b = _write_manifest(tmp_path / "b.csv", [_row(1)])

        with pytest.raises(ms.ManifestError, match="idx"):
            ms.load_manifest_rows([a, b])

    def test_empty_manifest_is_rejected(self, tmp_path):
        path = _write_manifest(tmp_path / "m.csv", [])

        with pytest.raises(ms.ManifestError, match="空"):
            ms.load_manifest_rows([path])

    @pytest.mark.parametrize("override", [
        {"image_path": "../evil.jpg"},
        {"image_path": "data/images/a.png"},
        {"part_name": "eyeliner_middle"},
        {"sub_key": "side"},
        {"point_index": 7},
        {"point_index": -1},
        {"face_index": -1},
        {"px": "nan"},
        {"qy": "inf"},
        {"t": "abc"},
        {"idx": "x"},
        {"t": ""},
    ])
    def test_invalid_values_are_rejected_with_line_number(self, tmp_path, override):
        path = _write_manifest(tmp_path / "m.csv", [_row(0), {**_row(1), **override}])

        with pytest.raises(ms.ManifestError, match=r"m\.csv:3"):
            ms.load_manifest_rows([path])

    def test_short_row_is_a_manifest_error_not_a_crash(self, tmp_path):
        """列が足りない行（途中で切れた CSV 等）でも AttributeError にならず、行番号付きで報告する。"""
        path = tmp_path / "m.csv"
        path.write_text(",".join(FULL_HEADER) + "\n0,data/images/a.jpg,0\n", encoding="utf-8")

        with pytest.raises(ms.ManifestError, match=r"m\.csv:2"):
            ms.load_manifest_rows([path])

    def test_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(ms.ManifestError, match="見つかりません"):
            ms.load_manifest_rows([tmp_path / "nope.csv"])


class TestImageIds:

    def test_returns_sorted_unique_ids(self, tmp_path):
        path = _write_manifest(
            tmp_path / "m.csv",
            [_row(0, "bbb"), _row(1, "aaa"), _row(2, "bbb")],
        )
        rows = ms.load_manifest_rows([path])

        assert ms.image_ids(rows) == ["aaa", "bbb"]


class TestRealRepositoryManifests:
    """リポジトリに同梱の最新 manifest が検証を通ること（同梱データの回帰）。"""

    def test_latest_bundled_manifest_loads(self):
        source = ms.find_latest_manifest(ROOT / "experiments")

        rows = ms.load_manifest_rows(source.paths)

        assert source.exp_id >= 32
        assert len(rows) > 1000
        assert all(r.image_path == f"data/images/{r.image_id}.jpg" for r in rows)
