"""tools/data_management/oi_filter.py のテスト。

アノテーション CSV による画像 ID の事前フィルタと、条件ハッシュ付きキャッシュ。
"""
from __future__ import annotations

import csv
import dataclasses
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tools.data_management import oi_filter as oif  # noqa: E402

FACE = "/m/face"
PERSON = "/m/person"
DOLL = "/m/doll"
TOY = "/m/toy"
BOY = "/m/boy"
GIRL = "/m/girl"

CLASSES = {
    FACE: "Human face", PERSON: "Person", DOLL: "Doll", TOY: "Toy", BOY: "Boy", GIRL: "Girl",
    "/m/teddy": "Teddy bear", "/m/poster": "Poster", "/m/sculpt": "Sculpture",
    "/m/crib": "Infant bed",
}
HEADER = [
    "ImageID", "Source", "LabelName", "Confidence", "XMin", "XMax", "YMin", "YMax",
    "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside",
]


def _box(image_id, label=FACE, w=0.3, h=0.3, x=0.1, y=0.1, **flags) -> list:
    row = {
        "ImageID": image_id, "Source": "xclick", "LabelName": label, "Confidence": "1",
        "XMin": x, "XMax": x + w, "YMin": y, "YMax": y + h,
        "IsOccluded": 0, "IsTruncated": 0, "IsGroupOf": 0, "IsDepiction": 0, "IsInside": 0,
    }
    row.update(flags)
    return [row[c] for c in HEADER]


def _write_inputs(tmp_path: Path, boxes: list[list], classes=CLASSES) -> tuple[Path, Path]:
    classes_csv = tmp_path / "classes.csv"
    with open(classes_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows((mid, name) for mid, name in classes.items())
    ann_csv = tmp_path / "ann.csv"
    with open(ann_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(boxes)
    return classes_csv, ann_csv


def _run(tmp_path, boxes, candidates=None, config=None, classes=CLASSES) -> set[str]:
    classes_csv, ann_csv = _write_inputs(tmp_path, boxes, classes)
    if candidates is None:
        candidates = {row[0] for row in boxes}
    return oif.filter_image_ids(candidates, ann_csv, classes_csv, config or oif.FilterConfig())


class TestFilterConfig:

    def test_defaults_match_the_documented_conditions(self):
        config = oif.FilterConfig()

        assert config.required_classes == ("Human face",)
        assert config.min_face_area == 0.01
        assert (config.min_face_aspect, config.max_face_aspect) == (0.2, 2.0)
        assert (config.min_faces, config.max_faces) == (1, 4)
        assert config.exclude_children is False
        assert set(config.excluded_classes) == {
            "Doll", "Toy", "Teddy bear", "Poster", "Sculpture", "Infant bed",
        }
        assert set(config.allowed_licenses) == {
            "https://creativecommons.org/publicdomain/zero/1.0",
            "https://creativecommons.org/licenses/by/2.0",
        }

    def test_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            oif.FilterConfig().max_faces = 9  # type: ignore[misc]

    def test_children_are_excluded_only_on_request(self):
        assert "Boy" not in oif.FilterConfig().effective_excluded_classes()
        assert {"Boy", "Girl"} <= oif.FilterConfig(exclude_children=True).effective_excluded_classes()

    def test_cache_key_is_stable_and_condition_sensitive(self):
        base = oif.FilterConfig()

        assert base.cache_key() == oif.FilterConfig().cache_key()
        assert len(base.cache_key()) == 12
        assert dataclasses.replace(base, max_faces=5).cache_key() != base.cache_key()
        assert dataclasses.replace(base, exclude_children=True).cache_key() != base.cache_key()
        assert dataclasses.replace(base, min_face_area=0.05).cache_key() != base.cache_key()

    @pytest.mark.parametrize("kwargs", [
        {"min_face_area": -0.1},
        {"min_face_area": 1.5},
        {"min_face_aspect": 3.0, "max_face_aspect": 2.0},
        {"min_face_aspect": 0.0},
        {"min_faces": 0},
        {"min_faces": 5, "max_faces": 4},
        {"required_classes": ()},
    ])
    def test_invalid_config_is_rejected(self, kwargs):
        with pytest.raises(ValueError):
            oif.FilterConfig(**kwargs)


class TestFilterImageIds:

    def test_single_valid_face_passes(self, tmp_path):
        assert _run(tmp_path, [_box("a1")]) == {"a1"}

    def test_requires_the_face_label(self, tmp_path):
        boxes = [_box("a1", label=PERSON), _box("b2")]

        assert _run(tmp_path, boxes) == {"b2"}

    def test_person_boxes_are_not_counted_as_faces(self, tmp_path):
        boxes = [_box("a1")] + [_box("a1", label=PERSON) for _ in range(6)]

        assert _run(tmp_path, boxes) == {"a1"}

    @pytest.mark.parametrize("label", [DOLL, TOY, "/m/teddy", "/m/poster", "/m/sculpt", "/m/crib"])
    def test_excluded_object_classes_drop_the_image(self, tmp_path, label):
        boxes = [_box("a1"), _box("a1", label=label), _box("b2")]

        assert _run(tmp_path, boxes) == {"b2"}

    def test_children_kept_by_default_and_dropped_on_request(self, tmp_path):
        boxes = [_box("a1"), _box("a1", label=BOY), _box("b2"), _box("b2", label=GIRL), _box("c3")]

        assert _run(tmp_path, boxes) == {"a1", "b2", "c3"}
        assert _run(tmp_path, boxes, config=oif.FilterConfig(exclude_children=True)) == {"c3"}

    def test_face_area_threshold(self, tmp_path):
        boxes = [_box("small", w=0.09, h=0.09), _box("edge", w=0.1, h=0.1), _box("big", w=0.5, h=0.5)]

        assert _run(tmp_path, boxes) == {"edge", "big"}

    def test_face_aspect_range(self, tmp_path):
        boxes = [
            _box("tall", w=0.1, h=0.6),    # 0.166 < 0.2
            _box("wide", w=0.7, h=0.3),    # 2.33 > 2.0
            _box("ok_tall", w=0.2, h=0.9), # 0.222
            _box("ok_wide", w=0.6, h=0.31),
        ]

        assert _run(tmp_path, boxes) == {"ok_tall", "ok_wide"}

    @pytest.mark.parametrize("flag", ["IsDepiction", "IsGroupOf", "IsInside"])
    def test_attribute_flags_invalidate_a_face(self, tmp_path, flag):
        boxes = [_box("a1", **{flag: 1}), _box("b2")]

        assert _run(tmp_path, boxes) == {"b2"}

    def test_occluded_and_truncated_flags_are_ignored(self, tmp_path):
        assert _run(tmp_path, [_box("a1", IsOccluded=1, IsTruncated=1)]) == {"a1"}

    def test_face_count_limit_counts_only_valid_faces(self, tmp_path):
        five_valid = [_box("many", x=0.1 * i, w=0.15, h=0.3) for i in range(5)]
        four_valid = [_box("four", x=0.1 * i, w=0.15, h=0.3) for i in range(4)]
        three_valid_two_tiny = (
            [_box("mixed", x=0.1 * i, w=0.15, h=0.3) for i in range(3)]
            + [_box("mixed", w=0.02, h=0.02), _box("mixed", w=0.02, h=0.02)]
        )

        assert _run(tmp_path, five_valid + four_valid + three_valid_two_tiny) == {"four", "mixed"}

    def test_images_without_any_valid_face_are_dropped(self, tmp_path):
        boxes = [_box("a1", w=0.02, h=0.02), _box("b2", IsDepiction=1)]

        assert _run(tmp_path, boxes) == set()

    def test_only_candidate_ids_are_considered(self, tmp_path):
        boxes = [_box("a1"), _box("b2")]

        assert _run(tmp_path, boxes, candidates={"b2", "not_in_csv"}) == {"b2"}

    def test_custom_limits_are_honoured(self, tmp_path):
        boxes = [_box("a1", x=0.0), _box("a1", x=0.4), _box("b2")]
        config = oif.FilterConfig(min_faces=2, max_faces=2)

        assert _run(tmp_path, boxes, config=config) == {"a1"}

    def test_malformed_rows_are_skipped_not_fatal(self, tmp_path):
        bad = _box("a1")
        bad[HEADER.index("XMin")] = "abc"
        short = ["b2", "xclick", FACE]

        assert _run(tmp_path, [bad, short, _box("c3")], candidates={"a1", "b2", "c3"}) == {"c3"}

    def test_zero_height_box_is_invalid_not_a_crash(self, tmp_path):
        assert _run(tmp_path, [_box("a1", h=0.0), _box("b2")]) == {"b2"}

    def test_unknown_required_class_is_an_error(self, tmp_path):
        classes = {k: v for k, v in CLASSES.items() if v != "Human face"}

        with pytest.raises(ValueError, match="Human face"):
            _run(tmp_path, [_box("a1")], classes=classes)

    def test_unknown_excluded_class_only_warns(self, tmp_path, caplog):
        classes = {k: v for k, v in CLASSES.items() if v != "Infant bed"}

        with caplog.at_level(logging.WARNING):
            result = _run(tmp_path, [_box("a1")], classes=classes)

        assert result == {"a1"}
        assert "Infant bed" in caplog.text

    def test_missing_columns_are_reported(self, tmp_path):
        classes_csv, _ = _write_inputs(tmp_path, [])
        ann = tmp_path / "broken.csv"
        ann.write_text("ImageID,LabelName\na1,/m/face\n", encoding="utf-8")

        with pytest.raises(ValueError, match="XMin"):
            oif.filter_image_ids({"a1"}, ann, classes_csv, oif.FilterConfig())


class TestFilteredIdsCache:

    def test_path_embeds_split_and_condition_hash(self, tmp_path):
        config = oif.FilterConfig()

        path = oif.filtered_ids_cache_path("train", tmp_path, config)

        assert path == tmp_path / f"train_filtered_{config.cache_key()}.txt"
        other = oif.filtered_ids_cache_path("train", tmp_path, dataclasses.replace(config, max_faces=3))
        assert other != path

    def test_ids_round_trip_sorted(self, tmp_path):
        path = tmp_path / "ids.txt"

        oif.save_ids({"b", "a", "c"}, path)

        assert path.read_text(encoding="utf-8") == "a\nb\nc\n"
        assert oif.load_ids(path) == {"a", "b", "c"}

    def test_computes_once_then_reuses_cache(self, tmp_path):
        calls = []

        def compute():
            calls.append(1)
            return {"a", "b"}

        config = oif.FilterConfig()
        first = oif.load_or_compute_filtered_ids("val", tmp_path, config, compute=compute)
        second = oif.load_or_compute_filtered_ids("val", tmp_path, config, compute=compute)

        assert first == second == {"a", "b"}
        assert len(calls) == 1

    def test_refresh_recomputes(self, tmp_path):
        results = iter([{"a"}, {"z"}])
        config = oif.FilterConfig()
        oif.load_or_compute_filtered_ids("val", tmp_path, config, compute=lambda: next(results))

        again = oif.load_or_compute_filtered_ids(
            "val", tmp_path, config, compute=lambda: next(results), refresh=True)

        assert again == {"z"}

    def test_different_conditions_do_not_share_cache(self, tmp_path):
        base = oif.FilterConfig()
        oif.load_or_compute_filtered_ids("val", tmp_path, base, compute=lambda: {"a"})

        other = oif.load_or_compute_filtered_ids(
            "val", tmp_path, dataclasses.replace(base, exclude_children=True), compute=lambda: {"b"})

        assert other == {"b"}

    def test_empty_result_is_cached_too(self, tmp_path):
        calls = []
        config = oif.FilterConfig()

        def compute():
            calls.append(1)
            return set()

        oif.load_or_compute_filtered_ids("val", tmp_path, config, compute=compute)
        result = oif.load_or_compute_filtered_ids("val", tmp_path, config, compute=compute)

        assert result == set()
        assert len(calls) == 1
