"""tools/annotation_support/t_pred_webui/t_pred_db.py の list_hints/get_hint/count_filtered テスト。"""
from __future__ import annotations

from pathlib import Path

import pytest

from tools.annotation_support.t_pred_webui.t_pred_db import (
    count_filtered,
    get_hint,
    init_db,
    list_hints,
    upsert,
)


@pytest.fixture()
def db_path(tmp_path) -> Path:
    path = tmp_path / "t_pred_hints.db"
    init_db(path)
    return path


def _upsert_row(
    db_path: Path,
    image_path: str,
    part_name: str = "eyeliner_left",
    *,
    repr_idx: int = 0,
    min_t_pred: float = 0.3,
    max_t_pred: float = 0.8,
    n_extreme: int = 0,
    n_points: int = 7,
) -> None:
    upsert(
        image_path=image_path,
        part_name=part_name,
        repr_idx=repr_idx,
        min_t_pred=min_t_pred,
        max_t_pred=max_t_pred,
        n_extreme=n_extreme,
        n_points=n_points,
        scan_model="model_a",
        db_path=db_path,
    )


# ---------------------------------------------------------------------------
# list_hints
# ---------------------------------------------------------------------------

class TestListHints:
    def test_db_not_exists_returns_empty(self, tmp_path):
        missing = tmp_path / "missing.db"
        assert list_hints(missing) == []

    def test_empty_db_returns_empty(self, db_path):
        assert list_hints(db_path) == []

    def test_basic_listing(self, db_path):
        _upsert_row(db_path, "a.jpg")
        _upsert_row(db_path, "b.jpg")
        rows = list_hints(db_path)
        assert len(rows) == 2

    def test_excludes_unscanned_rows(self, db_path):
        """scanned_at が NULL の行（=未スキャン）は除外される。"""
        _upsert_row(db_path, "a.jpg")
        # scanned_at が NULL の行を直接挿入
        import sqlite3
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "INSERT INTO t_pred_hints (image_path, part_name) VALUES (?, ?)",
                ("unscanned.jpg", "eyeliner_left"),
            )
        rows = list_hints(db_path)
        assert len(rows) == 1
        assert rows[0]["image_path"] == "a.jpg"

    def test_filter_min_t_max(self, db_path):
        """min_t_max 指定時は min_t_pred <= min_t_max の行のみ返す。"""
        _upsert_row(db_path, "low.jpg", min_t_pred=-0.5)
        _upsert_row(db_path, "high.jpg", min_t_pred=0.9)
        rows = list_hints(db_path, min_t_max=0.0)
        assert [r["image_path"] for r in rows] == ["low.jpg"]

    def test_filter_max_t_min(self, db_path):
        """max_t_min 指定時は max_t_pred >= max_t_min の行のみ返す。"""
        _upsert_row(db_path, "low.jpg", max_t_pred=0.5)
        _upsert_row(db_path, "high.jpg", max_t_pred=1.5)
        rows = list_hints(db_path, max_t_min=1.2)
        assert [r["image_path"] for r in rows] == ["high.jpg"]

    def test_filter_min_n_extreme(self, db_path):
        """min_n_extreme 指定時は n_extreme >= min_n_extreme の行のみ返す。"""
        _upsert_row(db_path, "none.jpg", n_extreme=0)
        _upsert_row(db_path, "some.jpg", n_extreme=3)
        rows = list_hints(db_path, min_n_extreme=1)
        assert [r["image_path"] for r in rows] == ["some.jpg"]

    def test_filters_combine_with_and(self, db_path):
        _upsert_row(db_path, "match.jpg", min_t_pred=-0.5, max_t_pred=1.5, n_extreme=2)
        _upsert_row(db_path, "no_match_low.jpg", min_t_pred=0.9, max_t_pred=1.5, n_extreme=2)
        rows = list_hints(db_path, min_t_max=0.0, max_t_min=1.2, min_n_extreme=1)
        assert [r["image_path"] for r in rows] == ["match.jpg"]

    def test_filters_none_means_ignored(self, db_path):
        """フィルタ引数が None の場合は無視される（全件対象）。"""
        _upsert_row(db_path, "a.jpg", min_t_pred=-5.0, max_t_pred=5.0, n_extreme=99)
        rows = list_hints(db_path, min_t_max=None, max_t_min=None, min_n_extreme=None)
        assert len(rows) == 1

    @pytest.mark.parametrize(
        "sort_by", ["min_t_pred", "max_t_pred", "n_extreme", "n_points", "scanned_at"]
    )
    def test_sort_by_whitelisted_columns(self, db_path, sort_by):
        _upsert_row(db_path, "a.jpg", min_t_pred=0.1, max_t_pred=0.2, n_extreme=1, n_points=5)
        _upsert_row(db_path, "b.jpg", min_t_pred=0.5, max_t_pred=0.9, n_extreme=2, n_points=7)
        rows = list_hints(db_path, sort_by=sort_by)
        assert len(rows) == 2  # 例外なく実行できる

    def test_sort_by_invalid_raises(self, db_path):
        _upsert_row(db_path, "a.jpg")
        with pytest.raises(ValueError):
            list_hints(db_path, sort_by="image_path; DROP TABLE t_pred_hints")

    def test_sort_descending_default(self, db_path):
        _upsert_row(db_path, "low.jpg", n_extreme=1)
        _upsert_row(db_path, "high.jpg", n_extreme=5)
        rows = list_hints(db_path, sort_by="n_extreme", descending=True)
        assert [r["image_path"] for r in rows] == ["high.jpg", "low.jpg"]

    def test_sort_ascending(self, db_path):
        _upsert_row(db_path, "low.jpg", n_extreme=1)
        _upsert_row(db_path, "high.jpg", n_extreme=5)
        rows = list_hints(db_path, sort_by="n_extreme", descending=False)
        assert [r["image_path"] for r in rows] == ["low.jpg", "high.jpg"]

    def test_limit(self, db_path):
        for i in range(5):
            _upsert_row(db_path, f"img{i}.jpg")
        rows = list_hints(db_path, limit=2)
        assert len(rows) == 2

    def test_offset(self, db_path):
        for i in range(5):
            _upsert_row(db_path, f"img{i}.jpg", n_extreme=i)
        rows_all = list_hints(db_path, sort_by="n_extreme", descending=False, limit=100)
        rows_offset = list_hints(db_path, sort_by="n_extreme", descending=False, limit=100, offset=2)
        assert rows_offset == rows_all[2:]

    def test_zero_limit_returns_empty(self, db_path):
        _upsert_row(db_path, "a.jpg")
        assert list_hints(db_path, limit=0) == []


# ---------------------------------------------------------------------------
# get_hint
# ---------------------------------------------------------------------------

class TestGetHint:
    def test_found(self, db_path):
        _upsert_row(db_path, "a.jpg", "eyeliner_left")
        row = get_hint(db_path, "a.jpg", "eyeliner_left")
        assert row is not None
        assert row["image_path"] == "a.jpg"
        assert row["part_name"] == "eyeliner_left"

    def test_not_found(self, db_path):
        assert get_hint(db_path, "nonexistent.jpg", "eyeliner_left") is None

    def test_db_not_exists(self, tmp_path):
        missing = tmp_path / "missing.db"
        assert get_hint(missing, "a.jpg", "eyeliner_left") is None

    def test_wrong_part_name_not_found(self, db_path):
        _upsert_row(db_path, "a.jpg", "eyeliner_left")
        assert get_hint(db_path, "a.jpg", "eyeliner_right") is None


# ---------------------------------------------------------------------------
# count_filtered
# ---------------------------------------------------------------------------

class TestCountFiltered:
    def test_db_not_exists_returns_zero(self, tmp_path):
        missing = tmp_path / "missing.db"
        assert count_filtered(missing) == 0

    def test_empty_db_returns_zero(self, db_path):
        assert count_filtered(db_path) == 0

    def test_counts_all_without_filter(self, db_path):
        _upsert_row(db_path, "a.jpg")
        _upsert_row(db_path, "b.jpg")
        assert count_filtered(db_path) == 2

    def test_matches_filtered_list_length(self, db_path):
        _upsert_row(db_path, "match.jpg", min_t_pred=-0.5, n_extreme=2)
        _upsert_row(db_path, "no_match.jpg", min_t_pred=0.9, n_extreme=2)
        n = count_filtered(db_path, min_t_max=0.0, min_n_extreme=1)
        rows = list_hints(db_path, min_t_max=0.0, min_n_extreme=1, limit=1000)
        assert n == len(rows) == 1

    def test_excludes_unscanned(self, db_path):
        _upsert_row(db_path, "a.jpg")
        import sqlite3
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "INSERT INTO t_pred_hints (image_path, part_name) VALUES (?, ?)",
                ("unscanned.jpg", "eyeliner_left"),
            )
        assert count_filtered(db_path) == 1

    def test_invalid_sort_by_not_applicable(self, db_path):
        """count_filtered はソート不要なので sort_by 引数を要求しない。"""
        _upsert_row(db_path, "a.jpg")
        assert count_filtered(db_path) == 1
