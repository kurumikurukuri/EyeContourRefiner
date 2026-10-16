"""eye_contour.training.oof の単体テスト。"""
from __future__ import annotations

import csv
from pathlib import Path

from eye_contour.training.oof import OOF_CSV_FIELDNAMES, OofPrediction, write_oof_csv


def _make_prediction(**overrides) -> OofPrediction:
    base = dict(
        idx="1",
        image_path="data/images/a.jpg",
        part_name="eyeliner_left",
        sub_key="upper",
        point_index="3",
        t_true=0.5,
        t_pred=0.52,
        fold_idx=1,
    )
    base.update(overrides)
    return OofPrediction(**base)


class TestWriteOofCsv:

    def test_writes_header_with_expected_column_order(self, tmp_path: Path):
        out = tmp_path / "oof.csv"
        write_oof_csv([_make_prediction()], out)
        with open(out, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        assert header == OOF_CSV_FIELDNAMES

    def test_round_trips_values(self, tmp_path: Path):
        out = tmp_path / "oof.csv"
        pred = _make_prediction(idx="42", t_true=0.123456, t_pred=0.654321, fold_idx=3)
        write_oof_csv([pred], out)
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        row = rows[0]
        assert row["idx"] == "42"
        assert float(row["t_true"]) == pred.t_true
        assert float(row["t_pred"]) == pred.t_pred
        assert int(row["fold_idx"]) == 3

    def test_writes_multiple_rows_in_order(self, tmp_path: Path):
        out = tmp_path / "oof.csv"
        preds = [_make_prediction(idx=str(i), fold_idx=(i % 5) + 1) for i in range(10)]
        write_oof_csv(preds, out)
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert [r["idx"] for r in rows] == [str(i) for i in range(10)]

    def test_creates_parent_directory(self, tmp_path: Path):
        out = tmp_path / "nested" / "dir" / "oof.csv"
        write_oof_csv([_make_prediction()], out)
        assert out.exists()

    def test_empty_predictions_writes_header_only(self, tmp_path: Path):
        out = tmp_path / "oof.csv"
        write_oof_csv([], out)
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows == [OOF_CSV_FIELDNAMES]
