"""predictions CSV の列名の別名対応のテスト。

scripts/infer.py は真値・予測値を true_t_raw / pred_t_raw で出力するが、共有評価ツール
（compute_passrate / paired_ttest / generate_summary）は t_true / t_pred を読む。
どちらの形式の CSV でも同じ結果になること、どちらでもない CSV は原因の分かるエラーで
止まることを保証する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from _common import read_predictions_csv  # noqa: E402
from compute_passrate import _read_predictions_csv, compute_band_stats  # noqa: E402
from paired_ttest import paired_comparison  # noqa: E402

TRUES = [0.10, 0.40, 0.60, 0.80, 0.95, 1.10]
BASE_PREDS = [0.12, 0.39, 0.65, 0.78, 0.99, 1.08]
TGT_PREDS = [0.11, 0.40, 0.61, 0.79, 0.96, 1.10]


def _frame(true_col: str, pred_col: str, preds: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "idx": [1, 2, 3, 4, 5, 6],
        "image_path": [f"{i}.jpg" for i in range(6)],
        true_col: TRUES,
        pred_col: preds,
    })


def _write(path: Path, df: pd.DataFrame) -> Path:
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def db_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "px": [100.0] * 6, "py": [100.0] * 6,
            "qx": [110.0] * 6, "qy": [100.0] * 6,
            "eye_scale": [50.0] * 6,
        },
        index=pd.Index([1, 2, 3, 4, 5, 6], name="idx"),
    )


class TestReadPredictionsCsv:

    def test_canonical_columns_are_returned_unchanged(self, tmp_path):
        path = _write(tmp_path / "p.csv", _frame("t_true", "t_pred", BASE_PREDS))

        df = read_predictions_csv(path)

        assert list(df.columns) == ["idx", "image_path", "t_true", "t_pred"]
        assert df["t_pred"].tolist() == BASE_PREDS

    def test_infer_py_columns_are_renamed(self, tmp_path):
        path = _write(tmp_path / "p.csv", _frame("true_t_raw", "pred_t_raw", BASE_PREDS))

        df = read_predictions_csv(path)

        assert list(df.columns) == ["idx", "image_path", "t_true", "t_pred"]
        assert df["t_true"].tolist() == TRUES
        assert df["t_pred"].tolist() == BASE_PREDS

    def test_only_the_missing_column_is_renamed(self, tmp_path):
        path = _write(tmp_path / "p.csv", _frame("t_true", "pred_t_raw", BASE_PREDS))

        df = read_predictions_csv(path)

        assert list(df.columns) == ["idx", "image_path", "t_true", "t_pred"]

    def test_canonical_column_wins_over_the_alias(self, tmp_path):
        frame = _frame("t_true", "t_pred", BASE_PREDS)
        frame["pred_t_raw"] = TGT_PREDS
        path = _write(tmp_path / "p.csv", frame)

        df = read_predictions_csv(path)

        assert df["t_pred"].tolist() == BASE_PREDS
        assert "pred_t_raw" in df.columns
        assert list(df.columns).count("t_pred") == 1

    def test_unrelated_columns_are_kept(self, tmp_path):
        frame = _frame("true_t_raw", "pred_t_raw", BASE_PREDS)
        frame["pred_t_norm"] = 0.5
        path = _write(tmp_path / "p.csv", frame)

        df = read_predictions_csv(path)

        assert "pred_t_norm" in df.columns
        assert "image_path" in df.columns

    def test_missing_columns_raise_with_file_and_columns_in_message(self, tmp_path):
        frame = _frame("t_true", "t_pred", BASE_PREDS).rename(columns={"t_pred": "score"})
        path = _write(tmp_path / "broken.csv", frame)

        with pytest.raises(ValueError) as excinfo:
            read_predictions_csv(path)

        message = str(excinfo.value)
        assert "broken.csv" in message
        assert "t_pred" in message
        assert "score" in message  # 実際にあった列も出す

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_predictions_csv(tmp_path / "nope.csv")


class TestSharedToolsAcceptInferPyCsv:

    def test_paired_comparison_gives_same_result_for_both_column_layouts(self, tmp_path, db_df):
        canonical_base = _write(tmp_path / "cb.csv", _frame("t_true", "t_pred", BASE_PREDS))
        canonical_tgt = _write(tmp_path / "ct.csv", _frame("t_true", "t_pred", TGT_PREDS))
        legacy_base = _write(tmp_path / "lb.csv", _frame("true_t_raw", "pred_t_raw", BASE_PREDS))
        legacy_tgt = _write(tmp_path / "lt.csv", _frame("true_t_raw", "pred_t_raw", TGT_PREDS))

        expected = paired_comparison(canonical_base, canonical_tgt, db_df)
        actual = paired_comparison(legacy_base, legacy_tgt, db_df)

        assert "error" not in actual
        assert actual["overall"] == expected["overall"]
        assert actual["by_band"] == expected["by_band"]
        assert actual["n_common"] == expected["n_common"] == 6

    def test_paired_comparison_error_ratio_metric_accepts_infer_py_csv(self, tmp_path, db_df):
        base = _write(tmp_path / "b.csv", _frame("true_t_raw", "pred_t_raw", BASE_PREDS))
        tgt = _write(tmp_path / "t.csv", _frame("true_t_raw", "pred_t_raw", TGT_PREDS))

        result = paired_comparison(base, tgt, db_df, metric="error_ratio")

        assert "error" not in result
        assert result["n_common"] == 6

    def test_compute_passrate_reader_renames_columns(self, tmp_path, db_df):
        path = _write(tmp_path / "p.csv", _frame("true_t_raw", "pred_t_raw", BASE_PREDS))

        pred_df = _read_predictions_csv(path)

        assert pred_df is not None
        stats = compute_band_stats(pred_df, db_df)
        assert stats["n_valid"] == 6

    def test_compute_passrate_reader_reports_unusable_csv_instead_of_raising(
        self, tmp_path, capsys
    ):
        frame = _frame("t_true", "t_pred", BASE_PREDS).rename(columns={"t_pred": "score"})
        path = _write(tmp_path / "broken.csv", frame)

        assert _read_predictions_csv(path) is None
        assert "broken.csv" in capsys.readouterr().out
