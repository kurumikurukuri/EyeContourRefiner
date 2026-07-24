"""scripts/compute_passrate.py のユニットテスト。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from compute_passrate import (  # noqa: E402
    THRESHOLD_STD,
    THRESHOLD_STRICT,
    T_BANDS,
    compute_error_ratio,
    compute_passrate,
    compute_band_stats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_df():
    """最小の DB index DataFrame。"""
    return pd.DataFrame(
        {
            "px":  [100.0, 200.0, 300.0],
            "py":  [100.0, 200.0, 300.0],
            "qx":  [110.0, 190.0, 290.0],
            "qy":  [100.0, 200.0, 300.0],
            "eye_scale": [50.0, 100.0, 0.0],  # idx=3 は eye_scale=0 → 無効
        },
        index=pd.Index([1, 2, 3], name="idx"),
    )


@pytest.fixture
def pred_df():
    """predictions.csv 相当の DataFrame。"""
    return pd.DataFrame({
        "idx": [1, 2, 3, 999],
        "t_true": [0.5, 0.8, 0.3, 0.5],
        "t_pred": [0.5, 0.82, 0.32, 0.5],
    })


# ---------------------------------------------------------------------------
# T_BANDS
# ---------------------------------------------------------------------------

class TestBands:
    def test_band_count(self):
        assert len(T_BANDS) == 6

    def test_band_boundaries_exhaustive(self):
        """全 t 値がいずれかの band に属する（重複なし）。"""
        test_values = [-0.5, 0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0, 1.1, 1.2, 1.5]
        for t in test_values:
            matches = [name for name, fn in T_BANDS if fn(t)]
            assert len(matches) == 1, f"t={t} matched {matches}"

    def test_band_names(self):
        names = [b[0] for b in T_BANDS]
        assert names == ["t<0", "0-0.3", "0.3-0.7", "0.7-1.0", "1.0-1.2", "t>=1.2"]


# ---------------------------------------------------------------------------
# compute_error_ratio
# ---------------------------------------------------------------------------

class TestComputeErrorRatio:
    def test_zero_error(self, pred_df, db_df):
        """t_pred == t_true → error_ratio = 0"""
        df = pred_df.copy()
        df["t_pred"] = df["t_true"]
        er = compute_error_ratio(df, db_df)
        # idx=1: pq_dist=10, eye_scale=50 → ratio=0
        assert er.iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_exact_error(self, pred_df, db_df):
        """手計算で検証: idx=1, t_pred=0.5, t_true=0.5 → error=0"""
        er = compute_error_ratio(pred_df, db_df)
        # idx=1: pq_dist=10 (|110-100|), eye_scale=50, dt=0 → 0
        assert er.iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_positive_error(self, pred_df, db_df):
        """idx=2: dt=0.02, pq_dist=10, eye_scale=100 → ratio=0.002"""
        er = compute_error_ratio(pred_df, db_df)
        # idx=2: px=200, qx=190 → pq_dist=10, dt=0.02, eye_scale=100
        expected = 0.02 * 10 / 100  # = 0.002
        assert er.iloc[1] == pytest.approx(expected, rel=1e-9)

    def test_invalid_eye_scale_excluded(self, pred_df, db_df):
        """eye_scale=0 のサンプルは NaN"""
        er = compute_error_ratio(pred_df, db_df)
        assert np.isnan(er.iloc[2])  # idx=3

    def test_missing_from_db_excluded(self, pred_df, db_df):
        """DB に存在しない idx は NaN"""
        er = compute_error_ratio(pred_df, db_df)
        assert np.isnan(er.iloc[3])  # idx=999


# ---------------------------------------------------------------------------
# compute_passrate
# ---------------------------------------------------------------------------

class TestComputePassrate:
    def test_all_pass(self):
        er = pd.Series([0.0, 0.005, 0.010])
        assert compute_passrate(er, 0.02) == 100.0

    def test_none_pass(self):
        er = pd.Series([0.05, 0.06, 0.07])
        assert compute_passrate(er, 0.02) == 0.0

    def test_partial_pass(self):
        er = pd.Series([0.01, 0.02, 0.03])
        assert compute_passrate(er, 0.015) == pytest.approx(100 / 3, rel=1e-6)

    def test_nan_handling(self):
        er = pd.Series([0.01, np.nan, 0.03, np.nan])
        assert compute_passrate(er, 0.02) == pytest.approx(50.0, rel=1e-6)

    def test_all_nan(self):
        er = pd.Series([np.nan, np.nan])
        assert np.isnan(compute_passrate(er, 0.02))


# ---------------------------------------------------------------------------
# compute_band_stats
# ---------------------------------------------------------------------------

class TestComputeBandStats:
    def test_basic_structure(self, pred_df, db_df):
        stats = compute_band_stats(pred_df, db_df)
        assert "n_total" in stats
        assert "n_valid" in stats
        assert "n_missing" in stats
        assert "pass_std" in stats
        assert "pass_strict" in stats
        assert "mae_raw_t" in stats
        assert "bands" in stats
        assert len(stats["bands"]) == 6

    def test_n_total(self, pred_df, db_df):
        stats = compute_band_stats(pred_df, db_df)
        assert stats["n_total"] == 4

    def test_n_missing(self, pred_df, db_df):
        stats = compute_band_stats(pred_df, db_df)
        # idx=999 is missing from DB
        assert stats["n_missing"] == 1

    def test_n_valid(self, pred_df, db_df):
        stats = compute_band_stats(pred_df, db_df)
        # idx=1,2 valid (idx=3 eye_scale=0, idx=999 missing)
        assert stats["n_valid"] == 2

    def test_mae_raw_t(self, pred_df, db_df):
        """idx=1 dt=0, idx=2 dt=0.02 → mae=0.01"""
        stats = compute_band_stats(pred_df, db_df)
        assert stats["mae_raw_t"] == pytest.approx(0.01, rel=1e-9)

    def test_band_coverage(self, pred_df, db_df):
        stats = compute_band_stats(pred_df, db_df)
        # idx=1: t=0.5 → "0.3-0.7"
        # idx=2: t=0.8 → "0.7-1.0"
        assert stats["bands"]["0.3-0.7"]["n"] == 1
        assert stats["bands"]["0.7-1.0"]["n"] == 1
        assert stats["bands"]["t<0"]["n"] == 0

    def test_zero_valid_samples(self, db_df):
        """全サンプルが無効な場合。"""
        df = pd.DataFrame({
            "idx": [999],
            "t_true": [0.5],
            "t_pred": [0.5],
        })
        stats = compute_band_stats(df, db_df)
        assert stats["n_valid"] == 0
        assert np.isnan(stats["pass_std"])

    def test_thresholds_are_canonical(self):
        assert THRESHOLD_STD == 0.01788
        assert THRESHOLD_STRICT == 0.01604


# ---------------------------------------------------------------------------
# Integration: passrate matches expected values for known case
# ---------------------------------------------------------------------------

class TestPassrateKnownValues:
    def test_perfect_prediction(self, db_df):
        """完全予測 → passrate 100%"""
        df = pd.DataFrame({
            "idx": [1, 2],
            "t_true": [0.5, 0.8],
            "t_pred": [0.5, 0.8],
        })
        stats = compute_band_stats(df, db_df)
        assert stats["pass_std"] == 100.0
        assert stats["pass_strict"] == 100.0
        assert stats["mae_raw_t"] == 0.0

    def test_large_errors(self, db_df):
        """大誤差 → passrate 0%"""
        df = pd.DataFrame({
            "idx": [1, 2],
            "t_true": [0.5, 0.8],
            "t_pred": [0.0, 0.0],
        })
        er = compute_error_ratio(df, db_df)
        # どちらも error_ratio >> threshold → passrate = 0%
        assert compute_passrate(er, THRESHOLD_STD) == 0.0
