"""scripts/paired_ttest.py のユニットテスト。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from _common import load_db_index  # noqa: E402
from paired_ttest import (  # noqa: E402
    sig,
    paired_comparison,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_df():
    return pd.DataFrame(
        {
            "px":  [100.0, 200.0, 300.0],
            "py":  [100.0, 200.0, 300.0],
            "qx":  [110.0, 190.0, 290.0],
            "qy":  [100.0, 200.0, 300.0],
            "eye_scale": [50.0, 100.0, 80.0],
        },
        index=pd.Index([1, 2, 3], name="idx"),
    )


@pytest.fixture
def temp_csv_dir(tmp_path, db_df):
    """一時ディレクトリに baseline と target の predictions.csv を作成。"""
    base_df = pd.DataFrame({
        "idx": [1, 2, 3],
        "image_path": ["a.jpg", "b.jpg", "c.jpg"],
        "part_name": ["eyeliner_right"] * 3,
        "sub_key": ["upper"] * 3,
        "point_index": [0, 1, 2],
        "t_true": [0.5, 0.8, 0.3],
        "t_pred": [0.5, 0.81, 0.3],  # base: error [0, 0.01, 0]
    })
    tgt_df = pd.DataFrame({
        "idx": [1, 2, 3],
        "image_path": ["a.jpg", "b.jpg", "c.jpg"],
        "part_name": ["eyeliner_right"] * 3,
        "sub_key": ["upper"] * 3,
        "point_index": [0, 1, 2],
        "t_true": [0.5, 0.8, 0.3],
        "t_pred": [0.5, 0.80, 0.3],  # target: error [0, 0, 0] → better
    })

    base_dir = tmp_path / "baseline"
    tgt_dir = tmp_path / "target"
    base_dir.mkdir()
    tgt_dir.mkdir()

    base_csv = base_dir / "predictions.csv"
    tgt_csv = tgt_dir / "predictions.csv"
    base_df.to_csv(base_csv, index=False)
    tgt_df.to_csv(tgt_csv, index=False)

    return base_csv, tgt_csv


# ---------------------------------------------------------------------------
# sig
# ---------------------------------------------------------------------------

class TestSig:
    def test_highly_significant(self):
        assert sig(0.0001) == "***"

    def test_moderately_significant(self):
        assert sig(0.005) == "**"

    def test_significant(self):
        assert sig(0.03) == "*"

    def test_not_significant(self):
        assert sig(0.10) == "n.s."

    def test_boundary_p001(self):
        assert sig(0.001) == "**"  # < 0.001 is ***, so 0.001 is **

    def test_boundary_p01(self):
        assert sig(0.01) == "*"  # < 0.01 is **, so 0.01 is *

    def test_boundary_p05(self):
        assert sig(0.05) == "n.s."  # < 0.05 is *, so 0.05 is n.s.


# ---------------------------------------------------------------------------
# paired_comparison
# ---------------------------------------------------------------------------

class TestPairedComparison:
    def test_identical_returns_zero_diff(self, temp_csv_dir, db_df):
        """同じファイル同士の比較 → MAE 差ゼロ、t-test は degenerate（p=1.0）"""
        base_csv, _ = temp_csv_dir
        result = paired_comparison(base_csv, base_csv, db_df, metric="raw_t")
        assert result["n_common"] == 3
        overall = result["overall"]
        assert overall["mae_base"] == pytest.approx(overall["mae_target"], rel=1e-6)
        assert overall["improvement_pct"] == pytest.approx(0.0, abs=1e-6)
        # 同一配列の t-test は degenerate: NaN ではなく p=1.0 を返すこと
        assert not np.isnan(overall["t_stat"])
        assert not np.isnan(overall["ttest_p"])
        assert overall["ttest_p"] == pytest.approx(1.0)

    def test_better_target(self, temp_csv_dir, db_df):
        """target が baseline より良い → improvement_pct > 0"""
        base_csv, tgt_csv = temp_csv_dir
        result = paired_comparison(base_csv, tgt_csv, db_df, metric="raw_t")
        overall = result["overall"]
        # base MAE > target MAE → improvement > 0
        assert overall["improvement_pct"] > 0

    def test_structure(self, temp_csv_dir, db_df):
        """結果 dict の構造確認"""
        base_csv, tgt_csv = temp_csv_dir
        result = paired_comparison(base_csv, tgt_csv, db_df, metric="raw_t")
        assert "baseline" in result
        assert "target" in result
        assert "metric" in result
        assert "n_common" in result
        assert "overall" in result
        assert "by_band" in result
        assert len(result["by_band"]) == 6

        overall = result["overall"]
        for key in ("mae_base", "mae_target", "improvement_pct", "t_stat", "ttest_p", "sig"):
            assert key in overall, f"missing key: {key}"

    def test_error_ratio_metric(self, temp_csv_dir, db_df):
        """error_ratio メトリックで比較"""
        base_csv, tgt_csv = temp_csv_dir
        result = paired_comparison(base_csv, tgt_csv, db_df, metric="error_ratio")
        assert result["metric"] == "error_ratio"
        assert result["n_common"] > 0
        overall = result["overall"]
        # passrate fields should be present for error_ratio
        assert overall["pass_std_base"] is not None
        assert overall["pass_std_target"] is not None

    def test_too_few_samples(self, db_df, tmp_path):
        """サンプル不足 → error キーが返る"""
        df = pd.DataFrame({
            "idx": [1],
            "t_true": [0.5],
            "t_pred": [0.5],
        })
        csv = tmp_path / "tiny.csv"
        df.to_csv(csv, index=False)
        result = paired_comparison(csv, csv, db_df, metric="raw_t")
        assert "error" in result

    def test_raw_t_excludes_nan_rows(self, db_df, tmp_path):
        """raw_t で NaN 行があっても全体が NaN → p=1.0 に化けず、NaN 行を除外して検定する。"""
        n = 20
        t_true = np.linspace(0.1, 0.9, n)
        base_pred = t_true + 0.05 + np.linspace(0.0, 0.01, n)
        tgt_pred = t_true + 0.01
        base_pred[0] = np.nan
        base_csv = tmp_path / "base.csv"
        tgt_csv = tmp_path / "tgt.csv"
        pd.DataFrame({"idx": range(n), "t_true": t_true, "t_pred": base_pred}).to_csv(base_csv, index=False)
        pd.DataFrame({"idx": range(n), "t_true": t_true, "t_pred": tgt_pred}).to_csv(tgt_csv, index=False)

        result = paired_comparison(base_csv, tgt_csv, db_df, metric="raw_t")

        assert result["n_common"] == n - 1
        assert not np.isnan(result["overall"]["mae_base"])
        assert result["overall"]["ttest_p"] < 0.001

    def test_non_overlapping_idx(self, db_df, tmp_path):
        """共通 idx なし → error"""
        base_df = pd.DataFrame({
            "idx": [1, 2],
            "t_true": [0.5, 0.8],
            "t_pred": [0.5, 0.8],
        })
        tgt_df = pd.DataFrame({
            "idx": [3, 4],
            "t_true": [0.3, 0.6],
            "t_pred": [0.3, 0.6],
        })
        base_csv = tmp_path / "base.csv"
        tgt_csv = tmp_path / "tgt.csv"
        base_df.to_csv(base_csv, index=False)
        tgt_df.to_csv(tgt_csv, index=False)
        result = paired_comparison(base_csv, tgt_csv, db_df, metric="raw_t")
        assert "error" in result

    def test_band_stats_structure(self, temp_csv_dir, db_df):
        base_csv, tgt_csv = temp_csv_dir
        result = paired_comparison(base_csv, tgt_csv, db_df, metric="raw_t")
        for bname in ["t<0", "0-0.3", "0.3-0.7", "0.7-1.0", "1.0-1.2", "t>=1.2"]:
            assert bname in result["by_band"]
            b = result["by_band"][bname]
            assert "n" in b
            assert "sig" in b


# ---------------------------------------------------------------------------
# DB loading
# ---------------------------------------------------------------------------

class TestLoadDbIndex:
    def test_returns_dataframe(self):
        """DB_PATH が存在する場合 DataFrame を返す。"""
        db_path = ROOT / "data" / "samples.db"
        if not db_path.exists():
            pytest.skip("samples.db が存在しません")
        df = load_db_index(db_path)
        assert isinstance(df, pd.DataFrame)
        assert "px" in df.columns
        assert "eye_scale" in df.columns
        assert df.index.name == "idx"
