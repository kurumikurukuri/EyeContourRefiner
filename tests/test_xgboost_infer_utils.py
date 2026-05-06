"""xgboost_baseline: infer.py の純粋関数のテスト。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analyses.xgboost_baseline.scr.infer import (
    OutlierConfig,
    _rank_normalize,
    compute_outlier_score,
    prepare_features,
)


def _cfg(**overrides) -> OutlierConfig:
    base = dict(
        top_k=10,
        weight_uncertainty=0.4,
        weight_range_violation=0.3,
        weight_iforest=0.3,
        t_valid_min=0.0,
        t_valid_max=1.5,
    )
    base.update(overrides)
    return OutlierConfig(**base)


class TestRankNormalize:
    def test_sorted_maps_to_0_1(self):
        r = _rank_normalize(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        np.testing.assert_allclose(r, [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_ties_get_average_rank(self):
        r = _rank_normalize(np.array([5.0, 5.0, 5.0]))
        np.testing.assert_allclose(r, [0.5, 0.5, 0.5])

    def test_empty_returns_empty(self):
        r = _rank_normalize(np.array([]))
        assert len(r) == 0


class TestComputeOutlierScore:
    def test_returns_expected_columns(self):
        n = 5
        out = compute_outlier_score(
            pred_lo=np.zeros(n),
            pred_hi=np.ones(n),
            pred_med=np.full(n, 0.5),
            iforest_score=np.zeros(n),
            cfg=_cfg(),
        )
        expected = {
            "pred_lo",
            "pred_med",
            "pred_hi",
            "score_uncertainty",
            "score_range_violation",
            "score_iforest",
            "score",
        }
        assert expected.issubset(set(out.columns))

    def test_high_uncertainty_increases_score(self):
        # 4 サンプル中 1 つだけ uncertainty が巨大
        lo = np.array([0.4, 0.4, 0.4, 0.4])
        hi = np.array([0.5, 0.5, 0.5, 10.0])
        med = np.full(4, 0.5)
        if_score = np.zeros(4)
        out = compute_outlier_score(lo, hi, med, if_score, _cfg())
        assert out["score"].iloc[3] > out["score"].iloc[0]

    def test_range_violation_flags_out_of_valid_range(self):
        # t_valid_min=0.0, t_valid_max=1.5 -> med=2.0 は違反、0.5 は正常
        lo = np.zeros(3)
        hi = np.ones(3)
        med = np.array([0.5, 0.5, 2.0])
        if_score = np.zeros(3)
        out = compute_outlier_score(lo, hi, med, if_score, _cfg())
        assert out["score"].iloc[2] > out["score"].iloc[0]

    def test_iforest_negative_means_higher_outlier(self):
        lo = np.zeros(3)
        hi = np.ones(3)
        med = np.full(3, 0.5)
        # score_samples: 高い=正常、低い=異常。符号反転後、低いものが高スコアになる。
        if_score = np.array([-0.5, 0.0, 0.5])
        out = compute_outlier_score(lo, hi, med, if_score, _cfg())
        assert out["score"].iloc[0] > out["score"].iloc[2]

    def test_weights_sum_scale_not_required(self):
        # 重みが 1.0 でなくても正しく重み付き和になる
        lo = np.zeros(3)
        hi = np.array([1.0, 2.0, 3.0])
        med = np.full(3, 0.5)
        if_score = np.zeros(3)
        cfg = _cfg(weight_uncertainty=1.0, weight_range_violation=0.0, weight_iforest=0.0)
        out = compute_outlier_score(lo, hi, med, if_score, cfg)
        # uncertainty の順位が単調なので score も単調
        assert out["score"].iloc[0] < out["score"].iloc[1] < out["score"].iloc[2]


class TestPrepareFeatures:
    def test_missing_column_filled_with_median(self):
        df = pd.DataFrame(
            [
                {
                    "px": 0,
                    "py": 0,
                    "qx": 3,
                    "qy": 4,
                    "eye_scale": 10,
                    "point_index": 3,
                    "part_name": "eyeliner_left",
                    "sub_key": "upper",
                }
            ]
        )
        medians = pd.Series({"face_yaw": 1.23, "pq_dist": 5.0, "point_index": 3.0})
        feat = prepare_features(df, ["face_yaw", "pq_dist", "point_index"], medians)
        assert feat["face_yaw"].iloc[0] == pytest.approx(1.23)
        assert feat["pq_dist"].iloc[0] == pytest.approx(5.0)
        assert feat["point_index"].iloc[0] == pytest.approx(3.0)

    def test_derived_columns_added(self):
        df = pd.DataFrame(
            [
                {
                    "px": 0,
                    "py": 0,
                    "qx": 3,
                    "qy": 4,
                    "eye_scale": 10,
                    "point_index": 3,
                    "part_name": "eyeliner_left",
                    "sub_key": "upper",
                }
            ]
        )
        medians = pd.Series({"pq_dist": 0.0, "is_upper": 0.0})
        feat = prepare_features(df, ["pq_dist", "is_upper"], medians)
        assert feat["pq_dist"].iloc[0] == pytest.approx(5.0)
        assert feat["is_upper"].iloc[0] == pytest.approx(1.0)
