"""xgboost_baseline: features.py のテスト。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analyses.xgboost_baseline.scr.features import (
    FeatureConfig,
    add_categorical_features,
    add_cyclical_features,
    add_geometry_features,
    add_interaction_features,
    build_feature_matrix,
    enrich_all,
    select_feature_columns,
)


def _base_row(**overrides) -> dict:
    base = {
        "idx": 0,
        "image_path": "a.jpg",
        "face_index": 0,
        "part_name": "eyeliner_left",
        "sub_key": "upper",
        "point_index": 3,
        "px": 100.0,
        "py": 200.0,
        "qx": 103.0,
        "qy": 204.0,
        "true_t_raw": 0.5,
        "eye_scale": 50.0,
        "norm_eye_scale": 0.1,
        "eye_bbox_w": 30.0,
        "eye_bbox_h": 10.0,
        "eye_aspect": 3.0,
        "face_yaw": 5.0,
        "face_pitch": -2.0,
        "face_roll": 0.5,
        "skip_flag": 0,
    }
    return {**base, **overrides}


# ---------------------------------------------------------------------------
# add_geometry_features
# ---------------------------------------------------------------------------
class TestAddGeometryFeatures:
    def test_pq_dist_pythagoras(self):
        df = pd.DataFrame([_base_row(px=0, py=0, qx=3, qy=4)])
        out = add_geometry_features(df)
        assert out["pq_dist"].iloc[0] == pytest.approx(5.0)

    def test_pq_angle_matches_atan2(self):
        df = pd.DataFrame([_base_row(px=0, py=0, qx=1, qy=1)])
        out = add_geometry_features(df)
        assert out["pq_angle"].iloc[0] == pytest.approx(math.pi / 4)

    def test_pq_ratio_with_eye_scale(self):
        df = pd.DataFrame([_base_row(px=0, py=0, qx=3, qy=4, eye_scale=10.0)])
        out = add_geometry_features(df)
        assert out["pq_ratio"].iloc[0] == pytest.approx(0.5)

    def test_pq_ratio_nan_when_eye_scale_zero(self):
        df = pd.DataFrame([_base_row(eye_scale=0.0)])
        out = add_geometry_features(df)
        assert np.isnan(out["pq_ratio"].iloc[0])

    def test_dist_from_center_edge_vs_middle(self):
        rows = [_base_row(point_index=pi) for pi in [0, 3, 6]]
        out = add_geometry_features(pd.DataFrame(rows))
        assert out["dist_from_center"].iloc[0] == pytest.approx(0.0)
        assert out["dist_from_center"].iloc[1] == pytest.approx(1.0)
        assert out["dist_from_center"].iloc[2] == pytest.approx(0.0)

    def test_does_not_mutate_input(self):
        df = pd.DataFrame([_base_row()])
        original_cols = set(df.columns)
        _ = add_geometry_features(df)
        assert set(df.columns) == original_cols


# ---------------------------------------------------------------------------
# add_categorical_features
# ---------------------------------------------------------------------------
class TestAddCategoricalFeatures:
    def test_is_upper_true(self):
        out = add_categorical_features(pd.DataFrame([_base_row(sub_key="upper")]))
        assert out["is_upper"].iloc[0] == 1

    def test_is_upper_false_for_lower(self):
        out = add_categorical_features(pd.DataFrame([_base_row(sub_key="lower")]))
        assert out["is_upper"].iloc[0] == 0

    def test_is_left_eye_true(self):
        out = add_categorical_features(
            pd.DataFrame([_base_row(part_name="eyeliner_left")])
        )
        assert out["is_left_eye"].iloc[0] == 1

    def test_is_left_eye_false_for_right(self):
        out = add_categorical_features(
            pd.DataFrame([_base_row(part_name="eyeliner_right")])
        )
        assert out["is_left_eye"].iloc[0] == 0


# ---------------------------------------------------------------------------
# add_cyclical_features
# ---------------------------------------------------------------------------
class TestAddCyclicalFeatures:
    def test_sin_cos_values(self):
        df = pd.DataFrame([_base_row(px=0, py=0, qx=1, qy=0)])
        out = add_geometry_features(df)
        out = add_cyclical_features(out)
        # angle = 0 → sin=0, cos=1
        assert out["pq_angle_sin"].iloc[0] == pytest.approx(0.0, abs=1e-9)
        assert out["pq_angle_cos"].iloc[0] == pytest.approx(1.0, abs=1e-9)

    def test_sin_cos_90_degrees(self):
        df = pd.DataFrame([_base_row(px=0, py=0, qx=0, qy=1)])
        out = add_geometry_features(df)
        out = add_cyclical_features(out)
        # angle = pi/2 → sin=1, cos=0
        assert out["pq_angle_sin"].iloc[0] == pytest.approx(1.0, abs=1e-9)
        assert out["pq_angle_cos"].iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_sin2_plus_cos2_equals_1(self):
        rows = [_base_row(px=0, py=0, qx=math.cos(a), qy=math.sin(a)) for a in [0.3, 1.2, 2.5, -0.7]]
        df = pd.DataFrame(rows)
        out = add_geometry_features(df)
        out = add_cyclical_features(out)
        s2c2 = out["pq_angle_sin"] ** 2 + out["pq_angle_cos"] ** 2
        np.testing.assert_allclose(s2c2.to_numpy(), 1.0, atol=1e-9)

    def test_does_not_mutate_input(self):
        df = pd.DataFrame([_base_row()])
        geo = add_geometry_features(df)
        original_cols = set(geo.columns)
        _ = add_cyclical_features(geo)
        assert set(geo.columns) == original_cols


# ---------------------------------------------------------------------------
# add_interaction_features
# ---------------------------------------------------------------------------
class TestAddInteractionFeatures:
    def _enriched(self, **overrides) -> pd.DataFrame:
        df = pd.DataFrame([_base_row(**overrides)])
        df = add_geometry_features(df)
        df = add_categorical_features(df)
        return df

    def test_upper_x_pq_ratio_for_upper(self):
        df = self._enriched(sub_key="upper", px=0, py=0, qx=5, qy=0, eye_scale=10.0)
        out = add_interaction_features(df)
        # is_upper=1, pq_ratio=5/10=0.5
        assert out["is_upper_x_pq_ratio"].iloc[0] == pytest.approx(0.5)

    def test_upper_x_pq_ratio_zero_for_lower(self):
        df = self._enriched(sub_key="lower", px=0, py=0, qx=5, qy=0, eye_scale=10.0)
        out = add_interaction_features(df)
        # is_upper=0 → interaction=0
        assert out["is_upper_x_pq_ratio"].iloc[0] == pytest.approx(0.0)

    def test_upper_x_dist_from_center(self):
        df = self._enriched(sub_key="upper", point_index=3)
        out = add_interaction_features(df)
        # is_upper=1, dist_from_center=1.0
        assert out["is_upper_x_dist_from_center"].iloc[0] == pytest.approx(1.0)

    def test_interaction_nan_propagates_when_pq_ratio_nan(self):
        df = self._enriched(eye_scale=0.0)  # pq_ratio = NaN
        out = add_interaction_features(df)
        assert np.isnan(out["is_upper_x_pq_ratio"].iloc[0])

    def test_does_not_mutate_input(self):
        df = self._enriched()
        original_cols = set(df.columns)
        _ = add_interaction_features(df)
        assert set(df.columns) == original_cols


# ---------------------------------------------------------------------------
# select_feature_columns
# ---------------------------------------------------------------------------
class TestSelectFeatureColumns:
    def _enriched_df(self, n: int = 10, **overrides) -> pd.DataFrame:
        rows = [_base_row(**overrides) for _ in range(n)]
        df = pd.DataFrame(rows)
        df = add_geometry_features(df)
        df = add_categorical_features(df)
        df = add_cyclical_features(df)
        df = add_interaction_features(df)
        return df

    def test_face_pose_excluded_when_all_null(self):
        df = pd.DataFrame(
            [_base_row(face_yaw=np.nan, face_pitch=np.nan, face_roll=np.nan) for _ in range(10)]
        )
        df = add_geometry_features(df)
        df = add_categorical_features(df)
        cols = select_feature_columns(df, FeatureConfig())
        assert "face_yaw" not in cols
        assert "pq_dist" in cols

    def test_face_pose_included_when_values_present(self):
        df = self._enriched_df()
        cols = select_feature_columns(df, FeatureConfig())
        assert "face_yaw" in cols

    def test_flags_disable_groups(self):
        df = self._enriched_df()
        cfg = FeatureConfig(use_geometry=False, use_face_pose=False)
        cols = select_feature_columns(df, cfg)
        assert "pq_dist" not in cols
        assert "face_yaw" not in cols
        assert "is_upper" in cols
        assert "eye_scale" in cols

    def test_cyclical_replaces_raw_angle(self):
        df = self._enriched_df()
        cfg = FeatureConfig(use_cyclical_angle=True)
        cols = select_feature_columns(df, cfg)
        assert "pq_angle" not in cols
        assert "pq_angle_sin" in cols
        assert "pq_angle_cos" in cols

    def test_no_cyclical_keeps_raw_angle(self):
        df = self._enriched_df()
        cfg = FeatureConfig(use_cyclical_angle=False)
        cols = select_feature_columns(df, cfg)
        assert "pq_angle" in cols
        assert "pq_angle_sin" not in cols
        assert "pq_angle_cos" not in cols

    def test_interactions_added_when_enabled(self):
        df = self._enriched_df()
        cfg = FeatureConfig(use_interactions=True)
        cols = select_feature_columns(df, cfg)
        assert "is_upper_x_pq_ratio" in cols
        assert "is_upper_x_dist_from_center" in cols

    def test_interactions_absent_when_disabled(self):
        df = self._enriched_df()
        cfg = FeatureConfig(use_interactions=False)
        cols = select_feature_columns(df, cfg)
        assert "is_upper_x_pq_ratio" not in cols


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------
class TestBuildFeatureMatrix:
    def test_returns_dataframe_and_columns(self):
        df = pd.DataFrame([_base_row() for _ in range(5)])
        feat, cols = build_feature_matrix(df, FeatureConfig())
        assert isinstance(feat, pd.DataFrame)
        assert len(feat) == 5
        assert set(feat.columns) == set(cols)

    def test_no_nan_after_fillna(self):
        rows = [_base_row() for _ in range(5)]
        rows[0]["eye_scale"] = np.nan
        df = pd.DataFrame(rows)
        feat, _ = build_feature_matrix(df, FeatureConfig())
        assert not feat.isna().any().any()

    def test_raises_when_no_features_available(self):
        df = pd.DataFrame([_base_row() for _ in range(3)])
        cfg = FeatureConfig(
            use_geometry=False,
            use_eye_scale=False,
            use_face_pose=False,
            use_categorical=False,
            use_point_index=False,
        )
        with pytest.raises(ValueError):
            build_feature_matrix(df, cfg)

    def test_cyclical_and_interactions_enabled(self):
        df = pd.DataFrame([_base_row() for _ in range(5)])
        cfg = FeatureConfig(use_cyclical_angle=True, use_interactions=True)
        feat, cols = build_feature_matrix(df, cfg)
        assert "pq_angle_sin" in cols
        assert "pq_angle_cos" in cols
        assert "pq_angle" not in cols
        assert "is_upper_x_pq_ratio" in cols
        assert not feat.isna().any().any()


# ---------------------------------------------------------------------------
# enrich_all
# ---------------------------------------------------------------------------
class TestEnrichAll:
    def test_no_cyclical_no_interactions(self):
        df = pd.DataFrame([_base_row()])
        cfg = FeatureConfig(use_cyclical_angle=False, use_interactions=False)
        out = enrich_all(df, cfg)
        assert "pq_dist" in out.columns
        assert "is_upper" in out.columns
        assert "pq_angle_sin" not in out.columns
        assert "is_upper_x_pq_ratio" not in out.columns

    def test_with_cyclical_and_interactions(self):
        df = pd.DataFrame([_base_row()])
        cfg = FeatureConfig(use_cyclical_angle=True, use_interactions=True)
        out = enrich_all(df, cfg)
        assert "pq_angle_sin" in out.columns
        assert "pq_angle_cos" in out.columns
        assert "is_upper_x_pq_ratio" in out.columns
        assert "is_upper_x_dist_from_center" in out.columns

    def test_does_not_mutate_input(self):
        df = pd.DataFrame([_base_row()])
        original_cols = set(df.columns)
        _ = enrich_all(df, FeatureConfig(use_cyclical_angle=True, use_interactions=True))
        assert set(df.columns) == original_cols
