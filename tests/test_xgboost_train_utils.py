"""xgboost_baseline: train.py の純粋関数のテスト。

XGBoost / sklearn / Optuna に依存しないロジック（apply_t_resolution, regression_metrics,
KFoldEnsemble, config loading）を pytest で検証する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analyses.xgboost_baseline.scr.train_exp015 import (
    KFoldEnsemble,
    KFoldConfig,
    OptunaConfig,
    XGBConfig,
    apply_t_resolution,
    regression_metrics,
    _kfold_cfg_from,
    _optuna_cfg_from,
    _xgb_cfg_from,
)


class TestApplyTResolution:
    def test_none_returns_input_unchanged(self):
        y = np.array([0.12, 0.55, 0.99])
        out = apply_t_resolution(y, None)
        np.testing.assert_allclose(out, y)

    def test_rounds_to_0_1(self):
        y = np.array([0.12, 0.55, 0.99, 0.24, 0.25])
        out = apply_t_resolution(y, 0.1)
        assert out[0] == pytest.approx(0.1)
        assert out[1] == pytest.approx(0.6)
        assert out[2] == pytest.approx(1.0)
        assert out[3] == pytest.approx(0.2)

    def test_rounds_to_0_05(self):
        y = np.array([0.12, 0.13, 0.17])
        out = apply_t_resolution(y, 0.05)
        assert out[0] == pytest.approx(0.10)
        assert out[1] == pytest.approx(0.15)
        assert out[2] == pytest.approx(0.15)

    def test_preserves_zero_and_one(self):
        y = np.array([0.0, 1.0])
        out = apply_t_resolution(y, 0.1)
        np.testing.assert_allclose(out, [0.0, 1.0])


class TestRegressionMetrics:
    def test_perfect_prediction(self):
        y = np.array([0.1, 0.5, 0.9])
        m = regression_metrics(y, y)
        assert m["mae"] == pytest.approx(0.0)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["r2"] == pytest.approx(1.0)

    def test_constant_offset(self):
        y = np.array([0.0, 0.5, 1.0])
        p = y + 0.1
        m = regression_metrics(y, p)
        assert m["mae"] == pytest.approx(0.1)
        assert m["rmse"] == pytest.approx(0.1)

    def test_r2_nan_when_zero_variance(self):
        y = np.array([0.5, 0.5, 0.5])
        p = np.array([0.6, 0.6, 0.6])
        m = regression_metrics(y, p)
        assert np.isnan(m["r2"])


# ---------------------------------------------------------------------------
# KFoldEnsemble
# ---------------------------------------------------------------------------
class _ConstantModel:
    """テスト用の定数予測モデル。"""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value)


class TestKFoldEnsemble:
    def test_predict_averages_models(self):
        models = [_ConstantModel(0.2), _ConstantModel(0.4), _ConstantModel(0.6)]
        ensemble = KFoldEnsemble(models)
        X = pd.DataFrame({"a": [1, 2, 3]})
        preds = ensemble.predict(X)
        np.testing.assert_allclose(preds, 0.4)

    def test_predict_shape_matches_input(self):
        models = [_ConstantModel(0.5), _ConstantModel(0.7)]
        ensemble = KFoldEnsemble(models)
        X = pd.DataFrame({"a": range(10)})
        preds = ensemble.predict(X)
        assert preds.shape == (10,)

    def test_single_model_ensemble(self):
        ensemble = KFoldEnsemble([_ConstantModel(0.3)])
        X = pd.DataFrame({"a": [1, 2]})
        np.testing.assert_allclose(ensemble.predict(X), 0.3)


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------
class TestXgbCfgFrom:
    def _minimal(self) -> dict:
        return {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

    def test_parses_required_fields(self):
        cfg = _xgb_cfg_from(self._minimal())
        assert cfg.n_estimators == 100
        assert cfg.max_depth == 4
        assert cfg.learning_rate == pytest.approx(0.1)

    def test_min_child_weight_defaults_to_1(self):
        cfg = _xgb_cfg_from(self._minimal())
        assert cfg.min_child_weight == 1

    def test_min_child_weight_parsed(self):
        d = {**self._minimal(), "min_child_weight": 5}
        cfg = _xgb_cfg_from(d)
        assert cfg.min_child_weight == 5


class TestKFoldCfgFrom:
    def test_defaults_when_empty(self):
        cfg = _kfold_cfg_from({})
        assert cfg.enabled is False
        assert cfg.n_splits == 5
        assert cfg.shuffle is True
        assert cfg.seed == 42

    def test_enabled_parsed(self):
        cfg = _kfold_cfg_from({"enabled": True, "n_splits": 3})
        assert cfg.enabled is True
        assert cfg.n_splits == 3


class TestOptunaCfgFrom:
    def test_defaults_when_empty(self):
        cfg = _optuna_cfg_from({})
        assert cfg.enabled is False
        assert cfg.n_trials == 50
        assert cfg.timeout is None

    def test_timeout_parsed(self):
        cfg = _optuna_cfg_from({"enabled": True, "n_trials": 20, "timeout": 300})
        assert cfg.enabled is True
        assert cfg.timeout == 300

    def test_timeout_none_when_null(self):
        cfg = _optuna_cfg_from({"timeout": None})
        assert cfg.timeout is None
