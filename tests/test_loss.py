"""eyeliner/evaluation/loss.py のユニットテスト。

損失・メトリクスのバグは全実験を汚染するため、カバレッジ 100% を目標とする。
"""
from __future__ import annotations

import pytest
import torch
import numpy as np

from eyeliner.evaluation.loss import (
    huber_t_loss,
    eye_normalized_smooth_l1_loss,
    get_loss_fn,
    LOSS_FN_REGISTRY,
)


# ===========================================================
# huber_t_loss
# ===========================================================

class TestHuberTLoss:

    def test_zero_error_is_zero(self):
        """誤差 0 のとき損失は 0。"""
        pred = torch.tensor([0.5, 0.3])
        y = torch.tensor([0.5, 0.3])
        loss = huber_t_loss(pred, y)
        assert abs(loss.item()) < 1e-7

    def test_delta_boundary_l2_region(self):
        """delta 未満の誤差は 0.5*dt^2/delta（二乗項）。"""
        delta = 0.05
        dt = 0.01  # < delta
        pred = torch.tensor([dt])
        y = torch.zeros(1)
        loss = huber_t_loss(pred, y, delta=delta)
        expected = 0.5 * dt**2 / delta
        assert abs(loss.item() - expected) < 1e-6

    def test_delta_boundary_l1_region(self):
        """delta 超の誤差は dt - 0.5*delta（線形項）。"""
        delta = 0.05
        dt = 0.10  # > delta
        pred = torch.tensor([dt])
        y = torch.zeros(1)
        loss = huber_t_loss(pred, y, delta=delta)
        expected = dt - 0.5 * delta
        assert abs(loss.item() - expected) < 1e-6

    def test_no_nan_backward(self):
        """backward() で NaN が発生しない。"""
        pred = torch.randn(32, requires_grad=True)
        y = torch.randn(32)
        loss = huber_t_loss(pred, y)
        loss.backward()
        assert not torch.isnan(pred.grad).any()

    def test_symmetric(self):
        """符号が逆の誤差で同じ損失値を返す。"""
        pred_pos = torch.tensor([0.1])
        pred_neg = torch.tensor([-0.1])
        y = torch.zeros(1)
        assert abs(huber_t_loss(pred_pos, y).item() - huber_t_loss(pred_neg, y).item()) < 1e-7

    def test_batch(self):
        """バッチ入力でも例外なく動作する。"""
        pred = torch.randn(16)
        y = torch.randn(16)
        loss = huber_t_loss(pred, y)
        assert loss.ndim == 0  # スカラー


# ===========================================================
# eye_normalized_smooth_l1_loss
# ===========================================================

class TestEyeNormalizedSmoothl1Loss:

    def test_zero_error_is_zero(self):
        """誤差 0 のとき損失は 0。"""
        pred = torch.tensor([0.5])
        y = torch.tensor([0.5])
        weight = torch.ones(1)
        loss = eye_normalized_smooth_l1_loss(pred, y, weight)
        assert abs(loss.item()) < 1e-7

    def test_weight_scaling(self):
        """weight=2 のとき損失は weight=1 の 2 倍になる。"""
        pred = torch.tensor([0.1])
        y = torch.zeros(1)
        w1 = torch.ones(1)
        w2 = torch.full((1,), 2.0)
        loss1 = eye_normalized_smooth_l1_loss(pred, y, w1)
        loss2 = eye_normalized_smooth_l1_loss(pred, y, w2)
        assert abs(loss2.item() / loss1.item() - 2.0) < 1e-5

    def test_no_nan_backward(self):
        """backward() で NaN が発生しない。"""
        pred = torch.randn(32, requires_grad=True)
        y = torch.randn(32)
        weight = torch.rand(32) + 0.1
        loss = eye_normalized_smooth_l1_loss(pred, y, weight)
        loss.backward()
        assert not torch.isnan(pred.grad).any()

    def test_huber_formula_l2_region(self):
        """delta 未満の誤差で二乗項になる（weight=1 のとき）。"""
        delta = 0.05
        dt = 0.01
        pred = torch.tensor([dt])
        y = torch.zeros(1)
        weight = torch.ones(1)
        loss = eye_normalized_smooth_l1_loss(pred, y, weight, delta=delta)
        expected = 0.5 * dt**2 / delta
        assert abs(loss.item() - expected) < 1e-6

    def test_huber_formula_l1_region(self):
        """delta 超の誤差で線形項になる（weight=1 のとき）。"""
        delta = 0.05
        dt = 0.10
        pred = torch.tensor([dt])
        y = torch.zeros(1)
        weight = torch.ones(1)
        loss = eye_normalized_smooth_l1_loss(pred, y, weight, delta=delta)
        expected = dt - 0.5 * delta
        assert abs(loss.item() - expected) < 1e-6

    def test_batch(self):
        """バッチ入力でも例外なく動作する。"""
        pred = torch.randn(16)
        y = torch.randn(16)
        weight = torch.rand(16) + 0.1
        loss = eye_normalized_smooth_l1_loss(pred, y, weight)
        assert loss.ndim == 0


# ===========================================================
# LOSS_FN_REGISTRY / get_loss_fn
# ===========================================================

class TestLossFnRegistry:

    def test_registry_contains_expected_keys(self):
        """レジストリに必要なキーが含まれる。"""
        assert "t" in LOSS_FN_REGISTRY
        assert "normalized_pixel" in LOSS_FN_REGISTRY

    def test_get_loss_fn_t(self):
        """'t' キーで huber_t_loss が返る。"""
        fn = get_loss_fn("t")
        assert fn is huber_t_loss

    def test_get_loss_fn_normalized_pixel(self):
        """'normalized_pixel' キーで eye_normalized_smooth_l1_loss が返る。"""
        fn = get_loss_fn("normalized_pixel")
        assert fn is eye_normalized_smooth_l1_loss

    def test_get_loss_fn_unknown_raises(self):
        """未知のキーは ValueError を送出する。"""
        with pytest.raises(ValueError, match="未知の損失関数"):
            get_loss_fn("nonexistent")
