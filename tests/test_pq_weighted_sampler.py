"""pq_weighted sampler / loss のユニットテスト。

EXP018 で追加した pq_ratio < threshold サンプルへの重み付け機能を検証する。
"""
from __future__ import annotations

import math

import pytest
import torch

from eyeliner.evaluation.loss import pq_weighted_huber_loss, get_loss_fn, LOSS_FN_REGISTRY


# ===========================================================
# pq_weighted_huber_loss
# ===========================================================

class TestPqWeightedHuberLoss:

    def test_zero_error_is_zero(self):
        pred = torch.tensor([0.3, 0.7])
        y = torch.tensor([0.3, 0.7])
        pq_w = torch.ones(2)
        loss = pq_weighted_huber_loss(pred, y, pq_w)
        assert abs(loss.item()) < 1e-7

    def test_weight_one_equals_huber_t_loss(self):
        """weight=1 のとき huber_t_loss と同値になる。"""
        from eyeliner.evaluation.loss import huber_t_loss
        pred = torch.randn(16)
        y = torch.randn(16)
        pq_w = torch.ones(16)
        loss_pq = pq_weighted_huber_loss(pred, y, pq_w)
        loss_t = huber_t_loss(pred, y)
        assert abs(loss_pq.item() - loss_t.item()) < 1e-6

    def test_higher_weight_increases_loss(self):
        """weight > 1 のサンプルは loss に大きく貢献する。"""
        pred = torch.tensor([0.1])
        y = torch.zeros(1)
        w_low = torch.tensor([1.0])
        w_high = torch.tensor([3.0])
        loss_low = pq_weighted_huber_loss(pred, y, w_low)
        loss_high = pq_weighted_huber_loss(pred, y, w_high)
        assert loss_high.item() > loss_low.item()

    def test_no_nan_backward(self):
        pred = torch.randn(32, requires_grad=True)
        y = torch.randn(32)
        pq_w = torch.rand(32) + 0.5
        loss = pq_weighted_huber_loss(pred, y, pq_w)
        loss.backward()
        assert not torch.isnan(pred.grad).any()

    def test_registered_in_loss_fn_registry(self):
        assert "pq_weighted" in LOSS_FN_REGISTRY
        fn = get_loss_fn("pq_weighted")
        assert fn is pq_weighted_huber_loss


# ===========================================================
# pq_weighted sampler weight 計算の検証（kfold ロジックの単体テスト）
# ===========================================================

class TestPqSamplerWeightLogic:
    """kfold.py の sampler 重み計算ロジックを独立して検証する。"""

    @staticmethod
    def _compute_pq_weights(pq_values: list[float], alpha: float, threshold: float) -> list[float]:
        return [
            1.0 + alpha * max(0.0, threshold - pq) / threshold
            for pq in pq_values
        ]

    def test_below_threshold_gets_higher_weight(self):
        """pq < threshold のサンプルは weight > 1 になる。"""
        weights = self._compute_pq_weights([0.05, 0.15], alpha=3.0, threshold=0.10)
        assert weights[0] > 1.0   # pq=0.05 < threshold → 重み大
        assert weights[1] == 1.0  # pq=0.15 > threshold → 重みなし

    def test_at_threshold_weight_is_one(self):
        """pq == threshold のとき weight = 1.0。"""
        weights = self._compute_pq_weights([0.10], alpha=3.0, threshold=0.10)
        assert abs(weights[0] - 1.0) < 1e-9

    def test_zero_pq_gives_max_weight(self):
        """pq = 0 のとき weight = 1 + alpha。"""
        alpha = 5.0
        weights = self._compute_pq_weights([0.0], alpha=alpha, threshold=0.10)
        assert abs(weights[0] - (1.0 + alpha)) < 1e-9

    def test_above_threshold_weight_is_one(self):
        """pq > threshold のサンプルは weight = 1.0 でクランプされる。"""
        weights = self._compute_pq_weights([0.50, 1.00], alpha=3.0, threshold=0.10)
        assert weights[0] == 1.0
        assert weights[1] == 1.0

    def test_low_pq_samples_appear_more(self):
        """pq < threshold のサンプルが weight 上位に来ることを assert。"""
        pq_values = [0.02, 0.04, 0.06, 0.20, 0.50, 0.80]
        weights = self._compute_pq_weights(pq_values, alpha=3.0, threshold=0.10)
        sorted_by_weight = sorted(range(len(pq_values)), key=lambda i: -weights[i])
        low_pq_indices = {i for i, pq in enumerate(pq_values) if pq < 0.10}
        # 上位 3 件がすべて low_pq（0.02, 0.04, 0.06）であることを確認
        assert set(sorted_by_weight[:3]) == low_pq_indices


# ===========================================================
# compute_aux_face_shape_meta
# ===========================================================

class TestComputeAuxFaceShapeMeta:

    @staticmethod
    def _make_record(px, py, qx, qy, t=0.5, part="eyeliner_left", sub_key="upper", pi=3):
        return {
            "image_path": "img.jpg",
            "face_index": 0,
            "part_name": part,
            "sub_key": sub_key,
            "point_index": pi,
            "px": px, "py": py,
            "qx": qx, "qy": qy,
            "t": t,
        }

    def test_pq_angle_cos_horizontal(self):
        """水平 PQ ベクトルで pq_angle_cos=1, pq_angle_sin=0。"""
        from eyeliner.data.weights import compute_eye_meta, compute_aux_face_shape_meta
        recs = [self._make_record(0, 0, 10, 0)]
        compute_eye_meta(recs)
        compute_aux_face_shape_meta(recs)
        assert abs(recs[0]["pq_angle_cos"] - 1.0) < 1e-6
        assert abs(recs[0]["pq_angle_sin"] - 0.0) < 1e-6

    def test_pq_angle_sin_vertical(self):
        """垂直 PQ ベクトルで pq_angle_cos=0, pq_angle_sin=1。"""
        from eyeliner.data.weights import compute_eye_meta, compute_aux_face_shape_meta
        recs = [self._make_record(0, 0, 0, 10)]
        compute_eye_meta(recs)
        compute_aux_face_shape_meta(recs)
        assert abs(recs[0]["pq_angle_cos"] - 0.0) < 1e-6
        assert abs(recs[0]["pq_angle_sin"] - 1.0) < 1e-6

    def test_position_polar_cos_upper0(self):
        """upper, point_index=0 → degree=22.5° → cos(22.5°)。"""
        from eyeliner.data.weights import compute_eye_meta, compute_aux_face_shape_meta
        recs = [self._make_record(0, 0, 10, 0, sub_key="upper", pi=0)]
        compute_eye_meta(recs)
        compute_aux_face_shape_meta(recs)
        expected_cos = math.cos(22.5 * math.pi / 180)
        assert abs(recs[0]["position_polar_cos"] - expected_cos) < 1e-6
