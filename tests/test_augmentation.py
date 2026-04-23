"""eyeliner/data/augmentation.py のテスト。"""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from eyeliner.data.augmentation import (
    AUG_ALL,
    AUG_ENDPOINT_BLEND,
    AUG_NONE,
    ENDPOINT_BLEND_RANGE,
    augment_endpoints,
)


def _reconstruct_r(px: float, py: float, qx: float, qy: float, t: float) -> tuple[float, float]:
    """R = t·P + (1 - t)·Q."""
    return t * px + (1.0 - t) * qx, t * py + (1.0 - t) * qy


class TestAugmentEndpoints:
    """augment_endpoints の単体テスト。"""

    def test_flag_off_returns_identity(self):
        """AUG_ENDPOINT_BLEND フラグ OFF のとき入力がそのまま返る。"""
        random.seed(0)
        out = augment_endpoints(
            10.0, 20.0, 30.0, 40.0, 0.5,
            label_t_min=-0.5, label_t_max=1.5,
            aug_flags=AUG_NONE,
        )
        assert out == (10.0, 20.0, 30.0, 40.0, 0.5)

    def test_flag_on_perturbs_values(self):
        """AUG_ENDPOINT_BLEND フラグ ON のとき端点 or t が変化しうる。"""
        random.seed(42)
        changed = False
        for _ in range(20):
            px, py, qx, qy, t = 10.0, 20.0, 30.0, 40.0, 0.5
            out = augment_endpoints(
                px, py, qx, qy, t,
                label_t_min=-0.5, label_t_max=1.5,
                aug_flags=AUG_ENDPOINT_BLEND,
            )
            if out != (px, py, qx, qy, t):
                changed = True
                break
        assert changed, "20 回中 1 回も値が変化しないのは異常"

    def test_geometric_invariant(self):
        """計算後の R' = t'·P' + (1-t')·Q' が元の R と一致する。"""
        random.seed(123)
        for _ in range(200):
            px = random.uniform(0, 500)
            py = random.uniform(0, 500)
            qx = random.uniform(0, 500)
            qy = random.uniform(0, 500)
            t = random.uniform(-0.4, 1.4)

            r_before = _reconstruct_r(px, py, qx, qy, t)
            new_px, new_py, new_qx, new_qy, new_t = augment_endpoints(
                px, py, qx, qy, t,
                label_t_min=-0.5, label_t_max=1.5,
                aug_flags=AUG_ENDPOINT_BLEND,
            )
            r_after = _reconstruct_r(new_px, new_py, new_qx, new_qy, new_t)

            assert math.isclose(r_before[0], r_after[0], abs_tol=1e-6), \
                f"Rx invariant 破れ: {r_before} → {r_after}"
            assert math.isclose(r_before[1], r_after[1], abs_tol=1e-6), \
                f"Ry invariant 破れ: {r_before} → {r_after}"

    def test_denominator_bounds(self):
        """内部で使う 1 - a - b が 0.5〜1.5 に収まる（反転しない）ことを間接的に検証。

        端点変換式から逆算して a, b を復元し、範囲をチェックする。
        """
        random.seed(7)
        for _ in range(500):
            px, py, qx, qy = 0.0, 0.0, 100.0, 0.0
            t_in = 0.5
            new_px, new_py, new_qx, new_qy, new_t = augment_endpoints(
                px, py, qx, qy, t_in,
                label_t_min=-100.0, label_t_max=100.0,  # 範囲を広く取り fallback を回避
                aug_flags=AUG_ENDPOINT_BLEND,
            )
            # Q' = a·P + (1-a)·Q  →  a = (Q' - Q) / (P - Q) = (new_qx - 100) / (0 - 100)
            a = (new_qx - 100.0) / (0.0 - 100.0)
            # P' = b·Q + (1-b)·P  →  b = (P' - P) / (Q - P) = (new_px - 0) / (100 - 0)
            b = (new_px - 0.0) / (100.0 - 0.0)
            denom = 1.0 - a - b
            assert 0.5 - 1e-9 <= denom <= 1.5 + 1e-9, \
                f"denom 範囲外: a={a}, b={b}, denom={denom}"

    def test_identity_fallback_when_t_out_of_range(self):
        """t' が label_t 範囲外になった場合は identity fallback。"""
        # blend_range=0.25, t が端に近いと t' が範囲外になりやすい状況を作る
        random.seed(99)
        px, py, qx, qy = 0.0, 0.0, 100.0, 0.0
        # label_t 範囲を狭めに取り、fallback を発生させる
        label_t_min, label_t_max = 0.45, 0.55
        t_in = 0.5

        fallback_count = 0
        for _ in range(100):
            out = augment_endpoints(
                px, py, qx, qy, t_in,
                label_t_min=label_t_min, label_t_max=label_t_max,
                aug_flags=AUG_ENDPOINT_BLEND,
            )
            if out == (px, py, qx, qy, t_in):
                fallback_count += 1
            # fallback でない場合は必ず範囲内
            else:
                assert label_t_min <= out[4] <= label_t_max

        # 狭い範囲では fallback が発生するはず
        assert fallback_count > 0

    def test_blend_range_constant(self):
        """ENDPOINT_BLEND_RANGE が 0.25 である。"""
        assert ENDPOINT_BLEND_RANGE == 0.25

    def test_aug_all_includes_endpoint_blend(self):
        """AUG_ALL に AUG_ENDPOINT_BLEND が含まれる。"""
        assert AUG_ALL & AUG_ENDPOINT_BLEND == AUG_ENDPOINT_BLEND

    def test_identity_when_a_b_zero(self, monkeypatch):
        """a = b = 0 のとき入力が保存される。"""
        monkeypatch.setattr(
            "eyeliner.data.augmentation.random.uniform",
            lambda lo, hi: 0.0,
        )
        px, py, qx, qy, t = 10.0, 20.0, 30.0, 40.0, 0.5
        out = augment_endpoints(
            px, py, qx, qy, t,
            label_t_min=-0.5, label_t_max=1.5,
            aug_flags=AUG_ENDPOINT_BLEND,
        )
        assert out == pytest.approx((px, py, qx, qy, t), abs=1e-9)
