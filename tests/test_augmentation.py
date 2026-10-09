"""eyeliner/data/augmentation.py のテスト。"""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from eyeliner.data.augmentation import (
    AUG_ALL,
    AUG_ENDPOINT_BLEND,
    AUG_FLIP,
    AUG_NOISE,
    AUG_NONE,
    ENDPOINT_BLEND_RANGE,
    augment,
    augment_endpoints,
)
from eyeliner.training.trainer import set_seed


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


class TestEndpointBlendRangeParam:
    """EXP021 Phase2: endpoint_blend_range パラメータの sweep 対応。"""

    def test_range_zero_is_identity(self):
        """endpoint_blend_range=0.0 はフラグ ON でも identity（a,b∈[-0,0]=0）。"""
        random.seed(0)
        px, py, qx, qy, t = 10.0, 20.0, 30.0, 40.0, 0.5
        for _ in range(20):
            out = augment_endpoints(
                px, py, qx, qy, t,
                label_t_min=-0.5, label_t_max=1.5,
                aug_flags=AUG_ENDPOINT_BLEND,
                endpoint_blend_range=0.0,
            )
            assert out == pytest.approx((px, py, qx, qy, t), abs=1e-12)

    def test_larger_range_perturbs_more(self):
        """range が大きいほど平均摂動量（|Δt|）が大きい。"""
        def mean_abs_dt(r, seed):
            random.seed(seed)
            total = 0.0
            n = 400
            for _ in range(n):
                _, _, _, _, nt = augment_endpoints(
                    0.0, 0.0, 100.0, 0.0, 0.5,
                    label_t_min=-100.0, label_t_max=100.0,  # fallback 回避
                    aug_flags=AUG_ENDPOINT_BLEND,
                    endpoint_blend_range=r,
                )
                total += abs(nt - 0.5)
            return total / n

        small = mean_abs_dt(0.10, seed=1)
        large = mean_abs_dt(0.50, seed=1)
        assert large > small

    def test_default_matches_module_constant(self):
        """デフォルト引数は ENDPOINT_BLEND_RANGE と同じ挙動（同一シードで一致）。"""
        random.seed(5)
        out_default = augment_endpoints(
            0.0, 0.0, 100.0, 0.0, 0.5,
            label_t_min=-100.0, label_t_max=100.0,
            aug_flags=AUG_ENDPOINT_BLEND,
        )
        random.seed(5)
        out_explicit = augment_endpoints(
            0.0, 0.0, 100.0, 0.0, 0.5,
            label_t_min=-100.0, label_t_max=100.0,
            aug_flags=AUG_ENDPOINT_BLEND,
            endpoint_blend_range=ENDPOINT_BLEND_RANGE,
        )
        assert out_default == pytest.approx(out_explicit, abs=1e-12)


class TestNoiseRngSeedControlled:
    """AUG_NOISE が set_seed() の制御下にあることの回帰テスト。

    NOISE だけが `np.random.default_rng()`（引数なし）を呼んでおり、set_seed() が固定する
    legacy グローバル状態の外にあった。そのため seed を固定しても NOISE の乱数列は
    run ごとに変わり、dataset_dual の枝A/B同期（RNG 状態の保存・復元）も素通りしていた。
    """

    def _patch(self) -> np.ndarray:
        return np.full((8, 8, 3), 0.5, dtype=np.float32)

    def test_same_seed_gives_same_output(self):
        """同一 seed で 2 回呼ぶと完全一致する。"""
        set_seed(0)
        out_a, _ = augment(self._patch(), 0.5, aug_flags=AUG_NOISE)
        set_seed(0)
        out_b, _ = augment(self._patch(), 0.5, aug_flags=AUG_NOISE)
        assert np.array_equal(out_a, out_b)

    def test_different_seed_gives_different_output(self):
        """seed を変えれば乱数列も変わる（seed が実際に効いていることの確認）。"""
        set_seed(0)
        out_a, _ = augment(self._patch(), 0.5, aug_flags=AUG_NOISE)
        set_seed(1)
        out_b, _ = augment(self._patch(), 0.5, aug_flags=AUG_NOISE)
        assert not np.array_equal(out_a, out_b)

    def test_noise_reproducible_via_legacy_state_save_restore(self):
        """dataset_dual の枝A/B同期と同じ手順（legacy RNG 状態の保存・復元）で再現できる。

        NOISE の振幅は random.uniform、実現値は np.random から引かれるため、
        両方の legacy 状態を戻す必要がある。dataset_dual.py が枝Aのもとで保存し
        枝Bの前に復元しているのはこの2つ。
        """
        random.seed(3)
        np_state = np.random.get_state()
        py_state = random.getstate()

        out_a, _ = augment(self._patch(), 0.5, aug_flags=AUG_NOISE)

        np.random.set_state(np_state)
        random.setstate(py_state)
        out_b, _ = augment(self._patch(), 0.5, aug_flags=AUG_NOISE)

        assert np.array_equal(out_a, out_b)

    def test_noise_is_actually_applied(self):
        """sigma>0 のとき値が変化する（テスト自体が空振りしていないこと）。"""
        set_seed(0)
        out, _ = augment(self._patch(), 0.5, aug_flags=AUG_NOISE)
        assert not np.allclose(out, 0.5)

    def test_dtype_and_shape_preserved(self):
        """float32 / 入力 shape を維持する。"""
        set_seed(0)
        out, _ = augment(self._patch(), 0.5, aug_flags=AUG_NOISE)
        assert out.dtype == np.float32
        assert out.shape == (8, 8, 3)

    def test_t_unaffected_by_noise(self):
        """NOISE は t を変えない。"""
        set_seed(0)
        _, t_out = augment(self._patch(), 0.37, aug_flags=AUG_NOISE)
        assert t_out == pytest.approx(0.37)


class TestFlipLabelTransform:
    """AUG_FLIP のラベル変換（t → t_min + t_max - t）と水平反転の直接テスト。

    以前は flip ラベル変換に直接テストが無く、`t_max - t`（t_min 欠落）等に書き間違えても
    検出できなかった。ここでは非対称な [t_min, t_max] でも反転式が厳密に成り立つことを確認する。
    """

    def _patch(self) -> np.ndarray:
        return np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3) / 255.0

    def test_flip_reflects_t_label(self, monkeypatch):
        """t が区間中点で反転する（t' = t_min + t_max - t）。"""
        monkeypatch.setattr("eyeliner.data.augmentation.random.random", lambda: 0.0)
        _, t_out = augment(self._patch(), 0.3, t_min=0.0, t_max=1.0, aug_flags=AUG_FLIP)
        assert t_out == pytest.approx(0.7)

    def test_flip_respects_asymmetric_range(self, monkeypatch):
        """非対称な [t_min, t_max] でも t_min + t_max - t が成り立つ。"""
        monkeypatch.setattr("eyeliner.data.augmentation.random.random", lambda: 0.0)
        _, t_out = augment(self._patch(), 0.5, t_min=0.2, t_max=1.1, aug_flags=AUG_FLIP)
        assert t_out == pytest.approx(0.2 + 1.1 - 0.5)

    def test_flip_horizontally_mirrors_patch(self, monkeypatch):
        """パッチが水平反転される。"""
        monkeypatch.setattr("eyeliner.data.augmentation.random.random", lambda: 0.0)
        patch = self._patch()
        out, _ = augment(patch, 0.5, aug_flags=AUG_FLIP)
        assert np.array_equal(out, patch[:, ::-1])

    def test_no_flip_preserves_t(self):
        """フラグ OFF のとき t は不変。"""
        _, t_out = augment(self._patch(), 0.3, aug_flags=AUG_NONE)
        assert t_out == pytest.approx(0.3)
