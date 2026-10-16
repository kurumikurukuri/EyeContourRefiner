"""utils/patch_utils.py のユニットテスト。

face-makeup ライブラリと共有のため、カバレッジ 100% を目標とする。
"""
from __future__ import annotations

import numpy as np
import pytest

from eye_contour.data.patch_utils import (
    PATCH_SIZE,
    T_MIN,
    T_MAX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    extract_patch,
    compute_t_from_point,
    reconstruct_point,
    build_display_patch,
    overlay_landmarks,
    get_norm_stats,
)


# ===========================================================
# extract_patch
# ===========================================================

class TestExtractPatch:

    def _make_image(self, h=480, w=640):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_output_shape(self):
        """出力パッチの shape が PATCH_SIZE と一致する。"""
        img = self._make_image()
        p = np.array([100.0, 100.0])
        q = np.array([200.0, 100.0])
        patch = extract_patch(img, p[0], p[1], q[0], q[1])
        assert patch.shape == (*PATCH_SIZE, 3)

    def test_output_dtype(self):
        """出力が float32 で値域が [0, 1] の範囲内。"""
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        patch = extract_patch(img, 100.0, 100.0, 200.0, 100.0)
        assert patch.dtype == np.float32
        assert patch.min() >= 0.0
        assert patch.max() <= 1.0 + 1e-6  # 浮動小数点の誤差許容

    def test_pq_same_returns_gray(self):
        """P == Q のとき、均一な 0.5 パッチを返す。"""
        img = self._make_image()
        patch = extract_patch(img, 100.0, 100.0, 100.0, 100.0)
        assert patch.shape == (*PATCH_SIZE, 3)
        np.testing.assert_allclose(patch, 0.5, atol=1e-6)

    def test_arbitrary_angle(self):
        """斜め方向の P-Q でも shape が正しい。"""
        img = self._make_image()
        patch = extract_patch(img, 100.0, 100.0, 150.0, 150.0)
        assert patch.shape == (*PATCH_SIZE, 3)

    def test_border_region_handled(self):
        """画像境界付近でも例外なく処理できる。"""
        img = self._make_image()
        patch = extract_patch(img, 5.0, 5.0, 10.0, 5.0)
        assert patch.shape == (*PATCH_SIZE, 3)


# ===========================================================
# compute_t_from_point / reconstruct_point 往復テスト
# ===========================================================

class TestRoundTrip:

    @pytest.mark.parametrize("t_input", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_roundtrip_parametric(self, t_input):
        """様々な t で compute_t → reconstruct の往復が成立する。"""
        px, py = 100.0, 100.0
        qx, qy = 200.0, 100.0
        rx = t_input * px + (1.0 - t_input) * qx
        ry = t_input * py + (1.0 - t_input) * qy

        t_calc = compute_t_from_point(px, py, qx, qy, rx, ry)
        r_reconstructed = reconstruct_point(px, py, qx, qy, t_calc)

        np.testing.assert_allclose(r_reconstructed[0], rx, atol=1e-5)
        np.testing.assert_allclose(r_reconstructed[1], ry, atol=1e-5)

    def test_roundtrip_diagonal(self):
        """斜め方向の P-Q でも往復テストが成立する。"""
        px, py = 100.0, 200.0
        qx, qy = 300.0, 400.0
        t_input = 0.3
        rx = t_input * px + (1.0 - t_input) * qx
        ry = t_input * py + (1.0 - t_input) * qy

        t_calc = compute_t_from_point(px, py, qx, qy, rx, ry)
        r_reconstructed = reconstruct_point(px, py, qx, qy, t_calc)

        np.testing.assert_allclose(r_reconstructed[0], rx, atol=1e-5)
        np.testing.assert_allclose(r_reconstructed[1], ry, atol=1e-5)


class TestComputeT:

    def test_midpoint_is_half(self):
        """P と Q の中点で t=0.5 になる。"""
        px, py = 100.0, 100.0
        qx, qy = 200.0, 100.0
        rx, ry = 150.0, 100.0
        t = compute_t_from_point(px, py, qx, qy, rx, ry)
        assert abs(t - 0.5) < 1e-6

    def test_at_p_returns_one(self):
        """R == P のとき t=1.0。"""
        px, py = 100.0, 100.0
        qx, qy = 200.0, 100.0
        t = compute_t_from_point(px, py, qx, qy, px, py)
        assert abs(t - 1.0) < 1e-6

    def test_at_q_returns_zero(self):
        """R == Q のとき t=0.0。"""
        px, py = 100.0, 100.0
        qx, qy = 200.0, 100.0
        t = compute_t_from_point(px, py, qx, qy, qx, qy)
        assert abs(t - 0.0) < 1e-6

    def test_clamp_below_t_min(self):
        """t < T_MIN の入力は T_MIN にクランプされる。"""
        px, py = 100.0, 100.0
        qx, qy = 200.0, 100.0
        # Q をはるかに超えた点
        rx, ry = 500.0, 100.0
        t = compute_t_from_point(px, py, qx, qy, rx, ry)
        assert t >= T_MIN

    def test_clamp_above_t_max(self):
        """t > T_MAX の入力は T_MAX にクランプされる。"""
        px, py = 100.0, 100.0
        qx, qy = 200.0, 100.0
        # P をはるかに超えた点
        rx, ry = -500.0, 100.0
        t = compute_t_from_point(px, py, qx, qy, rx, ry)
        assert t <= T_MAX

    def test_pq_same_returns_one(self):
        """P == Q のとき t=1.0 を返す（ゼロ除算防止）。"""
        t = compute_t_from_point(100.0, 100.0, 100.0, 100.0, 150.0, 150.0)
        assert t == 1.0


class TestReconstructPoint:

    def test_t_one_returns_p(self):
        """t=1.0 のとき R == P。"""
        px, py = 100.0, 200.0
        qx, qy = 300.0, 400.0
        rx, ry = reconstruct_point(px, py, qx, qy, 1.0)
        assert abs(rx - px) < 1e-10
        assert abs(ry - py) < 1e-10

    def test_t_zero_returns_q(self):
        """t=0.0 のとき R == Q。"""
        px, py = 100.0, 200.0
        qx, qy = 300.0, 400.0
        rx, ry = reconstruct_point(px, py, qx, qy, 0.0)
        assert abs(rx - qx) < 1e-10
        assert abs(ry - qy) < 1e-10


# ===========================================================
# build_display_patch
# ===========================================================

class TestBuildDisplayPatch:

    def test_output_shape(self):
        """出力 shape が PATCH_SIZE * scale になる。"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        scale = 6
        result = build_display_patch(img, 100.0, 100.0, 200.0, 100.0, display_scale=scale)
        h, w = PATCH_SIZE
        assert result.shape == (h * scale, w * scale, 3)

    def test_output_dtype_uint8(self):
        """出力が uint8 型。"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = build_display_patch(img, 100.0, 100.0, 200.0, 100.0)
        assert result.dtype == np.uint8


# ===========================================================
# overlay_landmarks
# ===========================================================

class TestOverlayLandmarks:

    def test_does_not_modify_input(self):
        """入力画像を変更しないことを確認（コピーを返す）。"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        original = img.copy()
        overlay_landmarks(img, 100.0, 100.0, 200.0, 100.0)
        np.testing.assert_array_equal(img, original)

    def test_output_shape_same(self):
        """出力 shape が入力と同じ。"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = overlay_landmarks(img, 100.0, 100.0, 200.0, 100.0)
        assert result.shape == img.shape

    def test_with_r_point(self):
        """R 点を指定しても例外なく処理できる。"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = overlay_landmarks(img, 100.0, 100.0, 200.0, 100.0, 150.0, 100.0)
        assert result.shape == img.shape


# ===========================================================
# get_norm_stats
# ===========================================================

class TestGetNormStats:

    def test_imagenet_returns_expected(self):
        """imagenet モードで ImageNet 統計が返る。"""
        mean, std = get_norm_stats("imagenet")
        assert mean == IMAGENET_MEAN
        assert std == IMAGENET_STD

    def test_none_returns_zeros_and_ones(self):
        """none モードで mean=0, std=1 が返る。"""
        mean, std = get_norm_stats("none")
        assert mean == [0.0, 0.0, 0.0]
        assert std == [1.0, 1.0, 1.0]

    def test_unknown_raises(self):
        """未知の norm_type は ValueError を送出する。"""
        with pytest.raises(ValueError):
            get_norm_stats("unknown_type")

    def test_dataset_without_json_raises(self):
        """dataset モードで JSON なしは FileNotFoundError を送出する。"""
        with pytest.raises(FileNotFoundError):
            get_norm_stats("dataset", stats_json_path=None)
