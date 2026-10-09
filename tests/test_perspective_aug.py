"""EXP027 面外回転（pitch/yaw）透視変換オーグメンテーションのテスト。

Phase 1: homography_from_pitch_yaw / warp_pq_with_homography / extract_patch(H=...)
Phase 2: expand_perspective_records / PERSPECTIVE_PATTERNS_8
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from eyeliner.data.patch_utils import (
    PATCH_SIZE,
    T_MIN,
    T_MAX,
    compute_t_from_point,
    extract_patch,
    homography_from_pitch_yaw,
    reconstruct_point,
    warp_pq_with_homography,
)
from eyeliner.training.dual_kfold import (
    PERSPECTIVE_PATTERNS_8,
    expand_perspective_records,
)


# ===========================================================
# homography_from_pitch_yaw
# ===========================================================

class TestHomographyFromPitchYaw:
    """homography_from_pitch_yaw のユニットテスト。"""

    def test_returns_3x3_matrix(self):
        """出力は (3, 3) 行列。"""
        H = homography_from_pitch_yaw(5.0, 0.0, cx=150.0, cy=200.0, eye_scale=100.0)
        assert H.shape == (3, 3)

    def test_identity_at_zero_angles(self):
        """dp=0, dy=0 → H = I（単位行列）。"""
        H = homography_from_pitch_yaw(0.0, 0.0, cx=150.0, cy=200.0, eye_scale=100.0)
        np.testing.assert_allclose(H, np.eye(3), atol=1e-10)

    def test_nonzero_pitch_gives_nonidentity(self):
        """dp≠0 → H ≠ I。"""
        H = homography_from_pitch_yaw(5.0, 0.0, cx=150.0, cy=200.0, eye_scale=100.0)
        assert not np.allclose(H, np.eye(3), atol=1e-3)

    def test_nonzero_yaw_gives_nonidentity(self):
        """dy≠0 → H ≠ I。"""
        H = homography_from_pitch_yaw(0.0, 5.0, cx=150.0, cy=200.0, eye_scale=100.0)
        assert not np.allclose(H, np.eye(3), atol=1e-3)

    def test_h_is_invertible(self):
        """H は可逆行列（det ≠ 0）。"""
        H = homography_from_pitch_yaw(5.0, 3.0, cx=150.0, cy=200.0, eye_scale=100.0)
        assert abs(np.linalg.det(H)) > 1e-6

    def test_bottom_row_nonzero_for_nonzero_rotation(self):
        """非ゼロ回転では H の第3行が [0, 0, 1] でない（真の透視変換）。"""
        H = homography_from_pitch_yaw(0.0, 5.0, cx=150.0, cy=200.0, eye_scale=100.0)
        # 透視変換では第3行に非零の h31 または h32 が入る
        assert not np.allclose(H[2, :], [0.0, 0.0, 1.0], atol=1e-6)

    def test_alpha_scales_focal_length(self):
        """alpha が 2 倍になると透視効果が変わる（H が変わる）。"""
        H1 = homography_from_pitch_yaw(5.0, 0.0, cx=150.0, cy=200.0, eye_scale=100.0, alpha=2.0)
        H2 = homography_from_pitch_yaw(5.0, 0.0, cx=150.0, cy=200.0, eye_scale=100.0, alpha=4.0)
        assert not np.allclose(H1, H2, atol=1e-6)

    @pytest.mark.parametrize("dp,dy,alpha", [
        (5.0, 0.0, 0.5), (0.0, 5.0, 1.0), (5.0, 5.0, 2.0), (-5.0, 3.0, 10.0), (3.54, -3.54, 100.0),
    ])
    def test_pivot_is_fixed_point(self, dp, dy, alpha):
        """(cx, cy) は常に H の不動点（顔パッチは自分の中心を軸に回転するモデル）。

        カメラ自体が回転するモデル（H = K・R・K⁻¹）は、この不動点性を満たさない
        （回転角固定でも alpha に比例して中心が無限にドリフトしてしまう）。
        """
        cx, cy = 150.0, 200.0
        H = homography_from_pitch_yaw(dp, dy, cx=cx, cy=cy, eye_scale=100.0, alpha=alpha)
        v = H @ np.array([cx, cy, 1.0])
        x2, y2 = v[0] / v[2], v[1] / v[2]
        assert x2 == pytest.approx(cx, abs=1e-6)
        assert y2 == pytest.approx(cy, abs=1e-6)

    def test_large_alpha_converges_to_pure_affine(self):
        """alpha → ∞ で第3行が [0, 0, 1] に収束する（透視要素が消え、純粋なアフィンに近づく）。

        「望遠・orthographic に近づく」という設計意図は、ピボット固定を前提として
        初めて意味を持つ（ピボット自体が動いてしまうモデルでは alpha を大きくしても
        歪みは収まらない）。
        """
        cx, cy, eye_scale = 150.0, 200.0, 100.0
        H_small = homography_from_pitch_yaw(0.0, 5.0, cx, cy, eye_scale, alpha=1.0)
        H_large = homography_from_pitch_yaw(0.0, 5.0, cx, cy, eye_scale, alpha=1000.0)
        perspective_term_small = abs(H_small[2, 0]) + abs(H_small[2, 1] - 0.0)
        perspective_term_large = abs(H_large[2, 0]) + abs(H_large[2, 1] - 0.0)
        assert perspective_term_large < perspective_term_small / 100

    def test_large_alpha_local_scale_converges_to_cosine(self):
        """alpha → ∞ でピボット近傍の水平スケールが cos(yaw) に収束する（純粋な affine squash）。"""
        cx, cy, eye_scale = 150.0, 200.0, 100.0
        dy_deg = 5.0
        H = homography_from_pitch_yaw(0.0, dy_deg, cx, cy, eye_scale, alpha=100000.0)
        v0 = H @ np.array([cx, cy, 1.0])
        v1 = H @ np.array([cx + eye_scale, cy, 1.0])
        x0, x1 = v0[0] / v0[2], v1[0] / v1[2]
        local_scale = (x1 - x0) / eye_scale
        assert local_scale == pytest.approx(math.cos(math.radians(dy_deg)), abs=1e-3)


# ===========================================================
# warp_pq_with_homography
# ===========================================================

class TestWarpPQWithHomography:
    """warp_pq_with_homography のユニットテスト。"""

    def test_identity_h_preserves_all_points(self):
        """H=I のとき P, Q, R がそのまま返る。"""
        H = np.eye(3, dtype=np.float64)
        px, py, qx, qy, rx, ry = 100.0, 200.0, 200.0, 200.0, 150.0, 200.0
        out = warp_pq_with_homography(H, px, py, qx, qy, rx, ry)
        assert out == pytest.approx((px, py, qx, qy, rx, ry), abs=1e-9)

    def test_returns_6_floats(self):
        """戻り値は (px', py', qx', qy', rx', ry') の 6 要素タプル。"""
        H = np.eye(3, dtype=np.float64)
        out = warp_pq_with_homography(H, 100.0, 200.0, 200.0, 200.0, 150.0, 200.0)
        assert len(out) == 6
        for v in out:
            assert isinstance(v, float)

    def test_t_prime_equals_t_for_identity_h(self):
        """H=I のとき t' == t（同一性不変）。"""
        H = np.eye(3, dtype=np.float64)
        px, py = 100.0, 200.0
        qx, qy = 200.0, 200.0
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            rx, ry = reconstruct_point(px, py, qx, qy, t)
            px_w, py_w, qx_w, qy_w, rx_w, ry_w = warp_pq_with_homography(
                H, px, py, qx, qy, rx, ry
            )
            t_prime = compute_t_from_point(px_w, py_w, qx_w, qy_w, rx_w, ry_w)
            assert abs(t_prime - t) < 1e-9, f"t={t}: t'={t_prime} != t"

    def test_t_prime_differs_for_yaw_rotation(self):
        """H≠I（yaw回転）のとき t' ≠ t（透視変換は比率非保存）。"""
        cx, cy = 150.0, 200.0
        eye_scale = 100.0
        H = homography_from_pitch_yaw(0.0, 5.0, cx=cx, cy=cy, eye_scale=eye_scale)

        # P と Q を眼中心の左右に対称に置く
        px, py = 100.0, 200.0
        qx, qy = 200.0, 200.0
        t = 0.5
        rx, ry = reconstruct_point(px, py, qx, qy, t)

        px_w, py_w, qx_w, qy_w, rx_w, ry_w = warp_pq_with_homography(
            H, px, py, qx, qy, rx, ry
        )
        t_prime = compute_t_from_point(px_w, py_w, qx_w, qy_w, rx_w, ry_w)

        # 5° の yaw 回転では t' は t から有意にズレる（比率非保存）
        assert abs(t_prime - t) > 1e-3, f"t' と t の差が小さすぎる: t={t}, t'={t_prime}"

    def test_t_prime_differs_for_pitch_rotation(self):
        """H≠I（pitch回転）のとき t' ≠ t。"""
        cx, cy = 150.0, 200.0
        eye_scale = 100.0
        H = homography_from_pitch_yaw(5.0, 0.0, cx=cx, cy=cy, eye_scale=eye_scale)

        # 斜め方向の P-Q で確認
        px, py = 100.0, 180.0
        qx, qy = 200.0, 220.0
        t = 0.5
        rx, ry = reconstruct_point(px, py, qx, qy, t)

        px_w, py_w, qx_w, qy_w, rx_w, ry_w = warp_pq_with_homography(
            H, px, py, qx, qy, rx, ry
        )
        t_prime = compute_t_from_point(px_w, py_w, qx_w, qy_w, rx_w, ry_w)
        assert abs(t_prime - t) > 1e-3, f"t={t}: t'={t_prime}"


# ===========================================================
# extract_patch(H=...)
# ===========================================================

class TestExtractPatchWithHomography:
    """extract_patch の H パラメータのテスト。"""

    @pytest.fixture
    def test_image(self):
        np.random.seed(42)
        return np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

    def test_h_none_matches_original_api(self, test_image):
        """H=None のとき既存の extract_patch と出力が同一（後方互換）。"""
        px, py = 200.0, 240.0
        qx, qy = 300.0, 240.0
        patch_original = extract_patch(test_image, px, py, qx, qy)
        patch_h_none = extract_patch(test_image, px, py, qx, qy, H=None)
        np.testing.assert_array_equal(patch_original, patch_h_none)

    def test_identity_h_matches_no_h(self, test_image):
        """H=I のとき H=None と数値的にほぼ同一のパッチが返る。

        warpAffine と warpPerspective は実装が異なるため uint8 量子化ステップ（≈1/255）
        程度の差が生じる。許容値は 5/255 ≈ 0.02 とする。
        """
        px, py = 200.0, 240.0
        qx, qy = 300.0, 240.0
        H = np.eye(3, dtype=np.float64)

        patch_no_h = extract_patch(test_image, px, py, qx, qy)
        patch_with_h = extract_patch(test_image, px, py, qx, qy, H=H)

        np.testing.assert_allclose(patch_no_h, patch_with_h, atol=0.02)

    def test_output_shape_unchanged_with_h(self, test_image):
        """H 指定時も出力 shape は PATCH_SIZE と同一。"""
        px, py = 250.0, 240.0
        qx, qy = 350.0, 240.0
        H = homography_from_pitch_yaw(5.0, 3.0, cx=300.0, cy=240.0, eye_scale=100.0)
        patch = extract_patch(test_image, px, py, qx, qy, H=H)
        assert patch.shape == (*PATCH_SIZE, 3)

    def test_output_dtype_float32_with_h(self, test_image):
        """H 指定時も出力が float32 で [0, 1] 範囲内。"""
        px, py = 250.0, 240.0
        qx, qy = 350.0, 240.0
        H = homography_from_pitch_yaw(5.0, 0.0, cx=300.0, cy=240.0, eye_scale=100.0)
        patch = extract_patch(test_image, px, py, qx, qy, H=H)
        assert patch.dtype == np.float32
        assert patch.min() >= 0.0
        assert patch.max() <= 1.0 + 1e-6

    def test_nonidentity_h_gives_different_patch(self, test_image):
        """H≠I のとき H=None と異なるパッチが返る。"""
        px, py = 250.0, 240.0
        qx, qy = 350.0, 240.0
        cx, cy = (px + qx) / 2, (py + qy) / 2

        H = homography_from_pitch_yaw(5.0, 0.0, cx=cx, cy=cy, eye_scale=100.0)
        t = 0.5
        rx, ry = reconstruct_point(px, py, qx, qy, t)
        px_w, py_w, qx_w, qy_w, _, _ = warp_pq_with_homography(
            H, px, py, qx, qy, rx, ry
        )

        patch_orig = extract_patch(test_image, px_w, py_w, qx_w, qy_w)
        patch_persp = extract_patch(test_image, px_w, py_w, qx_w, qy_w, H=H)

        assert not np.allclose(patch_orig, patch_persp, atol=0.01), \
            "H≠I なのにパッチが同一"

    def test_fused_warp_formula_correct_with_linear_image(self):
        """M_fused = M_aff @ H の変換式を線形画像で検証する。

        img[y, x, 0] = x / (W-1) の線形画像では、バイリニア補間が正確に x_src / (W-1) を
        返す（y 依存なし、線形のため）。これにより src_x = (M_fused⁻¹ · (u,v,1))[0] / w
        が cv2 の実際のサンプル位置と一致することを確認できる。
        """
        import math
        W_img, H_img = 1024, 768

        # Channel 0: x / (W-1)  → bilinear で x_src / (W-1) になる
        # Channel 1: y / (H-1)  → bilinear で y_src / (H-1) になる
        xs = np.linspace(0.0, 1.0, W_img, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, H_img, dtype=np.float32)
        img_lin = np.zeros((H_img, W_img, 3), dtype=np.float32)
        img_lin[:, :, 0] = xs[None, :]  # x-gradient
        img_lin[:, :, 1] = ys[:, None]  # y-gradient
        img_lin[:, :, 2] = 0.5

        px, py = 400.0, 384.0
        qx, qy = 450.0, 384.0   # |PQ| = 50
        eye_scale = 50.0
        cx = (px + qx) / 2      # 425.0
        cy = (py + qy) / 2      # 384.0
        H = homography_from_pitch_yaw(5.0, 3.0, cx, cy, eye_scale, alpha=2.0)

        t = 0.5
        rx, ry = reconstruct_point(px, py, qx, qy, t)
        px_w, py_w, qx_w, qy_w, _, _ = warp_pq_with_homography(
            H, px, py, qx, qy, rx, ry
        )

        # M_aff と M_fused を手動計算（extract_patch 内部と同一ロジック）
        ph, pw = PATCH_SIZE
        dx = qx_w - px_w
        dy = qy_w - py_w
        norm = math.hypot(dx, dy)
        ux, uy = dx / norm, dy / norm
        scale = (pw - 1) / ((T_MAX - T_MIN) * norm)
        sx = T_MAX * px_w + (1.0 - T_MAX) * qx_w
        sy = T_MAX * py_w + (1.0 - T_MAX) * qy_w
        ap = (ph - 1) / (1.0 * norm)
        M_aff_f64 = np.array([
            [scale * ux, scale * uy, -scale * (ux * sx + uy * sy)],
            [-uy * ap, ux * ap, ap * (uy * sx - ux * sy) + ph // 2],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        M_fused = M_aff_f64 @ H

        # fused warp を直接 cv2.warpPerspective で実行（extract_patch は /255 するため直接使わない）
        ph_s, pw_s = PATCH_SIZE
        patch = cv2.warpPerspective(img_lin, M_fused, (pw_s, ph_s), flags=cv2.INTER_LINEAR)

        # 期待値: M_fused⁻¹ @ (u, v, 1) / w で得られる (x_src, y_src)
        # line 0: x_src / (W-1), ch 1: y_src / (H-1)
        M_fused_inv = np.linalg.inv(M_fused)
        check_pixels = [(pw // 2, ph // 2), (pw // 4, ph // 4), (3 * pw // 4, 3 * ph // 4)]
        for u, v in check_pixels:
            homog = M_fused_inv @ np.array([u, v, 1.0], dtype=np.float64)
            x_src = homog[0] / homog[2]
            y_src = homog[1] / homog[2]
            expected_x = x_src / (W_img - 1)
            expected_y = y_src / (H_img - 1)
            # 線形画像でのバイリニア補間は理論値と float32 精度 (atol≈0.001) で一致
            assert abs(float(patch[v, u, 0]) - expected_x) < 0.002, (
                f"({u},{v}): ch0 expected {expected_x:.6f}, got {patch[v,u,0]:.6f}"
            )
            assert abs(float(patch[v, u, 1]) - expected_y) < 0.002, (
                f"({u},{v}): ch1 expected {expected_y:.6f}, got {patch[v,u,1]:.6f}"
            )

    def test_pq_same_with_h_returns_gray(self, test_image):
        """P==Q のとき H 有無に関わらず 0.5 パッチ（既存挙動を継承）。"""
        H = homography_from_pitch_yaw(5.0, 0.0, cx=150.0, cy=200.0, eye_scale=100.0)
        patch = extract_patch(test_image, 200.0, 200.0, 200.0, 200.0, H=H)
        np.testing.assert_allclose(patch, 0.5, atol=1e-6)


# ===========================================================
# PERSPECTIVE_PATTERNS_8 と expand_perspective_records
# ===========================================================

class TestPerspectivePatterns8:
    """PERSPECTIVE_PATTERNS_8 定数のテスト。"""

    def test_has_8_patterns(self):
        """8 パターンあること。"""
        assert len(PERSPECTIVE_PATTERNS_8) == 8

    def test_axis_patterns_present(self):
        """軸 4 方向（±5°, 0）が含まれる。"""
        axis = [(5.0, 0.0), (-5.0, 0.0), (0.0, 5.0), (0.0, -5.0)]
        for dp, dy in axis:
            assert any(
                math.isclose(p[0], dp, abs_tol=1e-9) and math.isclose(p[1], dy, abs_tol=1e-9)
                for p in PERSPECTIVE_PATTERNS_8
            ), f"軸パターン ({dp}, {dy}) が見つからない"

    def test_diagonal_patterns_present(self):
        """対角 4 方向（±5/√2°）が含まれる。"""
        d = 5.0 / math.sqrt(2)
        diag = [(d, d), (d, -d), (-d, d), (-d, -d)]
        for dp, dy in diag:
            assert any(
                math.isclose(p[0], dp, abs_tol=1e-9) and math.isclose(p[1], dy, abs_tol=1e-9)
                for p in PERSPECTIVE_PATTERNS_8
            ), f"対角パターン ({dp:.4f}, {dy:.4f}) が見つからない"

    def test_diagonal_magnitude_is_5deg(self):
        """対角パターンの合成角度が ≈5°（各軸 5/√2°）。"""
        d = 5.0 / math.sqrt(2)
        magnitude = math.sqrt(d ** 2 + d ** 2)
        assert math.isclose(magnitude, 5.0, abs_tol=1e-9)


class TestExpandPerspectiveRecords:
    """expand_perspective_records のユニットテスト。"""

    def test_empty_patterns_returns_base_unchanged(self):
        """patterns=[] のとき base records をそのまま返す。"""
        records = [{"image_path": "a.jpg", "t": 0.5}]
        result = expand_perspective_records(records, patterns=[])
        assert result == records

    def test_expansion_count_single_record(self):
        """1 record + 8 patterns → 9 records（1 base + 8 複製）。"""
        records = [{"image_path": "a.jpg"}]
        result = expand_perspective_records(records, PERSPECTIVE_PATTERNS_8)
        assert len(result) == 1 + 8

    def test_expansion_count_multiple_records(self):
        """N records + 8 patterns → N × 9 records。"""
        n = 5
        records = [{"image_path": f"{i}.jpg"} for i in range(n)]
        result = expand_perspective_records(records, PERSPECTIVE_PATTERNS_8)
        assert len(result) == n * (1 + len(PERSPECTIVE_PATTERNS_8))

    def test_base_record_has_no_persp_tag(self):
        """展開後の最初のレコード（base）に _persp タグがない。"""
        records = [{"image_path": "a.jpg"}]
        result = expand_perspective_records(records, PERSPECTIVE_PATTERNS_8)
        assert "_persp_dp" not in result[0]
        assert "_persp_dy" not in result[0]

    def test_aug_records_have_persp_tags(self):
        """aug レコードに _persp_dp / _persp_dy がある。"""
        records = [{"image_path": "a.jpg"}]
        result = expand_perspective_records(records, PERSPECTIVE_PATTERNS_8)
        for aug_rec in result[1:]:
            assert "_persp_dp" in aug_rec, "aug レコードに _persp_dp がない"
            assert "_persp_dy" in aug_rec, "aug レコードに _persp_dy がない"

    def test_all_8_patterns_in_result(self):
        """8 パターンが全て展開後に存在する。"""
        records = [{"image_path": "a.jpg"}]
        result = expand_perspective_records(records, PERSPECTIVE_PATTERNS_8)
        found = [(r["_persp_dp"], r["_persp_dy"]) for r in result if "_persp_dp" in r]
        assert len(found) == 8
        for dp, dy in PERSPECTIVE_PATTERNS_8:
            assert any(
                math.isclose(f[0], dp, abs_tol=1e-9) and math.isclose(f[1], dy, abs_tol=1e-9)
                for f in found
            ), f"パターン ({dp:.4f}, {dy:.4f}) が展開後に見つからない"

    def test_base_record_content_preserved_in_aug_copies(self):
        """aug レコードは base の全フィールドを保持（shallow copy）。"""
        records = [{"image_path": "a.jpg", "t": 0.7, "eye_scale": 100.0}]
        result = expand_perspective_records(records, PERSPECTIVE_PATTERNS_8)
        for r in result:
            assert r["image_path"] == "a.jpg"
            assert r["t"] == 0.7
            assert r["eye_scale"] == 100.0

    def test_aug_rec_is_independent_from_base(self):
        """aug レコードを変更しても base レコードに影響しない（shallow copy）。"""
        base = {"image_path": "a.jpg", "t": 0.5}
        records = [base]
        result = expand_perspective_records(records, PERSPECTIVE_PATTERNS_8)
        # aug レコードを変更
        result[1]["t"] = 999.0
        # base は不変
        assert base["t"] == 0.5

    def test_interleaved_order(self):
        """展開結果は [base1, aug1_1, ..., aug1_8, base2, aug2_1, ..., aug2_8] の順。"""
        records = [{"id": 0}, {"id": 1}]
        result = expand_perspective_records(records, PERSPECTIVE_PATTERNS_8)
        # 最初の 9 件: base0 + 8 aug of base0
        assert result[0]["id"] == 0
        assert "_persp_dp" not in result[0]
        for r in result[1:9]:
            assert r["id"] == 0
            assert "_persp_dp" in r
        # 次の 9 件: base1 + 8 aug of base1
        assert result[9]["id"] == 1
        assert "_persp_dp" not in result[9]
        for r in result[10:18]:
            assert r["id"] == 1
            assert "_persp_dp" in r
