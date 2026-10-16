"""dataset_dual Branch B の t' 配線の単体テスト。

検証:
  1. invert_pq_rescale_t ∘ compute_pq_rescale が任意 t を round-trip 復元する。
  2. EyeContourPatchDualDataset の Branch B ラベルが label_space により切り替わる:
       "t_prime" → 正規化済み clamp(t')
       "raw_t"   → 正規化済み clamp(生t)
  3. 極端 t（生tは枝Bレンジ外）でも t_prime なら表現可能、raw_t は飽和する。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from eye_contour.data.patch_utils import compute_pq_rescale, invert_pq_rescale_t
from eye_contour.data.dataset_dual import BranchConfig, EyeContourPatchDualDataset


# ---------------------------------------------------------------------------
# 1. round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("t", [-2.454, -1.0, -0.3, 0.0, 0.5, 0.95, 1.2, 1.86])
def test_pq_rescale_t_roundtrip(t):
    px, py, qx, qy = 100.0, 200.0, 102.47, 200.0  # |PQ| = 2.47px（極小）
    rec = {"eye_scale": 150.0}
    _, _, _, _, t_prime, pq_prime = compute_pq_rescale(
        px, py, qx, qy, t, rec, "eye_scale", 0.2
    )
    pq_dist = math.hypot(qx - px, qy - py)
    t_back = invert_pq_rescale_t(t_prime, pq_dist, pq_prime)
    assert t_back == pytest.approx(t, abs=1e-9)


def test_pq_rescale_compresses_extreme_into_range():
    """|PQ|小 の極端 t が t' では [-0.25,1.25] 内に圧縮されること。"""
    px, py, qx, qy = 100.0, 200.0, 102.47, 200.0
    rec = {"eye_scale": 150.0}
    _, _, _, _, t_prime, _ = compute_pq_rescale(px, py, qx, qy, -2.454, rec, "eye_scale", 0.2)
    assert -0.25 <= t_prime <= 1.25  # 枝B レンジ内（生 -2.454 は枝Bレンジ外）


# ---------------------------------------------------------------------------
# 2 & 3. dataset_dual ラベル切替
# ---------------------------------------------------------------------------

def _make_rec(t: float, px=100.0, py=200.0, qx=102.47, qy=200.0, eye_scale=150.0) -> dict:
    pq_dist = math.hypot(qx - px, qy - py)
    return {
        "image_path": "dummy.jpg",
        "px": px, "py": py, "qx": qx, "qy": qy, "t": t,
        "sub_key": "upper", "point_index": 3,
        "pq_ratio": pq_dist / eye_scale, "eye_scale": eye_scale,
        "weight": 1.0, "val_weight": 1.0,
    }


def _make_dataset(rec: dict, label_space: str) -> EyeContourPatchDualDataset:
    cfg_a = BranchConfig(
        patch_t_min=-2.0, patch_t_max=3.0, label_t_min=-1.0, label_t_max=2.0,
        pq_rescale_mode="none", patch_size=(64, 64),
    )
    cfg_b = BranchConfig(
        patch_t_min=-0.5, patch_t_max=1.5, label_t_min=-0.25, label_t_max=1.25,
        pq_rescale_mode="eye_scale", pq_rescale_factor=0.2,
        label_space=label_space, patch_size=(64, 64),
    )
    ds = EyeContourPatchDualDataset([rec], cfg_a, cfg_b, augment_data=False, preload_images=False)
    # 画像 IO を避けてダミー画像を注入（ラベルは画像非依存）
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    ds._image_cache["dummy.jpg"] = img
    ds._img_diag_map["dummy.jpg"] = math.sqrt(100 * 100)
    return ds


def _expected_y_b(t: float, rec: dict, label_space: str) -> float:
    min_b, max_b, rng = -0.25, 1.25, 1.5
    pq_dist = math.hypot(rec["qx"] - rec["px"], rec["qy"] - rec["py"])
    xxx = max(rec["eye_scale"] * 0.2, 1.0)
    target = (0.5 + (t - 0.5) * pq_dist / xxx) if label_space == "t_prime" else t
    t_b = max(min_b, min(max_b, target))
    return (t_b - min_b) / rng


@pytest.mark.parametrize("t", [0.7, -0.3, -2.454, 1.86])
def test_branch_b_label_tprime(t):
    rec = _make_rec(t)
    ds = _make_dataset(rec, "t_prime")
    out = ds[0]
    y_b = float(out[5].item())
    assert y_b == pytest.approx(_expected_y_b(t, rec, "t_prime"), abs=1e-5)
    # pq_dist / pq_prime_dist が返ること
    pq_dist, pq_prime = float(out[12].item()), float(out[13].item())
    assert pq_dist == pytest.approx(math.hypot(rec["qx"] - rec["px"], rec["qy"] - rec["py"]), abs=1e-4)
    assert pq_prime == pytest.approx(max(rec["eye_scale"] * 0.2, 1.0), abs=1e-4)


@pytest.mark.parametrize("t", [0.7, -0.3, -2.454, 1.86])
def test_branch_b_label_raw_t(t):
    rec = _make_rec(t)
    ds = _make_dataset(rec, "raw_t")
    out = ds[0]
    y_b = float(out[5].item())
    assert y_b == pytest.approx(_expected_y_b(t, rec, "raw_t"), abs=1e-5)


def test_tprime_recovers_extreme_but_raw_saturates():
    """生 t=-2.454: raw_t は y_b=0 に飽和、t_prime は逆変換で -2.454 を復元できる。"""
    t = -2.454
    rec = _make_rec(t)

    # raw_t: 飽和（y_b == 0）
    out_raw = _make_dataset(rec, "raw_t")[0]
    assert float(out_raw[5].item()) == pytest.approx(0.0, abs=1e-6)

    # t_prime: 逆変換で復元
    out_tp = _make_dataset(rec, "t_prime")[0]
    y_b, pq_dist, pq_prime = (float(out_tp[5].item()), float(out_tp[12].item()), float(out_tp[13].item()))
    t_prime = y_b * 1.5 + (-0.25)
    t_recovered = invert_pq_rescale_t(t_prime, pq_dist, pq_prime)
    assert t_recovered == pytest.approx(t, abs=1e-4)
