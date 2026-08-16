"""eyeliner/data/source_degradation.py のテスト（EXP031）。

最重要の不変条件は **「ROI 切り出しがパッチ内容を変えないこと」**。これが崩れると
SD1〜SD3 で「劣化の効果」と「ROI 切り出しの副作用」が交絡し、paired t-test の解釈が
不能になる。

厳密なビット一致は成立しない。座標を平行移動すると warpAffine の固定小数点丸めが
変わるためで、実測（ランダム 6000 ケース）で最大 8/255・画素の 0.85% 以下。
そこで不変条件を 2 つに分けて検定する:

  1. footprint 含有性 — マージンを 16 倍にしてもパッチが 1 bit も変わらないこと（厳密）
  2. 再サンプリング誤差 — 直接抽出との差が実測上限 9/255 以内であること（許容付き）

さらに「劣化の効果 >> 再サンプリング誤差」を確認し、交絡がないことを担保する。
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from eyeliner.data.patch_utils import extract_patch
from eyeliner.data.source_degradation import (
    JPEG_BLOCK_PX,
    SourceDegradeConfig,
    SourceDegradeSample,
    compute_patch_source_bbox,
    degrade_region,
    sample_source_degradation,
)

# EXP031 の実構成（枝A / 枝B）
BRANCH_A = dict(t_min=-2.0, t_max=3.0, perp_scale=2.0, patch_size=(64, 64))
BRANCH_B = dict(t_min=-0.5, t_max=1.5, perp_scale=1.0, patch_size=(64, 64))


def _make_image(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    """再現可能な擬似テクスチャ画像（一様でないと補間差分が検出できない）。"""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _extract_via_bbox(
    image: np.ndarray,
    px: float, py: float, qx: float, qy: float,
    branch: dict,
    sample: SourceDegradeSample,
    margin_px: int = 8,
) -> np.ndarray:
    bbox = compute_patch_source_bbox(
        px, py, qx, qy, branch["t_min"], branch["t_max"], branch["perp_scale"],
        margin_px=margin_px,
    )
    assert bbox is not None
    degraded = degrade_region(image, bbox, sample)
    assert degraded is not None
    roi, x0, y0 = degraded
    return extract_patch(
        roi, px - x0, py - y0, qx - x0, qy - y0,
        patch_size=branch["patch_size"],
        t_min=branch["t_min"], t_max=branch["t_max"], perp_scale=branch["perp_scale"],
    )


def _extract_direct(
    image: np.ndarray,
    px: float, py: float, qx: float, qy: float,
    branch: dict,
) -> np.ndarray:
    return extract_patch(
        image, px, py, qx, qy,
        patch_size=branch["patch_size"],
        t_min=branch["t_min"], t_max=branch["t_max"], perp_scale=branch["perp_scale"],
    )


# ---------------------------------------------------------------------------
# 不変条件: bbox は footprint を完全に含む
# ---------------------------------------------------------------------------
#
# ROI 切り出しは座標を (x0, y0) 平行移動するため、warpAffine の固定小数点座標の
# 丸めが変わり、画素の 0.1% 前後に差が出る。これは footprint 不足ではないので、
# 含有性の検定は「マージンを 16 倍にしてもパッチが 1 bit も変わらないこと」で行う
# （もし footprint が足りていなければ、マージンを増やせば結果が変わるはず）。

PQ_CASES = [
    (320.0, 240.0, 329.0, 241.0),    # |PQ|≈9（DB 中央値相当）、ほぼ水平
    (300.0, 200.0, 304.0, 205.0),    # |PQ|≈6.4、斜め
    (320.0, 240.0, 341.0, 236.0),    # |PQ|≈21（p95 相当）
    (100.0, 100.0, 92.0, 112.0),     # 逆向き
    (4.0, 4.0, 12.0, 6.0),           # 左上端: bbox が画像外へはみ出す
    (636.0, 476.0, 628.0, 470.0),    # 右下端
]
PQ_IDS = ["median", "diagonal", "large", "reversed", "top_left_edge", "bottom_right_edge"]

# warpAffine の固定小数点丸めによる差の上限。
# ランダム 6000 ケース（画像端・画像外を含む）の実測最大が 8/255 だったため 9/255 とする。
RESAMPLE_TOL = 9.0 / 255.0


@pytest.mark.parametrize("branch", [BRANCH_A, BRANCH_B], ids=["branch_a", "branch_b"])
@pytest.mark.parametrize("pq", PQ_CASES, ids=PQ_IDS)
def test_bbox_contains_full_footprint(branch: dict, pq: tuple) -> None:
    """マージンを 8→128 に増やしてもパッチが変わらない ＝ footprint を含みきっている。"""
    image = _make_image()
    px, py, qx, qy = pq
    tight = _extract_via_bbox(image, px, py, qx, qy, branch, SourceDegradeSample(), margin_px=8)
    loose = _extract_via_bbox(image, px, py, qx, qy, branch, SourceDegradeSample(), margin_px=128)
    np.testing.assert_array_equal(tight, loose)


@pytest.mark.parametrize("branch", [BRANCH_A, BRANCH_B], ids=["branch_a", "branch_b"])
@pytest.mark.parametrize("pq", PQ_CASES, ids=PQ_IDS)
def test_identity_matches_direct_within_resampling_tolerance(branch: dict, pq: tuple) -> None:
    """劣化なしなら ROI 経由と直接抽出は再サンプリング誤差の範囲で一致する。"""
    image = _make_image()
    px, py, qx, qy = pq
    direct = _extract_direct(image, px, py, qx, qy, branch)
    via_bbox = _extract_via_bbox(image, px, py, qx, qy, branch, SourceDegradeSample())
    assert via_bbox.shape == direct.shape
    np.testing.assert_allclose(via_bbox, direct, atol=RESAMPLE_TOL)
    mismatch_frac = float((via_bbox != direct).mean())
    assert mismatch_frac < 0.01, f"差分画素が多すぎる: {mismatch_frac:.4%}"


@pytest.mark.parametrize(
    "sample",
    [
        SourceDegradeSample(jpeg_quality=30),
        SourceDegradeSample(res_scale=2.5),
        SourceDegradeSample(jpeg_quality=30, res_scale=2.5),
    ],
    ids=["jpeg", "res", "both"],
)
def test_degradation_dominates_resampling_artifact(sample: SourceDegradeSample) -> None:
    """劣化の効果が再サンプリング誤差より 1 桁以上大きいことを確認する。

    これが成り立たないと、SD1〜SD3 の差が「劣化の効果」なのか「ROI 切り出しの
    副作用」なのか区別できなくなる。
    """
    image = _make_image()
    px, py, qx, qy = 320.0, 240.0, 329.0, 241.0
    direct = _extract_direct(image, px, py, qx, qy, BRANCH_A)
    identity = _extract_via_bbox(image, px, py, qx, qy, BRANCH_A, SourceDegradeSample())
    degraded = _extract_via_bbox(image, px, py, qx, qy, BRANCH_A, sample)

    artifact = float(np.abs(identity - direct).mean())
    effect = float(np.abs(degraded - identity).mean())
    assert effect > 10 * artifact, f"劣化の効果={effect:.6f} が再サンプリング誤差={artifact:.6f} に対し小さすぎる"


# ---------------------------------------------------------------------------
# 劣化が実際に効いていること
# ---------------------------------------------------------------------------

def test_jpeg_changes_patch() -> None:
    image = _make_image()
    px, py, qx, qy = 320.0, 240.0, 329.0, 241.0
    direct = _extract_direct(image, px, py, qx, qy, BRANCH_A)
    degraded = _extract_via_bbox(
        image, px, py, qx, qy, BRANCH_A, SourceDegradeSample(jpeg_quality=30),
    )
    assert degraded.shape == direct.shape
    assert not np.array_equal(degraded, direct)


def test_res_jitter_changes_patch() -> None:
    image = _make_image()
    px, py, qx, qy = 320.0, 240.0, 329.0, 241.0
    direct = _extract_direct(image, px, py, qx, qy, BRANCH_A)
    degraded = _extract_via_bbox(
        image, px, py, qx, qy, BRANCH_A, SourceDegradeSample(res_scale=2.5),
    )
    assert degraded.shape == direct.shape
    assert not np.array_equal(degraded, direct)


def test_degrade_region_does_not_mutate_source() -> None:
    image = _make_image()
    before = image.copy()
    bbox = compute_patch_source_bbox(320.0, 240.0, 329.0, 241.0, -2.0, 3.0, 2.0)
    assert bbox is not None
    assert degrade_region(image, bbox, SourceDegradeSample(jpeg_quality=40, res_scale=2.0)) is not None
    np.testing.assert_array_equal(image, before)


# ---------------------------------------------------------------------------
# bbox の性質
# ---------------------------------------------------------------------------

def test_bbox_origin_is_jpeg_block_aligned() -> None:
    image = _make_image()
    for px, py, qx, qy in [(321.0, 243.0, 330.0, 244.0), (117.0, 89.0, 125.0, 95.0)]:
        bbox = compute_patch_source_bbox(px, py, qx, qy, -2.0, 3.0, 2.0)
        assert bbox is not None
        x0, y0, _, _ = bbox
        assert x0 % JPEG_BLOCK_PX == 0
        assert y0 % JPEG_BLOCK_PX == 0


def test_bbox_none_when_p_equals_q() -> None:
    assert compute_patch_source_bbox(10.0, 10.0, 10.0, 10.0, -2.0, 3.0, 2.0) is None


# ---------------------------------------------------------------------------
# 設定の検証とサンプリング
# ---------------------------------------------------------------------------

def test_disabled_config_is_not_enabled() -> None:
    assert not SourceDegradeConfig().enabled
    assert SourceDegradeConfig(jpeg=True).enabled
    assert SourceDegradeConfig(res_jitter=True).enabled


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(jpeg_prob=1.5),
        dict(res_jitter_prob=-0.1),
        dict(jpeg_quality_min=50, jpeg_quality_max=40),
        dict(jpeg_quality_min=0),
        dict(res_scale_min=0.5),
        dict(res_scale_min=2.0, res_scale_max=1.5),
    ],
)
def test_validate_rejects_bad_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SourceDegradeConfig(**kwargs).validate()


def test_sample_respects_disabled_flags() -> None:
    rng = random.Random(0)
    cfg = SourceDegradeConfig(jpeg=True, jpeg_prob=1.0, res_jitter=False)
    for _ in range(20):
        s = sample_source_degradation(cfg, rng)
        assert s.jpeg_quality is not None
        assert s.res_scale is None


def test_sample_is_deterministic_for_same_seed() -> None:
    cfg = SourceDegradeConfig(jpeg=True, res_jitter=True)
    a = [sample_source_degradation(cfg, random.Random(7)) for _ in range(1)]
    b = [sample_source_degradation(cfg, random.Random(7)) for _ in range(1)]
    assert a == b


def test_sample_quality_within_range() -> None:
    rng = random.Random(1)
    cfg = SourceDegradeConfig(jpeg=True, jpeg_prob=1.0, jpeg_quality_min=30, jpeg_quality_max=90)
    for _ in range(200):
        s = sample_source_degradation(cfg, rng)
        assert 30 <= s.jpeg_quality <= 90


def test_sample_scale_within_range_or_none() -> None:
    rng = random.Random(2)
    cfg = SourceDegradeConfig(res_jitter=True, res_jitter_prob=1.0,
                              res_scale_min=1.0, res_scale_max=2.5)
    for _ in range(200):
        s = sample_source_degradation(cfg, rng)
        assert s.res_scale is None or 1.0 < s.res_scale <= 2.5
