"""dataset_dual への PQ 回転摂動（EXP035）組み込みのテスト。"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from eye_contour.data.augmentation import AUG_PQ_JITTER
from eye_contour.data.dataset_dual import BranchConfig, EyeContourPatchDualDataset

REC = {
    "image_path": "dummy.jpg",
    "px": 70.0, "py": 60.0, "qx": 50.0, "qy": 62.0, "t": 0.4,
    "sub_key": "upper", "point_index": 3,
    "pq_ratio": 0.13, "eye_scale": 150.0,
    "weight": 1.0, "val_weight": 1.0,
}


def _make_dataset(**kwargs) -> EyeContourPatchDualDataset:
    cfg_a = BranchConfig(
        patch_t_min=-2.0, patch_t_max=3.0, label_t_min=-1.0, label_t_max=2.0,
        pq_rescale_mode="none", patch_size=(32, 32),
    )
    cfg_b = BranchConfig(
        patch_t_min=-0.5, patch_t_max=1.5, label_t_min=-0.25, label_t_max=1.25,
        pq_rescale_mode="eye_scale", pq_rescale_factor=0.2,
        label_space="t_prime", patch_size=(32, 32),
    )
    ds = EyeContourPatchDualDataset([dict(REC)], cfg_a, cfg_b, preload_images=False, **kwargs)
    rng = np.random.default_rng(0)
    ds._image_cache["dummy.jpg"] = rng.integers(0, 255, (120, 140, 3), dtype=np.uint8)
    ds._img_diag_map["dummy.jpg"] = math.sqrt(120 * 140)
    return ds


def test_constructor_defaults_leave_jitter_disabled():
    """pq_jitter 未指定なら、AUG_PQ_JITTER が立っていても何も起きない（既存実験を変えない）。"""
    a = _make_dataset(augment_data=True, aug_flags=AUG_PQ_JITTER)[0]
    b = _make_dataset(augment_data=False)[0]
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_jitter_changes_patches_but_not_labels():
    base = _make_dataset(augment_data=False)[0]
    ds = _make_dataset(
        augment_data=True, aug_flags=AUG_PQ_JITTER,
        pq_jitter_rot_deg=10.0, pq_jitter_prob=1.0, pq_jitter_seed=5,
    )
    out = ds[0]
    # パッチ（x_a, x_b）は回転で変わる
    assert not torch.equal(out[0], base[0])
    assert not torch.equal(out[1], base[1])
    # ラベル y_a, y_b・|PQ| は不変（t 不変・回転は長さ保存）
    assert torch.allclose(out[4], base[4])
    assert torch.allclose(out[5], base[5])
    assert torch.allclose(out[12], base[12], atol=1e-4)
    assert torch.allclose(out[13], base[13], atol=1e-4)


def test_flag_off_ignores_jitter_params():
    ds = _make_dataset(
        augment_data=True, aug_flags=0,
        pq_jitter_rot_deg=10.0, pq_jitter_prob=1.0,
    )
    base = _make_dataset(augment_data=False)[0]
    out = ds[0]
    assert torch.equal(out[0], base[0]) and torch.equal(out[1], base[1])


def test_eval_dataset_never_jitters():
    """augment_data=False（val / hold-out / 推論）では摂動しない。"""
    ds = _make_dataset(
        augment_data=False, aug_flags=AUG_PQ_JITTER,
        pq_jitter_rot_deg=10.0, pq_jitter_prob=1.0,
    )
    base = _make_dataset(augment_data=False)[0]
    out = ds[0]
    assert torch.equal(out[0], base[0]) and torch.equal(out[1], base[1])


def test_same_seed_reproducible_and_different_seed_differs():
    kw = dict(augment_data=True, aug_flags=AUG_PQ_JITTER, pq_jitter_rot_deg=10.0, pq_jitter_prob=1.0)
    a = _make_dataset(pq_jitter_seed=1, **kw)[0]
    b = _make_dataset(pq_jitter_seed=1, **kw)[0]
    c = _make_dataset(pq_jitter_seed=2, **kw)[0]
    assert torch.equal(a[0], b[0])
    assert not torch.equal(a[0], c[0])


def test_jitter_does_not_consume_global_random_stream():
    """専用 RNG を使うため、他 aug の乱数列（グローバル random）を乱さない。"""
    import random
    ds = _make_dataset(
        augment_data=True, aug_flags=AUG_PQ_JITTER,
        pq_jitter_rot_deg=10.0, pq_jitter_prob=0.5, pq_jitter_seed=3,
    )
    random.seed(99)
    expected = random.random()
    random.seed(99)
    ds[0]
    assert random.random() == expected


# ---------------------------------------------------------------------------
# endpoint blend との併用（現行dual標準 aug_flags=63 は ENDPOINT_BLEND を含む）
# ---------------------------------------------------------------------------

def _getitem_with_seed(seed: int, aug_flags: int, rot_deg: float):
    import random
    ds = _make_dataset(
        augment_data=True, aug_flags=aug_flags,
        pq_jitter_rot_deg=rot_deg, pq_jitter_prob=1.0, pq_jitter_seed=11,
    )
    random.seed(seed)
    np.random.seed(seed)
    return ds[0]


def test_with_endpoint_blend_labels_equal_blend_only():
    """blend + jitter のラベルは、同じ乱数での blend のみと一致する（jitter は t を変えない）。

    blend はグローバル random、jitter は専用 RNG なので、同じ seed なら blend の抽選は同一。
    """
    from eye_contour.data.augmentation import AUG_ENDPOINT_BLEND
    for seed in range(5):
        only = _getitem_with_seed(seed, AUG_ENDPOINT_BLEND, 0.0)
        both = _getitem_with_seed(seed, AUG_ENDPOINT_BLEND | AUG_PQ_JITTER, 10.0)
        for k in (4, 5, 12, 13):  # y_a, y_b, |PQ|, |P'Q'|
            assert torch.allclose(only[k], both[k], atol=1e-4), (seed, k)
        assert not torch.equal(only[0], both[0])  # パッチは回転で変わる


def test_jitter_rotates_about_blended_r():
    """blend 後の (P, Q, t) から R を復元して回転する（元の t から復元しない）。"""
    import random

    from eye_contour.data.augmentation import (
        AUG_ENDPOINT_BLEND,
        augment_endpoints,
        augment_pq_jitter,
    )
    from eye_contour.data.patch_utils import reconstruct_point

    for seed in range(20):
        random.seed(seed)
        px, py, qx, qy, t = augment_endpoints(
            REC["px"], REC["py"], REC["qx"], REC["qy"], REC["t"],
            label_t_min=-1.0, label_t_max=2.0, aug_flags=AUG_ENDPOINT_BLEND,
        )
        r_blend = reconstruct_point(px, py, qx, qy, t)
        r_orig = reconstruct_point(REC["px"], REC["py"], REC["qx"], REC["qy"], REC["t"])
        assert r_blend == pytest.approx(r_orig, abs=1e-9)  # blend は R を保存
        out = augment_pq_jitter(
            px, py, qx, qy, t, aug_flags=AUG_PQ_JITTER, rot_deg=15.0, prob=1.0,
            rng=__import__("random").Random(seed),
        )
        assert reconstruct_point(*out) == pytest.approx(r_orig, abs=1e-9)
        assert out[4] == t
        # blend 後の |PQ| が回転で保たれる（枝B の t' = 0.5 + (t-0.5)·|PQ|/xxx が壊れない）
        assert math.hypot(out[0] - out[2], out[1] - out[3]) == pytest.approx(math.hypot(px - qx, py - qy))
