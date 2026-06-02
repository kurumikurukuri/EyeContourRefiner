"""EXP020 デュアルブランチ実装の単体テスト。

検証:
  1. CrossFuseDualModel の blocks_a / blocks_b が重みを共有していないこと
  2. 完全忠実版で使う core EyelinerPatchDualDataset の AUG_FLIP が枝A/B で同期すること
     （EXP020 は eyeliner/data/dataset_dual.py を採用。枝別 t空間/逆変換は
      test_exp020_branch_presets.py で別途検証）
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

_EXP020_SCR = Path(__file__).parent.parent / "experiments" / "EXP020_dual_branch" / "scr"
if str(_EXP020_SCR) not in sys.path:
    sys.path.insert(0, str(_EXP020_SCR))

from eyeliner.models.eyeliner_cnn_dual_branch import CrossFuseDualModel
from eyeliner.data.augmentation import AUG_FLIP
from eyeliner.data.dataset_dual import BranchConfig, EyelinerPatchDualDataset


# ===========================================================
# Bug 1: CrossFuseDualModel のブランチ間重み共有
# ===========================================================

class TestCrossFuseDualModelIndependence:

    def test_conv_params_are_independent(self):
        """blocks_a[i] と blocks_b[i] の Conv2d が別インスタンスである。"""
        model = CrossFuseDualModel(num_conv_layers=3)
        for i in range(len(model.blocks_a)):
            conv_a = next(m for m in model.blocks_a[i].modules() if isinstance(m, nn.Conv2d))
            conv_b = next(m for m in model.blocks_b[i].modules() if isinstance(m, nn.Conv2d))
            assert conv_a is not conv_b, \
                f"Block {i}: blocks_a と blocks_b が同じ Conv2d インスタンスを共有"
            assert conv_a.weight.data_ptr() != conv_b.weight.data_ptr(), \
                f"Block {i}: Conv2d weight が同一メモリを指す（重み共有バグ）"

    def test_bn_params_are_independent(self):
        """blocks_a[i] と blocks_b[i] の BatchNorm2d が別インスタンスである。"""
        model = CrossFuseDualModel(num_conv_layers=3, use_batchnorm=True)
        for i in range(len(model.blocks_a)):
            bns_a = [m for m in model.blocks_a[i].modules() if isinstance(m, nn.BatchNorm2d)]
            bns_b = [m for m in model.blocks_b[i].modules() if isinstance(m, nn.BatchNorm2d)]
            assert len(bns_a) == 1 and len(bns_b) == 1
            bn_a, bn_b = bns_a[0], bns_b[0]
            assert bn_a is not bn_b
            assert bn_a.weight.data_ptr() != bn_b.weight.data_ptr()

    def test_initial_weights_differ_between_branches(self):
        """独立初期化なら blocks_a[0] と blocks_b[0] の Conv2d 重みは異なる。"""
        model = CrossFuseDualModel(num_conv_layers=2)
        conv_a = next(m for m in model.blocks_a[0].modules() if isinstance(m, nn.Conv2d))
        conv_b = next(m for m in model.blocks_b[0].modules() if isinstance(m, nn.Conv2d))
        assert not torch.allclose(conv_a.weight, conv_b.weight)

    def test_forward_output_shape(self):
        model = CrossFuseDualModel(num_conv_layers=2)
        model.eval()
        x_a = torch.randn(2, 3, 64, 64)
        x_b = torch.randn(2, 3, 64, 64)
        is_upper = torch.zeros(2, 1)
        with torch.no_grad():
            pred_a, pred_b = model(x_a, x_b, is_upper)
        assert pred_a.shape == (2, 1)
        assert pred_b.shape == (2, 1)

    def test_forward_no_nan(self):
        model = CrossFuseDualModel(num_conv_layers=2)
        model.eval()
        x_a = torch.randn(2, 3, 64, 64)
        x_b = torch.randn(2, 3, 64, 64)
        is_upper = torch.zeros(2, 1)
        with torch.no_grad():
            pred_a, pred_b = model(x_a, x_b, is_upper)
        assert not torch.isnan(pred_a).any()
        assert not torch.isnan(pred_b).any()


# ===========================================================
# Bug 2: core EyelinerPatchDualDataset の AUG_FLIP 同期
# ===========================================================

class TestCoreDualDatasetFlipSync:
    """完全忠実版が採用する core dataset で flip が枝A/Bで同期することを検証。"""

    @pytest.fixture(scope="class")
    def img_and_rec(self):
        H, W = 100, 200
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:, W // 2:, :] = 200  # 右半分明るい
        rec = {
            "image_path": "test.jpg",
            "px": 10.0, "py": 50.0, "qx": 190.0, "qy": 50.0,  # 水平 PQ
            "t": 0.5, "sub_key": "upper", "point_index": 3,
            "pq_ratio": 180.0 / 200.0, "eye_scale": 200.0,
            "weight": 1.0, "val_weight": 1.0,
        }
        return img, rec

    def _make_ds(self, img, rec, augment_data):
        cfg_a = BranchConfig(
            patch_t_min=-2.0, patch_t_max=3.0, label_t_min=-1.0, label_t_max=2.0,
            pq_rescale_mode="none", patch_size=(32, 32),
        )
        cfg_b = BranchConfig(
            patch_t_min=-0.5, patch_t_max=1.5, label_t_min=-0.25, label_t_max=1.25,
            pq_rescale_mode="eye_scale", pq_rescale_factor=0.2,
            label_space="t_prime", patch_size=(32, 32),
        )
        ds = EyelinerPatchDualDataset(
            [rec], cfg_a, cfg_b, augment_data=augment_data,
            aug_flags=AUG_FLIP, preload_images=False,
        )
        ds._image_cache["test.jpg"] = img
        ds._img_diag_map["test.jpg"] = math.sqrt(img.shape[0] * img.shape[1])
        return ds

    @staticmethod
    def _detect_flip(x: torch.Tensor, x_base: torch.Tensor) -> bool:
        x_mirror = x_base.flip(-1)
        return (x - x_mirror).abs().mean().item() < (x - x_base).abs().mean().item()

    def test_flip_is_synchronized_between_branches(self, img_and_rec):
        img, rec = img_and_rec
        ds_base = self._make_ds(img, rec, augment_data=False)
        out_base = ds_base[0]
        x_a_base, x_b_base = out_base[0], out_base[1]

        ds_aug = self._make_ds(img, rec, augment_data=True)
        mismatches = 0
        flip_fired = False
        for seed in range(100):
            random.seed(seed)
            np.random.seed(seed)
            out = ds_aug[0]
            x_a, x_b = out[0], out[1]
            a_flipped = self._detect_flip(x_a, x_a_base)
            b_flipped = self._detect_flip(x_b, x_b_base)
            if a_flipped:
                flip_fired = True
            if a_flipped != b_flipped:
                mismatches += 1
        assert flip_fired, "AUG_FLIP が一度も発火しなかった"
        assert mismatches == 0, f"枝A/B のフリップ非同期: {mismatches}/100"
