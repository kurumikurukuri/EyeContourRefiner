"""eyeliner/evaluation/metrics.py と eyeliner/models/ のテスト。"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from unittest.mock import patch as mock_patch

from eyeliner.evaluation.metrics import (
    cv_mae,
    pixel_mae,
    relative_mae,
    holdout_mae,
)
from eyeliner.models.eyeliner_cnn import EyelinerCNN
from eyeliner.models.transfer import TransferModel, BACKBONE_CHOICES
from eyeliner.models.ensemble import EnsembleModel, build_model, load_model, save_model
import tempfile
from pathlib import Path


# ===========================================================
# cv_mae
# ===========================================================

class TestCvMae:

    def test_single_fold(self):
        """単一 fold の MAE をそのまま返す。"""
        assert abs(cv_mae([0.05]) - 0.05) < 1e-10

    def test_multiple_folds_mean(self):
        """複数 fold の平均を返す。"""
        folds = [0.04, 0.06, 0.05]
        assert abs(cv_mae(folds) - 0.05) < 1e-10

    def test_returns_float(self):
        """float を返す。"""
        result = cv_mae([0.03, 0.07])
        assert isinstance(result, float)


# ===========================================================
# pixel_mae
# ===========================================================

class TestPixelMae:

    def _make_points(self, n=10):
        p = np.zeros((n, 2))
        q = np.column_stack([np.full(n, 100.0), np.zeros(n)])  # pq_dist = 100
        return p, q

    def test_zero_error_is_zero(self):
        """t_pred == t_true のとき 0。"""
        n = 5
        p, q = self._make_points(n)
        t = np.full(n, 0.5)
        assert abs(pixel_mae(t, t, p, q)) < 1e-10

    def test_scales_with_pq_dist(self):
        """pq_dist が 2 倍になると pixel_mae も 2 倍。"""
        p1 = np.zeros((5, 2))
        q1 = np.column_stack([np.full(5, 100.0), np.zeros(5)])
        q2 = np.column_stack([np.full(5, 200.0), np.zeros(5)])
        t_pred = np.full(5, 0.6)
        t_true = np.full(5, 0.5)
        m1 = pixel_mae(t_pred, t_true, p1, q1)
        m2 = pixel_mae(t_pred, t_true, p1, q2)
        assert abs(m2 / m1 - 2.0) < 1e-6

    def test_returns_float(self):
        p, q = self._make_points()
        result = pixel_mae(np.zeros(10), np.zeros(10), p, q)
        assert isinstance(result, float)


# ===========================================================
# relative_mae
# ===========================================================

class TestRelativeMae:

    def test_zero_error_is_zero(self):
        """t_pred == t_true のとき 0。"""
        n = 5
        p = np.zeros((n, 2))
        q = np.column_stack([np.full(n, 100.0), np.zeros(n)])
        eye_scales = np.full(n, 200.0)
        t = np.full(n, 0.5)
        assert abs(relative_mae(t, t, p, q, eye_scales)) < 1e-10

    def test_normalizes_by_eye_scale(self):
        """eye_scale が 2 倍になると relative_mae は半分になる。"""
        n = 5
        p = np.zeros((n, 2))
        q = np.column_stack([np.full(n, 100.0), np.zeros(n)])
        t_pred = np.full(n, 0.6)
        t_true = np.full(n, 0.5)
        es1 = np.full(n, 100.0)
        es2 = np.full(n, 200.0)
        m1 = relative_mae(t_pred, t_true, p, q, es1)
        m2 = relative_mae(t_pred, t_true, p, q, es2)
        assert abs(m2 / m1 - 0.5) < 1e-6

    def test_zero_eye_scale_handled(self):
        """eye_scale=0 でもゼロ除算が発生しない（clamp して max(scale, 1.0) で安全処理）。"""
        n = 3
        p = np.zeros((n, 2))
        q = np.column_stack([np.full(n, 100.0), np.zeros(n)])
        t_pred = np.full(n, 0.6)
        t_true = np.full(n, 0.5)
        es = np.zeros(n)
        result = relative_mae(t_pred, t_true, p, q, es)
        assert np.isfinite(result)


# ===========================================================
# holdout_mae (モックで軽量テスト)
# ===========================================================

class TestHoldoutMae:

    def _make_loader(self, n: int, y_val: float):
        from torch.utils.data import TensorDataset, DataLoader
        import math
        x = torch.zeros(n, 3, 64, 128)
        is_upper = torch.zeros(n, 1)
        pq_ratio = torch.ones(n, 1) * 0.5
        y = torch.full((n, 1), y_val)
        weight = torch.ones(n, 1)
        val_weight = torch.ones(n, 1)
        dist_from_center = torch.zeros(n, 1)
        norm_eye_scale = torch.ones(n, 1) * 0.01
        point_polar_cos = torch.ones(n, 1) * math.cos(math.pi / 4)
        point_polar_sin = torch.ones(n, 1) * math.sin(math.pi / 4)
        ds = TensorDataset(
            x, is_upper, pq_ratio, y, weight, val_weight,
            dist_from_center, norm_eye_scale, point_polar_cos, point_polar_sin,
        )
        return DataLoader(ds, batch_size=4)

    def test_perfect_prediction_is_zero(self):
        """完璧な予測のとき holdout_mae == 0。"""
        class PerfectModel(torch.nn.Module):
            def forward(self, x, is_upper, *args, **kwargs):
                return torch.full((x.size(0), 1), 0.5)

        model = PerfectModel()
        loader = self._make_loader(8, 0.5)
        mae = holdout_mae(model, loader, torch.device("cpu"))
        assert abs(mae) < 1e-6

    def test_constant_error(self):
        """一定の誤差 0.1 のとき holdout_mae == 0.1。"""
        class OffsetModel(torch.nn.Module):
            def forward(self, x, is_upper, *args, **kwargs):
                return torch.full((x.size(0), 1), 0.6)  # y=0.5 に対して +0.1

        model = OffsetModel()
        loader = self._make_loader(8, 0.5)
        mae = holdout_mae(model, loader, torch.device("cpu"))
        assert abs(mae - 0.1) < 1e-5


# ===========================================================
# EyelinerCNN モデル
# ===========================================================

class TestEyelinerCNN:

    def test_forward_output_shape(self):
        """forward の出力 shape が (B, 1)。"""
        model = EyelinerCNN()
        model.eval()
        x = torch.zeros(4, 3, 64, 128)
        is_upper = torch.zeros(4, 1)
        with torch.no_grad():
            out = model(x, is_upper)
        assert out.shape == (4, 1)

    def test_forward_no_nan(self):
        """forward で NaN が発生しない。"""
        model = EyelinerCNN()
        model.eval()
        x = torch.randn(4, 3, 64, 128)
        is_upper = torch.zeros(4, 1)
        with torch.no_grad():
            out = model(x, is_upper)
        assert not torch.isnan(out).any()

    def test_batch_size_1(self):
        """バッチサイズ 1 でも動作する。"""
        model = EyelinerCNN()
        model.eval()
        x = torch.zeros(1, 3, 64, 128)
        is_upper = torch.zeros(1, 1)
        with torch.no_grad():
            out = model(x, is_upper)
        assert out.shape == (1, 1)


# ===========================================================
# TransferModel
# ===========================================================

class TestTransferModel:

    @pytest.mark.parametrize("backbone", BACKBONE_CHOICES[:1])  # mobilenet のみ（速度優先）
    def test_forward_output_shape(self, backbone):
        """forward の出力 shape が (B, 1)。"""
        model = TransferModel(backbone, pretrained=False)
        model.eval()
        x = torch.zeros(2, 3, 64, 128)
        is_upper = torch.zeros(2, 1)
        with torch.no_grad():
            out = model(x, is_upper)
        assert out.shape == (2, 1)

    def test_freeze_and_unfreeze(self):
        """freeze/unfreeze が正しく動作する。"""
        model = TransferModel("mobilenet_v3_small", pretrained=False)
        model.freeze_backbone()
        for p in model.backbone.parameters():
            assert not p.requires_grad
        model.unfreeze_backbone()
        for p in model.backbone.parameters():
            assert p.requires_grad


# ===========================================================
# EnsembleModel
# ===========================================================

class TestEnsembleModel:

    def test_forward_output_shape(self):
        """forward の出力が各モデルの平均（shape は単一モデルと同じ）。"""
        models = [EyelinerCNN() for _ in range(3)]
        ensemble = EnsembleModel(models)
        ensemble.eval()
        x = torch.zeros(4, 3, 64, 128)
        is_upper = torch.zeros(4, 1)
        with torch.no_grad():
            out = ensemble(x, is_upper)
        assert out.shape == (4, 1)

    def test_save_load_roundtrip(self, tmp_path):
        """save_model / load_model の往復で重みが保存・復元される。"""
        model = EyelinerCNN()
        model.eval()
        path = tmp_path / "test_model.pt"
        save_model(model, "custom", path)

        loaded = load_model(path)
        loaded.eval()

        x = torch.randn(2, 3, 64, 128)
        is_upper = torch.zeros(2, 1)
        with torch.no_grad():
            out_orig = model(x, is_upper)
            out_loaded = loaded(x, is_upper)
        torch.testing.assert_close(out_orig, out_loaded)

    def test_build_model_custom(self):
        """'custom' で EyelinerCNN が生成される。"""
        model = build_model("custom")
        assert isinstance(model, EyelinerCNN)

    def test_build_model_transfer(self):
        """バックボーン名で TransferModel が生成される。"""
        model = build_model("mobilenet_v3_small", pretrained=False)
        assert isinstance(model, TransferModel)


# ===========================================================
# EyelinerPatchDataset crop/label 範囲分離
# ===========================================================

class TestEyelinerPatchDatasetLabelRange:

    def _make_record(self, t: float) -> dict:
        return {
            "image_path": "dummy.jpg",
            "px": "0", "py": "0", "qx": "100", "qy": "0",
            "t": str(t),
            "sub_key": "upper",
            "point_index": "3",
            "pq_ratio": "0.5",
            "weight": "1.0",
            "val_weight": "1.0",
        }

    def test_label_range_clips_t(self):
        """label_t_min/max でtがクリップされ、正規化される。"""
        from eyeliner.data.dataset import EyelinerPatchDataset
        rec = self._make_record(t=2.0)  # label 範囲外（-0.5〜1.5）
        dummy_image = np.zeros((100, 200, 3), dtype=np.uint8)
        with mock_patch("cv2.imread", return_value=dummy_image):
            ds = EyelinerPatchDataset(
                [rec],
                preload_images=False,
                patch_t_min=-1.0, patch_t_max=2.0,
                label_t_min=-0.5, label_t_max=1.5,
                use_t_norm=True,
            )
        # t=2.0 は label_t_max=1.5 にクリップ → t_norm = (1.5 - (-0.5)) / (1.5 - (-0.5)) = 1.0
        _, _, _, y, *_ = ds[0]
        assert abs(y.item() - 1.0) < 1e-5

    def test_label_range_normalizes_independently_of_crop(self):
        """crop 範囲が広くても label 範囲で正規化された値が返る。"""
        from eyeliner.data.dataset import EyelinerPatchDataset
        rec = self._make_record(t=0.5)
        dummy_image = np.zeros((100, 200, 3), dtype=np.uint8)
        with mock_patch("cv2.imread", return_value=dummy_image):
            ds = EyelinerPatchDataset(
                [rec],
                preload_images=False,
                patch_t_min=-1.0, patch_t_max=2.0,   # crop: -1〜2（range=3）
                label_t_min=-0.5, label_t_max=1.5,    # label: -0.5〜1.5（range=2）
                use_t_norm=True,
            )
        # t=0.5 → t_norm = (0.5 - (-0.5)) / (1.5 - (-0.5)) = 1.0 / 2.0 = 0.5
        _, _, _, y, *_ = ds[0]
        assert abs(y.item() - 0.5) < 1e-5

    def test_no_label_range_falls_back_to_crop_range(self):
        """label_t_min/max=None のとき crop 範囲と同じ動作（後方互換）。"""
        from eyeliner.data.dataset import EyelinerPatchDataset
        rec = self._make_record(t=0.5)
        dummy_image = np.zeros((100, 200, 3), dtype=np.uint8)
        with mock_patch("cv2.imread", return_value=dummy_image):
            ds = EyelinerPatchDataset(
                [rec],
                preload_images=False,
                patch_t_min=-0.5, patch_t_max=1.5,
                label_t_min=None, label_t_max=None,
                use_t_norm=True,
            )
        # label = crop → t_norm = (0.5 - (-0.5)) / (1.5 - (-0.5)) = 0.5
        _, _, _, y, *_ = ds[0]
        assert abs(y.item() - 0.5) < 1e-5


# ===========================================================
# EyelinerCNN num_conv_layers
# ===========================================================

@pytest.mark.parametrize("n", [2, 3, 4])
def test_eyeliner_cnn_num_conv_layers_output_shape(n):
    """num_conv_layers=2/3/4 で出力 shape が (B, 1) かつ NaN なし。"""
    model = EyelinerCNN(num_conv_layers=n)
    model.eval()
    x = torch.randn(2, 3, 64, 128)
    is_upper = torch.zeros(2, 1)
    with torch.no_grad():
        out = model(x, is_upper)
    assert out.shape == (2, 1), f"num_conv_layers={n}: shape={out.shape}"
    assert not torch.isnan(out).any()
