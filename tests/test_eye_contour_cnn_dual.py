"""eye_contour.models.eye_contour_cnn_dual の base_channel / pool_width 単体テスト。

EXP022（base_channel x pool_width スイープ）向けに追加されたパラメータが
forward の出力 shape を変えないこと、特徴次元が想定どおりスケールすることを確認する。
"""
from __future__ import annotations

import torch

from eye_contour.models.eye_contour_cnn_dual import EyeContourCNNDual, _build_features


class TestBuildFeatures:

    def test_default_matches_legacy_32_32(self):
        _, cnn_flat = _build_features(num_conv_layers=4, use_batchnorm=True)
        assert cnn_flat == 256 * 32  # channels[-1]=32*2^3=256, pool_width=32

    def test_custom_base_channel_scales_channel_count(self):
        module, cnn_flat = _build_features(
            num_conv_layers=4, use_batchnorm=True, base_channel=16, pool_width=32,
        )
        assert cnn_flat == 128 * 32  # channels[-1]=16*2^3=128

    def test_custom_pool_width_scales_flat_dim(self):
        _, cnn_flat = _build_features(
            num_conv_layers=4, use_batchnorm=True, base_channel=32, pool_width=1,
        )
        assert cnn_flat == 256 * 1

    def test_pool_width_one_output_shape(self):
        module, cnn_flat = _build_features(
            num_conv_layers=4, use_batchnorm=True, base_channel=16, pool_width=1,
        )
        x = torch.randn(2, 3, 64, 128)
        out = module(x)
        assert out.shape == (2, 128, 1, 1)
        assert cnn_flat == 128


class TestEyeContourCNNDual:

    def _forward(self, model: EyeContourCNNDual, batch: int = 2):
        x_a = torch.randn(batch, 3, 64, 128)
        x_b = torch.randn(batch, 3, 64, 128)
        is_upper = torch.ones(batch, 1)
        return model(x_a, x_b, is_upper)

    def test_default_base_channel_pool_width(self):
        model = EyeContourCNNDual(fusion_mode="independent", num_conv_layers=4)
        assert model.base_channel == 32
        assert model.pool_width == 32
        pred_a, pred_b = self._forward(model)
        assert pred_a.shape == (2, 1)
        assert pred_b.shape == (2, 1)

    def test_custom_base_channel_and_pool_width(self):
        model = EyeContourCNNDual(
            fusion_mode="independent", num_conv_layers=4,
            base_channel=16, pool_width=8,
        )
        pred_a, pred_b = self._forward(model)
        assert pred_a.shape == (2, 1)
        assert pred_b.shape == (2, 1)

    def test_pool_width_one_forward_runs(self):
        model = EyeContourCNNDual(
            fusion_mode="independent", num_conv_layers=4,
            base_channel=16, pool_width=1,
        )
        pred_a, pred_b = self._forward(model)
        assert pred_a.shape == (2, 1)
        assert pred_b.shape == (2, 1)

    def test_shared_fusion_mode_with_custom_params(self):
        model = EyeContourCNNDual(
            fusion_mode="shared", num_conv_layers=4,
            base_channel=16, pool_width=16,
        )
        pred_a, pred_b = self._forward(model)
        assert pred_a.shape == (2, 1)
        assert pred_b.shape == (2, 1)

    def test_attention_fusion_mode_with_custom_params(self):
        model = EyeContourCNNDual(
            fusion_mode="attention", num_conv_layers=4,
            base_channel=16, pool_width=16,
        )
        pred_a, pred_b, alpha = self._forward(model)
        assert pred_a.shape == (2, 1)
        assert pred_b.shape == (2, 1)
        assert alpha.shape == (2, 1)
