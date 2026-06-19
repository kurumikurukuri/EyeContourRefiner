"""EyelinerCNNDual: 2系統入力（生PQ + P'Q'）デュアルブランチ CNN。"""
from __future__ import annotations

import torch
import torch.nn as nn

FUSION_MODES = ("shared", "independent", "attention")


def _build_features(
    num_conv_layers: int, use_batchnorm: bool,
    base_channel: int = 32, pool_width: int = 32,
) -> tuple[nn.Sequential, int]:
    """CNN特徴抽出器を構築する。

    Returns:
        (module, cnn_flat): モジュールとフラット特徴次元数
    """
    channels = [base_channel * (2 ** i) for i in range(num_conv_layers)]
    layers: list[nn.Module] = []
    in_ch = 3
    for i, out_ch in enumerate(channels):
        layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU())
        if i < num_conv_layers - 1:
            layers.append(nn.MaxPool2d(2))
        in_ch = out_ch
    layers.append(nn.AdaptiveAvgPool2d((1, pool_width)))
    return nn.Sequential(*layers), channels[-1] * pool_width


def _build_head(in_dim: int, dropout_rate: float, use_sigmoid: bool) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(in_dim, 128),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(128, 1),
    ]
    if use_sigmoid:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class EyelinerCNNDual(nn.Module):
    """2系統パッチを受け取るデュアルブランチ CNN 回帰モデル。

    各ブランチが独立した出力ヘッドを持ち、それぞれ生 t_raw 空間で予測する。
    融合は t_raw レベルで行うため、枝Aのラベル範囲が出力全体を制限しない。

    fusion_mode:
        "shared"      : 1つのCNNバックボーンを両枝で共有。
        "independent" : 各枝が独立したCNNバックボーンを持つ。
        "attention"   : 独立バックボーン + α重みで推論時フュージョン。

    入力:
        x_a    : (B, 3, H, W) 枝A パッチ（生PQ座標系）
        x_b    : (B, 3, H, W) 枝B パッチ（P'Q'座標系）
        is_upper: (B, 1)

    出力:
        D1/D2: (pred_a, pred_b)          各 (B, 1), t_norm ∈ [0, 1]
        D3:   (pred_a, pred_b, alpha)    alpha は (B, 1) の融合重み
    """

    def __init__(
        self,
        fusion_mode: str = "independent",
        num_conv_layers: int = 4,
        use_sigmoid: bool = True,
        use_batchnorm: bool = True,
        dropout_rate: float = 0.3,
        base_channel: int = 32,
        pool_width: int = 32,
    ) -> None:
        super().__init__()
        if fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {FUSION_MODES}")

        self.fusion_mode = fusion_mode
        self.num_conv_layers = num_conv_layers
        self.use_sigmoid = use_sigmoid
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        self.base_channel = base_channel
        self.pool_width = pool_width

        if fusion_mode == "shared":
            self.features, cnn_flat = _build_features(
                num_conv_layers, use_batchnorm, base_channel, pool_width,
            )
        else:
            self.features_a, cnn_flat = _build_features(
                num_conv_layers, use_batchnorm, base_channel, pool_width,
            )
            self.features_b, _ = _build_features(
                num_conv_layers, use_batchnorm, base_channel, pool_width,
            )

        self._cnn_flat = cnn_flat
        head_in = cnn_flat + 1  # features + is_upper

        self.head_a = _build_head(head_in, dropout_rate, use_sigmoid)
        self.head_b = _build_head(head_in, dropout_rate, use_sigmoid)

        if fusion_mode == "attention":
            # α ゲート: 推論時に枝A と枝B の重みを学習する
            self.alpha_net = nn.Linear(2 * cnn_flat, 1)

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        is_upper: torch.Tensor,
    ) -> tuple:
        B = x_a.size(0)

        if self.fusion_mode == "shared":
            fa = self.features(x_a).view(B, -1)
            fb = self.features(x_b).view(B, -1)
        else:
            fa = self.features_a(x_a).view(B, -1)
            fb = self.features_b(x_b).view(B, -1)

        head_in_a = torch.cat([fa, is_upper], dim=1)
        head_in_b = torch.cat([fb, is_upper], dim=1)

        pred_a = self.head_a(head_in_a)
        pred_b = self.head_b(head_in_b)

        if self.fusion_mode == "attention":
            alpha = torch.sigmoid(self.alpha_net(torch.cat([fa, fb], dim=1)))
            return pred_a, pred_b, alpha

        return pred_a, pred_b
