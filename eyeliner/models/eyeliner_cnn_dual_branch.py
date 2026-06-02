"""CrossFuseDualModel — 多段階 cross-stitch 融合デュアルブランチ CNN。"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossStitch(nn.Module):
    """Cross-stitch 融合: concat([f_a, f_b]) → 1×1 Conv → split([f_a', f_b'])。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fusion = nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=False)

    def forward(
        self, f_a: torch.Tensor, f_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.fusion(torch.cat([f_a, f_b], dim=1))
        return fused.chunk(2, dim=1)


def _build_head(in_dim: int, dropout_rate: float, use_sigmoid: bool) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(in_dim, 64),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(64, 1),
    ]
    if use_sigmoid:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class CrossFuseDualModel(nn.Module):
    """2系統入力のデュアルブランチ CNN。各 ConvBlock 後に CrossStitch で中間融合を行う。

    入力:
        x_a    : (B, 3, H, W) 枝A パッチ
        x_b    : (B, 3, H, W) 枝B パッチ
        is_upper: (B, 1)
    出力:
        (pred_a, pred_b) — 各 (B, 1)、t_norm ∈ [0, 1]
    """

    FUSION_MODE = "crossfuse"

    def __init__(
        self,
        num_conv_layers: int = 4,
        use_sigmoid: bool = True,
        use_batchnorm: bool = True,
        dropout_rate: float = 0.3,
        use_is_upper: bool = True,
    ) -> None:
        super().__init__()
        channels = [32 * (2 ** i) for i in range(num_conv_layers)]

        self.blocks_a = nn.ModuleList()
        self.blocks_b = nn.ModuleList()
        self.stitches = nn.ModuleList()

        def _make_block(c_in: int, c_out: int) -> nn.Sequential:
            layers: list[nn.Module] = [nn.Conv2d(c_in, c_out, 3, padding=1)]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        in_ch = 3
        for out_ch in channels:
            self.blocks_a.append(_make_block(in_ch, out_ch))
            self.blocks_b.append(_make_block(in_ch, out_ch))
            self.stitches.append(CrossStitch(out_ch))
            in_ch = out_ch

        self.pools = nn.ModuleList([
            nn.MaxPool2d(2) for _ in range(num_conv_layers - 1)
        ])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 32))
        cnn_flat = channels[-1] * 32

        head_in = cnn_flat + (1 if use_is_upper else 0)
        self.head_a = _build_head(head_in, dropout_rate, use_sigmoid)
        self.head_b = _build_head(head_in, dropout_rate, use_sigmoid)
        self.use_is_upper = use_is_upper

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        is_upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f_a, f_b = x_a, x_b
        for i in range(len(self.blocks_a)):
            f_a = self.blocks_a[i](f_a)
            f_b = self.blocks_b[i](f_b)
            f_a, f_b = self.stitches[i](f_a, f_b)
            if i < len(self.pools):
                f_a = self.pools[i](f_a)
                f_b = self.pools[i](f_b)

        f_a = self.avgpool(f_a).view(f_a.size(0), -1)
        f_b = self.avgpool(f_b).view(f_b.size(0), -1)

        if self.use_is_upper:
            pred_a = self.head_a(torch.cat([f_a, is_upper], dim=1))
            pred_b = self.head_b(torch.cat([f_b, is_upper], dim=1))
        else:
            pred_a = self.head_a(f_a)
            pred_b = self.head_b(f_b)

        return pred_a, pred_b
