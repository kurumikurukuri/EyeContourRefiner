"""TransferModel — 事前学習済みバックボーンを使った転移学習モデル。

対応バックボーン:
  - mobilenet_v3_small
  - efficientnet_b0
  - efficientnet_b1
  - efficientnet_b2
  - efficientnet_v2_s
  - efficientnet_v2_m
  - resnet18
"""
from __future__ import annotations

import torch
import torch.nn as nn

# 各バックボーンの最終特徴マップのチャンネル数
BACKBONE_FEAT_DIMS: dict[str, int] = {
    "mobilenet_v3_small": 576,
    "efficientnet_b0": 1280,
    "efficientnet_b1": 1280,
    "efficientnet_b2": 1408,
    "efficientnet_v2_s": 1280,
    "efficientnet_v2_m": 1280,
    "resnet18": 512,
}

BACKBONE_CHOICES = list(BACKBONE_FEAT_DIMS.keys())


def _build_backbone_features(backbone_name: str, pretrained: bool) -> nn.Module:
    """事前学習済みバックボーンの特徴抽出部分（classifier ヘッドを除く）を返す。"""
    if backbone_name == "mobilenet_v3_small":
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        return model.features

    if backbone_name == "efficientnet_b0":
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        return model.features

    if backbone_name == "efficientnet_b1":
        from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
        weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
        model = efficientnet_b1(weights=weights)
        return model.features

    if backbone_name == "efficientnet_b2":
        from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = efficientnet_b2(weights=weights)
        return model.features

    if backbone_name == "efficientnet_v2_s":
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = efficientnet_v2_s(weights=weights)
        return model.features

    if backbone_name == "efficientnet_v2_m":
        from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
        weights = EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
        model = efficientnet_v2_m(weights=weights)
        return model.features

    if backbone_name == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        return nn.Sequential(*list(model.children())[:-2])

    raise ValueError(f"未知のバックボーン: {backbone_name!r}  利用可能: {BACKBONE_CHOICES}")


class TransferModel(nn.Module):
    """事前学習済みバックボーンを使った転移学習モデル。

    アーキテクチャ:
        backbone → AdaptiveAvgPool2d(1, 32) → flatten
        → concat(is_upper, [pq_ratio]) → regressor → t

    入力: (B, 3, 64, 128) RGB パッチ + (B, 1) is_upper フラグ
          + オプションで (B, 1) pq_ratio（pq_dist / eye_scale）
    出力: (B, 1) スカラー t
    """

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool = True,
        use_pq_ratio: bool = False,
        use_sigmoid: bool = False,
        use_distance_from_center: bool = False,
        use_norm_eye_scale: bool = False,
        use_is_upper: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.use_pq_ratio = use_pq_ratio
        self.use_sigmoid = use_sigmoid
        self.use_distance_from_center = use_distance_from_center
        self.use_norm_eye_scale = use_norm_eye_scale
        self.use_is_upper = use_is_upper
        feat_dim = BACKBONE_FEAT_DIMS[backbone_name]

        self.backbone = _build_backbone_features(backbone_name, pretrained)
        self.pool = nn.AdaptiveAvgPool2d((1, 32))

        scalar_in = int(use_is_upper) + int(use_pq_ratio) + int(use_distance_from_center) + int(use_norm_eye_scale)
        regressor_in = feat_dim * 32 + scalar_in
        regressor_layers: list[nn.Module] = [
            nn.Linear(regressor_in, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        ]
        if use_sigmoid:
            regressor_layers.append(nn.Sigmoid())
        self.regressor = nn.Sequential(*regressor_layers)

    def forward(
        self,
        x: torch.Tensor,
        is_upper: torch.Tensor,
        pq_ratio: torch.Tensor | None = None,
        dist_from_center: torch.Tensor | None = None,
        norm_eye_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feat = self.backbone(x)           # (B, C, H', W')
        feat = self.pool(feat)            # (B, C, 1, 32)
        feat = feat.view(x.size(0), -1)  # (B, C*32)
        scalars = [feat]
        if self.use_is_upper:
            scalars.append(is_upper)
        if self.use_pq_ratio and pq_ratio is not None:
            scalars.append(pq_ratio)
        if self.use_distance_from_center and dist_from_center is not None:
            scalars.append(dist_from_center)
        if self.use_norm_eye_scale and norm_eye_scale is not None:
            scalars.append(norm_eye_scale)
        feat = torch.cat(scalars, dim=1)
        return self.regressor(feat)       # (B, 1)

    def freeze_backbone(self) -> None:
        """バックボーンのパラメータを凍結する（Phase 1 転移学習）。"""
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        """バックボーンのパラメータを解凍する（Phase 2 fine-tuning）。"""
        for p in self.backbone.parameters():
            p.requires_grad_(True)
