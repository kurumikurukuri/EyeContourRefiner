"""EdgeCNN — 軽量カスタム CNN（転移学習なし）。"""
from __future__ import annotations

import torch
import torch.nn as nn


class EyelinerCNN(nn.Module):
    """軽量な可変深度 CNN + 全結合回帰ヘッド。

    入力: (B, 3, 64, 128) RGB パッチ + (B, 1) is_upper フラグ
          + オプションで (B, 1) pq_ratio（pq_dist / eye_scale）
          + オプションで (B, 1) norm_eye_scale（eye_scale / sqrt(img_H * img_W)）
    出力: (B, 1) スカラー t ∈ [0, 1]（P-Q ライン上の正規化位置）
    """

    AUX_INPUT_DIM = 11  # pq_angle_cos/sin, polar_cos/sin, yaw/pitch/roll sin/cos, is_upper

    def __init__(
        self,
        use_pq_ratio: bool = False,
        use_sigmoid: bool = False,
        use_batchnorm: bool = True,
        num_conv_layers: int = 3,
        use_distance_from_center: bool = False,
        use_norm_eye_scale: bool = False,
        use_is_upper: bool = True,
        dropout_rate: float = 0.3,
        use_aux_face_shape: bool = False,
        pool_width: int = 32,
    ) -> None:
        super().__init__()
        self.use_pq_ratio = use_pq_ratio
        self.use_sigmoid = use_sigmoid
        self.use_batchnorm = use_batchnorm
        self.num_conv_layers = num_conv_layers
        self.use_distance_from_center = use_distance_from_center
        self.use_norm_eye_scale = use_norm_eye_scale
        self.use_is_upper = use_is_upper
        self.dropout_rate = dropout_rate
        self.use_aux_face_shape = use_aux_face_shape
        self.aux_input_dim = self.AUX_INPUT_DIM if use_aux_face_shape else 0
        self.pool_width = pool_width

        def _bn(channels: int) -> nn.Module:
            return nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()

        channels = [32 * (2 ** i) for i in range(num_conv_layers)]  # [32,64] / [32,64,128] / [32,64,128,256]
        layers: list[nn.Module] = []
        in_ch = 3
        for i, out_ch in enumerate(channels):
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), _bn(out_ch), nn.ReLU()]
            if i < num_conv_layers - 1:
                layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool2d((1, pool_width)))
        self.features = nn.Sequential(*layers)

        cnn_flat = channels[-1] * pool_width
        aux_dim = self.AUX_INPUT_DIM if use_aux_face_shape else 0
        scalar_in = int(use_is_upper) + int(use_pq_ratio) + int(use_distance_from_center) + int(use_norm_eye_scale) + aux_dim
        regressor_layers: list[nn.Module] = [
            nn.Linear(cnn_flat + scalar_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
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
        aux_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feat = self.features(x).view(x.size(0), -1)   # (B, 4096)
        scalars = [feat]
        if self.use_is_upper:
            scalars.append(is_upper)
        if self.use_pq_ratio:
            val = pq_ratio if pq_ratio is not None else torch.zeros(x.size(0), 1, device=x.device)
            scalars.append(val)
        if self.use_distance_from_center:
            val = dist_from_center if dist_from_center is not None else torch.zeros(x.size(0), 1, device=x.device)
            scalars.append(val)
        if self.use_norm_eye_scale:
            val = norm_eye_scale if norm_eye_scale is not None else torch.zeros(x.size(0), 1, device=x.device)
            scalars.append(val)
        if self.use_aux_face_shape:
            aux = aux_features if aux_features is not None else torch.zeros(x.size(0), self.AUX_INPUT_DIM, device=x.device)
            scalars.append(aux)
        feat = torch.cat(scalars, dim=1)
        return self.regressor(feat)                     # (B, 1)
