"""GateCNN — サンプルごとに「どの座標系の予測を信じるか」の重みを出力する軽量 CNN。

EXP019 の学習型ゲート用。凍結した複数専門家モデルの予測値を
ブレンドする重みベクトルを画像パッチと補助スカラー特徴量から学習する。

入力: (B, 3, H, W) RGB パッチ + (B, 1) is_upper + (B, aux_dim) 補助特徴量
出力: (B, num_experts) 重みベクトル（softmax、和=1）

使用例 (4 experts):
    w = gate(x, is_upper, aux_features)  # (B, 4)
    t_final = (w * t_experts).sum(dim=1)  # t_experts: (B, 4)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GateCNN(nn.Module):
    """画像パッチ + スカラー補助特徴からアンサンブル重みを予測する軽量ゲート CNN。

    Args:
        num_experts:     専門家（座標系）の数。2 なら sigmoid、3以上なら softmax
        num_conv_layers: 畳み込み層数
        aux_dim:         補助スカラー特徴量の次元数
        use_batchnorm:   BatchNorm の有無
        dropout_rate:    最終層の dropout 率
    """

    def __init__(
        self,
        num_experts: int = 4,
        num_conv_layers: int = 3,
        aux_dim: int = 3,
        use_batchnorm: bool = True,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_conv_layers = num_conv_layers
        self.aux_dim = aux_dim
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate

        def _bn(channels: int) -> nn.Module:
            return nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()

        # 特徴抽出バックボーン
        channels = [32 * (2 ** i) for i in range(num_conv_layers)]
        layers: list[nn.Module] = []
        in_ch = 3
        for i, out_ch in enumerate(channels):
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), _bn(out_ch), nn.ReLU()]
            if i < num_conv_layers - 1:
                layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)

        # ゲートヘッド
        cnn_channels = channels[-1]
        head_in = cnn_channels + 1 + aux_dim
        head_layers: list[nn.Module] = [
            nn.Linear(head_in, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 8),
            nn.ReLU(),
        ]
        if num_experts == 2:
            head_layers += [nn.Linear(8, 1), nn.Sigmoid()]
        else:
            head_layers += [nn.Linear(8, num_experts), nn.Softmax(dim=1)]
        self.head = nn.Sequential(*head_layers)

    def forward(
        self,
        x: torch.Tensor,
        is_upper: torch.Tensor,
        aux_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:           (B, 3, H, W) 正規化 RGB パッチ
            is_upper:    (B, 1) 上まぶたフラグ
            aux_features: (B, aux_dim) 補助スカラー特徴量

        Returns:
            w: (B, num_experts) 重みベクトル（和=1）。
               num_experts=2 のときは (B, 1) w ∈ [0,1]。
               t_final = w * t_x0 + (1-w) * t_x2  (2 experts)
               t_final = (w * t_experts).sum(dim=1)   (3+ experts)
        """
        feat = self.features(x).flatten(1)
        if aux_features is None:
            aux_features = torch.zeros(x.size(0), self.aux_dim, device=x.device)
        combined = torch.cat([feat, is_upper, aux_features], dim=1)
        return self.head(combined)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
