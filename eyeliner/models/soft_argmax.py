"""SoftArgmaxEdgeCNN — Spatial Soft-Argmax ヘッドを持つアイライン検出 CNN。

MLP 回帰ヘッドの代わりに 1D ヒートマップ + Spatial Soft-Argmax を使用する。
出力 t は構造的に [0, 1] に収まる（use_t_norm=True 必須）。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SpatialSoftArgmax1D(nn.Module):
    """1次元 Spatial Soft-Argmax。

    ヒートマップを softmax で確率分布化し、位置の期待値を返す。

    Args:
        W: ヒートマップの幅（特徴マップの空間次元）
    """

    def __init__(self, W: int) -> None:
        super().__init__()
        positions = torch.linspace(0.0, 1.0, W)  # (W,)
        self.register_buffer("positions", positions)

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        Args:
            heatmap: (B, 1, W) 空間ヒートマップ

        Returns:
            t: (B, 1) 位置期待値 ∈ [0, 1]
        """
        probs = torch.softmax(heatmap, dim=2)               # (B, 1, W)
        t = (probs * self.positions).sum(dim=2)             # (B, 1)
        return t


class SoftArgmaxEdgeCNN(nn.Module):
    """Spatial Soft-Argmax ヘッドを持つ軽量 CNN。

    EdgeCNN と同一の特徴抽出バックボーンを使用し、
    MLP 回帰ヘッドを 1D Conv + Spatial Soft-Argmax に置き換えたモデル。

    入力: (B, 3, 64, 128) RGB パッチ + (B, 1) is_upper フラグ
          + オプションで (B, 1) pq_ratio, (B, 1) dist_from_center
    出力: (B, 1) スカラー t ∈ [0, 1]（構造的に保証）

    注意:
        - use_t_norm=True（デフォルト）でのみ使用すること
        - use_logit_target=False で使用すること
        - use_sigmoid フラグは互換性のため保持するが機能しない
    """

    def __init__(
        self,
        use_pq_ratio: bool = False,
        use_sigmoid: bool = False,
        use_batchnorm: bool = True,
        num_conv_layers: int = 3,
        use_distance_from_center: bool = False,
        feat_width: int = 32,
    ) -> None:
        super().__init__()
        self.use_pq_ratio = use_pq_ratio
        self.use_sigmoid = use_sigmoid          # 互換性のみ・機能なし
        self.use_batchnorm = use_batchnorm
        self.num_conv_layers = num_conv_layers
        self.use_distance_from_center = use_distance_from_center
        self.feat_width = feat_width

        def _bn(channels: int) -> nn.Module:
            return nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()

        # EdgeCNN と同一の特徴抽出バックボーン
        channels = [32 * (2 ** i) for i in range(num_conv_layers)]
        layers: list[nn.Module] = []
        in_ch = 3
        for i, out_ch in enumerate(channels):
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), _bn(out_ch), nn.ReLU()]
            if i < num_conv_layers - 1:
                layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool2d((1, feat_width)))
        self.features = nn.Sequential(*layers)

        # スカラー入力チャネル数（is_upper は常に含む）
        n_scalars = 1 + int(use_pq_ratio) + int(use_distance_from_center)
        cnn_channels = channels[-1]

        # 1D ヒートマップヘッド
        self.head = nn.Sequential(
            nn.Conv1d(cnn_channels + n_scalars, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )

        self.soft_argmax = SpatialSoftArgmax1D(feat_width)

    def forward(
        self,
        x: torch.Tensor,
        is_upper: torch.Tensor,
        pq_ratio: torch.Tensor | None = None,
        dist_from_center: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:               (B, 3, H, W) 正規化 RGB パッチ
            is_upper:        (B, 1) 上まぶたフラグ
            pq_ratio:        (B, 1) pq_dist / eye_scale（オプション）
            dist_from_center:(B, 1) 端点距離スコア（オプション）

        Returns:
            t: (B, 1) ∈ [0, 1]
        """
        # 特徴抽出: (B, C, 1, W)
        feat = self.features(x)                             # (B, C, 1, W)
        feat = feat.squeeze(2)                              # (B, C, W)

        # スカラー入力をブロードキャストして channel 次元に concat
        scalars = [is_upper]
        if self.use_pq_ratio and pq_ratio is not None:
            scalars.append(pq_ratio)
        if self.use_distance_from_center and dist_from_center is not None:
            scalars.append(dist_from_center)

        # (B, 1) → (B, 1, W) にブロードキャスト
        scalar_channels = [s.unsqueeze(2).expand(-1, -1, self.feat_width) for s in scalars]
        x_cat = torch.cat([feat] + scalar_channels, dim=1)  # (B, C+n, W)

        # ヒートマップ生成 → soft-argmax
        heatmap = self.head(x_cat)                          # (B, 1, W)
        return self.soft_argmax(heatmap)                    # (B, 1)
