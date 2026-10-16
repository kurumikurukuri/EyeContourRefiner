"""EyeContourCNNDual: 2系統入力（生PQ + P'Q'）デュアルブランチ CNN。"""
from __future__ import annotations

import torch
import torch.nn as nn

FUSION_MODES = ("shared", "independent", "attention")
AUX_INPUT_DIM = 11  # pq_angle_cos/sin, polar_cos/sin, yaw/pitch/roll sin/cos, is_upper


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


class EyeContourCNNDual(nn.Module):
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
        use_aux_face_shape: bool = False,
        aux_dim: int | None = None,
        base_channel: int = 32,
        pool_width: int = 32,
        use_is_upper: bool = True,
    ) -> None:
        super().__init__()
        if fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {FUSION_MODES}")

        self.fusion_mode = fusion_mode
        self.num_conv_layers = num_conv_layers
        self.use_sigmoid = use_sigmoid
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        self.use_aux_face_shape = use_aux_face_shape
        self.base_channel = base_channel
        self.pool_width = pool_width
        # use_is_upper=False で is_upper concat を撤去（aux に集約する設計）。
        self.use_is_upper = use_is_upper
        # aux_dim 指定時（サブセット選択）はその次元、未指定なら use_aux_face_shape の
        # フル11次元（AUX_INPUT_DIM）で後方互換。
        self.aux_dim = aux_dim if aux_dim is not None else (AUX_INPUT_DIM if use_aux_face_shape else 0)

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
        head_in = cnn_flat + (1 if self.use_is_upper else 0) + self.aux_dim

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
        aux_features: torch.Tensor | None = None,
    ) -> tuple:
        B = x_a.size(0)

        if self.fusion_mode == "shared":
            fa = self.features(x_a).view(B, -1)
            fb = self.features(x_b).view(B, -1)
        else:
            fa = self.features_a(x_a).view(B, -1)
            fb = self.features_b(x_b).view(B, -1)

        parts_a: list[torch.Tensor] = [fa]
        parts_b: list[torch.Tensor] = [fb]
        if self.use_is_upper:
            parts_a.append(is_upper)
            parts_b.append(is_upper)
        if self.aux_dim > 0:
            if aux_features is None:
                raise ValueError(
                    f"aux_dim={self.aux_dim} だが forward に aux_features が渡されていない"
                )
            parts_a.append(aux_features)
            parts_b.append(aux_features)
        head_in_a = torch.cat(parts_a, dim=1) if len(parts_a) > 1 else fa
        head_in_b = torch.cat(parts_b, dim=1) if len(parts_b) > 1 else fb

        pred_a = self.head_a(head_in_a)
        pred_b = self.head_b(head_in_b)

        if self.fusion_mode == "attention":
            alpha = torch.sigmoid(self.alpha_net(torch.cat([fa, fb], dim=1)))
            return pred_a, pred_b, alpha

        return pred_a, pred_b
