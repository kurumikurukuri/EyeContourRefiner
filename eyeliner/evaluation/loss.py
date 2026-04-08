"""損失関数レジストリ。

新しい損失関数は LOSS_FN_REGISTRY に追加するだけで scripts/train.py 側の変更不要。
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


def huber_t_loss(pred: Tensor, y: Tensor, delta: float = 0.05) -> Tensor:
    """t空間での Huber 損失（正規化座標での直接回帰）。

    delta 未満の誤差は 0.5*dt^2/delta（二乗項）、
    delta 超は dt - 0.5*delta（線形項）として外れ値の影響を緩和する。
    """
    dt = (pred - y).abs()
    return torch.where(dt < delta, 0.5 * dt**2 / delta, dt - 0.5 * delta).mean()


def t_weighted_huber_loss(
    pred: Tensor,
    y: Tensor,
    t_weight: Tensor,
    delta: float = 0.05,
) -> Tensor:
    """t値の位置に応じた重み付き Huber 損失。

    輪郭端点付近（t が 0 や 1 に近い）のサンプルに高い重みを与えることで、
    端点予測の精度向上を狙う。
    t_weight は呼び出し元で計算済みのテンソル（例: 1 + α * |y - 0.5|）。
    """
    dt = (pred - y).abs()
    loss = torch.where(dt < delta, 0.5 * dt**2 / delta, dt - 0.5 * delta)
    return (loss * t_weight).mean()


def eye_normalized_smooth_l1_loss(
    pred: Tensor,
    y: Tensor,
    weight: Tensor,
    delta: float = 0.05,
) -> Tensor:
    """目スケール・アノテーション密度で正規化した Smooth L1（Huber）損失。

    weight = pq_dist / (eye_scale * n_pq) で目の大きさと密度を正規化する。
    delta 未満の誤差は二乗項、delta 超は線形項。
    """
    dt = (pred - y).abs()
    loss = torch.where(dt < delta, 0.5 * dt**2 / delta, dt - 0.5 * delta)
    return (loss * weight).mean()


# 新損失関数はここに追加するだけ（スクリプト変更不要）
LOSS_FN_REGISTRY: dict[str, Callable] = {
    "t": huber_t_loss,
    "t_weighted": t_weighted_huber_loss,
    "normalized_pixel": eye_normalized_smooth_l1_loss,
}


def get_loss_fn(name: str) -> Callable:
    """名前で損失関数を取得する。未知の名前は ValueError を送出。"""
    if name not in LOSS_FN_REGISTRY:
        raise ValueError(
            f"未知の損失関数: {name!r}. 利用可能: {list(LOSS_FN_REGISTRY)}"
        )
    return LOSS_FN_REGISTRY[name]
