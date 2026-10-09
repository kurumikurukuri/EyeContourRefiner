"""損失関数レジストリ。

新しい損失関数は LOSS_FN_REGISTRY に追加するだけで scripts/train.py 側の変更不要。
"""
from __future__ import annotations

import functools
import math
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


# ---------------------------------------------------------------------------
# EXP021: 「空間 × カーネル」2軸直積の損失
#
#   空間＼カーネル | L1          | Huber             | Wing
#   --------------|-------------|-------------------|--------
#   正規化t       | l1_t_loss   | huber_t_loss (t)  | wing_t_loss
#   目正規化px    | error_ratio | error_ratio_huber | wing_px_loss
#
# px 空間は残差を per-sample の scale = (label_t_max - label_t_min) * pq_ratio で
# スケールしたもの（pq_ratio = pq_dist / eye_scale）。scale=1 のとき px 空間損失は
# 対応する t 空間損失に厳密一致する（test_loss.py の不変量で保証）。
# ---------------------------------------------------------------------------

def l1_t_loss(pred: Tensor, y: Tensor) -> Tensor:
    """t空間での L1（MAE）損失。Huber の遷移点なし版（カーネル軸の対照）。"""
    return (pred - y).abs().mean()


def wing_t_loss(pred: Tensor, y: Tensor, omega: float = 0.1, epsilon: float = 0.01) -> Tensor:
    """t空間での Wing 損失（Feng et al. 2018, 顔ランドマーク回帰向け）。

        wing(x) = omega * ln(1 + |x|/epsilon)   if |x| < omega
                = |x| - C                        otherwise
        C = omega - omega * ln(1 + omega/epsilon)

    omega が非線形領域の幅、epsilon が原点付近の曲率を決める。
    小〜中誤差を対数的に強調し、Huber/L1 より端点精度を高める狙い。
    """
    dt = (pred - y).abs()
    c = omega - omega * math.log(1.0 + omega / epsilon)
    return torch.where(dt < omega, omega * torch.log1p(dt / epsilon), dt - c).mean()


def error_ratio_loss(pred: Tensor, y: Tensor, scale: Tensor) -> Tensor:
    """目正規化ピクセル空間での L1 損失（ユーザ提案の error_ratio）。

        loss = mean( |t_pred_raw - t_true_raw| * pq_dist / eye_scale )
             = mean( |pred - y| * scale )

    scale = (label_t_max - label_t_min) * pq_ratio を呼び出し元で計算して渡す。
    R 点の「目スケール正規化ピクセル局所化誤差」を直接最小化する評価整合型損失。
    """
    return ((pred - y).abs() * scale).mean()


def error_ratio_huber_loss(pred: Tensor, y: Tensor, scale: Tensor, delta: float = 0.05) -> Tensor:
    """目正規化ピクセル空間での Huber 損失（error_ratio の Huber カーネル版）。

    残差を px 単位 r = (pred - y) * scale にスケールしてから Huber を適用する。
    delta は px 空間での二次→線形遷移点。
    """
    r = (pred - y) * scale
    d = r.abs()
    return torch.where(d < delta, 0.5 * d**2 / delta, d - 0.5 * delta).mean()


def wing_px_loss(pred: Tensor, y: Tensor, scale: Tensor, omega: float = 0.1, epsilon: float = 0.01) -> Tensor:
    """目正規化ピクセル空間での Wing 損失（error_ratio と Wing の合成）。

    残差を px 単位 r = (pred - y) * scale にスケールしてから Wing を適用する。
    """
    r = (pred - y) * scale
    d = r.abs()
    c = omega - omega * math.log(1.0 + omega / epsilon)
    return torch.where(d < omega, omega * torch.log1p(d / epsilon), d - c).mean()


def dead_zone_huber_loss(
    pred: Tensor,
    y: Tensor,
    epsilon: float = 0.01,
    delta: float = 0.05,
) -> Tensor:
    """デッドゾーン付き Huber 損失（epsilon-insensitive Huber）。

    |残差| < epsilon の場合は損失 = 0（許容精度内なら勾配を与えない）。
    epsilon 以上の残差に対してシフトした Huber を適用する。

        loss = 0                              if |r| < epsilon
             = 0.5 * r'^2 / delta             if epsilon <= |r| < epsilon + delta
             = r' - 0.5 * delta               if |r| >= epsilon + delta

    ここで r' = |r| - epsilon（シフト済み残差）。

    アイライナー検出における「ほぼ正解（例: 残差 < 1px）なら penalize しない」
    というユースケースに対応する。epsilon=0.01 は正規化 t スケールで約 0.64px 相当。
    """
    dt = (pred - y).abs()
    dt_shifted = (dt - epsilon).clamp(min=0.0)
    return torch.where(
        dt_shifted < delta,
        0.5 * dt_shifted**2 / delta,
        dt_shifted - 0.5 * delta,
    ).mean()


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


def pq_weighted_huber_loss(
    pred: Tensor,
    y: Tensor,
    pq_w: Tensor,
    delta: float = 0.05,
) -> Tensor:
    """pq_ratio が小さいサンプルの損失を強調する Huber 損失。

    pq_w は呼び出し元で計算済みの重みテンソル。
    重み公式: 1.0 + alpha * max(0, threshold - pq_ratio) / threshold
    """
    dt = (pred - y).abs()
    loss = torch.where(dt < delta, 0.5 * dt**2 / delta, dt - 0.5 * delta)
    return (loss * pq_w).mean()


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
    "pq_weighted": pq_weighted_huber_loss,
    "normalized_pixel": eye_normalized_smooth_l1_loss,
    "dead_zone": dead_zone_huber_loss,
    # EXP021: 空間 × カーネル直積
    "t_l1": l1_t_loss,
    "wing": wing_t_loss,
    "error_ratio": error_ratio_loss,
    "error_ratio_huber": error_ratio_huber_loss,
    "wing_px": wing_px_loss,
}


def get_loss_fn(name: str, **kwargs) -> Callable:
    """名前で損失関数を取得する。未知の名前は ValueError を送出。

    kwargs を渡すと functools.partial でパラメータを固定した関数を返す。
    例: get_loss_fn("dead_zone", epsilon=0.01, delta=0.05)
    """
    if name not in LOSS_FN_REGISTRY:
        raise ValueError(
            f"未知の損失関数: {name!r}. 利用可能: {list(LOSS_FN_REGISTRY)}"
        )
    fn = LOSS_FN_REGISTRY[name]
    if kwargs:
        return functools.partial(fn, **kwargs)
    return fn
