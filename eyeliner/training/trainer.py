"""1エポック訓練・検証ループ。"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from eyeliner.evaluation.loss import get_loss_fn
from utils.patch_utils import IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """訓練設定（ハイパーパラメータ）。スクリプト側で定義し、trainer に渡す。"""

    backbone: str = "custom"
    lr: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 1e-4
    loss_fn: str = "normalized_pixel"
    freeze_epochs: int = 10
    epochs: int = 200
    patience: int = 50
    seed: int = 42
    round_t: int | None = None
    pretrained: bool = True
    norm_mean: list = field(default_factory=lambda: list(IMAGENET_MEAN))
    norm_std: list = field(default_factory=lambda: list(IMAGENET_STD))
    use_pq_ratio: bool = False
    use_sigmoid: bool = False
    use_batchnorm: bool = True
    num_conv_layers: int = 3
    use_t_weighted_sampler: bool = False
    t_weight_alpha: float = 3.0
    use_t_weighted_loss: bool = False
    t_loss_alpha: float = 3.0
    loss_epsilon: float = 0.01  # dead_zone 損失のデッドゾーン幅（正規化 t スケール）
    loss_delta: float = 0.05   # dead_zone / huber 損失の二次→線形遷移点
    use_distance_from_center: bool = False
    use_norm_eye_scale: bool = False
    use_logit_target: bool = False
    use_t_norm: bool = True
    patch_t_min: float = -0.5
    patch_t_max: float =  1.5
    label_t_min: float | None = None
    label_t_max: float | None = None
    patch_perp_scale: float = 1.0
    patch_h: int = 64
    patch_w: int = 128
    aug_flags: int = 0x4F  # AUG_ALL: 全オーグメンテーション有効


@dataclass
class EpochResult:
    """1エポックの訓練・検証結果。"""

    train_loss: float
    train_mae: float
    val_loss: float
    val_mae: float
    val_pixel_mae: float
    val_relative_mae: float


def set_seed(seed: int = 42) -> None:
    """再現性のためにランダムシードを固定する。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_fn_name: str,
    device: torch.device,
    use_pq_ratio: bool = False,
    use_distance_from_center: bool = False,
    use_norm_eye_scale: bool = False,
    use_logit_target: bool = False,
    t_loss_alpha: float = 3.0,
    loss_epsilon: float = 0.01,
    loss_delta: float = 0.05,
) -> tuple[float, float]:
    """1エポック訓練を実行する。

    Returns:
        (avg_train_loss, avg_train_mae)
    """
    model.train()
    if loss_fn_name == "dead_zone":
        loss_fn = get_loss_fn(loss_fn_name, epsilon=loss_epsilon, delta=loss_delta)
    else:
        loss_fn = get_loss_fn(loss_fn_name)
    total_loss = 0.0
    total_abs_error = 0.0
    total_elements = 0
    n_batches = 0

    for x, is_upper, pq_ratio, y, weight, _, dist_from_center, norm_eye_scale in loader:
        x = x.to(device)
        is_upper = is_upper.to(device)
        y = y.to(device)
        w_train = weight.to(device)

        optimizer.zero_grad()
        pq_ratio_dev = pq_ratio.to(device) if use_pq_ratio else None
        dfc_dev = dist_from_center.to(device) if use_distance_from_center else None
        nes_dev = norm_eye_scale.to(device) if use_norm_eye_scale else None
        pred = model(x, is_upper, pq_ratio_dev, dfc_dev, nes_dev)

        y_target = torch.logit(y.clamp(1e-4, 1 - 1e-4)) if use_logit_target else y
        if loss_fn_name in ("t", "dead_zone"):
            loss = loss_fn(pred, y_target)
        elif loss_fn_name == "t_weighted":
            t_w = 1.0 + t_loss_alpha * (y - 0.5).abs()
            loss = loss_fn(pred, y_target, t_w)
        else:
            loss = loss_fn(pred, y_target, w_train)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        with torch.no_grad():
            pred_t = torch.sigmoid(pred) if use_logit_target else pred
            total_abs_error += (pred_t - y).abs().sum().item()
            total_elements += y.numel()

    avg_loss = total_loss / max(n_batches, 1)
    avg_mae = total_abs_error / max(total_elements, 1)
    return avg_loss, avg_mae


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_pq_ratio: bool = False,
    use_distance_from_center: bool = False,
    use_norm_eye_scale: bool = False,
    use_logit_target: bool = False,
) -> EpochResult:
    """1エポック検証を実行する。

    Returns:
        EpochResult（val_mae を Early Stopping の基準として使用）
    """
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_val_weights: list[torch.Tensor] = []
    all_rel_weights: list[torch.Tensor] = []

    with torch.no_grad():
        for x, is_upper, pq_ratio, y, weight, val_weight, dist_from_center, norm_eye_scale in loader:
            x = x.to(device)
            is_upper = is_upper.to(device)
            pq_ratio_dev = pq_ratio.to(device) if use_pq_ratio else None
            dfc_dev = dist_from_center.to(device) if use_distance_from_center else None
            nes_dev = norm_eye_scale.to(device) if use_norm_eye_scale else None
            pred = model(x, is_upper, pq_ratio_dev, dfc_dev, nes_dev)
            if use_logit_target:
                pred = torch.sigmoid(pred)
            all_preds.append(pred.cpu())
            all_labels.append(y)
            all_val_weights.append(val_weight)
            all_rel_weights.append(weight)

    preds_t = torch.cat(all_preds)
    labels_t = torch.cat(all_labels)
    val_vws = torch.cat(all_val_weights)
    val_rws = torch.cat(all_rel_weights)

    dt = (preds_t - labels_t).abs()
    mae = dt.mean().item()
    pixel_mae = (dt * val_vws).mean().item()
    relative_mae = (dt * val_rws).mean().item()

    return EpochResult(
        train_loss=0.0,
        train_mae=0.0,
        val_loss=mae,
        val_mae=mae,
        val_pixel_mae=pixel_mae,
        val_relative_mae=relative_mae,
    )


