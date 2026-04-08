"""LR スケジューラ設定。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim


def make_optimizer_and_scheduler(
    model: nn.Module,
    phase: str,
    lr: float,
    weight_decay: float,
    steps_per_epoch: int,
    num_epochs: int,
) -> tuple[optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """フェーズに応じたオプティマイザと OneCycleLR スケジューラを作成する。

    phase:
      "freeze"   : 学習可能パラメータ（ヘッドのみ）を lr で最適化
      "finetune" : 全パラメータを lr/10 で fine-tuning
      "full"     : 全パラメータを lr で最適化（custom のみ）
    """
    if phase == "freeze":
        params = [p for p in model.parameters() if p.requires_grad]
        base_lr = lr / 10
        max_lr = lr
    elif phase == "finetune":
        params = list(model.parameters())
        base_lr = lr / 100
        max_lr = lr / 10
    else:  # "full"
        params = list(model.parameters())
        base_lr = lr / 10
        max_lr = lr

    optimizer = optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
    )
    return optimizer, scheduler
