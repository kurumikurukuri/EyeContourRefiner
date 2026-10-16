"""学習リジューム用のエポック単位チェックポイント保存・復元。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import optim


def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    best_mae: float,
    best_epoch: int,
    no_improve: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """epoch 完了時点の学習状態を path に保存する（中断時のリジューム用）。"""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_mae": best_mae,
            "best_epoch": best_epoch,
            "no_improve": no_improve,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path: str | Path) -> dict[str, Any] | None:
    """チェックポイントを読み込む。ファイルが存在しなければ None を返す。"""
    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def remove_checkpoint(path: str | Path) -> None:
    """チェックポイントファイルを削除する（存在しなくても no-op）。"""
    Path(path).unlink(missing_ok=True)
