"""dual_kfold の fold 学習に EMA / SWA の派生モデル生成を接続する（EXP034）。

派生モデルの評価・追加学習は乱数状態を保存・復元して行い、base（および次 fold）の
学習の乱数列に影響させない。
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from eyeliner.data.dataset_dual import BranchConfig
from eyeliner.models.dual_ensemble import load_dual_model
from eyeliner.training.dual_epoch import (
    make_bn_forward,
    save_variant_fold,
    train_one_epoch,
)
from eyeliner.training.weight_averaging import (
    EmaTracker,
    preserve_rng_state,
    run_swa,
    variant_dir,
)

if TYPE_CHECKING:
    from eyeliner.training.dual_kfold import DualTrainConfig

import logging

logger = logging.getLogger(__name__)


def track_ema_best(
    ema: EmaTracker,
    ema_best: dict[str, dict],
    val_mae_of: Callable[[torch.nn.Module], float],
    exp_dir: Path,
    fold_idx: int,
    epoch: int,
    config: DualTrainConfig,
) -> dict[str, dict]:
    """各 EMA モデルの val を評価し、最良を更新したものを保存した新しい ema_best を返す。"""
    updated: dict[str, dict] = {}
    with preserve_rng_state():
        for name, averaged in ema.items():
            mae = val_mae_of(averaged)
            logger.info("Fold %d Epoch %3d  [%s] val_mae=%.6f", fold_idx, epoch, name, mae)
            if mae < ema_best[name]["val_mae"]:
                info = {"val_mae": mae, "epoch": epoch}
                save_variant_fold(
                    variant_dir(exp_dir, name), fold_idx, averaged.module,
                    config.fusion_mode, config.__dict__, info,
                )
                updated[name] = info
            else:
                updated[name] = ema_best[name]
    return updated


def run_fold_swa(
    model_path: Path,
    train_loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    val_mae_of: Callable[[torch.nn.Module], float],
    exp_dir: Path,
    fold_idx: int,
    config: DualTrainConfig,
    cfg_a: BranchConfig,
    cfg_b: BranchConfig,
    aux_features: list[str],
    device: torch.device,
) -> None:
    """base 最良重みから SWA を行い `{exp_dir}_swa` に保存する（乱数状態は保存・復元）。"""
    with preserve_rng_state():
        base_model = load_dual_model(model_path).to(device)
        result = run_swa(
            base_model,
            lambda m, opt: train_one_epoch(
                m, train_loader, opt, None, loss_fn, config.fusion_mode,
                cfg_a, cfg_b, aux_features, device,
            ),
            swa_epochs=config.swa_epochs,
            swa_lr=config.swa_lr,
            weight_decay=config.weight_decay,
            bn_loader=train_loader,
            bn_forward=make_bn_forward(aux_features, device),
            eval_fn=val_mae_of,
        )
        swa_mae = val_mae_of(result.model)
    logger.info(
        "Fold %d: SWA val_mae=%.6f（%d epoch, lr=%.1e, 平均前 val=%s）",
        fold_idx, swa_mae, config.swa_epochs, config.swa_lr,
        ", ".join(f"{v:.6f}" for v in result.epoch_val_maes),
    )
    save_variant_fold(
        variant_dir(exp_dir, "swa"), fold_idx, result.model, config.fusion_mode,
        config.__dict__,
        {"val_mae": swa_mae, "swa_epoch_val_maes": result.epoch_val_maes},
    )
