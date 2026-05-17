"""K-Fold CV オーケストレーション。"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler
from eyeliner.data.dataset import EyelinerPatchDataset, load_manifest
from eyeliner.data.weights import compute_eye_meta
from eyeliner.models.ensemble import build_model, save_model
from eyeliner.training.scheduler import make_optimizer_and_scheduler
from eyeliner.training.trainer import (
    TrainConfig,
    EpochResult,
    set_seed,
    train_one_epoch,
    validate_one_epoch,
)

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """1 Fold の訓練結果。"""

    fold_idx: int
    best_val_mae: float
    best_epoch: int
    model_path: str


def run_kfold(
    manifest_path: str | Path,
    exp_dir: str | Path,
    config: TrainConfig,
    k_folds: int = 5,
    holdout_path: str | Path | None = None,
    max_train_samples: int | None = None,
    mid_t_keep_ratio: float = 1.0,
) -> list[FoldResult]:
    """K-Fold CV 訓練を実行し、各 Fold のベストモデルを exp_dir に保存する。

    Args:
        manifest_path:     manifest.csv のパス（訓練・CV 用、Hold-out を除いたもの）
        exp_dir:           実験ディレクトリ（モデル・ログを保存）
        config:            訓練設定
        k_folds:           Fold 数
        holdout_path:      Hold-out セットのパス（現在は未使用、Phase 2 で利用）
        max_train_samples: 訓練サンプル上限（None = 全件使用）。超過時はシード固定ランダムサブサンプリング
        mid_t_keep_ratio:  0.5 < t < 0.9 サンプルの保持率（1.0 = 全件保持）

    Returns:
        list[FoldResult] — 各 Fold の結果
    """
    set_seed(config.seed)

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    records = load_manifest(manifest_path)
    compute_eye_meta(records)

    if max_train_samples is not None and len(records) > max_train_samples:
        original_count = len(records)
        rng = np.random.RandomState(config.seed)
        chosen = sorted(rng.choice(original_count, size=max_train_samples, replace=False))
        records = [records[i] for i in chosen]
        logger.info("サブサンプリング: %d → %d サンプル", original_count, len(records))

    if mid_t_keep_ratio < 1.0:
        mid_mask = [0.5 < float(r["t"]) < 0.9 for r in records]
        mid_records = [r for r, m in zip(records, mid_mask) if m]
        other_records = [r for r, m in zip(records, mid_mask) if not m]
        n_keep = max(1, int(len(mid_records) * mid_t_keep_ratio))
        rng_mid = np.random.RandomState(config.seed + 1)
        chosen_mid = sorted(rng_mid.choice(len(mid_records), size=n_keep, replace=False))
        kept_mid = [mid_records[i] for i in chosen_mid]
        records = other_records + kept_mid
        rng_mid.shuffle(records)
        logger.info(
            "mid-t フィルタ (0.5<t<0.9): %d → %d 件 (keep_ratio=%.2f)",
            len(mid_records), n_keep, mid_t_keep_ratio,
        )

    logger.info("訓練データ: %d サンプル", len(records))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("デバイス: %s", device)

    groups = [r["image_path"] for r in records]
    use_transfer = config.backbone not in {"custom", "soft_argmax"}
    effective_freeze = config.freeze_epochs if use_transfer else 0

    if k_folds == 1:
        # k=1: GroupShuffleSplit による固定 80/20 分割（CV なし・高速）
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.seed)
        splits = list(gss.split(records, groups=groups))
        logger.info("k=1 モード: GroupShuffleSplit(test_size=0.2, seed=%d)", config.seed)
    else:
        kf = GroupKFold(n_splits=k_folds)
        splits = list(kf.split(records, groups=groups))

    fold_results: list[FoldResult] = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info("\n===== Fold %d/%d =====", fold + 1, k_folds)

        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]

        train_ds = EyelinerPatchDataset(
            train_records, augment_data=True,
            aug_flags=config.aug_flags,
            t_quant_step=config.t_quant_step,
            norm_mean=config.norm_mean,
            norm_std=config.norm_std,
            patch_t_min=config.patch_t_min,
            patch_t_max=config.patch_t_max,
            patch_perp_scale=config.patch_perp_scale,
            patch_size=(config.patch_h, config.patch_w),
            use_t_norm=config.use_t_norm,
            label_t_min=config.label_t_min,
            label_t_max=config.label_t_max,
        )
        val_ds = EyelinerPatchDataset(
            val_records, augment_data=False,
            t_quant_step=config.t_quant_step,
            norm_mean=config.norm_mean,
            norm_std=config.norm_std,
            patch_t_min=config.patch_t_min,
            patch_t_max=config.patch_t_max,
            patch_perp_scale=config.patch_perp_scale,
            patch_size=(config.patch_h, config.patch_w),
            use_t_norm=config.use_t_norm,
            label_t_min=config.label_t_min,
            label_t_max=config.label_t_max,
        )

        if config.use_t_weighted_sampler:
            t_values = [float(r["t"]) for r in train_records]
            # t は生値（raw）のため、重み計算前に [0,1] に正規化する
            t_min_eff = config.label_t_min if config.label_t_min is not None else min(t_values)
            t_max_eff = config.label_t_max if config.label_t_max is not None else max(t_values)
            t_range = max(t_max_eff - t_min_eff, 1e-6)
            t_norm = [(t - t_min_eff) / t_range for t in t_values]
            weights = [1.0 + config.t_weight_alpha * abs(tn - 0.5) for tn in t_norm]
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler)
            logger.info(
                "WeightedRandomSampler 有効: alpha=%.1f, weight range=[%.3f, %.3f]",
                config.t_weight_alpha, min(weights), max(weights),
            )
        else:
            train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

        model = build_model(
            config.backbone, pretrained=config.pretrained,
            use_pq_ratio=config.use_pq_ratio, use_sigmoid=config.use_sigmoid,
            use_batchnorm=config.use_batchnorm,
            num_conv_layers=config.num_conv_layers,
            use_distance_from_center=config.use_distance_from_center,
            use_norm_eye_scale=config.use_norm_eye_scale,
            use_is_upper=config.use_is_upper,
            dropout_rate=config.dropout_rate,
        ).to(device)

        # Phase 1: バックボーン凍結（転移学習時のみ）
        if effective_freeze > 0:
            model.freeze_backbone()
            phase1_epochs = min(effective_freeze, config.epochs)
            optimizer, scheduler = make_optimizer_and_scheduler(
                model, "freeze", config.lr, config.weight_decay,
                len(train_loader), phase1_epochs,
            )
        else:
            optimizer, scheduler = make_optimizer_and_scheduler(
                model, "full", config.lr, config.weight_decay,
                len(train_loader), config.epochs,
            )

        best_mae = float("inf")
        no_improve = 0
        best_epoch = 0
        phase_switched = False

        for epoch in range(config.epochs):

            # Phase 2 切り替え
            if effective_freeze > 0 and not phase_switched and epoch >= effective_freeze:
                remaining = config.epochs - effective_freeze
                if remaining <= 0:
                    break
                model.unfreeze_backbone()
                optimizer, scheduler = make_optimizer_and_scheduler(
                    model, "finetune", config.lr, config.weight_decay,
                    len(train_loader), remaining,
                )
                phase_switched = True
                no_improve = 0
                logger.info(
                    "Fold %d Epoch %d: バックボーン解凍 → fine-tuning 開始",
                    fold + 1, epoch + 1,
                )

            train_loss, train_mae = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                config.loss_fn, device,
                use_pq_ratio=config.use_pq_ratio,
                use_distance_from_center=config.use_distance_from_center,
                use_norm_eye_scale=config.use_norm_eye_scale,
                use_logit_target=config.use_logit_target,
                t_loss_alpha=config.t_loss_alpha,
                loss_epsilon=config.loss_epsilon,
                loss_delta=config.loss_delta,
            )
            result = validate_one_epoch(
                model, val_loader, device,
                use_pq_ratio=config.use_pq_ratio,
                use_distance_from_center=config.use_distance_from_center,
                use_norm_eye_scale=config.use_norm_eye_scale,
                use_logit_target=config.use_logit_target,
            )
            current_lr = scheduler.get_last_lr()[0]

            logger.info(
                "Fold %d Epoch %3d  train_loss=%.6f  train_mae=%.6f  "
                "val_mae=%.6f  lr=%.2e",
                fold + 1, epoch + 1, train_loss, train_mae, result.val_mae, current_lr,
            )

            # Early Stopping（MAE 基準）
            if result.val_mae < best_mae:
                best_mae = result.val_mae
                best_epoch = epoch + 1
                no_improve = 0
                model_path = exp_dir / f"eyeliner_detection_fold{fold + 1}.pt"
                save_model(model, config.backbone, model_path)
            else:
                no_improve += 1
                if no_improve >= config.patience:
                    logger.info("Early stopping triggered (fold %d)", fold + 1)
                    break

        fold_results.append(FoldResult(
            fold_idx=fold + 1,
            best_val_mae=best_mae,
            best_epoch=best_epoch,
            model_path=str(exp_dir / f"eyeliner_detection_fold{fold + 1}.pt"),
        ))
        logger.info("Fold %d: Best MAE=%.6f (epoch %d)", fold + 1, best_mae, best_epoch)

        del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # K-Fold サマリー
    mean_mae = float(np.mean([r.best_val_mae for r in fold_results]))
    std_mae = float(np.std([r.best_val_mae for r in fold_results]))
    logger.info("\n===== K-Fold Result =====")
    for r in fold_results:
        logger.info("Fold %d: Best MAE=%.6f", r.fold_idx, r.best_val_mae)
    logger.info("Mean MAE: %.6f +/- %.6f", mean_mae, std_mae)

    return fold_results
