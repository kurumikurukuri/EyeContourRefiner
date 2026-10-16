"""K-Fold CV オーケストレーション。"""
from __future__ import annotations

import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler
from eye_contour.data.dataset import EyeContourPatchDataset, load_manifest
from eye_contour.data.patch_utils import get_pq_prime_dist, invert_pq_rescale_t
from eye_contour.data.weights import compute_eye_meta, compute_aux_face_shape_meta
from eye_contour.models.ensemble import build_model, load_model, save_model
from eye_contour.training.checkpoint import load_checkpoint, remove_checkpoint, save_checkpoint
from eye_contour.training.oof import OofPrediction
from eye_contour.training.scheduler import make_optimizer_and_scheduler
from eye_contour.training.trainer import (
    TrainConfig,
    EpochResult,
    set_seed,
    train_one_epoch,
    validate_one_epoch,
)

logger = logging.getLogger(__name__)

_PROGRESS_FILENAME = "kfold_progress.json"


def _load_progress(exp_dir: Path) -> dict[int, "FoldResult"]:
    """exp_dir の進捗ファイルから完了済み fold の結果を読み込む（無ければ空 dict）。"""
    path = exp_dir / _PROGRESS_FILENAME
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): FoldResult(**v) for k, v in data.items()}


def _save_progress(exp_dir: Path, completed: dict[int, "FoldResult"]) -> None:
    """完了済み fold の結果を進捗ファイルに書き出す（次回起動時のスキップ判定用）。"""
    path = exp_dir / _PROGRESS_FILENAME
    data = {
        str(k): {
            "fold_idx": v.fold_idx,
            "best_val_mae": v.best_val_mae,
            "best_epoch": v.best_epoch,
            "model_path": v.model_path,
        }
        for k, v in completed.items()
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class FoldResult:
    """1 Fold の訓練結果。"""

    fold_idx: int
    best_val_mae: float
    best_epoch: int
    model_path: str


def _label_range_and_min(config: TrainConfig) -> tuple[float, float]:
    """目正規化px空間の損失・OOF生t変換用の (label_range, label_t_min) を計算する。

    t_norm 無効時は残差・予測が既に raw t なので range=1.0, min=0.0（無変換）。
    label_t_* 未指定時は patch_t_* に従う。
    """
    if config.use_t_norm:
        lt_min = config.label_t_min if config.label_t_min is not None else config.patch_t_min
        lt_max = config.label_t_max if config.label_t_max is not None else config.patch_t_max
        return max(lt_max - lt_min, 1e-6), lt_min
    return 1.0, 0.0


def _to_raw_t(t_pred: float, rec: dict, config: TrainConfig) -> float:
    """pq_rescale 学習モデルの予測 t'（P'Q' 座標系）を生 PQ 座標系の t に戻す。"""
    if config.pq_rescale_mode == "none":
        return t_pred
    pq_dist = float(np.hypot(float(rec["qx"]) - float(rec["px"]), float(rec["qy"]) - float(rec["py"])))
    pq_prime_dist = get_pq_prime_dist(rec, pq_dist, config.pq_rescale_mode, config.pq_rescale_factor)
    return invert_pq_rescale_t(t_pred, pq_dist, pq_prime_dist)


def _build_oof_predictions(
    val_records: list[dict],
    val_preds: torch.Tensor,
    config: TrainConfig,
    label_range: float,
    label_t_min: float,
    fold_idx: int,
) -> list[OofPrediction]:
    """検証セットの per-sample 予測（正規化空間）を生 t に変換し OofPrediction 化する。

    val_loader は shuffle=False のため、val_preds の順序は val_records と一致する。
    t_true は manifest の生 t（rec["t"]）をそのまま使う（infer_*.py と同じ convention）。
    pq_rescale_mode != "none" のモデルは P'Q' 座標系の t' を予測するため、生 t に逆変換して揃える。
    """
    preds = val_preds.reshape(-1).tolist()
    oof: list[OofPrediction] = []
    for rec, pred_norm in zip(val_records, preds):
        t_pred = pred_norm * label_range + label_t_min if config.use_t_norm else pred_norm
        t_pred = _to_raw_t(t_pred, rec, config)
        oof.append(OofPrediction(
            idx=str(rec.get("idx", "")),
            image_path=rec["image_path"],
            part_name=rec.get("part_name", ""),
            sub_key=rec.get("sub_key", ""),
            point_index=str(rec.get("point_index", "")),
            t_true=float(rec["t"]),
            t_pred=float(t_pred),
            fold_idx=fold_idx,
        ))
    return oof


def _predict_records(
    model_path: str | Path,
    records: list[dict],
    config: TrainConfig,
    label_range: float,
    label_t_min: float,
    device: torch.device,
    dl_kwargs: dict,
    fold_idx: int,
) -> list[OofPrediction]:
    """保存済みモデルで records を推論し OOF 予測（生t）を生成する。

    augment_data=False・shuffle=False の専用推論パス。_train_single_fold の早期終了判定
    （別の検証セットを使う場合がある）とは独立しており、どの records 集合にも使える。
    """
    ds = EyeContourPatchDataset(
        records, augment_data=False,
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
        use_aux_face_shape=config.use_aux_face_shape,
        pq_rescale_mode=config.pq_rescale_mode,
        pq_rescale_factor=config.pq_rescale_factor,
    )
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False, **dl_kwargs)

    model = load_model(model_path).to(device)
    model.eval()

    all_preds: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            x                = batch[0].to(device)
            is_upper         = batch[1].to(device)
            pq_ratio         = batch[2]
            dist_from_center = batch[6]
            norm_eye_scale   = batch[7]
            pq_ratio_dev = pq_ratio.to(device) if config.use_pq_ratio else None
            dfc_dev      = dist_from_center.to(device) if config.use_distance_from_center else None
            nes_dev      = norm_eye_scale.to(device) if config.use_norm_eye_scale else None
            aux_dev      = batch[8].to(device) if config.use_aux_face_shape else None
            pred = model(x, is_upper, pq_ratio_dev, dfc_dev, nes_dev, aux_dev)
            if config.use_logit_target:
                pred = torch.sigmoid(pred)
            all_preds.append(pred.cpu())

    preds = torch.cat(all_preds)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _build_oof_predictions(records, preds, config, label_range, label_t_min, fold_idx)


def _train_single_fold(
    train_records: list[dict],
    val_records: list[dict],
    fold_idx: int,
    exp_dir: Path,
    config: TrainConfig,
    label_range: float,
    device: torch.device,
    dl_kwargs: dict,
    use_transfer: bool,
    effective_freeze: int,
    resume: bool = True,
) -> FoldResult:
    """1 Fold 分の訓練を実行し、ベストモデル（val_mae 最小エポック）を exp_dir に保存する。

    OOF 予測の生成はこの関数の責務外（_predict_records を別途呼ぶこと）。
    train_records/val_records は早期終了の判定にのみ使われ、保存されるモデルは
    val_records を一切学習に使わない。

    resume=True かつ exp_dir に checkpoint_fold{fold_idx}.pt が残っている場合、
    そのエポックから学習を再開する（中断復旧用）。Fold が正常完了すると
    チェックポイントは削除される。
    """
    train_ds = EyeContourPatchDataset(
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
        use_aux_face_shape=config.use_aux_face_shape,
        pq_rescale_mode=config.pq_rescale_mode,
        pq_rescale_factor=config.pq_rescale_factor,
        endpoint_blend_range=config.endpoint_blend_range,
    )
    val_ds = EyeContourPatchDataset(
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
        use_aux_face_shape=config.use_aux_face_shape,
        pq_rescale_mode=config.pq_rescale_mode,
        pq_rescale_factor=config.pq_rescale_factor,
    )

    if config.use_t_weighted_sampler:
        t_values = [float(r["t"]) for r in train_records]
        # t は生値（raw）のため、重み計算前に [0,1] に正規化する
        t_min_eff = config.label_t_min if config.label_t_min is not None else min(t_values)
        t_max_eff = config.label_t_max if config.label_t_max is not None else max(t_values)
        t_range = max(t_max_eff - t_min_eff, 1e-6)
        t_norm = [(t - t_min_eff) / t_range for t in t_values]
        sample_weights = [1.0 + config.t_weight_alpha * abs(tn - 0.5) for tn in t_norm]
    else:
        sample_weights = [1.0] * len(train_records)

    if config.use_pq_weighted_sampler:
        pq_values = [float(r.get("pq_ratio", 1.0)) for r in train_records]
        pq_w = [1.0 + config.pq_weight_alpha * max(0.0, config.pq_threshold - pq) / config.pq_threshold for pq in pq_values]
        sample_weights = [sw * pw for sw, pw in zip(sample_weights, pq_w)]

    use_sampler = config.use_t_weighted_sampler or config.use_pq_weighted_sampler
    if use_sampler:
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler, **dl_kwargs)
        logger.info(
            "WeightedRandomSampler 有効: weight range=[%.3f, %.3f]",
            min(sample_weights), max(sample_weights),
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, **dl_kwargs)

    model = build_model(
        config.backbone, pretrained=config.pretrained,
        use_pq_ratio=config.use_pq_ratio, use_sigmoid=config.use_sigmoid,
        use_batchnorm=config.use_batchnorm,
        num_conv_layers=config.num_conv_layers,
        use_distance_from_center=config.use_distance_from_center,
        use_norm_eye_scale=config.use_norm_eye_scale,
        use_is_upper=config.use_is_upper,
        dropout_rate=config.dropout_rate,
        use_aux_face_shape=config.use_aux_face_shape,
        pool_width=config.pool_width,
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
    start_epoch = 0
    model_path = exp_dir / f"eyeliner_detection_fold{fold_idx}.pt"
    checkpoint_path = exp_dir / f"checkpoint_fold{fold_idx}.pt"

    ckpt = load_checkpoint(checkpoint_path) if resume else None
    if ckpt is not None:
        start_epoch = ckpt["epoch"]
        best_mae = ckpt["best_mae"]
        best_epoch = ckpt["best_epoch"]
        no_improve = ckpt["no_improve"]
        if bool(ckpt["extra"].get("phase_switched", False)) and effective_freeze > 0:
            # Phase 2（fine-tuning）に切り替わった状態から再開する場合、
            # optimizer/scheduler を fine-tuning フェーズの構造で作り直してから
            # state_dict をロードする（freeze フェーズの optimizer とは形が異なるため）。
            remaining = config.epochs - effective_freeze
            model.unfreeze_backbone()
            optimizer, scheduler = make_optimizer_and_scheduler(
                model, "finetune", config.lr, config.weight_decay,
                len(train_loader), remaining,
            )
            phase_switched = True
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        logger.info(
            "Fold %d: チェックポイントから再開 (epoch %d〜, best_mae=%.6f)",
            fold_idx, start_epoch + 1, best_mae,
        )

    for epoch in range(start_epoch, config.epochs):

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
                fold_idx, epoch + 1,
            )

        train_loss, train_mae = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            config.loss_fn, device,
            use_pq_ratio=config.use_pq_ratio,
            use_distance_from_center=config.use_distance_from_center,
            use_norm_eye_scale=config.use_norm_eye_scale,
            use_logit_target=config.use_logit_target,
            use_aux_face_shape=config.use_aux_face_shape,
            t_loss_alpha=config.t_loss_alpha,
            loss_epsilon=config.loss_epsilon,
            loss_delta=config.loss_delta,
            pq_weight_alpha=config.pq_weight_alpha,
            pq_threshold=config.pq_threshold,
            label_range=label_range,
        )
        result = validate_one_epoch(
            model, val_loader, device,
            use_pq_ratio=config.use_pq_ratio,
            use_distance_from_center=config.use_distance_from_center,
            use_norm_eye_scale=config.use_norm_eye_scale,
            use_logit_target=config.use_logit_target,
            use_aux_face_shape=config.use_aux_face_shape,
        )
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            "Fold %d Epoch %3d  train_loss=%.6f  train_mae=%.6f  "
            "val_mae=%.6f  lr=%.2e",
            fold_idx, epoch + 1, train_loss, train_mae, result.val_mae, current_lr,
        )

        # Early Stopping（MAE 基準）
        if result.val_mae < best_mae:
            best_mae = result.val_mae
            best_epoch = epoch + 1
            no_improve = 0
            save_model(model, config.backbone, model_path)
        else:
            no_improve += 1

        save_checkpoint(
            checkpoint_path, epoch=epoch + 1, model=model, optimizer=optimizer,
            scheduler=scheduler, best_mae=best_mae, best_epoch=best_epoch,
            no_improve=no_improve, extra={"phase_switched": phase_switched},
        )

        if no_improve >= config.patience:
            logger.info("Early stopping triggered (fold %d)", fold_idx)
            break

    fold_result = FoldResult(
        fold_idx=fold_idx,
        best_val_mae=best_mae,
        best_epoch=best_epoch,
        model_path=str(model_path),
    )
    logger.info("Fold %d: Best MAE=%.6f (epoch %d)", fold_idx, best_mae, best_epoch)
    remove_checkpoint(checkpoint_path)

    del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return fold_result


def run_kfold(
    manifest_path: str | Path,
    exp_dir: str | Path,
    config: TrainConfig,
    k_folds: int = 5,
    holdout_path: str | Path | None = None,
    max_train_samples: int | None = None,
    mid_t_keep_ratio: float = 1.0,
    db_path: "str | Path | None" = None,
    resume: bool = True,
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
        resume:            True の場合、exp_dir に残る進捗ファイル・チェックポイントから
                            再開する（完了済み fold はスキップ、中断中の fold は
                            エポック単位で再開）。False なら完全に最初から学習する。

    Returns:
        list[FoldResult] — 各 Fold の結果
    """
    set_seed(config.seed)

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    records = load_manifest(manifest_path)
    compute_eye_meta(records)
    if config.use_aux_face_shape:
        compute_aux_face_shape_meta(records, db_path)

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

    label_range, _ = _label_range_and_min(config)

    if k_folds == 1:
        # k=1: GroupShuffleSplit による固定 80/20 分割（CV なし・高速）
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.seed)
        splits = list(gss.split(records, groups=groups))
        logger.info("k=1 モード: GroupShuffleSplit(test_size=0.2, seed=%d)", config.seed)
    else:
        kf = GroupKFold(n_splits=k_folds)
        splits = list(kf.split(records, groups=groups))

    dl_kwargs = {"num_workers": config.num_workers, "persistent_workers": config.num_workers > 0}
    completed_folds = _load_progress(exp_dir) if resume else {}
    fold_results: list[FoldResult] = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        fold_idx = fold + 1

        if fold_idx in completed_folds:
            logger.info(
                "Fold %d/%d: 完了済み (best_val_mae=%.6f) のためスキップ",
                fold_idx, k_folds, completed_folds[fold_idx].best_val_mae,
            )
            fold_results.append(completed_folds[fold_idx])
            continue

        logger.info("\n===== Fold %d/%d =====", fold_idx, k_folds)

        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]

        fold_result = _train_single_fold(
            train_records, val_records, fold_idx, exp_dir, config,
            label_range, device, dl_kwargs,
            use_transfer, effective_freeze,
            resume=resume,
        )
        fold_results.append(fold_result)
        completed_folds[fold_idx] = fold_result
        _save_progress(exp_dir, completed_folds)

    # K-Fold サマリー
    mean_mae = float(np.mean([r.best_val_mae for r in fold_results]))
    std_mae = float(np.std([r.best_val_mae for r in fold_results]))
    logger.info("\n===== K-Fold Result =====")
    for r in fold_results:
        logger.info("Fold %d: Best MAE=%.6f", r.fold_idx, r.best_val_mae)
    logger.info("Mean MAE: %.6f +/- %.6f", mean_mae, std_mae)

    return fold_results
