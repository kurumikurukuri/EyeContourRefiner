"""デュアルブランチ K-Fold CV オーケストレーション。

eyeliner.training.kfold（単一ブランチ）に対応するデュアルブランチ版。
旧 experiments/EXP020_dual_branch/scr/train_exp020.py の train_dual() から
抽出（EXP020/EXP021 双方が import して再利用するため共有化）。

run_kfold() と異なり、モジュール定数（旧 MANIFEST_TRAIN 等）には一切依存せず、
manifest パス・出力先をすべて明示的な引数で受け取る。
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from eyeliner.data.dataset import load_manifest
from eyeliner.data.dataset_dual import BranchConfig, EyelinerPatchDualDataset
from eyeliner.data.weights import (
    DEFAULT_AUX_FEATURES,
    compute_aux_face_shape_meta,
    compute_eye_meta,
)
from eyeliner.evaluation.loss import get_loss_fn
from eyeliner.models.dual_ensemble import (
    get_dual_model,
    load_dual_model,
    save_dual_model,
)
from eyeliner.training.dual_branch import (
    branch_pred_to_raw_t,
    build_branch_configs,
    ensemble_raw_t,
    fold_ensemble_raw_t,
    model_pred_raw_t,
    normalize_to_branch_a,
)
from eyeliner.training.kfold import FoldResult
from eyeliner.training.oof import OofPrediction
from eyeliner.training.scheduler import make_optimizer_and_scheduler
from eyeliner.training.trainer import set_seed

logger = logging.getLogger(__name__)


@dataclass
class DualTrainConfig:
    """デュアルブランチ訓練設定（ハイパーパラメータ）。"""

    fusion_mode: str = "independent"
    pq_rescale_mode_b: str = "eye_scale"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    batch_size: int = 32
    k: int = 3
    patience: int = 50
    seed: int = 42
    num_conv_layers: int = 4
    use_sigmoid: bool = True
    use_batchnorm: bool = True
    dropout_rate: float = 0.3
    aug_flags: int = 127
    endpoint_blend_range: float = 0.25
    loss_fn: str = "t"
    num_workers: int = 0
    base_channel: int = 32
    pool_width: int = 32
    # aux は EXP018 と同一仕様（use_aux_face_shape=固定11次元の生 sin/cos）。
    use_aux_face_shape: bool = False
    # aux_features 指定時は canonical 特徴の部分集合（use_aux_face_shape=true と併用）。
    # 空のときは use_aux_face_shape=true ならデフォルト11次元、false なら aux 無し。
    aux_features: list = field(default_factory=list)
    use_is_upper: bool = True
    db_path: str = "data/samples.db"


def effective_aux_features(config: DualTrainConfig) -> list[str]:
    """config から実際に使用する aux 特徴名リストを導出する。

    use_aux_face_shape=False               → []（aux 無し）
    use_aux_face_shape=True かつ aux_features 空 → DEFAULT_AUX_FEATURES（11次元）
    use_aux_face_shape=True かつ aux_features 指定 → そのリスト（部分集合）
    """
    if not config.use_aux_face_shape:
        return []
    return list(config.aux_features) if config.aux_features else list(DEFAULT_AUX_FEATURES)


@dataclass
class DualKfoldResult:
    """run_dual_kfold() の戻り値。"""

    fold_results: list[FoldResult]
    n_train: int
    n_holdout: int
    train_images: list[str]
    holdout_images: list[str]
    cfg_a: BranchConfig
    cfg_b: BranchConfig
    holdout_mae: float | None = None
    holdout_mae_raw: float | None = None


def _train_single_dual_fold(
    train_records: list[dict],
    val_records: list[dict],
    fold_idx: int,
    exp_dir: Path,
    config: DualTrainConfig,
    cfg_a: BranchConfig,
    cfg_b: BranchConfig,
    device: torch.device,
    dl_kwargs: dict,
) -> FoldResult:
    """1 Fold 分のデュアルブランチ訓練を実行し、ベストモデル（val_mae 最小エポック）を保存する。

    OOF 予測の生成はこの関数の責務外（_predict_dual_records を別途呼ぶこと）。
    train_records/val_records は早期終了の判定にのみ使われ、保存されるモデルは
    val_records を一切学習に使わない。
    """
    loss_fn = get_loss_fn(config.loss_fn)
    aux_features = effective_aux_features(config)

    train_ds = EyelinerPatchDualDataset(
        train_records, cfg_a, cfg_b,
        augment_data=True, aug_flags=config.aug_flags,
        endpoint_blend_range=config.endpoint_blend_range,
        aux_features=aux_features,
    )
    val_ds = EyelinerPatchDualDataset(
        val_records, cfg_a, cfg_b,
        augment_data=False, aug_flags=config.aug_flags,
        endpoint_blend_range=config.endpoint_blend_range,
        aux_features=aux_features,
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, **dl_kwargs)

    model_kwargs = dict(
        num_conv_layers=config.num_conv_layers,
        use_sigmoid=config.use_sigmoid,
        use_batchnorm=config.use_batchnorm,
        dropout_rate=config.dropout_rate,
    )
    if config.fusion_mode != "crossfuse":
        # CrossFuseDualModel は base_channel/pool_width/aux を受け付けない（未対応）。
        model_kwargs["base_channel"] = config.base_channel
        model_kwargs["pool_width"] = config.pool_width
        model_kwargs["use_aux_face_shape"] = config.use_aux_face_shape
        model_kwargs["aux_dim"] = len(aux_features)
        model_kwargs["use_is_upper"] = config.use_is_upper
    model = get_dual_model(config.fusion_mode, **model_kwargs).to(device)

    optimizer, scheduler = make_optimizer_and_scheduler(
        model, "full", config.lr, config.weight_decay,
        len(train_loader), config.epochs,
    )

    best_mae = float("inf")
    no_improve = 0
    best_epoch = 0
    model_path = exp_dir / f"eyeliner_detection_fold{fold_idx}.pt"

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x_a = batch[0].to(device)
            x_b = batch[1].to(device)
            is_upper = batch[2].to(device)
            y_a = batch[4].to(device)
            y_b = batch[5].to(device)
            pq_dist = batch[12].to(device)
            pq_prime = batch[13].to(device)
            aux = batch[14].to(device) if aux_features else None

            if config.fusion_mode == "attention":
                pred_a, pred_b, alpha = model(x_a, x_b, is_upper, aux)
                t_a = branch_pred_to_raw_t(pred_a, cfg_a, pq_dist, pq_prime)
                t_b = branch_pred_to_raw_t(pred_b, cfg_b, pq_dist, pq_prime)
                ens_norm = normalize_to_branch_a(ensemble_raw_t(t_a, t_b, alpha), cfg_a)
                loss = loss_fn(pred_a, y_a) + loss_fn(pred_b, y_b) + loss_fn(ens_norm, y_a)
            else:
                pred_a, pred_b = model(x_a, x_b, is_upper, aux)
                loss = loss_fn(pred_a, y_a) + loss_fn(pred_b, y_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)

        model.eval()
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in val_loader:
                x_a = batch[0].to(device)
                x_b = batch[1].to(device)
                is_upper = batch[2].to(device)
                y_a = batch[4]
                pq_dist = batch[12].to(device)
                pq_prime = batch[13].to(device)
                aux = batch[14].to(device) if aux_features else None
                t_ens = model_pred_raw_t(
                    model, x_a, x_b, is_upper, config.fusion_mode,
                    cfg_a, cfg_b, pq_dist, pq_prime, aux,
                )
                pred_norm = normalize_to_branch_a(t_ens, cfg_a)
                all_preds.append(pred_norm.cpu())
                all_labels.append(y_a)

        preds_t = torch.cat(all_preds)
        labels_t = torch.cat(all_labels)
        val_mae = (preds_t - labels_t).abs().mean().item()

        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            "Fold %d Epoch %3d  loss=%.6f  val_mae=%.6f  lr=%.2e",
            fold_idx, epoch + 1, avg_loss, val_mae, current_lr,
        )

        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch + 1
            no_improve = 0
            save_dual_model(model, config.fusion_mode, config.__dict__, model_path)
        else:
            no_improve += 1
            if no_improve >= config.patience:
                logger.info("Early stopping (fold %d, epoch %d)", fold_idx, epoch + 1)
                break

    fold_result = FoldResult(
        fold_idx=fold_idx,
        best_val_mae=best_mae,
        best_epoch=best_epoch,
        model_path=str(model_path),
    )
    logger.info("Fold %d: Best MAE=%.6f (epoch %d)", fold_idx, best_mae, best_epoch)

    del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return fold_result


def _predict_dual_records(
    model_path: str | Path,
    records: list[dict],
    config: DualTrainConfig,
    cfg_a: BranchConfig,
    cfg_b: BranchConfig,
    device: torch.device,
    dl_kwargs: dict,
    fold_idx: int,
) -> list[OofPrediction]:
    """保存済みデュアルブランチモデルで records を推論し OOF 予測（生t）を生成する。

    model_pred_raw_t() は枝A/Bの予測を生t空間で合成して返すため、単一ブランチ版の
    _predict_records と異なり normalize_to_branch_a() による再正規化は不要
    （manifest の生 t と直接比較できる）。augment_data=False・shuffle=False の専用推論パス。
    """
    aux_features = effective_aux_features(config)
    ds = EyelinerPatchDualDataset(
        records, cfg_a, cfg_b,
        augment_data=False, aug_flags=config.aug_flags,
        endpoint_blend_range=config.endpoint_blend_range,
        aux_features=aux_features,
    )
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False, **dl_kwargs)

    model = load_dual_model(model_path).to(device)
    model.eval()

    all_t_pred: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            x_a = batch[0].to(device)
            x_b = batch[1].to(device)
            is_upper = batch[2].to(device)
            pq_dist = batch[12].to(device)
            pq_prime = batch[13].to(device)
            aux = batch[14].to(device) if aux_features else None
            t_ens = model_pred_raw_t(
                model, x_a, x_b, is_upper, config.fusion_mode,
                cfg_a, cfg_b, pq_dist, pq_prime, aux,
            )
            all_t_pred.append(t_ens.cpu())

    preds = torch.cat(all_t_pred).reshape(-1).tolist()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    oof: list[OofPrediction] = []
    for rec, t_pred in zip(records, preds):
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


def run_dual_kfold(
    manifest_train: str | Path,
    manifest_test: str | Path | None,
    exp_dir: str | Path,
    config: DualTrainConfig,
) -> DualKfoldResult:
    """デュアルブランチ K-Fold CV 訓練を実行し、各 Fold のベストモデルを exp_dir に保存する。

    Args:
        manifest_train: 訓練・CV 用 manifest.csv のパス
        manifest_test:  Hold-out 用 manifest.csv のパス（None ならホールドアウト評価をスキップ）
        exp_dir:        モデル・experiment.json を保存するディレクトリ
        config:         訓練設定

    Returns:
        DualKfoldResult
    """
    set_seed(config.seed)
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("デバイス: %s", device)
    logger.info("fusion_mode: %s  rescale_b: %s", config.fusion_mode, config.pq_rescale_mode_b)
    aux_features = effective_aux_features(config)

    records = load_manifest(manifest_train)
    compute_eye_meta(records)
    if aux_features:
        compute_aux_face_shape_meta(records, config.db_path)
    n_train = len(records)
    train_images = sorted({r["image_path"] for r in records})
    logger.info("訓練サンプル: %d", n_train)

    n_holdout = 0
    holdout_images: list[str] = []
    holdout_records: list[dict] | None = None
    if manifest_test is not None and Path(manifest_test).exists():
        holdout_records = load_manifest(manifest_test)
        compute_eye_meta(holdout_records)
        if aux_features:
            compute_aux_face_shape_meta(holdout_records, config.db_path)
        n_holdout = len(holdout_records)
        holdout_images = sorted({r["image_path"] for r in holdout_records})
        logger.info("hold-out: %d", n_holdout)
    else:
        logger.info("hold-out: なし")

    dl_kwargs = {"num_workers": config.num_workers, "persistent_workers": config.num_workers > 0}

    # 枝 A=EXP019 X0（生PQ, raw_t）, 枝 B=EXP019 X1/X2（P'Q', t_prime）を忠実に再現。
    cfg_a, cfg_b = build_branch_configs(config.pq_rescale_mode_b)

    groups = [r["image_path"] for r in records]
    kf = GroupKFold(n_splits=config.k)
    splits = list(kf.split(records, groups=groups))

    fold_results: list[FoldResult] = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info("----- Fold %d/%d -----", fold + 1, config.k)
        fold_result = _train_single_dual_fold(
            [records[i] for i in train_idx], [records[i] for i in val_idx],
            fold + 1, exp_dir, config, cfg_a, cfg_b, device, dl_kwargs,
        )
        fold_results.append(fold_result)

    mean_mae = float(np.mean([r.best_val_mae for r in fold_results]))
    logger.info("CV MAE: %.6f", mean_mae)

    holdout_mae: float | None = None
    holdout_mae_raw: float | None = None
    if holdout_records is not None and len(holdout_records) > 0:
        logger.info("Hold-out MAE 計算中...")
        try:
            fold_paths = sorted(exp_dir.glob("eyeliner_detection_fold*.pt"))
            models = [load_dual_model(p).to(device) for p in fold_paths]
            holdout_ds = EyelinerPatchDualDataset(
                holdout_records, cfg_a, cfg_b, augment_data=False, aug_flags=config.aug_flags,
                endpoint_blend_range=config.endpoint_blend_range,
                aux_features=aux_features,
            )
            holdout_loader = DataLoader(
                holdout_ds, batch_size=config.batch_size, shuffle=False, **dl_kwargs,
            )

            label_range_a = cfg_a.label_t_max - cfg_a.label_t_min
            abs_sum_norm = 0.0
            abs_sum_raw = 0.0
            count = 0
            offset = 0
            with torch.no_grad():
                for batch in holdout_loader:
                    x_a = batch[0].to(device)
                    x_b = batch[1].to(device)
                    is_upper = batch[2].to(device)
                    pq_dist = batch[12].to(device)
                    pq_prime = batch[13].to(device)
                    aux = batch[14].to(device) if aux_features else None
                    bsz = x_a.shape[0]
                    # 真値は生 t（クランプなし）を使う。batch[4]（y_a）は学習ロス用に
                    # cfg_a.label 範囲へクランプ済みのため、holdout 指標には使わない
                    # （クランプすると範囲外の t を持つサンプルの誤差が過小評価される）。
                    t_true_raw = torch.tensor(
                        [float(r["t"]) for r in holdout_records[offset:offset + bsz]],
                        dtype=torch.float32,
                    ).to(device).unsqueeze(1)
                    offset += bsz
                    t_ens = fold_ensemble_raw_t(
                        models, x_a, x_b, is_upper, config.fusion_mode,
                        cfg_a, cfg_b, pq_dist, pq_prime, aux,
                    )
                    abs_err_raw = (t_ens - t_true_raw).abs()
                    abs_sum_raw += abs_err_raw.sum().item()
                    abs_sum_norm += (abs_err_raw / label_range_a).sum().item()
                    count += t_true_raw.numel()
            holdout_mae = abs_sum_norm / max(count, 1)
            holdout_mae_raw = abs_sum_raw / max(count, 1)
            logger.info("Hold-out MAE: %.6f（正規化） / %.6f（生t）", holdout_mae, holdout_mae_raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hold-out MAE 計算失敗: %s", exc)

    return DualKfoldResult(
        fold_results=fold_results,
        n_train=n_train,
        n_holdout=n_holdout,
        train_images=train_images,
        holdout_images=holdout_images,
        cfg_a=cfg_a,
        cfg_b=cfg_b,
        holdout_mae=holdout_mae,
        holdout_mae_raw=holdout_mae_raw,
    )


def _branch_cfg_dict(cfg: BranchConfig) -> dict:
    """BranchConfig を experiment.json 用の dict に変換する。"""
    return {
        "patch_t_min": cfg.patch_t_min,
        "patch_t_max": cfg.patch_t_max,
        "patch_perp_scale": cfg.patch_perp_scale,
        "label_t_min": cfg.label_t_min,
        "label_t_max": cfg.label_t_max,
        "pq_rescale_mode": cfg.pq_rescale_mode,
        "pq_rescale_factor": cfg.pq_rescale_factor,
        "patch_size": list(cfg.patch_size),
        "label_space": cfg.label_space,
    }


def build_dual_experiment_record(
    config_name: str, config: DualTrainConfig, result: DualKfoldResult,
) -> dict:
    """run_dual_kfold() の結果から experiment.json 相当の dict を組み立てる。"""
    return {
        "exp_name": config_name,
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": {
            "fusion_mode": config.fusion_mode,
            "rescale_b": config.pq_rescale_mode_b,
            "branch_a": _branch_cfg_dict(result.cfg_a),
            "branch_b": _branch_cfg_dict(result.cfg_b),
            "num_conv_layers": config.num_conv_layers,
            "use_sigmoid": config.use_sigmoid,
            "use_batchnorm": config.use_batchnorm,
            "dropout_rate": config.dropout_rate,
            "base_channel": config.base_channel,
            "pool_width": config.pool_width,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "epochs": config.epochs,
            "batch": config.batch_size,
            "k": config.k,
            "patience": config.patience,
            "seed": config.seed,
            "aug_flags": config.aug_flags,
            "endpoint_blend_range": float(config.endpoint_blend_range),
            "loss_fn": config.loss_fn,
            "use_aux_face_shape": config.use_aux_face_shape,
            "aux_features": effective_aux_features(config),
            "use_is_upper": config.use_is_upper,
        },
        "data": {
            "n_train_samples": result.n_train,
            "n_holdout_samples": result.n_holdout,
            "train_images": result.train_images,
            "holdout_images": result.holdout_images,
        },
        "results": {
            # MAE は枝A正規化空間（EXP019/020 X0 の正規化 MAE と直接比較可能）。
            # 生 t MAE = normalized * (label_t_max - label_t_min)。
            "cv_mae": float(np.mean([r.best_val_mae for r in result.fold_results])),
            "cv_folds": [r.best_val_mae for r in result.fold_results],
            "holdout_mae": result.holdout_mae,
            "holdout_mae_raw_t": result.holdout_mae_raw,
        },
    }
