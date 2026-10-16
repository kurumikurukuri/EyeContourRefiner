"""デュアルブランチ K-Fold CV オーケストレーション。

eye_contour.training.kfold（単一ブランチ）に対応するデュアルブランチ版。
旧 experiments/EXP020_dual_branch/scr/train_exp020.py の train_dual() から
抽出（EXP020/EXP021 双方が import して再利用するため共有化）。

run_kfold() と異なり、モジュール定数（旧 MANIFEST_TRAIN 等）には一切依存せず、
manifest パス・出力先をすべて明示的な引数で受け取る。
"""
from __future__ import annotations

import gc
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from eye_contour.data.augmentation import (
    AUG_PQ_JITTER,
    AUG_SOURCE_JPEG,
    AUG_SOURCE_RES_JITTER,
    check_aug_flags,
)
from eye_contour.data.dataset import load_manifest
from eye_contour.data.dataset_dual import BranchConfig, EyeContourPatchDualDataset
from eye_contour.data.source_degradation import SourceDegradeConfig
from eye_contour.data.weights import (
    DEFAULT_AUX_FEATURES,
    compute_aux_face_shape_meta,
    compute_eye_meta,
)
from eye_contour.evaluation.loss import get_loss_fn
from eye_contour.models.dual_ensemble import (
    get_dual_model,
    load_dual_model,
    save_dual_model,
)
from eye_contour.training.checkpoint import (
    load_checkpoint,
    remove_checkpoint,
    save_checkpoint,
)
from eye_contour.training.dual_averaging import run_fold_swa, track_ema_best
from eye_contour.training.dual_branch import (
    build_branch_configs,
    fold_ensemble_raw_t,
    model_pred_raw_t,
)
from eye_contour.training.dual_epoch import evaluate_val_mae, train_one_epoch
from eye_contour.training.kfold import FoldResult
from eye_contour.training.oof import OofPrediction
from eye_contour.training.scheduler import make_optimizer_and_scheduler
from eye_contour.training.trainer import set_seed
from eye_contour.training.weight_averaging import EmaTracker

logger = logging.getLogger(__name__)

_PROGRESS_FILENAME = "dual_kfold_progress.json"


def _load_dual_progress(exp_dir: Path) -> dict[int, FoldResult]:
    """exp_dir の進捗ファイルから完了済み fold の結果を読み込む（無ければ空 dict）。"""
    path = exp_dir / _PROGRESS_FILENAME
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): FoldResult(**v) for k, v in data.items()}


def _save_dual_progress(exp_dir: Path, completed: dict[int, FoldResult]) -> None:
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

# EXP027: 8 方向の透視変換パターン（軸 4 方向 + 対角 4 方向、各 5° 相当）。
# (dp_deg, dy_deg): pitch（水平軸回り） × yaw（垂直軸回り）の角度 [degree]。
_D5 = 5.0 / math.sqrt(2)  # 対角成分: 5/√2° ≈ 3.54°
PERSPECTIVE_PATTERNS_8: list[tuple[float, float]] = [
    (5.0,  0.0),   # pitch up
    (-5.0, 0.0),   # pitch down
    (0.0,  5.0),   # yaw right
    (0.0, -5.0),   # yaw left
    (_D5,  _D5),   # diagonal ++
    (_D5, -_D5),   # diagonal +-
    (-_D5, _D5),   # diagonal -+
    (-_D5, -_D5),  # diagonal --
]


def expand_perspective_records(
    records: list[dict],
    patterns: list[tuple[float, float]],
) -> list[dict]:
    """各レコードを base + len(patterns) 複製に展開する（透視変換 aug 用）。

    各複製に {"_persp_dp": dp_deg, "_persp_dy": dy_deg} タグを付与する。
    base レコード（タグ無し）が先頭に来る順序を保証する。
    patterns が空の場合は records をそのまま返す。

    Args:
        records: 展開前のレコードリスト。
        patterns: (dp_deg, dy_deg) のタプルリスト。

    Returns:
        展開後のレコードリスト（長さ: len(records) × (1 + len(patterns))）。
    """
    if not patterns:
        return records
    expanded: list[dict] = []
    for rec in records:
        expanded.append(rec)
        for dp_deg, dy_deg in patterns:
            aug_rec = dict(rec)
            aug_rec["_persp_dp"] = dp_deg
            aug_rec["_persp_dy"] = dy_deg
            expanded.append(aug_rec)
    return expanded


def inject_label_noise(
    records: list[dict],
    sigma: float,
    seed: int,
) -> list[dict]:
    """train レコードの生 t ラベルに N(0, sigma^2) の Gaussian ノイズを加える（EXP028）。

    アノテーションノイズを模擬する。sigma<=0 の場合は records をそのまま返す。
    元の records は変更せず、t を書き換えたレコードの複製を返す。ノイズは
    seed から決定的に生成されるため、同一 config での再実行は再現可能。

    Args:
        records: 展開前の訓練レコードリスト（"t" キーを持つ dict のリスト）。
        sigma:   ノイズの標準偏差（生 t スケール）。
        seed:    乱数シード。

    Returns:
        t を書き換えたレコードのリスト（sigma<=0 の場合は入力をそのまま返す）。
    """
    if sigma <= 0:
        return records
    rng = np.random.default_rng(seed)
    noisy_records: list[dict] = []
    for rec in records:
        rec_noisy = dict(rec)
        rec_noisy["t"] = float(rec["t"]) + float(rng.normal(0.0, sigma))
        noisy_records.append(rec_noisy)
    return noisy_records


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
    # EXP030: BRIGHTNESS/BLUR/COLOR_TEMP/FLIP（基準確率0.5）の発火頻度に掛ける倍率。
    # NOISE・GRAYSCALE の発火確率は対象外（augment() 側で固定のまま）。
    aug_prob: float = 1.0
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
    # 面外回転透視変換オーグメンテーション。
    # True の場合、鼻先（顔中心）を homography の回転中心に使う（orbit 移動を含む）。
    # False（デフォルト）では眼中心を pivot にする（foreshortening のみ、位置不変）。
    perspective_aug: bool = False
    perspective_alpha: float = 2.0
    # perspective_patterns: 空 = 無効, 非空 = PERSPECTIVE_PATTERNS_8 などを渡す。
    perspective_patterns: list = field(default_factory=list)
    # EXP028: アノテーションノイズ模擬用。train レコードの生 t に N(0, sigma^2) を付与する
    # （0 = 無効）。manifest_test（hold-out）には適用しない。
    label_noise_sigma: float = 0.0
    # EXP031: 元画像ピクセル空間の劣化 aug。train のみ（augment_data=True）に適用する。
    # ON/OFF は aug_flags の AUG_SOURCE_JPEG (0x080) / AUG_SOURCE_RES_JITTER (0x100)
    # ビットで指定する。以下は発火確率と強度レンジのみ（既定値は EXP031 の採用値）。
    source_jpeg_prob: float = 0.5
    source_jpeg_quality_min: int = 30
    source_jpeg_quality_max: int = 90
    source_res_jitter_prob: float = 0.5
    source_res_scale_min: float = 1.0
    source_res_scale_max: float = 2.5
    # EXP034: 重み平均。いずれも既定は無効（base の学習は一切変わらない）。
    # ema_decays: 指定 decay ごとに EMA モデルを追跡し、val 最良を `{exp_dir}_ema{decay}` に保存。
    # swa_epochs>0: fold 完了後、base 最良重みから一定 lr=swa_lr で追加学習した各 epoch 末の
    # 重みを等重み平均し（BN 統計は再計算）、`{exp_dir}_swa` に保存する。
    ema_decays: list = field(default_factory=list)
    swa_epochs: int = 0
    swa_lr: float = 2.5e-4
    # EXP035: PQ 回転摂動（R 中心に P, Q を θ ~ U(±rot_deg) 回転、t は不変）。train のみ。
    # ON/OFF は aug_flags の AUG_PQ_JITTER (0x200) ビット、rot_deg=0 でも無効（既定は無効）。
    pq_jitter_rot_deg: float = 0.0
    pq_jitter_prob: float = 1.0


def validate_pq_jitter_config(config: DualTrainConfig) -> None:
    """PQ 回転摂動の設定を検証する（無音の no-op や学習/推論の不整合を事前に弾く）。

    - 強度は有限で範囲内（rot_deg >= 0, prob ∈ [0, 1]）。
    - AUG_PQ_JITTER ビットと rot_deg > 0 は両立させる（片方だけだと J0 と同一の無駄な run になる）。
    - 有効時は num_workers=0 必須（専用 RNG が worker に複製され、θ 系列が重複するため）。
    - 有効時は pq_angle_* の aux 特徴と併用不可（aux 角度はクリーン値のままパッチだけ回転し、
      推論時と食い違うため）。
    """
    rot, prob = config.pq_jitter_rot_deg, config.pq_jitter_prob
    if not math.isfinite(rot) or rot < 0:
        raise ValueError(f"pq_jitter_rot_deg は有限かつ 0 以上: {rot}")
    if not math.isfinite(prob) or not 0.0 <= prob <= 1.0:
        raise ValueError(f"pq_jitter_prob は [0, 1]: {prob}")

    bit_on = bool(config.aug_flags & AUG_PQ_JITTER)
    if bit_on != (rot > 0):
        raise ValueError(
            "AUG_PQ_JITTER ビット(0x200)と pq_jitter_rot_deg>0 は両方指定が必要: "
            f"aug_flags={config.aug_flags}, pq_jitter_rot_deg={rot}"
        )
    if not bit_on:
        return
    if config.num_workers > 0:
        raise ValueError("PQ 回転摂動の使用時は num_workers=0 が必要（乱数が worker 間で複製される）")
    stale = [f for f in effective_aux_features(config) if f.startswith("pq_angle")]
    if stale:
        raise ValueError(f"PQ 回転摂動は aux 特徴 {stale} と併用できません（回転後の角度と不整合）")


def effective_aux_features(config: DualTrainConfig) -> list[str]:
    """config から実際に使用する aux 特徴名リストを導出する。

    use_aux_face_shape=False               → []（aux 無し）
    use_aux_face_shape=True かつ aux_features 空 → DEFAULT_AUX_FEATURES（11次元）
    use_aux_face_shape=True かつ aux_features 指定 → そのリスト（部分集合）
    """
    if not config.use_aux_face_shape:
        return []
    return list(config.aux_features) if config.aux_features else list(DEFAULT_AUX_FEATURES)


def build_source_degrade_config(config: DualTrainConfig) -> SourceDegradeConfig | None:
    """aug_flags のビットと強度パラメータから SourceDegradeConfig を組み立てる。

    JPEG・解像度ジッタのビットがいずれも立っていない場合は None を返す
    （dataset 側で劣化パスに入らない）。
    設定値は validate() で検証し、不正なら ValueError を送出する。
    """
    jpeg = bool(config.aug_flags & AUG_SOURCE_JPEG)
    res_jitter = bool(config.aug_flags & AUG_SOURCE_RES_JITTER)
    if not (jpeg or res_jitter):
        return None
    cfg = SourceDegradeConfig(
        jpeg=jpeg,
        jpeg_prob=config.source_jpeg_prob,
        jpeg_quality_min=config.source_jpeg_quality_min,
        jpeg_quality_max=config.source_jpeg_quality_max,
        res_jitter=res_jitter,
        res_jitter_prob=config.source_res_jitter_prob,
        res_scale_min=config.source_res_scale_min,
        res_scale_max=config.source_res_scale_max,
    )
    cfg.validate()
    return cfg


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
    resume: bool = True,
) -> FoldResult:
    """1 Fold 分のデュアルブランチ訓練を実行し、ベストモデル（val_mae 最小エポック）を保存する。

    OOF 予測の生成はこの関数の責務外（_predict_dual_records を別途呼ぶこと）。
    train_records/val_records は早期終了の判定にのみ使われ、保存されるモデルは
    val_records を一切学習に使わない。

    resume=True かつ exp_dir に checkpoint_fold{fold_idx}.pt が残っている場合、
    そのエポックから学習を再開する（中断復旧用）。Fold が正常完了すると
    チェックポイントは削除される。
    """
    loss_fn = get_loss_fn(config.loss_fn)
    aux_features = effective_aux_features(config)

    # EXP031: 元画像空間の劣化は train のみ。val/hold-out/推論は augment_data=False なので
    # 劣化パスに入らず、config 間でテスト入力は完全に同一になる。
    source_degrade = build_source_degrade_config(config)
    train_ds = EyeContourPatchDualDataset(
        train_records, cfg_a, cfg_b,
        augment_data=True, aug_flags=config.aug_flags, aug_prob=config.aug_prob,
        endpoint_blend_range=config.endpoint_blend_range,
        aux_features=aux_features,
        perspective_alpha=config.perspective_alpha,
        source_degrade=source_degrade,
        source_degrade_seed=config.seed + fold_idx,
        pq_jitter_rot_deg=config.pq_jitter_rot_deg,
        pq_jitter_prob=config.pq_jitter_prob,
        pq_jitter_seed=config.seed + fold_idx,
    )
    val_ds = EyeContourPatchDualDataset(
        val_records, cfg_a, cfg_b,
        augment_data=False, aug_flags=config.aug_flags, aug_prob=config.aug_prob,
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
    ema = EmaTracker(model, config.ema_decays) if config.ema_decays else None
    ema_best: dict[str, dict] = {
        name: {"val_mae": float("inf"), "epoch": 0} for name, _ in (ema.items() if ema else [])
    }

    val_mae_of = partial(
        evaluate_val_mae, loader=val_loader, fusion_mode=config.fusion_mode,
        cfg_a=cfg_a, cfg_b=cfg_b, aux_features=aux_features, device=device,
    )

    best_mae = float("inf")
    no_improve = 0
    best_epoch = 0
    start_epoch = 0
    main_done = False
    model_path = exp_dir / f"eyeliner_detection_fold{fold_idx}.pt"
    checkpoint_path = exp_dir / f"checkpoint_fold{fold_idx}.pt"

    ckpt = load_checkpoint(checkpoint_path) if resume else None
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        best_mae = ckpt["best_mae"]
        best_epoch = ckpt["best_epoch"]
        no_improve = ckpt["no_improve"]
        extra = ckpt.get("extra") or {}
        main_done = bool(extra.get("main_done", False))
        if ema is not None:
            ema.load_state_dict(extra["ema"])
            ema_best = extra["ema_best"]
        logger.info(
            "Fold %d: チェックポイントから再開 (epoch %d〜, best_mae=%.6f, 本学習完了=%s)",
            fold_idx, start_epoch + 1, best_mae, main_done,
        )

    for epoch in range(config.epochs if main_done else start_epoch, config.epochs):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, config.fusion_mode,
            cfg_a, cfg_b, aux_features, device,
            on_step=ema.update if ema is not None else None,
        )
        val_mae = val_mae_of(model)
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

        if ema is not None:
            ema_best = track_ema_best(
                ema, ema_best, val_mae_of, exp_dir, fold_idx, epoch + 1, config,
            )

        main_done = no_improve >= config.patience
        save_checkpoint(
            checkpoint_path, epoch=epoch + 1, model=model, optimizer=optimizer,
            scheduler=scheduler, best_mae=best_mae, best_epoch=best_epoch,
            no_improve=no_improve,
            extra={
                "main_done": main_done,
                **({"ema": ema.state_dict(), "ema_best": ema_best} if ema is not None else {}),
            },
        )

        if main_done:
            logger.info("Early stopping (fold %d, epoch %d)", fold_idx, epoch + 1)
            break

    fold_result = FoldResult(
        fold_idx=fold_idx,
        best_val_mae=best_mae,
        best_epoch=best_epoch,
        model_path=str(model_path),
    )
    logger.info("Fold %d: Best MAE=%.6f (epoch %d)", fold_idx, best_mae, best_epoch)

    if config.swa_epochs > 0:
        run_fold_swa(
            model_path, train_loader, loss_fn, val_mae_of, exp_dir, fold_idx,
            config, cfg_a, cfg_b, aux_features, device,
        )
    remove_checkpoint(checkpoint_path)

    del model, optimizer, scheduler, ema, train_loader, val_loader, train_ds, val_ds
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
    ds = EyeContourPatchDualDataset(
        records, cfg_a, cfg_b,
        augment_data=False, aug_flags=config.aug_flags, aug_prob=config.aug_prob,
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
    resume: bool = True,
) -> DualKfoldResult:
    """デュアルブランチ K-Fold CV 訓練を実行し、各 Fold のベストモデルを exp_dir に保存する。

    Args:
        manifest_train: 訓練・CV 用 manifest.csv のパス
        manifest_test:  Hold-out 用 manifest.csv のパス（None ならホールドアウト評価をスキップ）
        exp_dir:        モデル・experiment.json を保存するディレクトリ
        config:         訓練設定
        resume:         True の場合、exp_dir に残る進捗ファイル・チェックポイントから
                         再開する（完了済み fold はスキップ、中断中の fold は
                         エポック単位で再開）。False なら完全に最初から学習する。

    Returns:
        DualKfoldResult
    """
    validate_pq_jitter_config(config)
    set_seed(config.seed)
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("デバイス: %s", device)
    logger.info("fusion_mode: %s  rescale_b: %s", config.fusion_mode, config.pq_rescale_mode_b)
    check_aug_flags(config.aug_flags)
    aux_features = effective_aux_features(config)

    records = load_manifest(manifest_train)
    if config.label_noise_sigma > 0:
        records = inject_label_noise(records, config.label_noise_sigma, config.seed)
        logger.info("EXP028: train t にラベルノイズ付与 sigma=%.3f", config.label_noise_sigma)
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

    completed_folds = _load_dual_progress(exp_dir) if resume else {}
    fold_results: list[FoldResult] = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        fold_idx = fold + 1

        if fold_idx in completed_folds:
            logger.info(
                "Fold %d/%d: 完了済み (best_val_mae=%.6f) のためスキップ",
                fold_idx, config.k, completed_folds[fold_idx].best_val_mae,
            )
            fold_results.append(completed_folds[fold_idx])
            continue

        logger.info("----- Fold %d/%d -----", fold_idx, config.k)
        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]

        # EXP027: 透視変換 aug → train のみ展開（val は無展開）。
        if config.perspective_aug and config.perspective_patterns:
            n_before = len(train_records)
            train_records = expand_perspective_records(train_records, config.perspective_patterns)
            logger.info(
                "Fold %d: train 展開 %d → %d（×%d）",
                fold_idx, n_before, len(train_records), 1 + len(config.perspective_patterns),
            )

        fold_result = _train_single_dual_fold(
            train_records, val_records,
            fold_idx, exp_dir, config, cfg_a, cfg_b, device, dl_kwargs,
            resume=resume,
        )
        fold_results.append(fold_result)
        completed_folds[fold_idx] = fold_result
        _save_dual_progress(exp_dir, completed_folds)

    mean_mae = float(np.mean([r.best_val_mae for r in fold_results]))
    logger.info("CV MAE: %.6f", mean_mae)

    holdout_mae: float | None = None
    holdout_mae_raw: float | None = None
    if holdout_records is not None and len(holdout_records) > 0:
        logger.info("Hold-out MAE 計算中...")
        try:
            fold_paths = sorted(exp_dir.glob("eyeliner_detection_fold*.pt"))
            models = [load_dual_model(p).to(device) for p in fold_paths]
            holdout_ds = EyeContourPatchDualDataset(
                holdout_records, cfg_a, cfg_b, augment_data=False, aug_flags=config.aug_flags,
                aug_prob=config.aug_prob,
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
            "aug_prob": config.aug_prob,
            "endpoint_blend_range": float(config.endpoint_blend_range),
            "loss_fn": config.loss_fn,
            "use_aux_face_shape": config.use_aux_face_shape,
            "aux_features": effective_aux_features(config),
            "use_is_upper": config.use_is_upper,
            "perspective_aug": config.perspective_aug,
            "perspective_alpha": config.perspective_alpha,
            "n_perspective_patterns": len(config.perspective_patterns),
            "label_noise_sigma": config.label_noise_sigma,
            "source_jpeg": bool(config.aug_flags & AUG_SOURCE_JPEG),
            "source_jpeg_prob": config.source_jpeg_prob,
            "source_jpeg_quality": [config.source_jpeg_quality_min, config.source_jpeg_quality_max],
            "source_res_jitter": bool(config.aug_flags & AUG_SOURCE_RES_JITTER),
            "source_res_jitter_prob": config.source_res_jitter_prob,
            "source_res_scale": [config.source_res_scale_min, config.source_res_scale_max],
            "ema_decays": list(config.ema_decays),
            "swa_epochs": config.swa_epochs,
            "swa_lr": config.swa_lr,
            "pq_jitter": bool(config.aug_flags & AUG_PQ_JITTER) and config.pq_jitter_rot_deg > 0,
            "pq_jitter_rot_deg": config.pq_jitter_rot_deg,
            "pq_jitter_prob": config.pq_jitter_prob,
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
