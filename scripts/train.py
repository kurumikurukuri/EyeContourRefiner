"""アイライン輪郭検出 CNN 訓練スクリプト（新版）。

eyeliner/ パッケージへの薄いラッパー。ハイパーパラメータの受け取りと
experiment.json の生成・オーケストレーションのみを行う。

モード:
  kfold    : K-Fold CV 訓練（デフォルト）
  ensemble : fold モデルをアンサンブル化して保存
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# プロジェクトルートを sys.path に追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from eyeliner.data.patch_utils import NORM_TYPE_CHOICES, get_norm_stats
from eyeliner.data.dataset import get_manifest_version, load_manifest
from eyeliner.data.weights import compute_eye_meta
from eyeliner.models.ensemble import (
    build_model, create_ensemble_from_dir, save_ensemble, load_model,
)
from eyeliner.training.kfold import run_kfold
from eyeliner.training.trainer import TrainConfig
from eyeliner.evaluation.metrics import holdout_mae as compute_holdout_mae

logger = logging.getLogger(__name__)

BACKBONE_CHOICES = ["custom", "soft_argmax", "mobilenet_v3_small", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_v2_s", "efficientnet_v2_m", "resnet18"]
LOSS_FN_CHOICES = ["t", "t_weighted", "normalized_pixel", "dead_zone"]


def make_exp_name(args: argparse.Namespace) -> str:
    """再現可能な実験名を自動生成する。"""
    arch = args.backbone.replace("_", "")
    lr_str = f"lr{str(args.lr).replace('.', '')}"
    wd_str = f"wd{str(args.weight_decay).replace('.', '')}"
    date_str = datetime.now().strftime("%Y%m%d")
    pq_str = "pqratio" if args.use_pq_ratio else "nopqratio"
    sigmoid_str = "sigmoid" if args.use_sigmoid else "nosigmoid"
    bn_str = "bn" if args.use_batchnorm else "nobn"
    depth_str = f"conv{args.num_conv_layers}"
    tws_str = f"tws{str(args.t_weight_alpha).replace('.', '')}" if args.use_t_weighted_sampler else "notws"
    twl_str = f"twl{str(args.t_loss_alpha).replace('.', '')}" if args.loss_fn == "t_weighted" else "notwl"
    dz_str = f"dz{str(args.loss_epsilon).replace('.', '')}d{str(args.loss_delta).replace('.', '')}" if args.loss_fn == "dead_zone" else ""
    res_str = f"res{args.patch_h}x{args.patch_w}"
    parts = [arch, lr_str, wd_str, pq_str, sigmoid_str, bn_str, depth_str, tws_str, twl_str]
    if dz_str:
        parts.append(dz_str)
    parts += [res_str, date_str]
    return "_".join(parts)


def build_experiment_record(
    args: argparse.Namespace,
    fold_results: list,
    holdout_mae_value: float | None,
    data_version: str,
    n_train_samples: int,
    n_holdout_samples: int,
    train_images: list[str],
    holdout_images: list[str],
) -> dict:
    """experiment.json の内容を構築する。"""
    return {
        "exp_name": args.exp_name,
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": {
            "model_arch": args.backbone,
            "lr": args.lr,
            "batch_size": args.batch,
            "weight_decay": args.weight_decay,
            "k_folds": args.k,
            "loss_fn": args.loss_fn,
            "freeze_epochs": args.freeze_epochs,
            "epochs": args.epochs,
            "patience": args.patience,
            "t_quant_step": args.t_quant_step,
            "norm_type": args.norm_type,
            "use_pq_ratio": args.use_pq_ratio,
            "use_sigmoid": args.use_sigmoid,
            "use_batchnorm": args.use_batchnorm,
            "num_conv_layers": args.num_conv_layers,
            "use_t_weighted_sampler": args.use_t_weighted_sampler,
            "t_weight_alpha": args.t_weight_alpha,
            "t_loss_alpha": args.t_loss_alpha,
            "loss_epsilon": args.loss_epsilon,
            "loss_delta": args.loss_delta,
            "use_distance_from_center": args.use_distance_from_center,
            "use_norm_eye_scale": args.use_norm_eye_scale,
            "use_logit_target": args.use_logit_target,
            "patch_h": args.patch_h,
            "patch_w": args.patch_w,
            "patch_t_min": args.patch_t_min,
            "patch_t_max": args.patch_t_max,
            "label_t_min": args.label_t_min,
            "label_t_max": args.label_t_max,
            "dropout_rate": args.dropout_rate,
        },
        "data": {
            "data_version": data_version,
            "n_train_samples": n_train_samples,
            "n_holdout_samples": n_holdout_samples,
            "train_images": train_images,
            "holdout_images": holdout_images,
        },
        "environment": {
            "seed": args.seed,
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
        },
        "results": {
            "cv_mae": float(np.mean([r.best_val_mae for r in fold_results])),
            "cv_folds": [r.best_val_mae for r in fold_results],
            "holdout_mae": holdout_mae_value,
        },
    }


def _save_prediction_train_csv(
    ensemble: torch.nn.Module,
    train_records: list[dict],
    exp_dir: Path,
    args: argparse.Namespace,
    norm_mean: list,
    norm_std: list,
    device: torch.device,
) -> None:
    """アンサンブルモデルで訓練データを推論し prediction_train.csv を保存する。"""
    from eyeliner.data.dataset import EyelinerPatchDataset
    from torch.utils.data import DataLoader

    compute_eye_meta(train_records)

    ds = EyelinerPatchDataset(
        train_records, augment_data=False,
        t_quant_step=args.t_quant_step, norm_mean=norm_mean, norm_std=norm_std,
        patch_t_min=args.patch_t_min,
        patch_t_max=args.patch_t_max,
        patch_perp_scale=args.patch_perp_scale,
        patch_size=(args.patch_h, args.patch_w),
        use_t_norm=getattr(args, "use_t_norm", True),
        label_t_min=args.label_t_min,
        label_t_max=args.label_t_max,
    )
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False)

    all_preds: list[float] = []
    ensemble.eval()
    with torch.no_grad():
        for x, is_upper, pq_ratio, *_ in loader:
            pred = ensemble(x.to(device), is_upper.to(device), pq_ratio.to(device))
            all_preds.extend(pred.squeeze(-1).cpu().tolist())

    csv_path = exp_dir / "prediction_train.csv"
    fieldnames = ["idx", "image_path", "part_name", "sub_key", "point_index", "t_true", "t_pred"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        eff_label_min = args.label_t_min if args.label_t_min is not None else args.patch_t_min
        eff_label_max = args.label_t_max if args.label_t_max is not None else args.patch_t_max
        for rec, t_pred in zip(train_records, all_preds):
            t_actual = float(rec["t"])
            t_norm = (t_actual - eff_label_min) / (eff_label_max - eff_label_min)
            writer.writerow({
                "idx":         rec.get("idx", ""),
                "image_path":  rec["image_path"],
                "part_name":   rec.get("part_name", ""),
                "sub_key":     rec.get("sub_key", ""),
                "point_index": rec.get("point_index", ""),
                "t_true":      f"{t_norm:.6f}",
                "t_pred":      f"{t_pred:.6f}",
            })
    logger.info("prediction_train.csv 保存: %s  (n=%d)", csv_path, len(all_preds))


def mode_kfold(args: argparse.Namespace) -> None:
    """K-Fold CV 訓練を実行する。"""
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error("manifest が見つかりません: %s", manifest_path)
        sys.exit(1)

    # 実験名と出力ディレクトリの決定
    exp_name = args.exp_name or make_exp_name(args)
    args.exp_name = exp_name
    exp_dir = Path(args.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 正規化統計の取得
    norm_mean, norm_std = get_norm_stats(args.norm_type, stats_json_path=args.norm_stats_json)
    logger.info(
        "正規化: %s  mean=%s  std=%s",
        args.norm_type,
        [f"{v:.4f}" for v in norm_mean],
        [f"{v:.4f}" for v in norm_std],
    )

    # データバージョン記録
    data_version = get_manifest_version(manifest_path)
    logger.info("データバージョン: %s", data_version)

    # Hold-out サンプル数カウント
    n_holdout_samples = 0
    if args.holdout and Path(args.holdout).exists():
        holdout_records = load_manifest(args.holdout)
        n_holdout_samples = len(holdout_records)

    # 訓練設定
    config = TrainConfig(
        backbone=args.backbone,
        lr=args.lr,
        batch_size=args.batch,
        weight_decay=args.weight_decay,
        loss_fn=args.loss_fn,
        freeze_epochs=args.freeze_epochs,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        t_quant_step=args.t_quant_step,
        pretrained=not args.no_pretrained,
        norm_mean=norm_mean,
        norm_std=norm_std,
        use_pq_ratio=args.use_pq_ratio,
        use_sigmoid=args.use_sigmoid,
        use_batchnorm=args.use_batchnorm,
        num_conv_layers=args.num_conv_layers,
        use_t_weighted_sampler=args.use_t_weighted_sampler,
        t_weight_alpha=args.t_weight_alpha,
        t_loss_alpha=args.t_loss_alpha,
        loss_epsilon=args.loss_epsilon,
        loss_delta=args.loss_delta,
        use_distance_from_center=args.use_distance_from_center,
        use_norm_eye_scale=args.use_norm_eye_scale,
        use_logit_target=args.use_logit_target,
        use_t_norm=args.use_t_norm,
        dropout_rate=args.dropout_rate,
        patch_t_min=args.patch_t_min,
        patch_t_max=args.patch_t_max,
        patch_perp_scale=args.patch_perp_scale,
        label_t_min=args.label_t_min,
        label_t_max=args.label_t_max,
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        aug_flags=args.aug_flags,
    )

    # K-Fold 訓練
    logger.info(
        "K-Fold 訓練開始: %s (%s)",
        exp_name,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    fold_results = run_kfold(
        manifest_path=manifest_path,
        exp_dir=exp_dir,
        config=config,
        k_folds=args.k,
    )

    # 訓練サンプル数・画像リスト取得
    train_records = load_manifest(manifest_path)
    n_train_samples = len(train_records)
    train_images = sorted({r["image_path"] for r in train_records})

    # Hold-out 画像リスト取得
    holdout_images: list[str] = []
    if args.holdout and Path(args.holdout).exists() and n_holdout_samples > 0:
        holdout_records_for_images = load_manifest(args.holdout)
        holdout_images = sorted({r["image_path"] for r in holdout_records_for_images})

    # Hold-out MAE 計算（Hold-out が存在する場合のみ）
    holdout_mae_value = None
    if args.holdout and Path(args.holdout).exists() and n_holdout_samples > 0:
        logger.info("Hold-out MAE を計算中...")
        # アンサンブルモデルで Hold-out 評価
        try:
            ensemble, backbone_name = create_ensemble_from_dir(exp_dir)
            ensemble.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ensemble = ensemble.to(device)

            holdout_records = load_manifest(args.holdout)
            compute_eye_meta(holdout_records)

            from eyeliner.data.dataset import EyelinerPatchDataset
            from torch.utils.data import DataLoader
            holdout_ds = EyelinerPatchDataset(
                holdout_records, augment_data=False,
                t_quant_step=args.t_quant_step, norm_mean=norm_mean, norm_std=norm_std,
                patch_t_min=args.patch_t_min,
                patch_t_max=args.patch_t_max,
                patch_perp_scale=args.patch_perp_scale,
                patch_size=(args.patch_h, args.patch_w),
                use_t_norm=args.use_t_norm,
                label_t_min=args.label_t_min,
                label_t_max=args.label_t_max,
            )
            holdout_loader = DataLoader(holdout_ds, batch_size=args.batch, shuffle=False)
            holdout_mae_value = compute_holdout_mae(
                ensemble, holdout_loader, device,
                use_pq_ratio=args.use_pq_ratio,
                use_distance_from_center=args.use_distance_from_center,
                use_norm_eye_scale=args.use_norm_eye_scale,
            )
            logger.info("holdout_mae: %.6f", holdout_mae_value)
        except Exception as exc:
            logger.warning("Hold-out MAE 計算失敗: %s", exc)

    # experiment.json 保存
    exp_record = build_experiment_record(
        args, fold_results, holdout_mae_value, data_version,
        n_train_samples, n_holdout_samples, train_images, holdout_images,
    )
    exp_json_path = exp_dir / "experiment.json"
    exp_json_path.write_text(
        json.dumps(exp_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("experiment.json 保存: %s", exp_json_path)

    # prediction_train.csv の保存（--save_prediction_train 指定時）
    if args.save_prediction_train:
        logger.info("prediction_train.csv を生成中...")
        try:
            ensemble, _ = create_ensemble_from_dir(exp_dir)
            ensemble.eval()
            device_pt = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ensemble = ensemble.to(device_pt)
            _save_prediction_train_csv(
                ensemble, train_records, exp_dir, args, norm_mean, norm_std, device_pt,
            )
        except Exception as exc:
            logger.warning("prediction_train.csv 生成失敗: %s", exc)

    logger.info("訓練完了: %s", exp_name)
    logger.info("  cv_mae     = %.6f", exp_record["results"]["cv_mae"])
    if holdout_mae_value is not None:
        logger.info("  holdout_mae = %.6f", holdout_mae_value)


def mode_ensemble(args: argparse.Namespace) -> None:
    """fold モデルをアンサンブル化して保存する。"""
    models_dir = Path(args.models_dir) if args.models_dir else Path(args.output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ensemble, backbone_name = create_ensemble_from_dir(models_dir)
    save_path = output_dir / "eyeliner_detection_ensemble.pt"
    save_ensemble(ensemble, backbone_name, save_path)
    logger.info("アンサンブルモデルを保存しました: %s", save_path)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="アイライン輪郭検出 CNN 訓練（新版）")
    parser.add_argument("--manifest", type=Path, help="manifest CSV（kfold モード用）")
    parser.add_argument("--holdout", type=Path, default=None, help="Hold-out CSV（汎化性能評価用）")
    parser.add_argument("--output_dir", default="models", type=Path, help="実験ディレクトリの親")
    parser.add_argument("--exp_name", default=None, help="実験名（省略時は自動生成）")
    parser.add_argument(
        "--mode", choices=["kfold", "ensemble"], default="kfold",
    )
    parser.add_argument(
        "--backbone", choices=BACKBONE_CHOICES, default="custom",
    )
    parser.add_argument("--freeze_epochs", default=10, type=int)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--k", default=5, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--patience", default=50, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--models_dir", type=Path, default=None, help="ensemble モード用")
    parser.add_argument(
        "--t_quant_step", type=float, default=None,
        help="生の物理 t 値の量子化ステップ（例: 0.1 / 0.02 / 0.01）。None で量子化なし",
    )
    parser.add_argument(
        "--round_t", type=int, default=None,
        help="(後方互換) --t_quant_step = 10**(-round_t) に翻訳される。新規実験では --t_quant_step を使うこと",
    )
    parser.add_argument(
        "--loss_fn", choices=LOSS_FN_CHOICES, default="normalized_pixel",
        dest="loss_fn",
    )
    # 後方互換: 旧スクリプトの --loss_space と同じ意味
    parser.add_argument(
        "--loss_space", choices=LOSS_FN_CHOICES, default=None,
        help="(後方互換) --loss_fn と同じ",
    )
    parser.add_argument(
        "--norm_type", choices=NORM_TYPE_CHOICES, default="imagenet",
    )
    parser.add_argument("--norm_stats_json", type=Path, default=None)
    parser.add_argument(
        "--use_pq_ratio", action="store_true",
        help="pq_dist/eye_scale をモデルの追加スカラー入力として使用する",
    )
    parser.add_argument(
        "--use_sigmoid", action="store_true",
        help="出力層に Sigmoid を追加して t を [0, 1] に制約する",
    )
    parser.add_argument(
        "--use_batchnorm", action="store_true", default=True,
        help="特徴抽出 CNN に BatchNorm2d を使用する（デフォルト: True）",
    )
    parser.add_argument(
        "--no_batchnorm", dest="use_batchnorm", action="store_false",
        help="特徴抽出 CNN の BatchNorm2d を無効化する",
    )
    parser.add_argument(
        "--num_conv_layers", type=int, default=3, choices=[2, 3, 4, 5],
        help="EyelinerCNN の畳み込み層数（default: 3）",
    )
    parser.add_argument(
        "--use_distance_from_center", action="store_true",
        help="端点度 min(pi, 6-pi)/3 をモデルの追加スカラー入力として使用する",
    )
    parser.add_argument(
        "--use_norm_eye_scale", action="store_true",
        help="eye_scale / sqrt(img_H * img_W) をモデルの追加スカラー入力として使用する",
    )
    parser.add_argument(
        "--use_logit_target", action="store_true",
        help="ターゲット t を logit 空間に変換して学習する（use_sigmoid は False にすること）",
    )
    parser.add_argument(
        "--no_t_norm", action="store_true",
        help="t_norm 正規化を無効化し、t_actual（生の物理 t 値）で直接学習・評価する",
    )
    parser.add_argument(
        "--use_t_weighted_sampler", action="store_true",
        help="t 値のエッジ（0 or 1 付近）サンプルを高確率でサンプリングする WeightedRandomSampler を使用する",
    )
    parser.add_argument(
        "--t_weight_alpha", type=float, default=3.0,
        help="WeightedRandomSampler の重み係数 α（weight = 1 + α × |t - 0.5|, default: 3.0）",
    )
    parser.add_argument(
        "--t_loss_alpha", type=float, default=3.0,
        help="t_weighted 損失の重み係数 α（weight = 1 + α × |y - 0.5|, default: 3.0）",
    )
    parser.add_argument(
        "--loss_epsilon", type=float, default=0.01,
        help="dead_zone 損失のデッドゾーン幅（正規化 t スケール, default: 0.01 ≈ 0.64px）",
    )
    parser.add_argument(
        "--loss_delta", type=float, default=0.05,
        help="dead_zone / huber 損失の二次→線形遷移点（default: 0.05）",
    )
    parser.add_argument(
        "--patch_t_min", type=float, default=-0.5,
        help="パッチ抽出（crop）の t 下限（default: -0.5）",
    )
    parser.add_argument(
        "--patch_t_max", type=float, default=1.5,
        help="パッチ抽出（crop）の t 上限（default: 1.5）",
    )
    parser.add_argument(
        "--label_t_min", type=float, default=None,
        help="ラベル正規化の t 下限（default: None → patch_t_min と同じ）",
    )
    parser.add_argument(
        "--label_t_max", type=float, default=None,
        help="ラベル正規化の t 上限（default: None → patch_t_max と同じ）",
    )
    parser.add_argument(
        "--patch_perp_scale", type=float, default=1.0,
        help="縦軸カバレッジ = perp_scale * |PQ| ピクセル（default: 1.0）",
    )
    parser.add_argument(
        "--patch_h", type=int, default=64,
        help="入力パッチ高さ（default: 64）",
    )
    parser.add_argument(
        "--patch_w", type=int, default=128,
        help="入力パッチ幅（default: 128）",
    )
    parser.add_argument(
        "--aug_flags", type=lambda x: int(x, 0), default=0x4F,
        help="オーグメンテーションのビットフラグ（0x4F=全有効, 0x00=全無効）",
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.3,
        help="回帰ヘッドの Dropout 率（default: 0.3）",
    )
    parser.add_argument(
        "--save_prediction_train", action="store_true",
        help="訓練セットへのアンサンブル推論結果を prediction_train.csv として保存する",
    )

    args = parser.parse_args()

    # --loss_space が指定された場合は --loss_fn として扱う（後方互換）
    if args.loss_space is not None:
        args.loss_fn = args.loss_space

    # --round_t エイリアス: 新規実装では --t_quant_step を使用する
    if args.round_t is not None:
        if args.t_quant_step is not None:
            parser.error("--round_t と --t_quant_step は同時指定できません。--t_quant_step を使ってください。")
        args.t_quant_step = 10.0 ** (-args.round_t)
        logger.warning(
            "--round_t=%d は非推奨です。--t_quant_step=%g に翻訳しました。",
            args.round_t, args.t_quant_step,
        )

    args.use_t_norm = not args.no_t_norm

    # --norm_type dataset かつ --norm_stats_json 未指定の場合は自動検索
    if args.norm_type == "dataset" and args.norm_stats_json is None:
        research_dir = Path(__file__).parent.parent / "tasks" / "research"
        candidates = sorted(research_dir.glob("*_dataset_normalization_stats.json"))
        if candidates:
            args.norm_stats_json = candidates[-1]
            logger.info("norm_stats_json を自動検出: %s", args.norm_stats_json)
        else:
            parser.error(
                "--norm_type dataset を使用するには先に "
                "scripts/compute_dataset_stats.py を実行してください。"
            )

    if args.mode == "ensemble":
        mode_ensemble(args)
    else:
        if not args.manifest:
            parser.error("--manifest は kfold モードで必須です。")
        mode_kfold(args)


if __name__ == "__main__":
    main()
