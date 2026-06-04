"""EXP020: デュアルブランチ fusion mode sweep 訓練スクリプト。

条件 (k=3):
    X0_none_baseline                : 生PQ座標系 単一CNN（比較基準）
    X0_eye_scale_baseline           : P'Q'(eye_scale) 単一CNN（比較基準）
    X1_independent_none_eye_scale   : 生PQ + eye_scale 独立2ブランチ出力平均
    X1_independent_none_eye_height  : 生PQ + eye_height 独立2ブランチ出力平均
    X2_crossfuse_none_eye_scale     : 生PQ + eye_scale cross-stitch融合
    X2_crossfuse_none_eye_height    : 生PQ + eye_height cross-stitch融合
    X3_attention_none_eye_scale     : 生PQ + eye_scale α重み出力融合
    X3_attention_none_eye_height    : 生PQ + eye_height α重み出力融合

使い方:
    # 全条件実行
    python experiments/EXP020_dual_branch/scr/train_exp020.py

    # 特定条件のみ実行
    python experiments/EXP020_dual_branch/scr/train_exp020.py --configs X0_none_baseline

事前に manifest を生成しておくこと:
    python scripts/split_holdout.py --exp 020
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent.parent.parent
SCR_DIR = Path(__file__).parent
EXP_DIR = Path(__file__).parent.parent
CONFIG_DIR = EXP_DIR / "config"
MANIFEST_DIR = EXP_DIR / "manifests"
OUTPUT_DIR = EXP_DIR / "outputs"
LOG_PATH = EXP_DIR / "log.txt"

MANIFEST_TRAIN = MANIFEST_DIR / "manifest_train.csv"
MANIFEST_TEST = MANIFEST_DIR / "manifest_test.csv"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCR_DIR))
from scripts.exp_utils import run_with_log  # noqa: E402
from eyeliner.data.dataset import load_manifest  # noqa: E402
from eyeliner.data.dataset_dual import EyelinerPatchDualDataset  # noqa: E402
from eyeliner.data.weights import compute_eye_meta  # noqa: E402
from eyeliner.evaluation.loss import get_loss_fn  # noqa: E402
from eyeliner.training.scheduler import make_optimizer_and_scheduler  # noqa: E402
from eyeliner.training.trainer import set_seed  # noqa: E402
from dual_branch_utils import (  # noqa: E402
    build_branch_configs,
    branch_pred_to_raw_t,
    ensemble_raw_t,
    model_pred_raw_t,
    fold_ensemble_raw_t,
    normalize_to_branch_a,
)

ALL_CONFIGS = [
    "X0_none_baseline",
    "X0_eye_scale_baseline",
    "X0_eye_height_baseline",
    "X1_independent_none_eye_scale",
    "X1_independent_none_eye_height",
    "X2_crossfuse_none_eye_scale",
    "X2_crossfuse_none_eye_height",
    "X3_attention_none_eye_scale",
    "X3_attention_none_eye_height",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_name: str) -> dict:
    path = CONFIG_DIR / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        import yaml
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Baseline (subprocess → scripts/train.py)
# ---------------------------------------------------------------------------

def build_baseline_cmd(config_name: str, cfg: dict, num_workers: int = 0) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--manifest",         str(MANIFEST_TRAIN),
        "--holdout",          str(MANIFEST_TEST),
        "--output_dir",       str(OUTPUT_DIR),
        "--exp_name",         config_name,
        "--backbone",         "custom",
        "--loss_fn",          cfg["loss_fn"],
        "--lr",               str(cfg["lr"]),
        "--weight_decay",     str(cfg["weight_decay"]),
        "--epochs",           str(cfg["epochs"]),
        "--batch",            str(cfg["batch"]),
        "--k",                str(cfg["k"]),
        "--patience",         str(cfg["patience"]),
        "--seed",             str(cfg["seed"]),
        "--num_conv_layers",  str(cfg.get("num_conv_layers", 4)),
        "--dropout_rate",     str(cfg.get("dropout_rate", 0.3)),
        "--aug_flags",        str(cfg.get("aug_flags", 127)),
        "--patch_t_min",      str(cfg["patch_t_min"]),
        "--patch_t_max",      str(cfg["patch_t_max"]),
        "--patch_perp_scale", str(cfg["patch_perp_scale"]),
        "--patch_h",          str(cfg["patch_h"]),
        "--patch_w",          str(cfg["patch_w"]),
        "--label_t_min",      str(cfg["label_t_min"]),
        "--label_t_max",      str(cfg["label_t_max"]),
        "--use_sigmoid",
    ]
    if not cfg.get("use_batchnorm", True):
        cmd.append("--no_batchnorm")
    if not cfg.get("use_is_upper", True):
        cmd.append("--no_is_upper")
    pq_mode = cfg.get("pq_rescale_mode", "none")
    if pq_mode != "none":
        cmd += ["--pq_rescale_mode", pq_mode]
        # factor を渡さないと scripts/train.py のデフォルト 1.0 になり EXP019 と不一致になる。
        cmd += ["--pq_rescale_factor", str(cfg.get("pq_rescale_factor", 1.0))]
    if num_workers > 0:
        cmd += ["--num_workers", str(num_workers)]
    return cmd


def run_baseline(config_name: str, cfg: dict, num_workers: int = 0) -> int:
    print(f"\n{'='*60}")
    print(f"  Baseline: {config_name}")
    print(f"{'='*60}")

    for path, label in [(MANIFEST_TRAIN, "manifest_train"), (MANIFEST_TEST, "manifest_test")]:
        if not path.exists():
            print(f"  [ERROR] {label} が見つかりません: {path}")
            print(f"         先に split_holdout.py --exp 020 を実行してください")
            return 1

    cmd = build_baseline_cmd(config_name, cfg, num_workers)
    print(f"  pq_rescale_mode: {cfg.get('pq_rescale_mode', 'none')}")
    print(f"  k:             {cfg['k']}")
    print(f"  コマンド: {' '.join(cmd)}")
    return run_with_log(cmd, LOG_PATH, cwd=ROOT)


# ---------------------------------------------------------------------------
# Dual training (self-contained)
# ---------------------------------------------------------------------------

def _get_dual_model(
    fusion_mode: str,
    num_conv_layers: int = 4,
    use_sigmoid: bool = True,
    use_batchnorm: bool = True,
    dropout_rate: float = 0.3,
) -> nn.Module:
    from eyeliner.models.eyeliner_cnn_dual import EyelinerCNNDual
    from eyeliner.models.eyeliner_cnn_dual_branch import CrossFuseDualModel

    if fusion_mode == "crossfuse":
        return CrossFuseDualModel(
            num_conv_layers=num_conv_layers,
            use_sigmoid=use_sigmoid,
            use_batchnorm=use_batchnorm,
            dropout_rate=dropout_rate,
        )
    return EyelinerCNNDual(
        fusion_mode=fusion_mode,
        num_conv_layers=num_conv_layers,
        use_sigmoid=use_sigmoid,
        use_batchnorm=use_batchnorm,
        dropout_rate=dropout_rate,
    )


def _save_dual_model(
    model: nn.Module, fusion_mode: str, cfg: dict, path: Path,
) -> None:
    torch.save({
        "type": "dual_model",
        "fusion_mode": fusion_mode,
        "num_conv_layers": cfg.get("num_conv_layers", 4),
        "use_sigmoid": cfg.get("use_sigmoid", True),
        "use_batchnorm": cfg.get("use_batchnorm", True),
        "dropout_rate": cfg.get("dropout_rate", 0.3),
        "state_dict": model.state_dict(),
    }, path)


def _load_dual_model(path: Path, fusion_mode: str) -> nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = _get_dual_model(
        fusion_mode,
        num_conv_layers=ckpt.get("num_conv_layers", 4),
        use_sigmoid=ckpt.get("use_sigmoid", True),
        use_batchnorm=ckpt.get("use_batchnorm", True),
        dropout_rate=ckpt.get("dropout_rate", 0.3),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _branch_cfg_dict(cfg) -> dict:
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


def _build_exp_json(
    config_name: str, cfg: dict, fold_results: list,
    holdout_mae: float | None, holdout_mae_raw: float | None,
    n_train: int, n_holdout: int,
    train_images: list[str], holdout_images: list[str],
    cfg_a=None, cfg_b=None,
) -> dict:
    return {
        "exp_name": config_name,
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": {
            "fusion_mode": cfg.get("fusion_mode", "baseline"),
            "rescale_b": cfg.get("pq_rescale_mode_b", "none"),
            "branch_a": _branch_cfg_dict(cfg_a) if cfg_a is not None else None,
            "branch_b": _branch_cfg_dict(cfg_b) if cfg_b is not None else None,
            "num_conv_layers": cfg.get("num_conv_layers", 4),
            "use_sigmoid": cfg.get("use_sigmoid", True),
            "use_batchnorm": cfg.get("use_batchnorm", True),
            "dropout_rate": cfg.get("dropout_rate", 0.3),
            "lr": cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "epochs": cfg["epochs"],
            "batch": cfg["batch"],
            "k": cfg["k"],
            "patience": cfg["patience"],
            "seed": cfg["seed"],
        },
        "data": {
            "n_train_samples": n_train,
            "n_holdout_samples": n_holdout,
            "train_images": train_images,
            "holdout_images": holdout_images,
        },
        "results": {
            # MAE は枝A正規化空間（EXP019/020 X0 の正規化 MAE と直接比較可能）。
            # 生 t MAE = normalized * (label_t_max - label_t_min)。
            "cv_mae": float(np.mean([r["val_mae"] for r in fold_results])),
            "cv_folds": [r["val_mae"] for r in fold_results],
            "holdout_mae": holdout_mae,
            "holdout_mae_raw_t": holdout_mae_raw,
        },
    }


def train_dual(config_name: str, cfg: dict, num_workers: int = 0) -> int:
    print(f"\n{'='*60}")
    print(f"  Dual: {config_name}")
    print(f"{'='*60}")

    for path, label in [(MANIFEST_TRAIN, "manifest_train"), (MANIFEST_TEST, "manifest_test")]:
        if not path.exists():
            print(f"  [ERROR] {label} が見つかりません: {path}")
            return 1

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  デバイス: {device}")

    fusion_mode = cfg["fusion_mode"]
    rescale_b = cfg.get("pq_rescale_mode_b", "eye_scale")
    print(f"  fusion_mode: {fusion_mode}")
    print(f"  rescale_b:   {rescale_b}")

    # manifest
    records = load_manifest(MANIFEST_TRAIN)
    compute_eye_meta(records)
    n_train = len(records)
    train_images = sorted({r["image_path"] for r in records})
    print(f"  訓練サンプル: {n_train}")

    n_holdout = 0
    holdout_images: list[str] = []
    holdout_records: list[dict] | None = None
    if MANIFEST_TEST.exists():
        holdout_records = load_manifest(MANIFEST_TEST)
        compute_eye_meta(holdout_records)  # core dataset は pq_ratio/eye_scale 等を要求
        n_holdout = len(holdout_records)
        holdout_images = sorted({r["image_path"] for r in holdout_records})
        print(f"  hold-out:     {n_holdout}")
    else:
        print("  hold-out:     なし")

    # DataLoader params
    dl_kwargs = {"num_workers": num_workers, "persistent_workers": num_workers > 0}

    # 枝 A=EXP019 X0（生PQ, raw_t）, 枝 B=EXP019 X1/X2（P'Q', t_prime）を忠実に再現。
    cfg_a, cfg_b = build_branch_configs(rescale_b)
    aug_flags = cfg.get("aug_flags", 127)
    print(f"  枝A: {cfg_a.pq_rescale_mode} label_space={cfg_a.label_space} "
          f"patch_t=[{cfg_a.patch_t_min},{cfg_a.patch_t_max}] perp={cfg_a.patch_perp_scale}")
    print(f"  枝B: {cfg_b.pq_rescale_mode}*{cfg_b.pq_rescale_factor} "
          f"label_space={cfg_b.label_space} "
          f"patch_t=[{cfg_b.patch_t_min},{cfg_b.patch_t_max}] perp={cfg_b.patch_perp_scale}")

    # K-Fold
    groups = [r["image_path"] for r in records]
    kf = GroupKFold(n_splits=cfg["k"])
    splits = list(kf.split(records, groups=groups))

    exp_dir = OUTPUT_DIR / config_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    fold_results: list[dict] = []
    loss_fn_name = cfg.get("loss_fn", "t")
    if loss_fn_name == "t":
        loss_fn = get_loss_fn("t")
    else:
        loss_fn = get_loss_fn(loss_fn_name)

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n----- Fold {fold + 1}/{cfg['k']} -----")

        train_ds = EyelinerPatchDualDataset(
            [records[i] for i in train_idx], cfg_a, cfg_b,
            augment_data=True, aug_flags=aug_flags,
        )
        val_ds = EyelinerPatchDualDataset(
            [records[i] for i in val_idx], cfg_a, cfg_b,
            augment_data=False, aug_flags=aug_flags,
        )
        train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True, **dl_kwargs)
        val_loader = DataLoader(val_ds, batch_size=cfg["batch"], shuffle=False, **dl_kwargs)

        model = _get_dual_model(
            fusion_mode,
            num_conv_layers=cfg.get("num_conv_layers", 4),
            use_sigmoid=cfg.get("use_sigmoid", True),
            use_batchnorm=cfg.get("use_batchnorm", True),
            dropout_rate=cfg.get("dropout_rate", 0.3),
        ).to(device)

        optimizer, scheduler = make_optimizer_and_scheduler(
            model, "full", cfg["lr"], cfg["weight_decay"],
            len(train_loader), cfg["epochs"],
        )

        best_mae = float("inf")
        no_improve = 0
        best_epoch = 0

        for epoch in range(cfg["epochs"]):
            # --- Train ---
            model.train()
            total_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                x_a = batch[0].to(device)
                x_b = batch[1].to(device)
                is_upper = batch[2].to(device)
                y_a = batch[4].to(device)          # 枝A ラベル（生t正規化）
                y_b = batch[5].to(device)          # 枝B ラベル（t'正規化）
                pq_dist = batch[12].to(device)
                pq_prime = batch[13].to(device)

                # 各枝はそれぞれの label_space で学習（EXP019 単一ブランチを忠実再現）。
                if fusion_mode == "attention":
                    pred_a, pred_b, alpha = model(x_a, x_b, is_upper)
                    t_a = branch_pred_to_raw_t(pred_a, cfg_a, pq_dist, pq_prime)
                    t_b = branch_pred_to_raw_t(pred_b, cfg_b, pq_dist, pq_prime)
                    ens_norm = normalize_to_branch_a(ensemble_raw_t(t_a, t_b, alpha), cfg_a)
                    loss = loss_fn(pred_a, y_a) + loss_fn(pred_b, y_b) + loss_fn(ens_norm, y_a)
                else:
                    pred_a, pred_b = model(x_a, x_b, is_upper)
                    loss = loss_fn(pred_a, y_a) + loss_fn(pred_b, y_b)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                n_batches += 1
            avg_loss = total_loss / max(n_batches, 1)

            # --- Validate（生t空間でアンサンブル → 枝A正規化空間で MAE）---
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
                    t_ens = model_pred_raw_t(
                        model, x_a, x_b, is_upper, fusion_mode,
                        cfg_a, cfg_b, pq_dist, pq_prime,
                    )
                    pred_norm = normalize_to_branch_a(t_ens, cfg_a)
                    all_preds.append(pred_norm.cpu())
                    all_labels.append(y_a)

            preds_t = torch.cat(all_preds)
            labels_t = torch.cat(all_labels)
            val_mae = (preds_t - labels_t).abs().mean().item()

            current_lr = scheduler.get_last_lr()[0]
            print(f"  Fold {fold + 1} Epoch {epoch + 1:3d}  "
                  f"loss={avg_loss:.6f}  val_mae={val_mae:.6f}  lr={current_lr:.2e}")

            # Early stopping
            if val_mae < best_mae:
                best_mae = val_mae
                best_epoch = epoch + 1
                no_improve = 0
                model_path = exp_dir / f"eyeliner_detection_fold{fold + 1}.pt"
                _save_dual_model(model, fusion_mode, cfg, model_path)
            else:
                no_improve += 1
                if no_improve >= cfg["patience"]:
                    print(f"  Early stopping (fold {fold + 1}, epoch {epoch + 1})")
                    break

        fold_results.append({
            "fold": fold + 1,
            "val_mae": best_mae,
            "best_epoch": best_epoch,
        })
        print(f"  Fold {fold + 1}: Best MAE={best_mae:.6f} (epoch {best_epoch})")

        del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Summary
    mean_mae = float(np.mean([r["val_mae"] for r in fold_results]))
    print(f"\n  CV MAE: {mean_mae:.6f}")
    for r in fold_results:
        print(f"    Fold {r['fold']}: {r['val_mae']:.6f}")

    # Hold-out MAE（fold モデルを生t空間でアンサンブル → 枝A正規化空間で MAE）
    holdout_mae_value: float | None = None
    holdout_mae_raw: float | None = None
    if holdout_records is not None and len(holdout_records) > 0:
        print("\n  Hold-out MAE 計算中...")
        try:
            fold_paths = sorted(exp_dir.glob("eyeliner_detection_fold*.pt"))
            models = [_load_dual_model(p, fusion_mode).to(device) for p in fold_paths]
            holdout_ds = EyelinerPatchDualDataset(
                holdout_records, cfg_a, cfg_b, augment_data=False, aug_flags=aug_flags,
            )
            holdout_loader = DataLoader(holdout_ds, batch_size=cfg["batch"], shuffle=False, **dl_kwargs)

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
                        models, x_a, x_b, is_upper, fusion_mode,
                        cfg_a, cfg_b, pq_dist, pq_prime,
                    )
                    abs_err_raw = (t_ens - t_true_raw).abs()
                    abs_sum_raw += abs_err_raw.sum().item()
                    abs_sum_norm += (abs_err_raw / label_range_a).sum().item()
                    count += t_true_raw.numel()
            holdout_mae_value = abs_sum_norm / max(count, 1)
            holdout_mae_raw = abs_sum_raw / max(count, 1)
            print(f"  Hold-out MAE: {holdout_mae_value:.6f}（正規化） / "
                  f"{holdout_mae_raw:.6f}（生t）")
        except Exception as exc:
            print(f"  [WARN] Hold-out MAE 計算失敗: {exc}")

    # experiment.json
    exp_record = _build_exp_json(
        config_name, cfg, fold_results, holdout_mae_value, holdout_mae_raw,
        n_train, n_holdout, train_images, holdout_images,
        cfg_a=cfg_a, cfg_b=cfg_b,
    )
    exp_json_path = exp_dir / "experiment.json"
    exp_json_path.write_text(
        json.dumps(exp_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  experiment.json 保存: {exp_json_path}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="EXP020 dual branch sweep 訓練")
    parser.add_argument(
        "--configs", nargs="+", default=ALL_CONFIGS,
        help=f"実行する設定名（デフォルト: 全条件）",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader の num_workers",
    )
    args = parser.parse_args()

    print(f"\nEXP020 dual_branch sweep 開始: {len(args.configs)} 条件")
    print(f"  output_dir: {OUTPUT_DIR}")
    print(f"  log:        {LOG_PATH}")
    print(f"  num_workers:{args.num_workers}")

    failed: list[str] = []
    for config_name in args.configs:
        if config_name not in ALL_CONFIGS:
            print(f"\n[ERROR] 未知の設定名: {config_name}")
            print(f"  有効な設定: {ALL_CONFIGS}")
            failed.append(config_name)
            continue

        cfg = load_config(config_name)

        if cfg["backbone"] == "custom":
            rc = run_baseline(config_name, cfg, num_workers=args.num_workers)
        else:
            rc = train_dual(config_name, cfg, num_workers=args.num_workers)

        if rc != 0:
            print(f"\n[WARN] {config_name} が終了コード {rc} で失敗")
            failed.append(config_name)

    print(f"\n{'='*60}")
    print(f"  EXP020 完了: {len(args.configs) - len(failed)}/{len(args.configs)} 成功")
    if failed:
        print(f"  失敗: {failed}")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
