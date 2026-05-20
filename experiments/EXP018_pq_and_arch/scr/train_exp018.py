"""EXP018: pq_ratio 困難サンプル対策 & アーキテクチャ強化 sweep 訓練スクリプト

条件 (k=3):
    X0_baseline          : EXP017 baseline と同一
    X1_pq_sampler_a3     : pq WeightedRandomSampler alpha=3
    X2_pq_sampler_a5     : pq WeightedRandomSampler alpha=5
    X3_pq_loss_weighted  : pq_weighted 損失 alpha=3
    Y1_custom_conv5_d4   : conv5層・dropout=0.4
    Y2_aux_face_shape    : 顔形状補助特徴量（11次元）concat — カスタム Python 実行

使い方:
    # 全条件を実行
    python scripts/sweep.py --exp 018

    # 特定条件のみ実行
    python experiments/EXP018_pq_and_arch/scr/train_exp018.py --configs X0_baseline X1_pq_sampler_a3

事前に manifest を生成しておくこと:
    python scripts/split_holdout.py --exp 018
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
CONFIG_DIR = EXP_DIR / "config"
MANIFEST_DIR = EXP_DIR / "manifests"
OUTPUT_DIR = EXP_DIR / "outputs"
LOG_PATH = EXP_DIR / "log.txt"

MANIFEST_TRAIN = MANIFEST_DIR / "manifest_train.csv"
MANIFEST_TEST = MANIFEST_DIR / "manifest_test.csv"

DB_PATH = ROOT / "data" / "samples.db"

sys.path.insert(0, str(ROOT))
from scripts.exp_utils import run_with_log  # noqa: E402

logger = logging.getLogger(__name__)


def load_config(config_name: str) -> dict:
    path = CONFIG_DIR / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_cmd(config_name: str, cfg: dict, num_workers: int = 0) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--manifest",         str(MANIFEST_TRAIN),
        "--holdout",          str(MANIFEST_TEST),
        "--output_dir",       str(OUTPUT_DIR),
        "--exp_name",         config_name,
        "--backbone",         cfg["backbone"],
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
        "--freeze_epochs",    str(cfg.get("freeze_epochs", 0)),
        "--aug_flags",        str(cfg.get("aug_flags", 127)),
        "--patch_t_min",      str(cfg["patch_t_min"]),
        "--patch_t_max",      str(cfg["patch_t_max"]),
        "--patch_perp_scale", str(cfg["patch_perp_scale"]),
        "--patch_h",          str(cfg["patch_h"]),
        "--patch_w",          str(cfg["patch_w"]),
    ]

    label_t_min = cfg.get("label_t_min")
    label_t_max = cfg.get("label_t_max")
    if label_t_min is not None:
        cmd += ["--label_t_min", str(label_t_min)]
    if label_t_max is not None:
        cmd += ["--label_t_max", str(label_t_max)]

    max_train_samples = cfg.get("max_train_samples")
    if max_train_samples is not None:
        cmd += ["--max_train_samples", str(max_train_samples)]

    mid_t_keep_ratio = cfg.get("mid_t_keep_ratio", 1.0)
    if mid_t_keep_ratio < 1.0:
        cmd += ["--mid_t_keep_ratio", str(mid_t_keep_ratio)]

    if cfg.get("use_pq_ratio"):
        cmd.append("--use_pq_ratio")
    if cfg.get("use_sigmoid"):
        cmd.append("--use_sigmoid")
    if not cfg.get("use_batchnorm", True):
        cmd.append("--no_batchnorm")
    if not cfg.get("use_is_upper", True):
        cmd.append("--no_is_upper")
    if cfg.get("use_t_weighted_sampler"):
        cmd.append("--use_t_weighted_sampler")
        cmd += ["--t_weight_alpha", str(cfg.get("t_weight_alpha", 3.0))]
    if cfg.get("use_pq_weighted_sampler"):
        cmd.append("--use_pq_weighted_sampler")
        cmd += ["--pq_weight_alpha", str(cfg.get("pq_weight_alpha", 3.0))]
        cmd += ["--pq_threshold",   str(cfg.get("pq_threshold", 0.10))]

    if num_workers > 0:
        cmd += ["--num_workers", str(num_workers)]

    return cmd


def run_standard(config_name: str, num_workers: int = 0) -> int:
    """X0〜Y1 条件: scripts/train.py 経由で実行する。"""
    print(f"\n{'='*60}")
    print(f"  実行: {config_name}")
    print(f"{'='*60}")

    cfg = load_config(config_name)

    for path, label in [(MANIFEST_TRAIN, "manifest_train"), (MANIFEST_TEST, "manifest_test")]:
        if not path.exists():
            print(f"  [ERROR] {label} が見つかりません: {path}")
            print(f"         先に split_holdout.py --exp 018 を実行してください")
            return 1

    cmd = build_train_cmd(config_name, cfg, num_workers)

    print(f"  loss_fn:                {cfg['loss_fn']}")
    print(f"  use_pq_weighted_sampler:{cfg.get('use_pq_weighted_sampler', False)}")
    print(f"  num_conv_layers:        {cfg.get('num_conv_layers', 4)}")
    print(f"  k:                      {cfg['k']}")
    print(f"  コマンド: {' '.join(cmd)}")
    return run_with_log(cmd, LOG_PATH, cwd=ROOT)


def run_y2(num_workers: int = 0) -> int:
    """Y2_aux_face_shape: 顔形状補助特徴量 (11次元) を concat する条件。
    scripts/train.py では補助特徴量パスに対応していないため直接 Python で実行する。
    """
    config_name = "Y2_aux_face_shape"
    print(f"\n{'='*60}")
    print(f"  実行: {config_name}  (カスタム Python パス)")
    print(f"{'='*60}")

    cfg = load_config(config_name)

    for path, label in [(MANIFEST_TRAIN, "manifest_train"), (MANIFEST_TEST, "manifest_test")]:
        if not path.exists():
            print(f"  [ERROR] {label} が見つかりません: {path}")
            return 1

    db_path = DB_PATH if DB_PATH.exists() else None
    if db_path is None:
        print(f"  [WARN] samples.db が見つかりません: {DB_PATH}")
        print(f"         face_yaw/pitch/roll は 0 にフォールバックします")

    from eyeliner.training.kfold import run_kfold
    from eyeliner.training.trainer import TrainConfig

    config = TrainConfig(
        backbone=cfg["backbone"],
        loss_fn=cfg["loss_fn"],
        lr=cfg["lr"],
        batch_size=cfg["batch"],
        weight_decay=cfg["weight_decay"],
        epochs=cfg["epochs"],
        patience=cfg["patience"],
        seed=cfg["seed"],
        num_conv_layers=cfg.get("num_conv_layers", 4),
        dropout_rate=cfg.get("dropout_rate", 0.3),
        use_sigmoid=cfg.get("use_sigmoid", True),
        use_batchnorm=cfg.get("use_batchnorm", True),
        use_is_upper=cfg.get("use_is_upper", True),
        use_aux_face_shape=True,
        aug_flags=cfg.get("aug_flags", 127),
        patch_t_min=cfg["patch_t_min"],
        patch_t_max=cfg["patch_t_max"],
        patch_perp_scale=cfg["patch_perp_scale"],
        patch_h=cfg["patch_h"],
        patch_w=cfg["patch_w"],
        label_t_min=cfg.get("label_t_min"),
        label_t_max=cfg.get("label_t_max"),
        num_workers=num_workers,
    )

    output_dir = OUTPUT_DIR / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(LOG_PATH), encoding="utf-8"),
        ],
    )

    fold_results = run_kfold(
        manifest_path=MANIFEST_TRAIN,
        exp_dir=output_dir,
        config=config,
        k_folds=cfg["k"],
        db_path=db_path,
    )

    mean_mae = sum(r.best_val_mae for r in fold_results) / len(fold_results)
    print(f"\n  Y2 完了: cv_mae={mean_mae:.6f}")
    for r in fold_results:
        print(f"    Fold {r.fold_idx}: best_val_mae={r.best_val_mae:.6f} (epoch {r.best_epoch})")

    # holdout MAE
    if MANIFEST_TEST.exists():
        try:
            import torch
            import json
            from eyeliner.models.ensemble import create_ensemble_from_dir
            from eyeliner.data.dataset import EyelinerPatchDataset, load_manifest
            from eyeliner.data.weights import compute_eye_meta, compute_aux_face_shape_meta
            from eyeliner.evaluation.metrics import holdout_mae as compute_holdout_mae
            from torch.utils.data import DataLoader

            ensemble, _ = create_ensemble_from_dir(output_dir)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ensemble = ensemble.to(device).eval()

            holdout_records = load_manifest(MANIFEST_TEST)
            compute_eye_meta(holdout_records)
            compute_aux_face_shape_meta(holdout_records, db_path)

            holdout_ds = EyelinerPatchDataset(
                holdout_records, augment_data=False,
                use_aux_face_shape=True,
                patch_t_min=config.patch_t_min,
                patch_t_max=config.patch_t_max,
                patch_perp_scale=config.patch_perp_scale,
                patch_size=(config.patch_h, config.patch_w),
                use_t_norm=config.use_t_norm,
                label_t_min=config.label_t_min,
                label_t_max=config.label_t_max,
            )
            holdout_loader = DataLoader(holdout_ds, batch_size=cfg["batch"], shuffle=False, num_workers=num_workers)
            h_mae = compute_holdout_mae(
                ensemble, holdout_loader, device,
                use_aux_face_shape=True,
            )
            print(f"  holdout_mae: {h_mae:.6f}")

            exp_record = {
                "exp_name": config_name,
                "results": {
                    "cv_mae": mean_mae,
                    "cv_folds": [r.best_val_mae for r in fold_results],
                    "holdout_mae": h_mae,
                },
            }
            (output_dir / "experiment.json").write_text(
                json.dumps(exp_record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            print(f"  [WARN] holdout MAE 計算失敗: {exc}")

    return 0


STANDARD_CONFIGS = ["X0_baseline", "X1_pq_sampler_a3", "X2_pq_sampler_a5", "X3_pq_loss_weighted", "Y1_custom_conv5_d4"]
Y2_CONFIG = "Y2_aux_face_shape"


def main() -> None:
    all_configs = STANDARD_CONFIGS + [Y2_CONFIG]

    parser = argparse.ArgumentParser(description="EXP018 sweep 訓練")
    parser.add_argument(
        "--configs", nargs="+", default=all_configs,
        help=f"実行する設定名（デフォルト: 全条件 {all_configs}）",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader の num_workers（default: 0）",
    )
    args = parser.parse_args()

    print(f"\nEXP018 sweep 開始: {len(args.configs)} 条件")
    print(f"  output_dir: {OUTPUT_DIR}")
    print(f"  log:        {LOG_PATH}")
    print(f"  num_workers:{args.num_workers}")

    failed: list[str] = []
    for config_name in args.configs:
        if config_name == Y2_CONFIG:
            rc = run_y2(num_workers=args.num_workers)
        else:
            rc = run_standard(config_name, num_workers=args.num_workers)
        if rc != 0:
            print(f"\n[WARN] {config_name} が終了コード {rc} で失敗")
            failed.append(config_name)

    print(f"\n{'='*60}")
    print(f"  EXP018 sweep 完了: {len(args.configs) - len(failed)}/{len(args.configs)} 成功")
    if failed:
        print(f"  失敗: {failed}")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
