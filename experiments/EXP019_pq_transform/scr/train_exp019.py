"""EXP019: P'Q' 再スケール座標系 sweep 訓練スクリプト

条件 (k=3):
    X0_baseline  : EXP018 X0_baseline と同一（比較基準）
    X1_eye_height : pq_rescale_mode=eye_height（eye_height 基準 P'Q' 再設定）
    X2_eye_scale : pq_rescale_mode=eye_scale（eye_scale 基準 P'Q' 再設定）
    X3_mean_pq   : pq_rescale_mode=mean_pq_dist（mean_pq_dist 基準 P'Q' 再設定）

使い方:
    # 全条件を実行
    python scripts/sweep.py --exp 019

    # 特定条件のみ実行
    python experiments/EXP019_pq_transform/scr/train_exp019.py --configs X0_baseline

事前に manifest を生成しておくこと:
    python scripts/split_holdout.py --exp 019
"""
from __future__ import annotations

import argparse
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

sys.path.insert(0, str(ROOT))
from scripts.exp_utils import run_with_log  # noqa: E402

ALL_CONFIGS = ["X0_baseline", "X1_eye_height", "X2_eye_scale", "X3_mean_pq"]


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

    pq_rescale_mode = cfg.get("pq_rescale_mode", "none")
    if pq_rescale_mode != "none":
        cmd += ["--pq_rescale_mode", pq_rescale_mode]
        cmd += ["--pq_rescale_factor", str(cfg.get("pq_rescale_factor", 1.0))]

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

    max_train_samples = cfg.get("max_train_samples")
    if max_train_samples is not None:
        cmd += ["--max_train_samples", str(max_train_samples)]

    mid_t_keep_ratio = cfg.get("mid_t_keep_ratio", 1.0)
    if mid_t_keep_ratio < 1.0:
        cmd += ["--mid_t_keep_ratio", str(mid_t_keep_ratio)]

    if num_workers > 0:
        cmd += ["--num_workers", str(num_workers)]

    return cmd


def run_condition(config_name: str, num_workers: int = 0) -> int:
    print(f"\n{'='*60}")
    print(f"  実行: {config_name}")
    print(f"{'='*60}")

    cfg = load_config(config_name)

    for path, label in [(MANIFEST_TRAIN, "manifest_train"), (MANIFEST_TEST, "manifest_test")]:
        if not path.exists():
            print(f"  [ERROR] {label} が見つかりません: {path}")
            print(f"         先に split_holdout.py --exp 019 を実行してください")
            return 1

    cmd = build_train_cmd(config_name, cfg, num_workers)

    print(f"  loss_fn:         {cfg['loss_fn']}")
    print(f"  pq_rescale_mode: {cfg.get('pq_rescale_mode', 'none')}")
    print(f"  k:             {cfg['k']}")
    print(f"  コマンド: {' '.join(cmd)}")
    return run_with_log(cmd, LOG_PATH, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP019 sweep 訓練")
    parser.add_argument(
        "--configs", nargs="+", default=ALL_CONFIGS,
        help=f"実行する設定名（デフォルト: 全条件 {ALL_CONFIGS}）",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader の num_workers（default: 0）",
    )
    args = parser.parse_args()

    print(f"\nEXP019 sweep 開始: {len(args.configs)} 条件")
    print(f"  output_dir: {OUTPUT_DIR}")
    print(f"  log:        {LOG_PATH}")
    print(f"  num_workers:{args.num_workers}")

    failed: list[str] = []
    for config_name in args.configs:
        rc = run_condition(config_name, num_workers=args.num_workers)
        if rc != 0:
            print(f"\n[WARN] {config_name} が終了コード {rc} で失敗")
            failed.append(config_name)

    print(f"\n{'='*60}")
    print(f"  EXP019 sweep 完了: {len(args.configs) - len(failed)}/{len(args.configs)} 成功")
    if failed:
        print(f"  失敗: {failed}")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
