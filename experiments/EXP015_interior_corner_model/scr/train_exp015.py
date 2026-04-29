"""EXP015: interior / corner 分離モデル 訓練スクリプト

point_index を「corner（0, 6）」と「interior（1–5）」に二分割し、
それぞれ専用モデルを学習する。

グループ定義:
    corner   : point_index 0, 6（端点・眼角）→ manifest_*_corner.csv
    interior : point_index 1–5（中間点）→ manifest_*_interior.csv

条件:
    corner          : corner デフォルト（patch_t -2.0〜3.0 / label_t -1〜2 / perp 2.0）
    corner_long     : corner × 長期学習（epochs=1000 / patience=250 / batch=16）
    corner_wide     : corner × ワイドパッチ（patch_t -2.5〜3.5 / label_t -2〜3 / perp 2.5）
    interior        : interior デフォルト（patch_t -2.0〜3.0 / perp 2.0）
    interior_narrow : interior × ナローパッチ（h=96 w=192 / perp 1.0 / label_t -0.5〜1.5）

使い方:
    # 全条件を実行
    python scripts/sweep.py --exp 015

    # 特定条件のみ実行
    python experiments/EXP015_interior_corner_model/scr/train_exp015.py --configs corner interior
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

MANIFEST_TRAIN_CORNER   = MANIFEST_DIR / "manifest_train_corner.csv"
MANIFEST_TEST_CORNER    = MANIFEST_DIR / "manifest_test_corner.csv"
MANIFEST_TRAIN_INTERIOR = MANIFEST_DIR / "manifest_train_interior.csv"
MANIFEST_TEST_INTERIOR  = MANIFEST_DIR / "manifest_test_interior.csv"

sys.path.insert(0, str(ROOT))
from scripts.exp_utils import run_with_log  # noqa: E402


def get_manifests(config_name: str) -> tuple[Path, Path]:
    """config 名のプレフィックスから (manifest_train, manifest_test) を返す。"""
    if config_name.startswith("corner"):
        return MANIFEST_TRAIN_CORNER, MANIFEST_TEST_CORNER
    if config_name.startswith("interior"):
        return MANIFEST_TRAIN_INTERIOR, MANIFEST_TEST_INTERIOR
    raise ValueError(
        f"config 名 '{config_name}' が 'corner' / 'interior' で始まっていません"
    )


def load_config(config_name: str) -> dict:
    path = CONFIG_DIR / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_cmd(config_name: str, cfg: dict) -> list[str]:
    manifest_train, manifest_test = get_manifests(config_name)
    output_dir = OUTPUT_DIR / config_name
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--manifest",         str(manifest_train),
        "--holdout",          str(manifest_test),
        "--output_dir",       str(output_dir),
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
        "--aug_flags",        str(cfg.get("aug_flags", 79)),
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

    if cfg.get("use_pq_ratio"):
        cmd.append("--use_pq_ratio")
    if cfg.get("use_norm_eye_scale"):
        cmd.append("--use_norm_eye_scale")
    if cfg.get("use_sigmoid"):
        cmd.append("--use_sigmoid")
    if not cfg.get("use_batchnorm", True):
        cmd.append("--no_batchnorm")
    if cfg.get("use_t_weighted_sampler"):
        cmd.append("--use_t_weighted_sampler")
        cmd += ["--t_weight_alpha", str(cfg.get("t_weight_alpha", 3.0))]

    return cmd


def run_experiment(config_name: str) -> int:
    print(f"\n{'='*60}")
    print(f"  実行: {config_name}")
    print(f"{'='*60}")

    cfg = load_config(config_name)
    manifest_train, manifest_test = get_manifests(config_name)

    for path, label in [(manifest_train, "manifest_train"), (manifest_test, "manifest_test")]:
        if not path.exists():
            print(f"  [ERROR] {label} が見つかりません: {path}")
            return 1

    cmd = build_train_cmd(config_name, cfg)
    output_dir = OUTPUT_DIR / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  manifest_train: {manifest_train.name}")
    print(f"  manifest_test:  {manifest_test.name}")
    print(f"  コマンド: {' '.join(cmd)}")
    return run_with_log(cmd, LOG_PATH, cwd=ROOT)


def main() -> None:
    all_configs = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))

    parser = argparse.ArgumentParser(description="EXP015 interior/corner 分離モデル 訓練")
    parser.add_argument(
        "--configs", nargs="+", default=all_configs,
        help=f"実行する設定名（デフォルト: 全条件 {all_configs}）",
    )
    args = parser.parse_args()

    print(f"\nEXP015 sweep 開始: {len(args.configs)} 条件")
    print(f"  output_dir: {OUTPUT_DIR}")
    print(f"  log:        {LOG_PATH}")

    failed = []
    for config_name in args.configs:
        rc = run_experiment(config_name)
        if rc != 0:
            print(f"\n[WARN] {config_name} が終了コード {rc} で失敗")
            failed.append(config_name)

    print(f"\n{'='*60}")
    print(f"  EXP015 sweep 完了: {len(args.configs) - len(failed)}/{len(args.configs)} 成功")
    if failed:
        print(f"  失敗: {failed}")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
