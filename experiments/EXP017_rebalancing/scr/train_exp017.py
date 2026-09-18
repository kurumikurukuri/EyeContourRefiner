"""EXP017: データ再バランス sweep 訓練スクリプト

易しいサンプル（0.5 < t < 0.9）の削減と t 極端値のオーバーサンプリングが
holdout test MAE に与える影響を測定する。

条件 (k=3):
    W0_baseline    : 変更なし（EXP016 S12 参照）
    W1_tw_a3       : t weighted sampler alpha=3
    W2_tw_a5       : t weighted sampler alpha=5
    W3_mid50       : mid-t 50% 削減
    W4_mid30       : mid-t 30% 削減
    W5_mid50_tw3   : mid-t 50% 削減 + t weighted sampler alpha=3

使い方:
    # 全条件を実行
    python scripts/sweep.py --exp 017

    # 特定条件のみ実行
    python experiments/EXP017_rebalancing/scr/train_exp017.py --configs W0_baseline W1_tw_a3

事前に manifest を生成しておくこと:
    python scripts/split_holdout.py --exp 017
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


def load_config(config_name: str) -> dict:
    path = CONFIG_DIR / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_cmd(config_name: str, cfg: dict) -> list[str]:
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

    t_weight_alpha = cfg.get("t_weight_alpha", 3.0)
    if cfg.get("use_t_weighted_sampler"):
        cmd.append("--use_t_weighted_sampler")
        cmd += ["--t_weight_alpha", str(t_weight_alpha)]
    if cfg.get("use_pq_ratio"):
        cmd.append("--use_pq_ratio")
    if cfg.get("use_sigmoid"):
        cmd.append("--use_sigmoid")
    if not cfg.get("use_batchnorm", True):
        cmd.append("--no_batchnorm")
    if cfg.get("use_distance_from_center"):
        cmd.append("--use_distance_from_center")
    if not cfg.get("use_is_upper", True):
        cmd.append("--no_is_upper")

    return cmd


def run_experiment(config_name: str) -> int:
    print(f"\n{'='*60}")
    print(f"  実行: {config_name}")
    print(f"{'='*60}")

    cfg = load_config(config_name)

    for path, label in [(MANIFEST_TRAIN, "manifest_train"), (MANIFEST_TEST, "manifest_test")]:
        if not path.exists():
            print(f"  [ERROR] {label} が見つかりません: {path}")
            print(f"         先に split_holdout.py --exp 017 を実行してください")
            return 1

    cmd = build_train_cmd(config_name, cfg)
    output_dir = OUTPUT_DIR / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    mid_ratio = cfg.get("mid_t_keep_ratio", 1.0)
    tw = cfg.get("use_t_weighted_sampler", False)
    tw_alpha = cfg.get("t_weight_alpha", 3.0)
    print(f"  mid_t_keep_ratio:       {mid_ratio}")
    print(f"  use_t_weighted_sampler: {tw} (alpha={tw_alpha})")
    print(f"  k:                      {cfg['k']}")
    print(f"  manifest_train:         {MANIFEST_TRAIN.name}")
    print(f"  manifest_test:          {MANIFEST_TEST.name}")
    print(f"  コマンド: {' '.join(cmd)}")
    return run_with_log(cmd, LOG_PATH, cwd=ROOT)


def main() -> None:
    all_configs = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))

    parser = argparse.ArgumentParser(description="EXP017 データ再バランス sweep 訓練")
    parser.add_argument(
        "--configs", nargs="+", default=all_configs,
        help=f"実行する設定名（デフォルト: 全条件 {all_configs}）",
    )
    args = parser.parse_args()

    print(f"\nEXP017 sweep 開始: {len(args.configs)} 条件")
    print(f"  output_dir: {OUTPUT_DIR}")
    print(f"  log:        {LOG_PATH}")

    failed = []
    for config_name in args.configs:
        rc = run_experiment(config_name)
        if rc != 0:
            print(f"\n[WARN] {config_name} が終了コード {rc} で失敗")
            failed.append(config_name)

    print(f"\n{'='*60}")
    print(f"  EXP017 sweep 完了: {len(args.configs) - len(failed)}/{len(args.configs)} 成功")
    if failed:
        print(f"  失敗: {failed}")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
