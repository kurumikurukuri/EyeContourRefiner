"""EXP013: ハイパーパラメータ組み合わせ検証（L7 ベース）

EXP010 L7_wide_patch_deep4（conv4, aug_flags=79）をベースラインとし、
EXP011 で単体効果が確認された lr=1e-3 / conv5 / twl=3 の組み合わせ効果を検証する。

条件:
    Q0_L7_baseline           : L7 再現（基準線）
    Q1_EXP012_baseline       : Q0 の patch_w=128→64（EXP012 baseline 相当）
    Q2_lr3                   : Q1 + lr=1e-3（EXP011 E6 を L7 ベースで再検証）
    Q3_conv5                 : Q1 + conv5（EXP011 F3 を L7 ベースで再検証）
    Q4_lr3_conv5_reg         : lr=1e-3 + conv5 + wd=1e-3 + dropout=0.1（L8 崩壊を正則化で回避）
    Q5_twl3                  : Q1 + t_weighted_loss alpha=3（EXP011 A5/C03 の loss 効果を L7 で再検証）
    Q6_best_combo            : lr=1e-3 + conv5 + twl=3 + wd=1e-3 + dropout=0.1（本命）
    Q7_best_combo_label      : Q6 + patch_t/label_t 範囲拡大（-1.5,2.5 / -1.0,2.0）
    Q8_best_combo_label_wide : Q7 + patch_w=96
    Q9_lr3_label_wide        : Q2 + patch_t/label_t 範囲拡大（-2.0,3.0 / -1.0,2.0）
    Q10_lr3_label_wide_scale : Q9 + perp_scale=2.0

前提:
    experiments/EXP013_hparam_combo/manifests/ に
    manifest_train.csv / manifest_test.csv が存在すること。
    （EXP010 の manifest を流用、または split_holdout.py --exp 013 で生成）

使い方:
    # 全条件を実行
    python scripts/sweep.py --exp 013

    # 特定条件のみ実行
    python experiments/EXP013_hparam_combo/scr/train_exp013.py --configs Q0_L7_baseline Q2_lr3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent  # リポジトリルート
EXP_DIR = Path(__file__).parent.parent             # EXP013_hparam_combo/
CONFIG_DIR = EXP_DIR / "config"
MANIFEST_DIR = EXP_DIR / "manifests"
OUTPUT_DIR = EXP_DIR / "outputs"
LOG_PATH = EXP_DIR / "log.txt"

MANIFEST_TRAIN = MANIFEST_DIR / "manifest_train.csv"
MANIFEST_TEST  = MANIFEST_DIR / "manifest_test.csv"

sys.path.insert(0, str(ROOT))
from scripts.exp_utils import run_with_log  # noqa: E402


def load_config(config_name: str) -> dict:
    path = CONFIG_DIR / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_cmd(config_name: str, cfg: dict) -> list[str]:
    """config dict から train.py 呼び出しコマンドを構築する。"""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--manifest",        str(MANIFEST_TRAIN),
        "--holdout",         str(MANIFEST_TEST),
        "--output_dir",      str(OUTPUT_DIR),
        "--exp_name",        config_name,
        "--backbone",        cfg["backbone"],
        "--loss_fn",         cfg["loss_fn"],
        "--lr",              str(cfg["lr"]),
        "--weight_decay",    str(cfg["weight_decay"]),
        "--epochs",          str(cfg["epochs"]),
        "--batch",           str(cfg["batch"]),
        "--k",               str(cfg["k"]),
        "--patience",        str(cfg["patience"]),
        "--seed",            str(cfg["seed"]),
        "--num_conv_layers", str(cfg.get("num_conv_layers", 4)),
        "--dropout_rate",    str(cfg.get("dropout_rate", 0.3)),
        "--freeze_epochs",   str(cfg.get("freeze_epochs", 0)),
        "--aug_flags",       str(cfg.get("aug_flags", 79)),
        "--patch_t_min",     str(cfg["patch_t_min"]),
        "--patch_t_max",     str(cfg["patch_t_max"]),
        "--patch_perp_scale", str(cfg["patch_perp_scale"]),
        "--patch_h",         str(cfg["patch_h"]),
        "--patch_w",         str(cfg["patch_w"]),
    ]

    label_t_min = cfg.get("label_t_min")
    label_t_max = cfg.get("label_t_max")
    if label_t_min is not None:
        cmd += ["--label_t_min", str(label_t_min)]
    if label_t_max is not None:
        cmd += ["--label_t_max", str(label_t_max)]

    if cfg["loss_fn"] == "t_weighted":
        cmd += ["--t_loss_alpha", str(cfg.get("t_loss_alpha", 3.0))]
    if cfg["loss_fn"] == "dead_zone":
        cmd += ["--loss_epsilon", str(cfg.get("loss_epsilon", 0.01))]
        cmd += ["--loss_delta",   str(cfg.get("loss_delta",   0.05))]

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
    cmd = build_train_cmd(config_name, cfg)

    output_dir = OUTPUT_DIR / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  コマンド: {' '.join(cmd)}")
    return run_with_log(cmd, LOG_PATH, cwd=ROOT)


def main() -> None:
    all_configs = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))

    parser = argparse.ArgumentParser(description="EXP013 ハイパーパラメータ組み合わせ検証")
    parser.add_argument(
        "--configs", nargs="+", default=all_configs,
        help=f"実行する設定名（デフォルト: 全条件 {all_configs}）",
    )
    args = parser.parse_args()

    for manifest, label in [(MANIFEST_TRAIN, "manifest_train.csv"),
                            (MANIFEST_TEST,  "manifest_test.csv")]:
        if not manifest.exists():
            print(f"\n[ERROR] {label} が見つかりません: {manifest}")
            print("EXP010 の manifest を流用してください:")
            print("  cp experiments/EXP010_data_v2_wide_patch/manifests/*.csv \\")
            print("     experiments/EXP013_hparam_combo/manifests/")
            sys.exit(1)

    print(f"\nEXP013 sweep 開始: {len(args.configs)} 条件")
    print(f"  manifest_train: {MANIFEST_TRAIN}")
    print(f"  manifest_test:  {MANIFEST_TEST}")
    print(f"  output_dir:     {OUTPUT_DIR}")
    print(f"  log:            {LOG_PATH}")

    failed = []
    for config_name in args.configs:
        rc = run_experiment(config_name)
        if rc != 0:
            print(f"\n[WARN] {config_name} が終了コード {rc} で失敗")
            failed.append(config_name)

    print(f"\n{'='*60}")
    print(f"  EXP013 sweep 完了: {len(args.configs) - len(failed)}/{len(args.configs)} 成功")
    if failed:
        print(f"  失敗: {failed}")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
