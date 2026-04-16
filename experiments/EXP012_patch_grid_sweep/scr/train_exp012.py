"""EXP012: パッチサイズ格子 sweep（108条件）

patch_t_min × patch_t_max × patch_perp_scale × patch_h × patch_w の
全組み合わせ (2×2×3×3×3 = 108条件) を k=1 / epochs=100 / patience=25 で探索する。

グループ構成:
    A: patch_t_min=-1.0,  patch_t_max=2.0   （ベースライン）
    B: patch_t_min=-1.0,  patch_t_max=2.25
    C: patch_t_min=-1.25, patch_t_max=2.0
    D: patch_t_min=-1.25, patch_t_max=2.25

命名規則:
    {group}_p{perp_scale}_h{patch_h}_w{patch_w}
    例: A_p10_h064_w128  （= EXP010 L1 / EXP011 M0 相当）

前提:
    experiments/EXP012_patch_grid_sweep/manifests/ に
    manifest_train.csv / manifest_test.csv が存在すること。
    （EXP011/manifests/ からコピー、または split_holdout.py --exp 012 で生成）

使い方:
    # 全条件
    python scripts/sweep.py --exp 012

    # 特定条件のみ
    python experiments/EXP012_patch_grid_sweep/scr/train_exp012.py --configs A_p10_h064_w128

    # グループまとめて (並行実行例: GPU1=A+B, GPU2=C+D)
    python experiments/EXP012_patch_grid_sweep/scr/train_exp012.py --group A B
    python experiments/EXP012_patch_grid_sweep/scr/train_exp012.py --group C D
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent  # リポジトリルート
EXP_DIR = Path(__file__).parent.parent             # EXP012_patch_grid_sweep/
CONFIG_DIR = EXP_DIR / "config"
MANIFEST_DIR = EXP_DIR / "manifests"
OUTPUT_DIR = EXP_DIR / "outputs"
LOG_PATH = EXP_DIR / "log.txt"

MANIFEST_TRAIN = MANIFEST_DIR / "manifest_train.csv"
MANIFEST_TEST  = MANIFEST_DIR / "manifest_test.csv"

_GROUP_PREFIXES = ["A", "B", "C", "D"]

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
    output_dir = OUTPUT_DIR / config_name
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--manifest",         str(MANIFEST_TRAIN),
        "--holdout",          str(MANIFEST_TEST),
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
        "--num_conv_layers",  str(cfg.get("num_conv_layers", 3)),
        "--freeze_epochs",    str(cfg.get("freeze_epochs", 0)),
        "--norm_type",        cfg.get("norm_type", "imagenet"),
        "--aug_flags",        str(cfg.get("aug_flags", 79)),
        "--patch_t_min",      str(cfg["patch_t_min"]),
        "--patch_t_max",      str(cfg["patch_t_max"]),
        "--patch_perp_scale", str(cfg["patch_perp_scale"]),
        "--patch_h",          str(cfg["patch_h"]),
        "--patch_w",          str(cfg["patch_w"]),
    ]

    if cfg.get("label_t_min") is not None:
        cmd += ["--label_t_min", str(cfg["label_t_min"])]
    if cfg.get("label_t_max") is not None:
        cmd += ["--label_t_max", str(cfg["label_t_max"])]

    if cfg.get("use_pq_ratio"):
        cmd.append("--use_pq_ratio")
    if cfg.get("use_norm_eye_scale"):
        cmd.append("--use_norm_eye_scale")
    if cfg.get("use_sigmoid"):
        cmd.append("--use_sigmoid")
    if not cfg.get("use_batchnorm", True):
        cmd.append("--no_batchnorm")

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


def filter_by_group(configs: list[str], group: str) -> list[str]:
    """グループ文字 (A/B/C/D) に一致する条件名を返す。"""
    prefix = group.upper()
    return [c for c in configs if c.upper().startswith(prefix + "_")]


def main() -> None:
    all_configs = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))

    parser = argparse.ArgumentParser(
        description="EXP012 パッチサイズ格子 sweep (k=1, 100 epoch)"
    )
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="実行する設定名（デフォルト: 全条件）",
    )
    parser.add_argument(
        "--group", nargs="+", default=None,
        metavar="GROUP",
        help=f"グループ文字で絞り込み（例: --group A B）。選択肢: {_GROUP_PREFIXES}",
    )
    args = parser.parse_args()

    if args.configs is not None:
        target_configs = args.configs
    elif args.group is not None:
        target_configs = []
        for g in args.group:
            target_configs += filter_by_group(all_configs, g)
        target_configs = sorted(set(target_configs))
    else:
        target_configs = all_configs

    if not target_configs:
        print("[ERROR] 実行対象の条件が見つかりません。")
        sys.exit(1)

    for manifest, label in [(MANIFEST_TRAIN, "manifest_train.csv"),
                            (MANIFEST_TEST,  "manifest_test.csv")]:
        if not manifest.exists():
            print(f"\n[ERROR] {label} が見つかりません: {manifest}")
            print("以下のいずれかで manifest を用意してください:")
            print("  1. EXP011 の manifests をコピー:")
            print("       cp experiments/EXP011_hparam_sweep/manifests/*.csv \\")
            print("          experiments/EXP012_patch_grid_sweep/manifests/")
            print("  2. split_holdout.py で新規生成:")
            print("       python scripts/split_holdout.py --exp 012 --method random --ratio 0.2 --seed 42")
            sys.exit(1)

    print(f"\nEXP012 sweep 開始: {len(target_configs)} 条件")
    print(f"  manifest_train : {MANIFEST_TRAIN}")
    print(f"  manifest_test  : {MANIFEST_TEST}")
    print(f"  output_dir     : {OUTPUT_DIR}")
    print(f"  log            : {LOG_PATH}")
    print(f"  条件           : {target_configs}")

    failed = []
    for config_name in target_configs:
        rc = run_experiment(config_name)
        if rc != 0:
            print(f"\n[WARN] {config_name} が終了コード {rc} で失敗")
            failed.append(config_name)

    print(f"\n{'='*60}")
    print(f"  EXP012 sweep 完了: {len(target_configs) - len(failed)}/{len(target_configs)} 成功")
    if failed:
        print(f"  失敗: {failed}")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
