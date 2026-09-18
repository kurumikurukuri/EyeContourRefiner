"""EXP008: norm_eye_scale 特徴量 sweep

画像解像度正規化済み eye_scale（eye_scale / sqrt(img_H * img_W)）を
モデルのスカラー入力として追加したときの効果を確認する。

条件:
    J1_baseline_nes  ← H1_baseline + use_norm_eye_scale=True

比較ベースライン: EXP006 H1_baseline（norm_eye_scale なし）

manifests: EXP006 の manifests を共有（同一 train/test 分割で比較を保証）

使い方:
    # 全条件を実行
    python scripts/sweep.py --exp 008

    # 特定条件のみ実行
    python experiments/EXP008_norm_eye_scale_sweep/scr/train_exp008.py --configs J1_baseline_nes

    # sweep.py 経由（推奨）
    python scripts/sweep.py --exp 008 --configs J1_baseline_nes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent  # eyelinerCNN/
EXP_DIR = Path(__file__).parent.parent             # EXP008_norm_eye_scale_sweep/
CONFIG_DIR = EXP_DIR / "config"
# EXP006 の manifests を共有（同一分割で公平な比較）
EXP006_DIR = ROOT / "experiments" / "EXP006_data_increase_sweep"
MANIFEST_TRAIN = EXP006_DIR / "manifests" / "manifest_train.csv"
MANIFEST_TEST  = EXP006_DIR / "manifests" / "manifest_test.csv"
OUTPUT_DIR     = EXP_DIR / "outputs"
LOG_PATH       = EXP_DIR / "log.txt"

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
        "--num_conv_layers", str(cfg.get("num_conv_layers", 3)),
        "--freeze_epochs",   str(cfg.get("freeze_epochs", 0)),
        "--aug_flags",       str(cfg.get("aug_flags", 79)),
        "--patch_t_min",     str(cfg["patch_t_min"]),
        "--patch_t_max",     str(cfg["patch_t_max"]),
        "--patch_perp_scale", str(cfg["patch_perp_scale"]),
        "--patch_h",         str(cfg["patch_h"]),
        "--patch_w",         str(cfg["patch_w"]),
    ]

    # dead_zone 損失パラメータ
    if cfg["loss_fn"] == "dead_zone":
        cmd += ["--loss_epsilon", str(cfg.get("loss_epsilon", 0.01))]
        cmd += ["--loss_delta",   str(cfg.get("loss_delta",   0.05))]

    # label_t が明示指定されている場合のみ渡す
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
        cmd += ["--t_loss_alpha",   str(cfg.get("t_loss_alpha",   3.0))]

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

    parser = argparse.ArgumentParser(description="EXP008 norm_eye_scale sweep")
    parser.add_argument(
        "--configs", nargs="+", default=all_configs,
        help=f"実行する設定名（デフォルト: 全条件 {all_configs}）",
    )
    args = parser.parse_args()

    print(f"EXP008 sweep開始: {len(args.configs)} 条件")
    print(f"  manifest_train: {MANIFEST_TRAIN}")
    print(f"  manifest_test:  {MANIFEST_TEST}")
    print(f"  output_dir:     {OUTPUT_DIR}")
    print(f"  log:            {LOG_PATH}")

    if not MANIFEST_TRAIN.exists():
        print(f"\n[ERROR] manifest_train.csv が見つかりません: {MANIFEST_TRAIN}")
        print("EXP006 の manifests を使用します。先に EXP006 の分割を実行してください:")
        print("  python scripts/split_holdout.py --method random --ratio 0.2 --seed 42 --exp 006")
        sys.exit(1)

    failed = []
    for config_name in args.configs:
        rc = run_experiment(config_name)
        if rc != 0:
            print(f"\n[WARN] {config_name} が終了コード {rc} で失敗")
            failed.append(config_name)

    print(f"\n{'='*60}")
    print(f"  EXP008 sweep完了: {len(args.configs) - len(failed)}/{len(args.configs)} 成功")
    if failed:
        print(f"  失敗: {failed}")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
