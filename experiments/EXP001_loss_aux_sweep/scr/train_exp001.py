"""EXP001: 損失関数・補助入力 sweep

実験条件（config/*.yaml）を順次実行する。

使い方:
    # 全条件を実行
    python experiments/EXP001_loss_aux_sweep/scr/train_exp001.py

    # 特定条件のみ実行
    python experiments/EXP001_loss_aux_sweep/scr/train_exp001.py --configs A1_baseline A2_t_loss

    # sweep.py 経由（推奨）
    python scripts/sweep.py --exp 001 --configs A1_baseline A2_t_loss A3_pq_ratio B1_sigmoid B2_pq_sigmoid B3_deep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent  # eyelinerCNN/
EXP_DIR = Path(__file__).parent.parent             # EXP001_loss_aux_sweep/
CONFIG_DIR = EXP_DIR / "config"
MANIFEST_TRAIN = EXP_DIR / "manifests" / "manifest_train.csv"
MANIFEST_TEST  = EXP_DIR / "manifests" / "manifest_test.csv"
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
    output_dir = OUTPUT_DIR / config_name
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--manifest",    str(MANIFEST_TRAIN),
        "--holdout",     str(MANIFEST_TEST),
        "--output_dir",  str(output_dir),
        "--exp_name",    config_name,
        "--backbone",    cfg["backbone"],
        "--loss_fn",     cfg["loss_fn"],
        "--lr",          str(cfg["lr"]),
        "--weight_decay",str(cfg["weight_decay"]),
        "--epochs",      str(cfg["epochs"]),
        "--batch",       str(cfg["batch"]),
        "--k",           str(cfg["k"]),
        "--patience",    str(cfg["patience"]),
        "--seed",        str(cfg["seed"]),
        "--num_conv_layers", str(cfg.get("num_conv_layers", 3)),
        "--freeze_epochs",   str(cfg.get("freeze_epochs", 0)),
    ]

    if cfg.get("use_pq_ratio"):
        cmd.append("--use_pq_ratio")
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

    parser = argparse.ArgumentParser(description="EXP001 sweep実験")
    parser.add_argument(
        "--configs", nargs="+", default=all_configs,
        help=f"実行する設定名（デフォルト: 全条件 {all_configs}）",
    )
    args = parser.parse_args()

    print(f"EXP001 sweep開始: {len(args.configs)} 条件")
    print(f"  manifest_train: {MANIFEST_TRAIN}")
    print(f"  manifest_test:  {MANIFEST_TEST}")
    print(f"  output_dir:     {OUTPUT_DIR}")
    print(f"  log:            {LOG_PATH}")

    if not MANIFEST_TRAIN.exists():
        print(f"\n[ERROR] manifest_train.csv が見つかりません: {MANIFEST_TRAIN}")
        print("先に以下を実行してください:")
        print(f"  python scripts/split_holdout.py --method random --ratio 0.2 --seed 42 --exp 001")
        sys.exit(1)

    failed = []
    for config_name in args.configs:
        rc = run_experiment(config_name)
        if rc != 0:
            print(f"\n[WARN] {config_name} が終了コード {rc} で失敗")
            failed.append(config_name)

    print(f"\n{'='*60}")
    print(f"  EXP001 sweep完了: {len(args.configs) - len(failed)}/{len(args.configs)} 成功")
    if failed:
        print(f"  失敗: {failed}")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
