"""EXP004: エッジケース学習改善 sweep

t正規化なし（use_t_norm=False）でモデルが生のt値を直接回帰させる。
sigmoid廃止で出力を非制約にし、t=0/1付近のエッジケースの学習を改善する。

sweep 軸:
    1. use_distance_from_center: point_index由来の端点位置特徴量
    2. use_t_weighted_sampler: エッジサンプルの過サンプリング（alpha=1.0）

条件一覧:
    E1_no_norm          : t正規化なし baseline
    E2_no_norm_dfc      : + distance_from_center
    E3_no_norm_sampler  : + t_weighted_sampler (alpha=1.0)
    E4_no_norm_dfc_sampler : dfc + sampler

ベースライン: EXP002 sig_tloss (CV=0.064264 / Holdout=0.064225)
  ※ MAEスケールが正規化t → 生tに変わるため直接比較にはスケール換算が必要

使い方:
    # 全条件を実行
    python experiments/EXP004_edge_case_sweep/scr/train_exp004.py

    # 特定条件のみ実行
    python experiments/EXP004_edge_case_sweep/scr/train_exp004.py --configs E1_no_norm E2_no_norm_dfc

    # sweep.py 経由（推奨）
    python scripts/sweep.py --exp 004 --configs E1_no_norm E2_no_norm_dfc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent  # eyelinerCNN/
EXP_DIR = Path(__file__).parent.parent             # EXP004_edge_case_sweep/
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
        "--patch_t_min",     str(cfg.get("patch_t_min", -0.5)),
        "--patch_t_max",     str(cfg.get("patch_t_max", 1.5)),
        "--patch_perp_scale", str(cfg.get("patch_perp_scale", 1.0)),
        "--patch_h",         str(cfg.get("patch_h", 64)),
        "--patch_w",         str(cfg.get("patch_w", 128)),
    ]

    # label_t が明示指定されている場合のみ渡す
    label_t_min = cfg.get("label_t_min")
    label_t_max = cfg.get("label_t_max")
    if label_t_min is not None:
        cmd += ["--label_t_min", str(label_t_min)]
    if label_t_max is not None:
        cmd += ["--label_t_max", str(label_t_max)]

    # フラグ系（True のときのみ追加）
    if cfg.get("use_pq_ratio"):
        cmd.append("--use_pq_ratio")
    if cfg.get("use_sigmoid"):
        cmd.append("--use_sigmoid")
    if not cfg.get("use_batchnorm", True):
        cmd.append("--no_batchnorm")

    # EXP004 固有: t正規化の無効化
    if not cfg.get("use_t_norm", True):
        cmd.append("--no_t_norm")

    # EXP004 固有: 端点位置特徴量
    if cfg.get("use_distance_from_center"):
        cmd.append("--use_distance_from_center")

    # エッジサンプリング
    if cfg.get("use_t_weighted_sampler"):
        cmd.append("--use_t_weighted_sampler")
        cmd += ["--t_weight_alpha", str(cfg.get("t_weight_alpha", 1.0))]
        cmd += ["--t_loss_alpha",   str(cfg.get("t_loss_alpha",   1.0))]

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

    parser = argparse.ArgumentParser(description="EXP004 エッジケース学習改善 sweep")
    parser.add_argument(
        "--configs", nargs="+", default=all_configs,
        help=f"実行する設定名（デフォルト: 全条件 {all_configs}）",
    )
    args = parser.parse_args()

    print(f"EXP004 sweep開始: {len(args.configs)} 条件")
    print(f"  manifest_train: {MANIFEST_TRAIN}")
    print(f"  manifest_test:  {MANIFEST_TEST}")
    print(f"  output_dir:     {OUTPUT_DIR}")
    print(f"  log:            {LOG_PATH}")

    if not MANIFEST_TRAIN.exists():
        print(f"\n[ERROR] manifest_train.csv が見つかりません: {MANIFEST_TRAIN}")
        print("先に以下を実行してください:")
        print("  python scripts/split_holdout.py --method random --ratio 0.2 --seed 42 --exp 004")
        sys.exit(1)

    failed = []
    for config_name in args.configs:
        rc = run_experiment(config_name)
        if rc != 0:
            print(f"\n[WARN] {config_name} が終了コード {rc} で失敗")
            failed.append(config_name)

    print(f"\n{'='*60}")
    print(f"  EXP004 sweep完了: {len(args.configs) - len(failed)}/{len(args.configs)} 成功")
    if failed:
        print(f"  失敗: {failed}")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
