"""EXP011: ハイパーパラメータ大規模 sweep（平均回帰・過学習対策）

k=1 固定分割 / epochs=100 / patience=25 で高速に 63 条件を探索する。
GroupShuffleSplit による 80/20 固定 val 分割を使用。

グループ構成:
    M0  : ベースライン（EXP010 L1 を k=1/100epoch 版に変換）
    A   : t_weighted loss alpha sweep (0.5–20)
    B   : WeightedRandomSampler alpha sweep (0.5–12)
    C   : sampler × loss 格子 (tws_alpha × t_loss_alpha ∈ {1,2,3,5}²)
    D   : dead_zone loss ε×δ sweep
    E   : lr × weight_decay 格子
    F   : num_conv_layers sweep (2/4/5)
    G   : patch_w / patch_perp_scale
    H   : batch size sweep (8/16/64/128)
    J   : aug_flags アブレーション
    L   : backbone 転移学習 sweep (efficientnet_b0 / mobilenet_v3_small / resnet18)

前提:
    experiments/EXP011_hparam_sweep/manifests/ に
    manifest_train.csv / manifest_test.csv が存在すること。
    （EXP010/manifests/ からコピー、または split_holdout.py --exp 011 で生成）

使い方:
    # 全条件
    python scripts/sweep.py --exp 011

    # 特定条件のみ
    python experiments/EXP011_hparam_sweep/scr/train_exp011.py --configs A1_twl05 B3_tws2

    # グループまとめて
    python experiments/EXP011_hparam_sweep/scr/train_exp011.py --group A
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent  # リポジトリルート
EXP_DIR = Path(__file__).parent.parent             # EXP011_hparam_sweep/
CONFIG_DIR = EXP_DIR / "config"
MANIFEST_DIR = EXP_DIR / "manifests"
OUTPUT_DIR = EXP_DIR / "outputs"
LOG_PATH = EXP_DIR / "log.txt"

MANIFEST_TRAIN = MANIFEST_DIR / "manifest_train.csv"
MANIFEST_TEST  = MANIFEST_DIR / "manifest_test.csv"

# 条件名のグループ prefix → グループ文字
_GROUP_PREFIXES = ["M", "A", "B", "C", "D", "E", "F", "G", "H", "J", "L"]

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
        "--norm_type",       cfg.get("norm_type", "imagenet"),
        "--aug_flags",       str(cfg.get("aug_flags", 79)),
        "--patch_t_min",     str(cfg["patch_t_min"]),
        "--patch_t_max",     str(cfg["patch_t_max"]),
        "--patch_perp_scale", str(cfg["patch_perp_scale"]),
        "--patch_h",         str(cfg["patch_h"]),
        "--patch_w",         str(cfg["patch_w"]),
    ]

    # label 正規化範囲
    if cfg.get("label_t_min") is not None:
        cmd += ["--label_t_min", str(cfg["label_t_min"])]
    if cfg.get("label_t_max") is not None:
        cmd += ["--label_t_max", str(cfg["label_t_max"])]

    # dead_zone 損失パラメータ
    if cfg["loss_fn"] == "dead_zone":
        cmd += ["--loss_epsilon", str(cfg.get("loss_epsilon", 0.01))]
        cmd += ["--loss_delta",   str(cfg.get("loss_delta",   0.05))]

    # t_weighted 損失: sampler の有無に関わらず t_loss_alpha を渡す
    if cfg["loss_fn"] == "t_weighted":
        cmd += ["--t_loss_alpha", str(cfg.get("t_loss_alpha", 3.0))]

    # WeightedRandomSampler（t_loss_alpha は上で処理済み、ここでは sampler フラグのみ）
    if cfg.get("use_t_weighted_sampler"):
        cmd.append("--use_t_weighted_sampler")
        cmd += ["--t_weight_alpha", str(cfg.get("t_weight_alpha", 3.0))]
        # sampler と t_weighted loss を同時使用する場合 (Group C 等)
        # loss_fn が t のとき t_loss_alpha は無視されるが、念のため渡す
        if cfg["loss_fn"] != "t_weighted":
            cmd += ["--t_loss_alpha", str(cfg.get("t_loss_alpha", 3.0))]

    # フラグ系オプション
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
    """グループ文字 (A/B/C/…) に一致する条件名を返す。"""
    prefix = group.upper()
    return [c for c in configs if c.upper().startswith(prefix)]


def main() -> None:
    all_configs = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))

    parser = argparse.ArgumentParser(
        description="EXP011 ハイパーパラメータ sweep (k=1, 100 epoch)"
    )
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="実行する設定名（デフォルト: 全条件）",
    )
    parser.add_argument(
        "--group", nargs="+", default=None,
        metavar="GROUP",
        help=f"グループ文字で絞り込み（例: --group A B C）。選択肢: {_GROUP_PREFIXES}",
    )
    args = parser.parse_args()

    # 実行対象の決定
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

    # 前提チェック
    for manifest, label in [(MANIFEST_TRAIN, "manifest_train.csv"),
                            (MANIFEST_TEST,  "manifest_test.csv")]:
        if not manifest.exists():
            print(f"\n[ERROR] {label} が見つかりません: {manifest}")
            print("以下のいずれかで manifest を用意してください:")
            print("  1. EXP010 の manifests をコピー:")
            print("       cp experiments/EXP010_data_v2_wide_patch/manifests/*.csv \\")
            print("          experiments/EXP011_hparam_sweep/manifests/")
            print("  2. split_holdout.py で新規生成:")
            print("       python scripts/split_holdout.py --exp 011 --method random --ratio 0.2 --seed 42")
            sys.exit(1)

    print(f"\nEXP011 sweep 開始: {len(target_configs)} 条件")
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
    print(f"  EXP011 sweep 完了: {len(target_configs) - len(failed)}/{len(target_configs)} 成功")
    if failed:
        print(f"  失敗: {failed}")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
