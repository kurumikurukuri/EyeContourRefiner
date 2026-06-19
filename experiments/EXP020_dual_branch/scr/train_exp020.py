"""EXP020: デュアルブランチ fusion mode sweep 訓練スクリプト。

条件 (k=3):
    X0_none_baseline                : 生PQ座標系 単一CNN（比較基準）
    X0_eye_scale_baseline           : P'Q'(eye_scale) 単一CNN（比較基準）
    X1_independent_none_eye_scale   : 生PQ + eye_scale 独立2ブランチ出力平均
    X1_independent_none_eye_height  : 生PQ + eye_height 独立2ブランチ出力平均
    X2_crossfuse_none_eye_scale     : 生PQ + eye_scale cross-stitch融合
    X2_crossfuse_none_eye_height    : 生PQ + eye_height cross-stitch融合
    X3_attention_none_eye_scale     : 生PQ + eye_scale α重み出力融合
    X3_attention_none_eye_height    : 生PQ + eye_height α重み出力融合

使い方:
    # 全条件実行
    python experiments/EXP020_dual_branch/scr/train_exp020.py

    # 特定条件のみ実行
    python experiments/EXP020_dual_branch/scr/train_exp020.py --configs X0_none_baseline

事前に manifest を生成しておくこと:
    python scripts/split_holdout.py --exp 020
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
SCR_DIR = Path(__file__).parent
EXP_DIR = Path(__file__).parent.parent
CONFIG_DIR = EXP_DIR / "config"
MANIFEST_DIR = EXP_DIR / "manifests"
OUTPUT_DIR = EXP_DIR / "outputs"
LOG_PATH = EXP_DIR / "log.txt"

MANIFEST_TRAIN = MANIFEST_DIR / "manifest_train.csv"
MANIFEST_TEST = MANIFEST_DIR / "manifest_test.csv"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCR_DIR))
from scripts.exp_utils import run_with_log  # noqa: E402
from eyeliner.training.dual_kfold import (  # noqa: E402
    DualTrainConfig,
    build_dual_experiment_record,
    run_dual_kfold,
)

ALL_CONFIGS = [
    "X0_none_baseline",
    "X0_eye_scale_baseline",
    "X0_eye_height_baseline",
    "X1_independent_none_eye_scale",
    "X1_independent_none_eye_height",
    "X2_crossfuse_none_eye_scale",
    "X2_crossfuse_none_eye_height",
    "X3_attention_none_eye_scale",
    "X3_attention_none_eye_height",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_name: str) -> dict:
    path = CONFIG_DIR / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        import yaml
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Baseline (subprocess → scripts/train.py)
# ---------------------------------------------------------------------------

def build_baseline_cmd(config_name: str, cfg: dict, num_workers: int = 0) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--manifest",         str(MANIFEST_TRAIN),
        "--holdout",          str(MANIFEST_TEST),
        "--output_dir",       str(OUTPUT_DIR),
        "--exp_name",         config_name,
        "--backbone",         "custom",
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
        "--aug_flags",        str(cfg.get("aug_flags", 127)),
        "--patch_t_min",      str(cfg["patch_t_min"]),
        "--patch_t_max",      str(cfg["patch_t_max"]),
        "--patch_perp_scale", str(cfg["patch_perp_scale"]),
        "--patch_h",          str(cfg["patch_h"]),
        "--patch_w",          str(cfg["patch_w"]),
        "--label_t_min",      str(cfg["label_t_min"]),
        "--label_t_max",      str(cfg["label_t_max"]),
        "--use_sigmoid",
    ]
    if not cfg.get("use_batchnorm", True):
        cmd.append("--no_batchnorm")
    if not cfg.get("use_is_upper", True):
        cmd.append("--no_is_upper")
    pq_mode = cfg.get("pq_rescale_mode", "none")
    if pq_mode != "none":
        cmd += ["--pq_rescale_mode", pq_mode]
        # factor を渡さないと scripts/train.py のデフォルト 1.0 になり EXP019 と不一致になる。
        cmd += ["--pq_rescale_factor", str(cfg.get("pq_rescale_factor", 1.0))]
    if num_workers > 0:
        cmd += ["--num_workers", str(num_workers)]
    return cmd


def run_baseline(config_name: str, cfg: dict, num_workers: int = 0) -> int:
    print(f"\n{'='*60}")
    print(f"  Baseline: {config_name}")
    print(f"{'='*60}")

    for path, label in [(MANIFEST_TRAIN, "manifest_train"), (MANIFEST_TEST, "manifest_test")]:
        if not path.exists():
            print(f"  [ERROR] {label} が見つかりません: {path}")
            print(f"         先に split_holdout.py --exp 020 を実行してください")
            return 1

    cmd = build_baseline_cmd(config_name, cfg, num_workers)
    print(f"  pq_rescale_mode: {cfg.get('pq_rescale_mode', 'none')}")
    print(f"  k:             {cfg['k']}")
    print(f"  コマンド: {' '.join(cmd)}")
    return run_with_log(cmd, LOG_PATH, cwd=ROOT)


# ---------------------------------------------------------------------------
# Dual training（eyeliner.training.dual_kfold への薄いラッパー）
# ---------------------------------------------------------------------------

def train_dual(
    config_name: str, cfg: dict, num_workers: int = 0,
    output_dir: Path | None = None, manifest_train: Path | None = None,
    manifest_test: Path | None = None,
) -> int:
    print(f"\n{'='*60}")
    print(f"  Dual: {config_name}")
    print(f"{'='*60}")

    _manifest_train = manifest_train if manifest_train is not None else MANIFEST_TRAIN
    _manifest_test = manifest_test if manifest_test is not None else MANIFEST_TEST
    _output_dir = output_dir if output_dir is not None else OUTPUT_DIR

    for path, label in [(_manifest_train, "manifest_train"), (_manifest_test, "manifest_test")]:
        if not path.exists():
            print(f"  [ERROR] {label} が見つかりません: {path}")
            return 1

    config = DualTrainConfig(
        fusion_mode=cfg["fusion_mode"],
        pq_rescale_mode_b=cfg.get("pq_rescale_mode_b", "eye_scale"),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch"],
        k=cfg["k"],
        patience=cfg["patience"],
        seed=cfg["seed"],
        num_conv_layers=cfg.get("num_conv_layers", 4),
        use_sigmoid=cfg.get("use_sigmoid", True),
        use_batchnorm=cfg.get("use_batchnorm", True),
        dropout_rate=cfg.get("dropout_rate", 0.3),
        aug_flags=cfg.get("aug_flags", 127),
        endpoint_blend_range=float(cfg.get("endpoint_blend_range", 0.25)),
        loss_fn=cfg.get("loss_fn", "t"),
        num_workers=num_workers,
    )
    print(f"  fusion_mode: {config.fusion_mode}")
    print(f"  rescale_b:   {config.pq_rescale_mode_b}")

    exp_dir = _output_dir / config_name
    result = run_dual_kfold(_manifest_train, _manifest_test, exp_dir, config)

    print(f"  訓練サンプル: {result.n_train}")
    print(f"  hold-out:     {result.n_holdout}")
    for r in result.fold_results:
        print(f"    Fold {r.fold_idx}: {r.best_val_mae:.6f}")
    if result.holdout_mae is not None:
        print(f"  Hold-out MAE: {result.holdout_mae:.6f}（正規化） / "
              f"{result.holdout_mae_raw:.6f}（生t）")

    exp_record = build_dual_experiment_record(config_name, config, result)
    exp_json_path = exp_dir / "experiment.json"
    exp_json_path.write_text(
        json.dumps(exp_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  experiment.json 保存: {exp_json_path}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="EXP020 dual branch sweep 訓練")
    parser.add_argument(
        "--configs", nargs="+", default=ALL_CONFIGS,
        help=f"実行する設定名（デフォルト: 全条件）",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader の num_workers",
    )
    args = parser.parse_args()

    print(f"\nEXP020 dual_branch sweep 開始: {len(args.configs)} 条件")
    print(f"  output_dir: {OUTPUT_DIR}")
    print(f"  log:        {LOG_PATH}")
    print(f"  num_workers:{args.num_workers}")

    failed: list[str] = []
    for config_name in args.configs:
        if config_name not in ALL_CONFIGS:
            print(f"\n[ERROR] 未知の設定名: {config_name}")
            print(f"  有効な設定: {ALL_CONFIGS}")
            failed.append(config_name)
            continue

        cfg = load_config(config_name)

        if cfg["backbone"] == "custom":
            rc = run_baseline(config_name, cfg, num_workers=args.num_workers)
        else:
            rc = train_dual(config_name, cfg, num_workers=args.num_workers)

        if rc != 0:
            print(f"\n[WARN] {config_name} が終了コード {rc} で失敗")
            failed.append(config_name)

    print(f"\n{'='*60}")
    print(f"  EXP020 完了: {len(args.configs) - len(failed)}/{len(args.configs)} 成功")
    if failed:
        print(f"  失敗: {failed}")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
