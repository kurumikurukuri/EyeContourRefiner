"""EXP022: base_channel x pool_width 容量/分解度スイープ。

EXP021 Phase3 ベスト構成（dual independent, loss=t_l1, aug=woflip(63), EBR=0.25）を
土台に、EyelinerCNNDual の base_channel（初期チャネル数）と pool_width
（AdaptiveAvgPool2d 出力幅）の各条件を config/bc{base_channel}_pw{pool_width}.yaml
として定義し、CV MAE / hold-out MAE を比較する。

base_channel=32, pool_width=32（config/bc32_pw32.yaml）は EXP021 Phase3 ベスト構成と
同一設定で EXP022 内で学習し直した条件であり、本スイープの比較ベースラインとして使う
（EXP022 の outputs/ 内で完結する）。

使い方:
    # config/ にある全条件
    python experiments/EXP022_channel_pool_sweep/scr/train_exp022.py

    # 部分実行
    python experiments/EXP022_channel_pool_sweep/scr/train_exp022.py \
        --configs bc16_pw1 bc16_pw8

    # スモークテスト（短縮）
    python experiments/EXP022_channel_pool_sweep/scr/train_exp022.py \
        --configs bc16_pw1 --epochs 2 --k 2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent
SCR_DIR = Path(__file__).parent
EXP_DIR = Path(__file__).parent.parent
CONFIG_DIR = EXP_DIR / "config"
MANIFEST_DIR = EXP_DIR / "manifests"
OUTPUT_DIR = EXP_DIR / "outputs"
MANIFEST_TRAIN = MANIFEST_DIR / "manifest_train.csv"
MANIFEST_TEST = MANIFEST_DIR / "manifest_test.csv"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCR_DIR))
from eyeliner.training.dual_kfold import (  # noqa: E402
    DualTrainConfig,
    build_dual_experiment_record,
    run_dual_kfold,
)
from infer_exp022 import infer_single  # noqa: E402
from exp022_compare import discover_variants, exp_name_for, summarize  # noqa: E402


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one(cfg: dict, exp_name: str, epochs: int | None, k: int | None, num_workers: int) -> Path:
    base_channel = cfg["base_channel"]
    pool_width = cfg["pool_width"]
    model_dir = OUTPUT_DIR / exp_name

    print(f"\n{'='*64}\n  {exp_name}  (base_channel={base_channel}, pool_width={pool_width})\n{'='*64}")

    config = DualTrainConfig(
        fusion_mode=cfg["fusion_mode"],
        pq_rescale_mode_b=cfg.get("pq_rescale_mode_b", "eye_scale"),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        epochs=epochs if epochs is not None else cfg["epochs"],
        batch_size=cfg["batch"],
        k=k if k is not None else cfg["k"],
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
        base_channel=base_channel,
        pool_width=pool_width,
    )
    result = run_dual_kfold(MANIFEST_TRAIN, MANIFEST_TEST, model_dir, config)
    exp_record = build_dual_experiment_record(exp_name, config, result)
    (model_dir / "experiment.json").write_text(
        json.dumps(exp_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  CV MAE: {exp_record['results']['cv_mae']:.6f}")
    if result.holdout_mae is not None:
        print(f"  Hold-out MAE: {result.holdout_mae:.6f}（正規化） / "
              f"{result.holdout_mae_raw:.6f}（生t）")
    return model_dir


def main() -> int:
    variant_names = [exp_name_for(bc, pw) for bc, pw in discover_variants()]

    parser = argparse.ArgumentParser(description="EXP022 base_channel x pool_width sweep")
    parser.add_argument("--configs", nargs="+", default=variant_names, choices=variant_names,
                        help="実行する config YAML 名（拡張子なし）のリスト")
    parser.add_argument("--epochs", type=int, default=None, help="config 既定を上書き（スモーク用）")
    parser.add_argument("--k", type=int, default=None, help="config 既定を上書き（スモーク用）")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="test_predictions.csv が既にある条件はスキップ（再開用）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if not MANIFEST_TRAIN.exists() or not MANIFEST_TEST.exists():
        print(f"[ERROR] manifest が見つかりません: {MANIFEST_DIR}")
        return 1

    print(f"EXP022 sweep: {len(args.configs)} 条件 = {args.configs}")

    for exp_name in args.configs:
        cfg = load_config(exp_name)
        model_dir = OUTPUT_DIR / exp_name

        if args.skip_existing and (model_dir / "test_predictions.csv").exists():
            print(f"  [skip] 既存: {exp_name}")
            continue

        if not args.skip_train:
            train_one(cfg, exp_name, args.epochs, args.k, args.num_workers)

        if not args.skip_infer:
            try:
                infer_single(model_dir, MANIFEST_TEST, model_dir / "test_predictions.csv", cfg, args.num_workers)
            except Exception as exc:  # noqa: BLE001
                print(f"  [ERROR] 推論失敗: {exp_name}: {exc}")
                return 1

    combos = [(load_config(name)["base_channel"], load_config(name)["pool_width"]) for name in args.configs]
    baseline_dir = OUTPUT_DIR / exp_name_for(32, 32)
    summarize(combos, baseline_dir)

    print("\nEXP022 sweep 完了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
