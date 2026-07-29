"""EXP030: データオーグメンテーション発火頻度 sweep — 訓練スクリプト。

EXP021 Phase2 で aug_flags（どのオーグメンテーションを有効にするか）を sweep し
FLIP off（aug_flags=63）が勝者と判明したが、有効化した各オーグメンテーションが
「どのくらいの頻度で発火するか」は一度も sweep されていない
（brightness/blur/color_temp/flip は全て固定 0.5 のまま）。

本実験は aug_flags=63 を固定し、BRIGHTNESS/BLUR/COLOR_TEMP/FLIP の基準発火確率
0.5 に掛ける倍率 aug_prob のみを 0.5/1.0/1.5（実効確率 0.25/0.5/0.75）で sweep する。
NOISE（常時発火）・GRAYSCALE（基準0.2）・ENDPOINT_BLEND（幾何系）・perspective_aug
（EXP027の軸）は対象外で全 config 共通に固定する。

テストセットは EXP029 で採用が決まった db_replica 固定 manifest
（manifests/manifest_{train,test}.csv、EXP029 の
manifest_{train,test}_db_replica.csv をコピーしたもの）を使用する。

使い方:
    python experiments/EXP030_aug_freq_sweep/scr/train_exp030.py             # 全 3 構成
    python experiments/EXP030_aug_freq_sweep/scr/train_exp030.py --configs AF1_p0p25 AF2_p0p75
    python experiments/EXP030_aug_freq_sweep/scr/train_exp030.py --epochs 2  # smoke
    python experiments/EXP030_aug_freq_sweep/scr/infer_exp030.py             # 推論
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
CONFIG_DIR = EXP_DIR / "config"
MANIFEST_DIR = EXP_DIR / "manifests"
OUTPUT_DIR = EXP_DIR / "outputs"
sys.path.insert(0, str(ROOT))

from eyeliner.training.dual_kfold import (  # noqa: E402
    DualTrainConfig,
    build_dual_experiment_record,
    run_dual_kfold,
)

logger = logging.getLogger(__name__)

# 3 構成 = aug_prob sweep。yaml ファイル名は arm 名と同一。
CONFIGS = {
    "AF0_baseline": "AF0_baseline",  # anchor = aug_prob 1.0（現行標準、実効確率0.5）
    "AF1_p0p25": "AF1_p0p25",        # aug_prob 0.5（実効確率0.25）
    "AF2_p0p75": "AF2_p0p75",        # aug_prob 1.5（実効確率0.75）
}


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_config(
    cfg: dict, epochs_override: int | None, num_workers: int = 0
) -> DualTrainConfig:
    return DualTrainConfig(
        fusion_mode=cfg["fusion_mode"],
        pq_rescale_mode_b=cfg.get("pq_rescale_mode_b", "eye_scale"),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        epochs=epochs_override if epochs_override else cfg["epochs"],
        batch_size=cfg["batch"], k=cfg["k"], patience=cfg["patience"], seed=cfg["seed"],
        num_conv_layers=cfg.get("num_conv_layers", 4),
        use_sigmoid=cfg.get("use_sigmoid", True),
        use_batchnorm=cfg.get("use_batchnorm", True),
        dropout_rate=cfg.get("dropout_rate", 0.3),
        aug_flags=cfg.get("aug_flags", 63),
        aug_prob=float(cfg.get("aug_prob", 1.0)),
        endpoint_blend_range=float(cfg.get("endpoint_blend_range", 0.25)),
        loss_fn=cfg.get("loss_fn", "t_l1"), num_workers=num_workers,
        base_channel=cfg["base_channel"], pool_width=cfg["pool_width"],
        use_aux_face_shape=bool(cfg.get("use_aux_face_shape", False)),
        use_is_upper=cfg.get("use_is_upper", True),
        db_path=str(ROOT / "data" / "samples.db"),
    )


def run_one(exp_name: str, epochs_override: int | None, num_workers: int = 0) -> dict:
    cfg = load_config(CONFIGS[exp_name])
    config = build_train_config(cfg, epochs_override, num_workers=num_workers)
    model_dir = OUTPUT_DIR / exp_name
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("===== %s  aug_prob=%.2f =====", exp_name, config.aug_prob)

    result = run_dual_kfold(MANIFEST_DIR / "manifest_train.csv",
                            MANIFEST_DIR / "manifest_test.csv", model_dir, config)
    rec = build_dual_experiment_record(exp_name, config, result)
    (model_dir / "experiment.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"aug_prob": config.aug_prob,
            "cv_mae": rec["results"]["cv_mae"],
            "holdout_mae_raw_t": rec["results"]["holdout_mae_raw_t"]}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--epochs", type=int, default=None, help="smoke 用 epoch 上書き")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers（Windows で num_workers>0 は spawn デッドロック "
                         "リスクあり。既定 0=安全）")
    ap.add_argument("--skip-train", action="store_true", help="学習済み前提でスキップ")
    args = ap.parse_args()

    summary: dict = {"configs": {}}
    if not args.skip_train:
        for exp_name in args.configs:
            summary["configs"][exp_name] = run_one(
                exp_name, args.epochs, num_workers=args.num_workers)

    print(f"\n{'='*64}\n  EXP030 結果サマリ\n{'='*64}")
    for exp_name, s in summary["configs"].items():
        print(f"  {exp_name:<15}  aug_prob={s['aug_prob']:.2f}  "
              f"cv_mae={s['cv_mae']:.6f}  holdout_raw_t={s['holdout_mae_raw_t']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "EXP030_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  保存: {OUTPUT_DIR / 'EXP030_summary.json'}")
    print(f"  passrate: python scripts/compute_passrate.py --exp EXP030")
    print(f"  arm 間比較: python scripts/paired_ttest.py --exp EXP030 --baseline AF0_baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
