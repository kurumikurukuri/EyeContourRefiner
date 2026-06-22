"""EXP024: 修正後DBでの bc32_pw8 再学習（学習のみ）。

学習だけを行い、fold モデルと experiment.json（訓練結果）を出力する。
推論・比較は別スクリプトに分離:
  - 推論: infer_exp024.py  → outputs/bc32_pw8/{train,test}_predictions.csv
  - 比較: compare_exp024.py → EXP022 との paired test + t-band breakdown（experiment.json に追記）

使い方:
    python experiments/EXP024_retrain_bc32_pw8/scr/train_exp024.py
    python experiments/EXP024_retrain_bc32_pw8/scr/train_exp024.py --num-workers 4
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


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="EXP024 bc32_pw8 再学習（学習のみ）")
    parser.add_argument("--config", default="bc32_pw8", help="config YAML 名（拡張子なし）")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    cfg_name = args.config
    cfg = load_config(cfg_name)
    model_dir = OUTPUT_DIR / cfg_name
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  EXP024: {cfg_name} retrain (修正後DB)")
    print(f"{'='*64}")

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
        num_workers=args.num_workers,
        base_channel=cfg["base_channel"],
        pool_width=cfg["pool_width"],
    )

    manifest_train = MANIFEST_DIR / "manifest_train.csv"
    manifest_test = MANIFEST_DIR / "manifest_test.csv"

    if not manifest_train.exists() or not manifest_test.exists():
        print(f"[ERROR] manifests not found in {MANIFEST_DIR}")
        return 1

    # ---------- training ----------
    result = run_dual_kfold(manifest_train, manifest_test, model_dir, config)
    exp_record = build_dual_experiment_record(cfg_name, config, result)
    (model_dir / "experiment.json").write_text(
        json.dumps(exp_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  CV MAE: {exp_record['results']['cv_mae']:.6f}")
    if result.holdout_mae is not None:
        print(f"  Hold-out MAE: {result.holdout_mae:.6f}（正規化） / "
              f"{result.holdout_mae_raw:.6f}（生t）")

    print(f"\n学習完了: {model_dir}")
    print("次のステップ:")
    print(f"  推論: python experiments/EXP024_retrain_bc32_pw8/scr/infer_exp024.py "
          f"--model-dir {model_dir.relative_to(ROOT)}")
    print(f"  比較: python experiments/EXP024_retrain_bc32_pw8/scr/compare_exp024.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
