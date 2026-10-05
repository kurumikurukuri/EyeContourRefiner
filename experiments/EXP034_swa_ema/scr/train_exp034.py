"""EXP034: 重み平均（EMA / SWA）— 訓練スクリプト。

現行dual標準は OneCycleLR（200 epoch 前提）の途中、lr が最大値付近のまま早期終了しており
（EXP033 のログで停止時 lr≈9.6e-4）、最終重みが揺れている。重み平均でこの揺れを打ち消せば、
推論コストを変えずに改善できるかを検証する。

1 回の学習（seed=42）から 4 モデルを作り、同一 run 内で比較する（EXP033 で判明した
同一 seed 再学習の非決定性を比較から排除するため）。

  base           早期終了で選んだ val 最良（= dual標準）
  base_ema0999   EMA decay=0.999
  base_ema09998  EMA decay=0.9998
  base_swa       base 最良から lr 一定 2.5e-4 で 30 epoch → 等重み平均 + BN 再計算

使い方:
    python experiments/EXP034_swa_ema/scr/train_exp034.py
    python experiments/EXP034_swa_ema/scr/train_exp034.py --epochs 2 --swa-epochs 1  # smoke
    python experiments/EXP034_swa_ema/scr/infer_exp034.py                            # 推論
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
from eyeliner.training.weight_averaging import (  # noqa: E402
    ema_variant_name,
    variant_dir,
)

logger = logging.getLogger(__name__)

CONFIG_NAME = "base"


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_config(
    cfg: dict,
    epochs_override: int | None,
    swa_epochs_override: int | None,
    num_workers: int = 0,
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
        ema_decays=[float(d) for d in cfg.get("ema_decays", [])],
        swa_epochs=swa_epochs_override if swa_epochs_override else int(cfg.get("swa_epochs", 0)),
        swa_lr=float(cfg.get("swa_lr", 2.5e-4)),
    )


def variant_names(config: DualTrainConfig) -> list[str]:
    names = [ema_variant_name(d) for d in config.ema_decays]
    return names + (["swa"] if config.swa_epochs > 0 else [])


def summarize_variant(out_dir: Path) -> dict:
    """派生モデルの fold ごとの val 最良を読み、cv_mae（枝A正規化 val MAE の平均）を返す。"""
    folds = json.loads((out_dir / "variant_folds.json").read_text(encoding="utf-8"))
    maes = [folds[k]["val_mae"] for k in sorted(folds)]
    return {"cv_mae": sum(maes) / len(maes), "folds": folds}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None, help="smoke 用 epoch 上書き")
    ap.add_argument("--swa-epochs", type=int, default=None, help="smoke 用 SWA epoch 上書き")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers（Windows で num_workers>0 は spawn デッドロック "
                         "リスクあり。既定 0=安全）")
    args = ap.parse_args()

    cfg = load_config(CONFIG_NAME)
    config = build_train_config(cfg, args.epochs, args.swa_epochs, num_workers=args.num_workers)
    model_dir = OUTPUT_DIR / CONFIG_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("===== %s  seed=%d  ema=%s  swa_epochs=%d  swa_lr=%.1e =====",
                CONFIG_NAME, config.seed, config.ema_decays, config.swa_epochs, config.swa_lr)

    result = run_dual_kfold(MANIFEST_DIR / "manifest_train.csv",
                            MANIFEST_DIR / "manifest_test.csv", model_dir, config)
    rec = build_dual_experiment_record(CONFIG_NAME, config, result)
    (model_dir / "experiment.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {CONFIG_NAME: {"cv_mae": rec["results"]["cv_mae"],
                             "holdout_mae_raw_t": rec["results"]["holdout_mae_raw_t"]}}
    for name in variant_names(config):
        summary[f"{CONFIG_NAME}_{name}"] = summarize_variant(variant_dir(model_dir, name))

    print(f"\n{'='*72}\n  EXP034 結果サマリ（cv_mae = 枝A正規化 val MAE）\n{'='*72}")
    for name, s in summary.items():
        print(f"  {name:<16}  cv_mae={s['cv_mae']:.6f}")

    (OUTPUT_DIR / "EXP034_summary.json").write_text(
        json.dumps({"configs": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  保存: {OUTPUT_DIR / 'EXP034_summary.json'}")
    print("  推論: python experiments/EXP034_swa_ema/scr/infer_exp034.py")
    print("  passrate: python scripts/compute_passrate.py --exp EXP034")
    print("  比較: python scripts/paired_ttest.py --exp EXP034 --baseline base")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
