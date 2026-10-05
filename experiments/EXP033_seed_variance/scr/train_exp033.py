"""EXP033: 現行dual標準の seed 間ばらつき測定 — 訓練スクリプト。

EXP029〜EXP032 はすべて 1 seed 学習で、有意差・帯別所見が真の効果か run-to-run
分散かを切り分けられないことが繰り返し課題になっていた。本実験は現行dual標準
（EXP031 SD0_baseline と同一構成）を seed だけ変えて 3 回学習し、seed 間の
ばらつき（＝今後の比較で「差」とみなすべき最小幅）を定量化する。

  dual_s42  seed=42（EXP031 SD0_baseline と同一 seed — 再現性チェックを兼ねる）
  dual_s43  seed=43
  dual_s44  seed=44

テストセットは EXP029 で採用が決まった db_replica 固定 manifest（EXP031 と同一）。

使い方:
    python experiments/EXP033_seed_variance/scr/train_exp033.py             # 全 3 seed
    python experiments/EXP033_seed_variance/scr/train_exp033.py --configs dual_s43
    python experiments/EXP033_seed_variance/scr/train_exp033.py --epochs 2  # smoke
    python experiments/EXP033_seed_variance/scr/infer_exp033.py             # 推論
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

# yaml ファイル名は config 名と同一。seed 以外は全 config 同一構成。
CONFIGS = ("dual_s42", "dual_s43", "dual_s44")


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
    cfg = load_config(exp_name)
    config = build_train_config(cfg, epochs_override, num_workers=num_workers)
    model_dir = OUTPUT_DIR / exp_name
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("===== %s  seed=%d =====", exp_name, config.seed)

    result = run_dual_kfold(MANIFEST_DIR / "manifest_train.csv",
                            MANIFEST_DIR / "manifest_test.csv", model_dir, config)
    rec = build_dual_experiment_record(exp_name, config, result)
    (model_dir / "experiment.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"seed": config.seed,
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
    args = ap.parse_args()

    summary: dict = {"configs": {}}
    for exp_name in args.configs:
        summary["configs"][exp_name] = run_one(
            exp_name, args.epochs, num_workers=args.num_workers)

    print(f"\n{'='*72}\n  EXP033 結果サマリ\n{'='*72}")
    for exp_name, s in summary["configs"].items():
        print(f"  {exp_name:<10}  seed={s['seed']:<4}  "
              f"cv_mae={s['cv_mae']:.6f}  holdout_raw_t={s['holdout_mae_raw_t']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "EXP033_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  保存: {OUTPUT_DIR / 'EXP033_summary.json'}")
    print("  推論: python experiments/EXP033_seed_variance/scr/infer_exp033.py")
    print("  passrate: python scripts/compute_passrate.py --exp EXP033")
    print("  seed 間比較: python scripts/paired_ttest.py --exp EXP033 --baseline dual_s42")
    print("  ばらつき集計: python experiments/EXP033_seed_variance/scr/seed_variance_exp033.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
