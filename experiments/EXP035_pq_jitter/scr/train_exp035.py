"""EXP035: PQ ランドマーク回転摂動 augmentation — 訓練スクリプト。

本番の P, Q は FaceMesh 推定値で PQ の向き（角度）の誤差を含むが、学習では再現していない。
R = tP + (1-t)Q を固定したまま、R を中心に P, Q を θ ~ U(±θmax) 回転した入力で学習し
（t は不変）、FaceMesh 誤差への頑健性と hold-out MAE の改善を検証する。

  J0_base    摂動なし（現行dual標準 = EXP033 dual_s42 / EXP034 base と同一構成）
  J1_weak    θmax = ±5°
  J2_strong  θmax = ±10°
  J3_xstrong θmax = ±15°（J2 までの単調改善を受けて追加）
  J4_rot20   θmax = ±20°（J3 でも悪化しないため、悪化に転じる点の探索として追加）
  J5_rot30   θmax = ±30°（同上）

テストセットは EXP034 と同一の db_replica 固定 manifest。

使い方:
    python experiments/EXP035_pq_jitter/scr/train_exp035.py                      # 全 6 config
    python experiments/EXP035_pq_jitter/scr/train_exp035.py --configs J1_weak
    python experiments/EXP035_pq_jitter/scr/train_exp035.py --epochs 2           # smoke
    python experiments/EXP035_pq_jitter/scr/infer_exp035.py                      # 推論
    python experiments/EXP035_pq_jitter/scr/robustness_exp035.py                 # 頑健性評価
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

# yaml ファイル名は config 名と同一。PQ 回転摂動の強度以外は全 config 同一構成。
CONFIGS = ("J0_base", "J1_weak", "J2_strong", "J3_xstrong", "J4_rot20", "J5_rot30")


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
        pq_jitter_rot_deg=float(cfg.get("pq_jitter_rot_deg", 0.0)),
        pq_jitter_prob=float(cfg.get("pq_jitter_prob", 1.0)),
    )


def run_one(exp_name: str, epochs_override: int | None, num_workers: int = 0) -> dict:
    cfg = load_config(exp_name)
    config = build_train_config(cfg, epochs_override, num_workers=num_workers)
    model_dir = OUTPUT_DIR / exp_name
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("===== %s  seed=%d  pq_jitter_rot_deg=%.1f =====",
                exp_name, config.seed, config.pq_jitter_rot_deg)

    result = run_dual_kfold(MANIFEST_DIR / "manifest_train.csv",
                            MANIFEST_DIR / "manifest_test.csv", model_dir, config)
    rec = build_dual_experiment_record(exp_name, config, result)
    (model_dir / "experiment.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"seed": config.seed, "pq_jitter_rot_deg": config.pq_jitter_rot_deg,
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

    print(f"\n{'='*72}\n  EXP035 結果サマリ\n{'='*72}")
    for exp_name, s in summary["configs"].items():
        print(f"  {exp_name:<10}  rot=±{s['pq_jitter_rot_deg']:<5}  "
              f"cv_mae={s['cv_mae']:.6f}  holdout_raw_t={s['holdout_mae_raw_t']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "EXP035_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  保存: {OUTPUT_DIR / 'EXP035_summary.json'}")
    print("  推論: python experiments/EXP035_pq_jitter/scr/infer_exp035.py")
    print("  passrate: python scripts/compute_passrate.py --exp EXP035")
    print("  比較: python scripts/paired_ttest.py --exp EXP035 --baseline J0_base")
    print("  頑健性: python experiments/EXP035_pq_jitter/scr/robustness_exp035.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
