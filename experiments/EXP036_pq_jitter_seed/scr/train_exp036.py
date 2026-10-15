"""EXP036: PQ 回転摂動 augmentation の 3 seed 確認 — 訓練スクリプト。

EXP035（1 seed スクリーニング）で有望だった PQ 回転摂動について、改善が seed ノイズを超えるか、
事後追加・最良選択のバイアスを割り引いても残るか、J3 と J4 のどちらが安定して良いかを確認する。

  J0_s{42,43,44}  摂動なし（現行dual標準 = EXP035 J0_base と同一構成）
  J3_s{42,43,44}  θmax = ±15°（EXP035 J3_xstrong と同一構成）
  J4_s{42,43,44}  θmax = ±20°（EXP035 J4_rot20 と同一構成）

seed 42 は EXP035 の再現確認を兼ねる。テストセットは EXP035 と同一の db_replica 固定 manifest。
実行順は seed ごと（s42 の 3 条件 → s43 → s44）で、途中経過でも同 seed 内の比較ができる。

使い方:
    python experiments/EXP036_pq_jitter_seed/scr/train_exp036.py                  # 全 9 config
    python experiments/EXP036_pq_jitter_seed/scr/train_exp036.py --configs J4_s43
    python experiments/EXP036_pq_jitter_seed/scr/train_exp036.py --epochs 2       # smoke
    python experiments/EXP036_pq_jitter_seed/scr/infer_exp036.py                  # 推論
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

# yaml ファイル名は config 名と同一。PQ 回転摂動の強度と seed 以外は全 config 同一構成。
CONFIGS = tuple(f"{c}_s{s}" for s in (42, 43, 44) for c in ("J0", "J3", "J4"))


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

    print(f"\n{'='*72}\n  EXP036 結果サマリ\n{'='*72}")
    for exp_name, s in summary["configs"].items():
        print(f"  {exp_name:<8}  rot=±{s['pq_jitter_rot_deg']:<5}  "
              f"cv_mae={s['cv_mae']:.6f}  holdout_raw_t={s['holdout_mae_raw_t']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "EXP036_train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  保存: {OUTPUT_DIR / 'EXP036_train_summary.json'}")
    print("  推論: python experiments/EXP036_pq_jitter_seed/scr/infer_exp036.py")
    print("  passrate: python scripts/compute_passrate.py --exp EXP036")
    print("  比較: python scripts/paired_ttest.py --exp EXP036 --baseline J0_s42 --targets J4_s42")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
