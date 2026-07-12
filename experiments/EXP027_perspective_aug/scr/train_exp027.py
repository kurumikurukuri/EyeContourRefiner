"""EXP027: 顔中心ピボット（鼻先 landmark #4）+ 固定パターン面外回転 perspective aug。

homography の回転中心を鼻先（MediaPipe face mesh landmark #4）に固定する。
これにより目領域の foreshortening だけでなく顔回転に伴う orbit 移動も再現される。

事前準備（必須）:
  1. DB に nose_x/nose_y を追加: python annotator/preprocessor.py --images data/images/
  2. manifest を再生成:          python scripts/split_holdout.py --exp 027

使い方:
    python experiments/EXP027_perspective_aug/scr/train_exp027.py
    python experiments/EXP027_perspective_aug/scr/train_exp027.py --configs fcp_p8
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
    PERSPECTIVE_PATTERNS_8,
    DualTrainConfig,
    build_dual_experiment_record,
    run_dual_kfold,
)

logger = logging.getLogger(__name__)

CONFIGS = {
    "fcp_p8":         "fcp_p8",
    "fcp_p8_b256":    "fcp_p8_b256",
}


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_config(cfg: dict, epochs_override: int | None, num_workers: int = 0) -> DualTrainConfig:
    use_persp = bool(cfg.get("perspective_aug", False))
    patterns = PERSPECTIVE_PATTERNS_8 if use_persp else []
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
        endpoint_blend_range=float(cfg.get("endpoint_blend_range", 0.25)),
        loss_fn=cfg.get("loss_fn", "t_l1"), num_workers=num_workers,
        base_channel=cfg["base_channel"], pool_width=cfg["pool_width"],
        use_aux_face_shape=bool(cfg.get("use_aux_face_shape", False)),
        use_is_upper=cfg.get("use_is_upper", True),
        db_path=str(ROOT / "data" / "samples.db"),
        perspective_aug=use_persp,
        perspective_alpha=float(cfg.get("perspective_alpha", 2.0)),
        perspective_patterns=patterns,
    )


def run_one(exp_name: str, epochs_override: int | None, num_workers: int = 0) -> dict:
    cfg = load_config(CONFIGS[exp_name])
    config = build_train_config(cfg, epochs_override, num_workers=num_workers)
    model_dir = OUTPUT_DIR / exp_name
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "===== %s  perspective_aug=%s  n_patterns=%d =====",
        exp_name, config.perspective_aug, len(config.perspective_patterns),
    )

    result = run_dual_kfold(
        MANIFEST_DIR / "manifest_train.csv",
        MANIFEST_DIR / "manifest_test.csv",
        model_dir, config,
    )
    rec = build_dual_experiment_record(exp_name, config, result)
    (model_dir / "experiment.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "perspective_aug": config.perspective_aug,
        "n_patterns": len(config.perspective_patterns),
        "cv_mae": rec["results"]["cv_mae"],
        "holdout_mae_raw_t": rec["results"]["holdout_mae_raw_t"],
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--epochs", type=int, default=None, help="smoke 用 epoch 上書き")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    summary: dict = {"configs": {}}
    for exp_name in args.configs:
        summary["configs"][exp_name] = run_one(
            exp_name, args.epochs, num_workers=args.num_workers
        )

    print(f"\n{'='*64}\n  EXP027 結果サマリ\n{'='*64}")
    for exp_name, s in summary["configs"].items():
        print(
            f"  {exp_name:<15}  face_pivot={s['perspective_aug']}  n_pat={s['n_patterns']}  "
            f"cv_mae={s['cv_mae']:.6f}  holdout_raw_t={s['holdout_mae_raw_t']}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "EXP027_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  保存: {OUTPUT_DIR / 'EXP027_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
