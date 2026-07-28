"""EXP029: 固定テストセット 5 パターン比較実験 — 訓練スクリプト.

EXP028 LN0_baseline と同一条件で、5 つの固定テストセット (Pattern A-E)
それぞれに対して同一モデルアーキテクチャを訓練する。

使い方:
    python experiments/EXP029_fixed_test/scr/train_exp029.py              # 全 5 パターン
    python experiments/EXP029_fixed_test/scr/train_exp029.py --patterns oof_error_hard db_replica  # 特定パターン
    python experiments/EXP029_fixed_test/scr/train_exp029.py --epochs 2   # smoke test
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

PATTERNS = ["oof_error_hard", "db_replica", "random"]
EXTRA_PATTERNS = ["oof_error_easy", "oof_error_half"]
ALL_PATTERNS = PATTERNS + EXTRA_PATTERNS


def load_config() -> dict:
    with open(CONFIG_DIR / "baseline.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_config(
    cfg: dict, epochs_override: int | None, num_workers: int = 0
) -> DualTrainConfig:
    return DualTrainConfig(
        fusion_mode=cfg["fusion_mode"],
        pq_rescale_mode_b=cfg.get("pq_rescale_mode_b", "eye_scale"),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        epochs=epochs_override if epochs_override else cfg["epochs"],
        batch_size=cfg["batch"], k=cfg["k"], patience=cfg["patience"],
        seed=cfg["seed"],
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
    )


def run_one(
    pattern: str, cfg: dict, epochs_override: int | None, num_workers: int = 0
) -> dict:
    config = build_train_config(cfg, epochs_override, num_workers=num_workers)
    model_dir = OUTPUT_DIR / pattern
    model_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = MANIFEST_DIR / f"manifest_train_{pattern}.csv"
    test_manifest = MANIFEST_DIR / f"manifest_test_{pattern}.csv"

    if not train_manifest.exists():
        raise FileNotFoundError(f"Train manifest not found: {train_manifest}")
    if not test_manifest.exists():
        raise FileNotFoundError(f"Test manifest not found: {test_manifest}")

    logger.info("===== Pattern %s =====", pattern)
    logger.info("  train: %s", train_manifest)
    logger.info("  test:  %s", test_manifest)

    result = run_dual_kfold(train_manifest, test_manifest, model_dir, config)
    rec = build_dual_experiment_record(pattern, config, result)
    (model_dir / "experiment.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "pattern": pattern,
        "cv_mae": rec["results"]["cv_mae"],
        "holdout_mae_raw_t": rec["results"]["holdout_mae_raw_t"],
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--patterns", nargs="+", default=PATTERNS,
        choices=ALL_PATTERNS,
        help=f"Patterns to train (default: {' '.join(PATTERNS)})",
    )
    ap.add_argument(
        "--epochs", type=int, default=None,
        help="Override epoch count (for smoke testing)",
    )
    ap.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers (0=safe on Windows)",
    )
    args = ap.parse_args()

    patterns = args.patterns

    cfg = load_config()
    logger.info("Config: baseline.yaml (EXP028 LN0_baseline equivalent)")
    logger.info("Patterns: %s", " ".join(patterns))

    summary_path = OUTPUT_DIR / "EXP029_summary.json"
    summary: dict = {"configs": {}}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.setdefault("configs", {})

    for pattern in patterns:
        summary["configs"][pattern] = run_one(
            pattern, cfg, args.epochs, num_workers=args.num_workers
        )

    print(f"\n{'='*64}")
    print(f"  EXP029 結果サマリ")
    print(f"{'='*64}")
    for pattern, s in summary["configs"].items():
        print(
            f"  Pattern {pattern}  "
            f"cv_mae={s['cv_mae']:.6f}  "
            f"holdout_raw_t={s['holdout_mae_raw_t']:.6f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  保存: {summary_path}")
    print(f"  passrate: python scripts/compute_passrate.py --exp EXP029")
    print(f"  比較: python scripts/paired_ttest.py --exp EXP029 --baseline oof_error_hard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
