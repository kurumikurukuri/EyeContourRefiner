"""EXP028: 訓練ラベル(t)への Gaussian ノイズ sweep（アノテーションノイズ模擬）。

EXP026 で k/batch 軸が飽和し、EXP023/024 の誤ラベル修正が t>=1.2 帯で有意改善した
ことから、本タスクは依然としてラベル品質に律速されていると考えられる。本実験は
その逆方向を検証する: 訓練ラベルに人為的な Gaussian ノイズを加えた場合、
holdout MAE（クリーンなラベルで評価）がどの程度悪化するかを sigma ごとに測定し、
ラベル品質の限界効用（＝アノテーション精度向上にどれだけ価値があるか）を定量化する。

3 構成 = LN0_baseline(sigma=0) / LN1_sigma0p01(sigma=0.01) / LN2_sigma0p05(sigma=0.05)。
他は全て EXP026 k3_b32 (= EXP025 A0 = EXP024 bc32_pw8 baseline) に固定。
ノイズは train manifest の t のみに適用され、manifest_test（hold-out）は常にクリーン。
LN0_baseline が anchor（sigma=0 は理論上 k3_b32 の再現）。

使い方:
    python experiments/EXP028_label_noise/scr/train_exp028.py             # 全 3 構成
    python experiments/EXP028_label_noise/scr/train_exp028.py --configs LN1_sigma0p01 LN2_sigma0p05
    python experiments/EXP028_label_noise/scr/train_exp028.py --epochs 2  # smoke
    python experiments/EXP028_label_noise/scr/infer_exp028.py             # 推論
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

# 3 構成 = label_noise_sigma sweep。yaml ファイル名は arm 名と同一。
CONFIGS = {
    "LN0_baseline": "LN0_baseline",  # anchor = sigma 0（k3_b32 の再現）
    "LN1_sigma0p01": "LN1_sigma0p01",
    "LN2_sigma0p05": "LN2_sigma0p05",
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
        endpoint_blend_range=float(cfg.get("endpoint_blend_range", 0.25)),
        loss_fn=cfg.get("loss_fn", "t_l1"), num_workers=num_workers,
        base_channel=cfg["base_channel"], pool_width=cfg["pool_width"],
        use_aux_face_shape=bool(cfg.get("use_aux_face_shape", False)),
        use_is_upper=cfg.get("use_is_upper", True),
        db_path=str(ROOT / "data" / "samples.db"),
        label_noise_sigma=float(cfg.get("label_noise_sigma", 0.0)),
    )


def run_one(exp_name: str, epochs_override: int | None, num_workers: int = 0) -> dict:
    cfg = load_config(CONFIGS[exp_name])
    config = build_train_config(cfg, epochs_override, num_workers=num_workers)
    model_dir = OUTPUT_DIR / exp_name
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "===== %s  label_noise_sigma=%.3f =====", exp_name, config.label_noise_sigma
    )

    result = run_dual_kfold(MANIFEST_DIR / "manifest_train.csv",
                            MANIFEST_DIR / "manifest_test.csv", model_dir, config)
    rec = build_dual_experiment_record(exp_name, config, result)
    (model_dir / "experiment.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"label_noise_sigma": config.label_noise_sigma,
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

    print(f"\n{'='*64}\n  EXP028 結果サマリ\n{'='*64}")
    for exp_name, s in summary["configs"].items():
        print(f"  {exp_name:<15}  sigma={s['label_noise_sigma']:.3f}  "
              f"cv_mae={s['cv_mae']:.6f}  holdout_raw_t={s['holdout_mae_raw_t']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "EXP028_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  保存: {OUTPUT_DIR / 'EXP028_summary.json'}")
    print(f"  passrate: python scripts/compute_passrate.py --exp EXP028")
    print(f"  arm 間比較: python scripts/paired_ttest.py --exp EXP028 --baseline LN0_baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
