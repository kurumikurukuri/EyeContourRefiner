"""EXP031: 元画像空間の劣化 aug（JPEG 圧縮 / 解像度ジッタ）— 訓練スクリプト。

パッチは元画像を補間拡大したものであるため、パッチ空間で JPEG 圧縮や解像度劣化を
定義しても元画像で実際に起きる劣化とは一致しない。本実験は劣化を **ワープ前の
元画像 ROI** に適用する 2 種を追加し、2×2 直積 + endpoint_blend 単独 ablation の
計 5 構成で評価する。

  SD0_baseline   劣化なし（= EXP030 AF0 / EXP029 baseline と同一構成の anchor）
  SD1_jpeg       JPEG 圧縮のみ      quality ∈ [30, 90], p=0.5
  SD2_resjitter  解像度ジッタのみ    scale ∈ [1.0, 2.5]（AREA↓ → LINEAR↑）, p=0.5
  SD3_both       両方（縮小 → 縮小サイズで JPEG → 拡大）
  SD4_endpoint_only  photometric 5種 off、ENDPOINT_BLEND のみ

劣化は train のみ（augment_data=True）に適用され、val / hold-out / 推論は
全 config で完全に同一の入力になる。テストセットは EXP029 で採用が決まった
db_replica 固定 manifest。

使い方:
    python experiments/EXP031_source_space_aug/scr/train_exp031.py             # 全 5 構成
    python experiments/EXP031_source_space_aug/scr/train_exp031.py --configs SD1_jpeg
    python experiments/EXP031_source_space_aug/scr/train_exp031.py --epochs 2  # smoke
    python experiments/EXP031_source_space_aug/scr/infer_exp031.py             # 推論
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

from eyeliner.data.augmentation import (  # noqa: E402
    AUG_SOURCE_JPEG,
    AUG_SOURCE_RES_JITTER,
)
from eyeliner.training.dual_kfold import (  # noqa: E402
    DualTrainConfig,
    build_dual_experiment_record,
    run_dual_kfold,
)

logger = logging.getLogger(__name__)

# 5 構成 = 劣化 2 種の 2×2 直積 + endpoint_blend 単独 ablation。
# yaml ファイル名は arm 名と同一のため、arm 名の列挙で足りる。
CONFIGS = (
    "SD0_baseline",    # anchor（劣化なし）
    "SD1_jpeg",        # JPEG のみ
    "SD2_resjitter",   # 解像度ジッタのみ
    "SD3_both",        # 両方
    "SD4_endpoint_only",  # photometric 5種 off、ENDPOINT_BLEND のみ
)


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
        # EXP031 の変動軸（ON/OFF は aug_flags のビット、ここは強度パラメータのみ）
        source_jpeg_prob=float(cfg.get("source_jpeg_prob", 0.5)),
        source_jpeg_quality_min=int(cfg.get("source_jpeg_quality_min", 30)),
        source_jpeg_quality_max=int(cfg.get("source_jpeg_quality_max", 90)),
        source_res_jitter_prob=float(cfg.get("source_res_jitter_prob", 0.5)),
        source_res_scale_min=float(cfg.get("source_res_scale_min", 1.0)),
        source_res_scale_max=float(cfg.get("source_res_scale_max", 2.5)),
    )


def _degrade_label(config: DualTrainConfig) -> str:
    parts: list[str] = []
    if config.aug_flags & AUG_SOURCE_JPEG:
        parts.append(
            f"jpeg(q{config.source_jpeg_quality_min}-{config.source_jpeg_quality_max}"
            f",p{config.source_jpeg_prob})"
        )
    if config.aug_flags & AUG_SOURCE_RES_JITTER:
        parts.append(
            f"res(s{config.source_res_scale_min}-{config.source_res_scale_max}"
            f",p{config.source_res_jitter_prob})"
        )
    return "+".join(parts) if parts else "none"


def run_one(exp_name: str, epochs_override: int | None, num_workers: int = 0) -> dict:
    cfg = load_config(exp_name)
    config = build_train_config(cfg, epochs_override, num_workers=num_workers)
    model_dir = OUTPUT_DIR / exp_name
    model_dir.mkdir(parents=True, exist_ok=True)
    label = _degrade_label(config)
    logger.info("===== %s  source_degrade=%s =====", exp_name, label)

    result = run_dual_kfold(MANIFEST_DIR / "manifest_train.csv",
                            MANIFEST_DIR / "manifest_test.csv", model_dir, config)
    rec = build_dual_experiment_record(exp_name, config, result)
    (model_dir / "experiment.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"source_degrade": label,
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

    print(f"\n{'='*72}\n  EXP031 結果サマリ\n{'='*72}")
    for exp_name, s in summary["configs"].items():
        print(f"  {exp_name:<15}  {s['source_degrade']:<34}  "
              f"cv_mae={s['cv_mae']:.6f}  holdout_raw_t={s['holdout_mae_raw_t']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "EXP031_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  保存: {OUTPUT_DIR / 'EXP031_summary.json'}")
    print("  passrate: python scripts/compute_passrate.py --exp EXP031")
    print("  arm 間比較: python scripts/paired_ttest.py --exp EXP031 --baseline SD0_baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
