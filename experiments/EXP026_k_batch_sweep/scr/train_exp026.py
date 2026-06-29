"""EXP026: k(=fold数) × batch size の 2D sweep（dual k-fold 再学習）。

EXP024/025 で容量軸(base_channel/pool_width/num_conv_layers)と入力特徴軸(aux)が
共に飽和し、タスクはデータ/ラベル律速と判明した。本実験は残る最適化系軸
「CV fold 数 k」と「batch size」が holdout raw-t MAE を動かすかを検証する。

仮説(事前知識): 容量・入力軸飽和の延長で、k/batch も概ね飽和（現行 k=3,batch=32
がほぼ最適）と予測する。ただし k は「fold あたり学習データ量」と「holdout
アンサンブル枚数」を同時に変える特殊軸で、純容量軸とは異なるため実測で確認する。

6 構成 = k{3,4,5} × batch{32,64}。他は全て EXP025 A0 (= EXP024 bc32_pw8 baseline)
に固定。k3_b32 が anchor（現行デフォルト）。arm 間の paired t-test 比較は
compare_exp026.py で実行する。

使い方:
    python experiments/EXP026_k_batch_sweep/scr/train_exp026.py             # 全 6 構成
    python experiments/EXP026_k_batch_sweep/scr/train_exp026.py --configs k3_b32 k4_b32
    python experiments/EXP026_k_batch_sweep/scr/train_exp026.py --epochs 2  # smoke
    python experiments/EXP026_k_batch_sweep/scr/infer_exp026.py             # 推論
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

# 6 構成 = k{3,4,5} × batch{32,64}。yaml ファイル名は arm 名と同一。
CONFIGS = {
    "k3_b32": "k3_b32",  # anchor = 現行デフォルト
    "k3_b64": "k3_b64",
    "k4_b32": "k4_b32",
    "k4_b64": "k4_b64",
    "k5_b32": "k5_b32",
    "k5_b64": "k5_b64",
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
    )


def run_one(exp_name: str, epochs_override: int | None, num_workers: int = 0) -> dict:
    cfg = load_config(CONFIGS[exp_name])
    config = build_train_config(cfg, epochs_override, num_workers=num_workers)
    model_dir = OUTPUT_DIR / exp_name
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("===== %s  k=%d batch=%d =====", exp_name, config.k, config.batch_size)

    result = run_dual_kfold(MANIFEST_DIR / "manifest_train.csv",
                            MANIFEST_DIR / "manifest_test.csv", model_dir, config)
    rec = build_dual_experiment_record(exp_name, config, result)
    (model_dir / "experiment.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"k": config.k, "batch": config.batch_size,
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

    print(f"\n{'='*64}\n  EXP026 結果サマリ\n{'='*64}")
    for exp_name, s in summary["configs"].items():
        print(f"  {exp_name}  k={s['k']} batch={s['batch']}  "
              f"cv_mae={s['cv_mae']:.6f}  holdout_raw_t={s['holdout_mae_raw_t']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "exp026_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  保存: {OUTPUT_DIR / 'exp026_summary.json'}")
    print(f"  arm 間比較: python experiments/EXP026_k_batch_sweep/scr/compare_exp026.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
