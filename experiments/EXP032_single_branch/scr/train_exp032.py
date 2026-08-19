"""EXP032: 単独ブランチ（dual fusionなし）— 現行ベストプラクティス再実験 訓練スクリプト。

EXP020でdual-branch化して以降（EXP021〜EXP031）に積み上がった改善
（t_l1損失・sigmoid出力ヘッド・aug_flags=63のphotometricブロック・k=3/batch=32・
OOF誤ラベル除去/label noise対応によるDBクリーニング・db_replica固定テストセット）は
すべてdual-branchモデルの上でのみ検証・採用されてきた。単独ブランチが最後に実際の
EXPで使われたのはEXP018で、EXP019のX2_eye_scale（pq_rescale_mode=eye_scale）が
単独座標系として最良だった（raw PQ比 -1.5%、汎化ギャップ最小）。

本実験は、その単独ブランチ構成（branch B = eye_scale、対照としてbranch A = 生PQ）を
現行dual標準ベースラインと同一のハイパーパラメータ・同一のdb_replica固定テストセットで
再学習し、現行dual-branch標準モデルとの差を公平に測定する。

branch の patch_t_min/max, label_t_min/max, patch_perp_scale, patch_size は
eyeliner/training/dual_branch.py の branch_a_x0() / branch_b_eye_scale() プリセットを
そのまま使う（dual側と値がズレないようにするため、config yamlには重複させない）。

使い方:
    python experiments/EXP032_single_branch/scr/train_exp032.py             # 両config
    python experiments/EXP032_single_branch/scr/train_exp032.py --configs SB_eyescale
    python experiments/EXP032_single_branch/scr/train_exp032.py --epochs 2  # smoke
    python experiments/EXP032_single_branch/scr/infer_exp032.py             # 推論
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

from eyeliner.training.dual_branch import branch_a_x0, branch_b_eye_scale  # noqa: E402
from eyeliner.training.kfold import TrainConfig, run_kfold  # noqa: E402

logger = logging.getLogger(__name__)

CONFIGS = {
    "SB_none": "SB_none",          # branch A 相当（生PQ）— 対照
    "SB_eyescale": "SB_eyescale",  # branch B 相当（eye_scale）— 本命
}

_BRANCH_BUILDERS = {
    "none": branch_a_x0,
    "eye_scale": branch_b_eye_scale,
}


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_config(
    cfg: dict, epochs_override: int | None, num_workers: int = 0
) -> TrainConfig:
    pq_rescale_mode = cfg.get("pq_rescale_mode", "none")
    branch = _BRANCH_BUILDERS[pq_rescale_mode]()
    return TrainConfig(
        backbone=cfg.get("backbone", "custom"),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        loss_fn=cfg.get("loss_fn", "t_l1"),
        epochs=epochs_override if epochs_override else cfg["epochs"],
        batch_size=cfg["batch"], patience=cfg["patience"], seed=cfg["seed"],
        use_sigmoid=cfg.get("use_sigmoid", True),
        use_batchnorm=cfg.get("use_batchnorm", True),
        num_conv_layers=cfg.get("num_conv_layers", 4),
        dropout_rate=cfg.get("dropout_rate", 0.3),
        use_is_upper=cfg.get("use_is_upper", True),
        use_t_norm=True,
        aug_flags=cfg.get("aug_flags", 63),
        endpoint_blend_range=float(cfg.get("endpoint_blend_range", 0.25)),
        pool_width=int(cfg.get("pool_width", 32)),
        num_workers=num_workers,
        # branch preset（dual_branch.py と同一値。ズレ防止のためここで直接取得）
        pq_rescale_mode=branch.pq_rescale_mode,
        pq_rescale_factor=branch.pq_rescale_factor,
        patch_t_min=branch.patch_t_min,
        patch_t_max=branch.patch_t_max,
        label_t_min=branch.label_t_min,
        label_t_max=branch.label_t_max,
        patch_perp_scale=branch.patch_perp_scale,
        patch_h=branch.patch_size[0],
        patch_w=branch.patch_size[1],
    )


def build_experiment_record(exp_name: str, config: TrainConfig, k_folds: int, fold_results) -> dict:
    mean_mae = sum(r.best_val_mae for r in fold_results) / len(fold_results)
    return {
        "exp_name": exp_name,
        "config": {
            "pq_rescale_mode": config.pq_rescale_mode,
            "pq_rescale_factor": config.pq_rescale_factor,
            "loss_fn": config.loss_fn,
            "num_conv_layers": config.num_conv_layers,
            "pool_width": config.pool_width,
            "aug_flags": config.aug_flags,
            "k_folds": k_folds,
            "seed": config.seed,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
        },
        "results": {
            "cv_mae": mean_mae,
            "fold_maes": [r.best_val_mae for r in fold_results],
        },
    }


def run_one(exp_name: str, epochs_override: int | None, num_workers: int = 0) -> dict:
    cfg = load_config(CONFIGS[exp_name])
    config = build_train_config(cfg, epochs_override, num_workers=num_workers)
    model_dir = OUTPUT_DIR / exp_name
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "===== %s  pq_rescale_mode=%s  label_range=[%.2f, %.2f] =====",
        exp_name, config.pq_rescale_mode, config.label_t_min, config.label_t_max,
    )

    k_folds = cfg["k"]
    fold_results = run_kfold(
        MANIFEST_DIR / "manifest_train.csv", model_dir, config,
        k_folds=k_folds, resume=True,
    )
    rec = build_experiment_record(exp_name, config, k_folds, fold_results)
    (model_dir / "experiment.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"cv_mae": rec["results"]["cv_mae"]}


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

    print(f"\n{'='*72}\n  EXP032 結果サマリ\n{'='*72}")
    for exp_name, s in summary["configs"].items():
        print(f"  {exp_name:<15}  cv_mae={s['cv_mae']:.6f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "EXP032_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  保存: {OUTPUT_DIR / 'EXP032_summary.json'}")
    print("  推論: python experiments/EXP032_single_branch/scr/infer_exp032.py")
    print("  passrate: python scripts/compute_passrate.py --exp EXP032")
    print("  単独ブランチ内比較: python scripts/paired_ttest.py --exp EXP032 --baseline SB_none --targets SB_eyescale")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
