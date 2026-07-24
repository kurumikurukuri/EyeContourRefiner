"""EXP025: aux 特徴量の効果検証（use_aux_face_shape + サブセット選択）。

configs:
  A0  aux 無し + is_upper 有（baseline、現 bc32_pw8 と同条件）
  A1  aux={face_pitch_sin, face_pitch_cos, position_polar_cos}, is_upper 無し
  A2  A1 + position_polar_sin（上下まぶた明示シグナル）
  A3  use_aux_face_shape フル11次元（is_upper は aux 内に内包）
  A4  aux 無し + is_upper 無し（最小構成 ablation）

各 config を同一 DB/manifest/seed で学習する。config 間の paired t-test 比較は
scripts/paired_ttest.py で実行する。

使い方:
    python experiments/EXP025_aux_redesign/scr/train_exp025.py             # 全 config
    python experiments/EXP025_aux_redesign/scr/train_exp025.py --configs A0 A1
    python experiments/EXP025_aux_redesign/scr/train_exp025.py --epochs 2  # smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
CONFIG_DIR = EXP_DIR / "config"
MANIFEST_DIR = EXP_DIR / "manifests"
OUTPUT_DIR = EXP_DIR / "outputs"
sys.path.insert(0, str(ROOT))

from eyeliner.data.dataset import load_manifest  # noqa: E402
from eyeliner.data.dataset_dual import EyelinerPatchDualDataset  # noqa: E402
from eyeliner.data.weights import compute_aux_face_shape_meta, compute_eye_meta  # noqa: E402
from eyeliner.models.dual_ensemble import load_dual_model  # noqa: E402
from eyeliner.training.dual_branch import build_branch_configs, fold_ensemble_raw_t  # noqa: E402
from eyeliner.training.dual_kfold import (  # noqa: E402
    DualTrainConfig,
    build_dual_experiment_record,
    effective_aux_features,
    run_dual_kfold,
)

logger = logging.getLogger(__name__)

CONFIGS = {
    "A0": "A0_baseline",
    "A1": "A1_pitch_polarcos",
    "A2": "A2_add_polarsin",
    "A3": "A3_aux_face_shape",
    "A4": "A4_no_features",
}


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_config(cfg: dict, epochs_override: int | None) -> DualTrainConfig:
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
        loss_fn=cfg.get("loss_fn", "t_l1"), num_workers=0,
        base_channel=cfg["base_channel"], pool_width=cfg["pool_width"],
        use_aux_face_shape=bool(cfg.get("use_aux_face_shape", False)),
        aux_features=list(cfg.get("aux_features", []) or []),
        use_is_upper=cfg.get("use_is_upper", True),
        db_path=str(ROOT / "data" / "samples.db"),
    )


def predict_one(model_dir: Path, out_csv: Path, config: DualTrainConfig) -> None:
    """fold アンサンブル + aux で test の per-sample 生t予測を出力する。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_manifest(MANIFEST_DIR / "manifest_test.csv")
    compute_eye_meta(records)
    aux_features = effective_aux_features(config)
    if aux_features:
        compute_aux_face_shape_meta(records, config.db_path)
    cfg_a, cfg_b = build_branch_configs(config.pq_rescale_mode_b)
    ds = EyelinerPatchDualDataset(
        records, cfg_a, cfg_b, augment_data=False, preload_images=False,
        aux_features=aux_features,
    )
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    fold_paths = sorted(model_dir.glob("eyeliner_detection_fold*.pt"))
    models = [load_dual_model(p).to(device) for p in fold_paths]

    preds: list[float] = []
    with torch.no_grad():
        for batch in loader:
            x_a, x_b, is_upper = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            pq_dist, pq_prime = batch[12].to(device), batch[13].to(device)
            aux = batch[14].to(device) if aux_features else None
            t_ens = fold_ensemble_raw_t(models, x_a, x_b, is_upper, config.fusion_mode,
                                        cfg_a, cfg_b, pq_dist, pq_prime, aux)
            preds.extend(t_ens.squeeze(-1).cpu().tolist())

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "t_true", "t_pred"])
        for rec, p in zip(records, preds):
            w.writerow([rec.get("idx", ""), rec["t"], f"{p:.6f}"])


def run_one(exp_name: str, epochs_override: int | None) -> dict:
    cfg = load_config(CONFIGS[exp_name])
    config = build_train_config(cfg, epochs_override)
    model_dir = OUTPUT_DIR / exp_name
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("===== %s  use_aux_face_shape=%s aux_features=%s use_is_upper=%s =====",
                exp_name, config.use_aux_face_shape, effective_aux_features(config), config.use_is_upper)

    result = run_dual_kfold(MANIFEST_DIR / "manifest_train.csv",
                            MANIFEST_DIR / "manifest_test.csv", model_dir, config)
    rec = build_dual_experiment_record(exp_name, config, result)
    (model_dir / "experiment.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    predict_one(model_dir, model_dir / "test_predictions.csv", config)
    return {"cv_mae": rec["results"]["cv_mae"],
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
    ap.add_argument("--skip-train", action="store_true", help="学習済み前提でスキップ")
    args = ap.parse_args()

    summary: dict = {"configs": {}}
    if not args.skip_train:
        for exp_name in args.configs:
            summary["configs"][exp_name] = run_one(exp_name, args.epochs)

    print(f"\n{'='*64}\n  EXP025 結果サマリ\n{'='*64}")
    for exp_name, s in summary["configs"].items():
        print(f"  {exp_name}  cv_mae={s['cv_mae']:.6f}  holdout_raw_t={s['holdout_mae_raw_t']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "exp025_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  保存: {OUTPUT_DIR / 'exp025_summary.json'}")
    print(f"  passrate: python scripts/compute_passrate.py --exp EXP025")
    print(f"  config 間比較: python scripts/paired_ttest.py --exp EXP025 --baseline A0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
