"""EXP025 推論スクリプト: aux 再設計 config の fold アンサンブルでテストセットを推論。

使い方:
    python experiments/EXP025_aux_redesign/scr/infer_exp025.py --config A0
    python experiments/EXP025_aux_redesign/scr/infer_exp025.py --config A2 --manifest path/to/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
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
DB_PATH = ROOT / "data" / "samples.db"

sys.path.insert(0, str(ROOT))

from eyeliner.data.dataset import load_manifest  # noqa: E402
from eyeliner.data.dataset_dual import EyelinerPatchDualDataset  # noqa: E402
from eyeliner.data.weights import (  # noqa: E402
    DEFAULT_AUX_FEATURES,
    compute_aux_face_shape_meta,
    compute_eye_meta,
)
from eyeliner.models.dual_ensemble import load_dual_model  # noqa: E402
from eyeliner.training.dual_branch import build_branch_configs, fold_ensemble_raw_t  # noqa: E402

CONFIGS = {
    "A0": "A0_baseline",
    "A1": "A1_pitch_polarcos",
    "A2": "A2_add_polarsin",
    "A3": "A3_aux_face_shape",
    "A4": "A4_no_features",
}


def infer_one(
    exp_name: str,
    manifest_path: Path,
    output_path: Path,
    num_workers: int = 0,
) -> float:
    """指定 config の fold アンサンブルで推論し CSV を保存する。MAE を返す。"""
    config_name = CONFIGS[exp_name]
    with open(CONFIG_DIR / f"{config_name}.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    use_aux_face_shape: bool = bool(cfg.get("use_aux_face_shape", False))
    aux_features: list[str] = list(cfg.get("aux_features", []) or [])
    if not aux_features and use_aux_face_shape:
        aux_features = list(DEFAULT_AUX_FEATURES)
    use_is_upper: bool = cfg.get("use_is_upper", True)
    fusion_mode: str = cfg.get("fusion_mode", "independent")
    pq_rescale_mode_b: str = cfg.get("pq_rescale_mode_b", "eye_scale")
    batch_size: int = cfg.get("batch", 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    records = load_manifest(manifest_path)
    compute_eye_meta(records)
    if aux_features:
        compute_aux_face_shape_meta(records, str(DB_PATH))
    print(f"サンプル数: {len(records)}")

    cfg_a, cfg_b = build_branch_configs(pq_rescale_mode_b)
    ds = EyelinerPatchDualDataset(
        records, cfg_a, cfg_b, augment_data=False, preload_images=False,
        aux_features=aux_features,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model_dir = OUTPUT_DIR / exp_name
    fold_paths = sorted(model_dir.glob("eyeliner_detection_fold*.pt"))
    if not fold_paths:
        raise FileNotFoundError(f"fold モデルが見つかりません: {model_dir}")
    models = [load_dual_model(p).to(device) for p in fold_paths]
    print(f"fold モデル数: {len(models)}")

    preds: list[float] = []
    with torch.no_grad():
        for batch in loader:
            x_a, x_b, is_upper = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            pq_dist, pq_prime = batch[12].to(device), batch[13].to(device)
            aux = batch[14].to(device) if aux_features else None
            t_ens = fold_ensemble_raw_t(
                models, x_a, x_b, is_upper, fusion_mode,
                cfg_a, cfg_b, pq_dist, pq_prime, aux,
            )
            preds.extend(t_ens.squeeze(-1).cpu().tolist())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["idx", "image_path", "part_name", "sub_key", "point_index", "t_true", "t_pred"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec, p in zip(records, preds):
            writer.writerow({
                "idx": rec.get("idx", ""),
                "image_path": rec.get("image_path", ""),
                "part_name": rec.get("part_name", ""),
                "sub_key": rec.get("sub_key", ""),
                "point_index": rec.get("point_index", ""),
                "t_true": f"{float(rec['t']):.6f}",
                "t_pred": f"{p:.6f}",
            })

    mae = sum(abs(p - float(rec["t"])) for p, rec in zip(preds, records)) / max(len(preds), 1)
    print(f"推論完了: config={exp_name}  aux_features={aux_features}  use_is_upper={use_is_upper}")
    print(f"MAE (生t): {mae:.6f}")
    print(f"保存: {output_path}")
    return mae


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="EXP025 aux 再設計 config 推論")
    parser.add_argument("--config", default="A0", choices=list(CONFIGS),
                        help="推論する config（デフォルト: A0）")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="省略時は train/test 両方を推論し、"
                             "train_predictions.csv / test_predictions.csv を出力")
    parser.add_argument("--output", type=Path, default=None, help="出力 CSV パス（--manifest 指定時のみ有効）")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.manifest is not None:
        output = args.output or OUTPUT_DIR / args.config / "test_predictions.csv"
        infer_one(args.config, args.manifest, output, args.num_workers)
        return 0

    infer_one(args.config, MANIFEST_DIR / "manifest_train.csv",
              OUTPUT_DIR / args.config / "train_predictions.csv", args.num_workers)
    infer_one(args.config, MANIFEST_DIR / "manifest_test.csv",
              OUTPUT_DIR / args.config / "test_predictions.csv", args.num_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
