"""EXP026 推論スクリプト: k × batch sweep の fold アンサンブルでテストセットを推論。

configs:
  k3_b32  k=3, batch=32（anchor）
  k3_b64  k=3, batch=64
  k4_b32  k=4, batch=32
  k4_b64  k=4, batch=64
  k5_b32  k=5, batch=32
  k5_b64  k=5, batch=64

使い方:
    python experiments/EXP026_k_batch_sweep/scr/infer_exp026.py              # 全 6 構成
    python experiments/EXP026_k_batch_sweep/scr/infer_exp026.py --configs k3_b32 k4_b32
    python experiments/EXP026_k_batch_sweep/scr/infer_exp026.py --configs k3_b32 --manifest path/to/manifest.csv
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
from eyeliner.data.weights import compute_aux_face_shape_meta, compute_eye_meta  # noqa: E402
from eyeliner.models.dual_ensemble import load_dual_model  # noqa: E402
from eyeliner.training.dual_branch import build_branch_configs, fold_ensemble_raw_t  # noqa: E402

CONFIGS = {
    "k3_b32": "k3_b32",
    "k3_b64": "k3_b64",
    "k4_b32": "k4_b32",
    "k4_b64": "k4_b64",
    "k5_b32": "k5_b32",
    "k5_b64": "k5_b64",
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
    use_is_upper: bool = cfg.get("use_is_upper", True)
    fusion_mode: str = cfg.get("fusion_mode", "independent")
    pq_rescale_mode_b: str = cfg.get("pq_rescale_mode_b", "eye_scale")
    batch_size: int = cfg.get("batch", 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    records = load_manifest(manifest_path)
    compute_eye_meta(records)
    if use_aux_face_shape:
        compute_aux_face_shape_meta(records, str(DB_PATH))
    print(f"サンプル数: {len(records)}")

    cfg_a, cfg_b = build_branch_configs(pq_rescale_mode_b)
    ds = EyelinerPatchDualDataset(
        records, cfg_a, cfg_b, augment_data=False, preload_images=False,
        use_aux_face_shape=use_aux_face_shape,
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
            aux = batch[14].to(device) if use_aux_face_shape else None
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
    print(f"推論完了: config={exp_name}  k={cfg['k']} batch={batch_size}  use_is_upper={use_is_upper}")
    print(f"MAE (生t): {mae:.6f}")
    print(f"保存: {output_path}")
    return mae


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="EXP026 k × batch sweep 推論")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS),
                        help="推論する config（省略時: 全 6 構成）")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="省略時は train/test 両方を推論し、"
                             "train_predictions.csv / test_predictions.csv を出力")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    print(f"\n{'='*64}\n  EXP026 推論\n{'='*64}")
    results: dict[str, float] = {}
    for exp_name in args.configs:
        if args.manifest is not None:
            output_path = OUTPUT_DIR / exp_name / "test_predictions.csv"
            mae = infer_one(exp_name, args.manifest, output_path, args.num_workers)
        else:
            infer_one(exp_name, MANIFEST_DIR / "manifest_train.csv",
                      OUTPUT_DIR / exp_name / "train_predictions.csv", args.num_workers)
            mae = infer_one(exp_name, MANIFEST_DIR / "manifest_test.csv",
                             OUTPUT_DIR / exp_name / "test_predictions.csv", args.num_workers)
        results[exp_name] = mae

    print(f"\n{'='*64}\n  EXP026 MAE サマリ\n{'='*64}")
    for exp_name, mae in results.items():
        cfg = CONFIGS[exp_name]
        print(f"  {exp_name:<10}  MAE={mae:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
