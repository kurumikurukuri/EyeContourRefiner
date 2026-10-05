"""EXP034 推論スクリプト: base と重み平均の派生モデル（EMA / SWA）の fold アンサンブルで推論する。

4 つの出力 config はすべて同一 run・同一構造のモデルで、学習設定は config/base.yaml。
manifest は augmentation 無効で読み込むため入力は全 config で同一で、paired t-test が成立する。

configs:
  base / base_ema0999 / base_ema09998 / base_swa

使い方:
    python experiments/EXP034_swa_ema/scr/infer_exp034.py              # 全 4 config
    python experiments/EXP034_swa_ema/scr/infer_exp034.py --configs base_swa
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
    compute_aux_face_shape_meta,
    compute_eye_meta,
)
from eyeliner.models.dual_ensemble import load_dual_model  # noqa: E402
from eyeliner.training.dual_branch import (  # noqa: E402
    build_branch_configs,
    fold_ensemble_raw_t,
)

# 出力 config 名 → 学習設定 yaml 名（派生モデルは base と同一 run・同一構造）
CONFIGS = {
    "base": "base",
    "base_ema0999": "base",
    "base_ema09998": "base",
    "base_swa": "base",
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
    # augment_data=False のため推論入力は全 config で同一。
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
    print(f"推論完了: config={exp_name}  yaml={config_name}  seed={cfg.get('seed')}")
    print(f"MAE (生t): {mae:.6f}")
    print(f"保存: {output_path}")
    return mae


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="EXP034 重み平均（EMA / SWA）推論")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS),
                        help="推論する config（省略時: 全 4 config）")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="省略時は train/test 両方を推論し、"
                             "train_predictions.csv / test_predictions.csv を出力")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    print(f"\n{'='*64}\n  EXP034 推論\n{'='*64}")
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

    print(f"\n{'='*64}\n  EXP034 MAE サマリ\n{'='*64}")
    for exp_name, mae in results.items():
        print(f"  {exp_name:<15}  MAE={mae:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
