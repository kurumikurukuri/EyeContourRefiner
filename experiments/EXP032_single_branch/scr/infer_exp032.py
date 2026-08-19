"""EXP032 推論スクリプト: 単独ブランチ（dual fusionなし）fold アンサンブルで推論する。

pq_rescale_mode=eye_scale（SB_eyescale）の予測は P'Q' 座標系の t'（label_space="t_prime"）
なので、db_replicaのraw t（生PQ座標系）とpaired t-testで比較できるよう
eyeliner.training.dual_branch.branch_pred_to_raw_t で生 t に逆変換してからCSVへ書き出す
（dual-branch推論・EXP019のtest_mae評価と同じ変換）。SB_none（pq_rescale_mode=none）は
label_space="raw_t"なのでdenormのみで生tになる。

manifest は augmentation 無効（augment_data=False）で読み込むため、推論入力は
学習時のaugと無関係に決定論的。

使い方:
    python experiments/EXP032_single_branch/scr/infer_exp032.py              # 全2構成
    python experiments/EXP032_single_branch/scr/infer_exp032.py --configs SB_eyescale
"""
from __future__ import annotations

import argparse
import csv
import math
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

from eyeliner.data.dataset import EyelinerPatchDataset, load_manifest  # noqa: E402
from eyeliner.data.patch_utils import get_pq_prime_dist  # noqa: E402
from eyeliner.data.weights import compute_eye_meta  # noqa: E402
from eyeliner.models.ensemble import create_ensemble_from_dir  # noqa: E402
from eyeliner.training.dual_branch import (  # noqa: E402
    branch_a_x0,
    branch_b_eye_scale,
    branch_pred_to_raw_t,
)

CONFIGS = {
    "SB_none": "SB_none",
    "SB_eyescale": "SB_eyescale",
}

_BRANCH_BUILDERS = {
    "none": branch_a_x0,
    "eye_scale": branch_b_eye_scale,
}


def infer_one(
    exp_name: str,
    manifest_path: Path,
    output_path: Path,
    num_workers: int = 0,
) -> float:
    """指定 config の fold アンサンブルで推論し、生 t で CSV を保存する。MAE(生t) を返す。"""
    config_name = CONFIGS[exp_name]
    with open(CONFIG_DIR / f"{config_name}.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pq_rescale_mode: str = cfg.get("pq_rescale_mode", "none")
    branch = _BRANCH_BUILDERS[pq_rescale_mode]()
    batch_size: int = cfg.get("batch", 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    records = load_manifest(manifest_path)
    compute_eye_meta(records)  # eye_scale / pq_ratio / weight / val_weight を付与
    print(f"サンプル数: {len(records)}")

    # 生 t への逆変換用: 各サンプルの |PQ|（生）と |P'Q'|（rescale後）
    orig_pq_dists = [
        math.sqrt((float(r["qx"]) - float(r["px"])) ** 2 + (float(r["qy"]) - float(r["py"])) ** 2)
        for r in records
    ]
    pq_prime_dists = [
        get_pq_prime_dist(r, d, pq_rescale_mode, branch.pq_rescale_factor)
        for r, d in zip(records, orig_pq_dists)
    ]

    ds = EyelinerPatchDataset(
        records, augment_data=False, preload_images=True,
        patch_t_min=branch.patch_t_min, patch_t_max=branch.patch_t_max,
        patch_perp_scale=branch.patch_perp_scale, patch_size=branch.patch_size,
        use_t_norm=True,
        label_t_min=branch.label_t_min, label_t_max=branch.label_t_max,
        pq_rescale_mode=branch.pq_rescale_mode, pq_rescale_factor=branch.pq_rescale_factor,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model_dir = OUTPUT_DIR / exp_name
    ensemble, _ = create_ensemble_from_dir(model_dir)
    ensemble = ensemble.to(device)
    ensemble.eval()
    print(f"fold モデル数: {len(ensemble._models)}")

    preds_norm: list[float] = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            is_upper = batch[1].to(device)
            pred = ensemble(x, is_upper)
            preds_norm.extend(pred.squeeze(-1).cpu().tolist())

    preds_norm_t = torch.tensor(preds_norm, dtype=torch.float32).unsqueeze(-1)
    pq_dist_t = torch.tensor(orig_pq_dists, dtype=torch.float32).unsqueeze(-1)
    pq_prime_t = torch.tensor(pq_prime_dists, dtype=torch.float32).unsqueeze(-1)
    t_raw = branch_pred_to_raw_t(preds_norm_t, branch, pq_dist_t, pq_prime_t).squeeze(-1).tolist()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["idx", "image_path", "part_name", "sub_key", "point_index", "t_true", "t_pred"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec, p in zip(records, t_raw):
            writer.writerow({
                "idx": rec.get("idx", ""),
                "image_path": rec.get("image_path", ""),
                "part_name": rec.get("part_name", ""),
                "sub_key": rec.get("sub_key", ""),
                "point_index": rec.get("point_index", ""),
                "t_true": f"{float(rec['t']):.6f}",
                "t_pred": f"{p:.6f}",
            })

    mae = sum(abs(p - float(rec["t"])) for p, rec in zip(t_raw, records)) / max(len(t_raw), 1)
    print(f"推論完了: config={exp_name}  pq_rescale_mode={pq_rescale_mode}")
    print(f"MAE (生t): {mae:.6f}")
    print(f"保存: {output_path}")
    return mae


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="EXP032 単独ブランチ 推論")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS),
                        help="推論する config（省略時: 全2構成）")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="省略時は train/test 両方を推論し、"
                             "train_predictions.csv / test_predictions.csv を出力")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    print(f"\n{'='*64}\n  EXP032 推論\n{'='*64}")
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

    print(f"\n{'='*64}\n  EXP032 MAE サマリ\n{'='*64}")
    for exp_name, mae in results.items():
        print(f"  {exp_name:<15}  MAE={mae:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
