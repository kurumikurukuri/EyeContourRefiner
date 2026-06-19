"""EXP021 推論: 単一ブランチ fold アンサンブルでテストセットを推論し生 t を出力。

none 系統     : pred(正規化) → denorm = 生 t。
eye_scale 系統: pred(正規化) → denorm = t' → invert_pq_rescale_t で生 t に逆変換。

出力 test_predictions.csv / train_predictions.csv の (t_true, t_pred) はいずれも生 t。
exp021_band_mae.py がバンド別 MAE と paired t-test に使う。
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from eyeliner.data.dataset import EyelinerPatchDataset, load_manifest  # noqa: E402
from eyeliner.data.weights import compute_eye_meta  # noqa: E402
from eyeliner.data.patch_utils import (  # noqa: E402
    get_pq_prime_dist, invert_pq_rescale_t, get_norm_stats,
)
from eyeliner.models.ensemble import create_ensemble_from_dir  # noqa: E402


def infer_single(
    model_dir: Path,
    manifest_path: Path,
    output_path: Path,
    cfg: dict,
    batch_size: int = 64,
    num_workers: int = 0,
) -> float:
    """単一ブランチ推論を実行し CSV を保存、生 t MAE を返す。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = load_manifest(manifest_path)
    compute_eye_meta(records)

    norm_mean, norm_std = get_norm_stats("imagenet")
    ds = EyelinerPatchDataset(
        records, augment_data=False,
        norm_mean=norm_mean, norm_std=norm_std,
        patch_t_min=cfg["patch_t_min"], patch_t_max=cfg["patch_t_max"],
        patch_perp_scale=cfg["patch_perp_scale"],
        patch_size=(cfg["patch_h"], cfg["patch_w"]),
        use_t_norm=True,
        label_t_min=cfg["label_t_min"], label_t_max=cfg["label_t_max"],
        pq_rescale_mode=cfg.get("pq_rescale_mode", "none"),
        pq_rescale_factor=cfg.get("pq_rescale_factor", 1.0),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    ensemble, _ = create_ensemble_from_dir(model_dir)
    ensemble.eval().to(device)

    preds_norm: list[float] = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            is_upper = batch[1].to(device)
            pred = ensemble(x, is_upper)
            preds_norm.extend(pred.squeeze(-1).cpu().tolist())

    label_min = float(cfg["label_t_min"])
    label_max = float(cfg["label_t_max"])
    rng = label_max - label_min
    mode = cfg.get("pq_rescale_mode", "none")
    factor = float(cfg.get("pq_rescale_factor", 1.0))

    rows = []
    total_err = 0.0
    for rec, pn in zip(records, preds_norm):
        val = pn * rng + label_min  # 生 t（none） or t'（eye_scale）
        if mode != "none":
            orig_pq = math.hypot(
                float(rec["qx"]) - float(rec["px"]),
                float(rec["qy"]) - float(rec["py"]),
            )
            ppd = get_pq_prime_dist(rec, orig_pq, mode, factor)
            t_pred = invert_pq_rescale_t(val, orig_pq, ppd)
        else:
            t_pred = val
        t_true = float(rec["t"])
        total_err += abs(t_pred - t_true)
        rows.append((rec, t_true, t_pred))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["idx", "image_path", "part_name", "sub_key", "point_index", "t_true", "t_pred"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec, t_true, t_pred in rows:
            writer.writerow({
                "idx":         rec.get("idx", ""),
                "image_path":  rec["image_path"],
                "part_name":   rec.get("part_name", ""),
                "sub_key":     rec.get("sub_key", ""),
                "point_index": rec.get("point_index", ""),
                "t_true":      f"{t_true:.6f}",
                "t_pred":      f"{t_pred:.6f}",
            })

    mae_raw = total_err / max(len(rows), 1)
    print(f"  推論: {output_path.parent.name}  生t MAE={mae_raw:.6f}  (n={len(rows)})")
    return mae_raw


def main() -> None:
    import yaml
    parser = argparse.ArgumentParser(description="EXP021 単一ブランチ推論")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="base_*.yaml")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="省略時は train/test 両方を推論し、"
                             "train_predictions.csv / test_predictions.csv を出力")
    parser.add_argument("--output", type=Path, default=None, help="出力 CSV パス（--manifest 指定時のみ有効）")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.manifest is not None:
        output = args.output or args.model_dir / "test_predictions.csv"
        infer_single(args.model_dir, args.manifest, output, cfg, args.batch, args.num_workers)
        return

    manifest_dir = args.model_dir.parent.parent / "manifests"
    infer_single(args.model_dir, manifest_dir / "manifest_train.csv",
                 args.model_dir / "train_predictions.csv", cfg, args.batch, args.num_workers)
    infer_single(args.model_dir, manifest_dir / "manifest_test.csv",
                 args.model_dir / "test_predictions.csv", cfg, args.batch, args.num_workers)


if __name__ == "__main__":
    main()
