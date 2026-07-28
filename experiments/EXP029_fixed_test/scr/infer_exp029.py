"""EXP029 推論スクリプト: 5 パターンの fold アンサンブルでテストセットを推論。

使い方:
    python experiments/EXP029_fixed_test/scr/infer_exp029.py              # 全 5 パターン
    python experiments/EXP029_fixed_test/scr/infer_exp029.py --patterns oof_error_hard db_replica
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

PATTERNS = ["oof_error_hard", "db_replica", "random"]
EXTRA_PATTERNS = ["oof_error_easy", "oof_error_half"]
ALL_PATTERNS = PATTERNS + EXTRA_PATTERNS


def infer_one(
    pattern: str,
    manifest_path: Path,
    output_path: Path,
    num_workers: int = 0,
) -> float:
    """指定パターンの fold アンサンブルで推論し CSV を保存する。MAE を返す。"""
    with open(CONFIG_DIR / "baseline.yaml", encoding="utf-8") as f:
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

    model_dir = OUTPUT_DIR / pattern
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
    print(f"推論完了: pattern={pattern}  MAE (生t): {mae:.6f}")
    print(f"保存: {output_path}")
    return mae


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="EXP029 fixed test set 推論")
    parser.add_argument(
        "--patterns", nargs="+", default=PATTERNS,
        choices=ALL_PATTERNS,
        help=f"推論するパターン（省略時: 全 {len(PATTERNS)} パターン）",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    patterns = args.patterns

    print(f"\n{'='*64}\n  EXP029 推論\n{'='*64}")
    results: dict[str, dict[str, float]] = {}
    for pattern in patterns:
        print(f"\n--- Pattern {pattern} ---")
        results[pattern] = {}
        try:
            mae = infer_one(
                pattern,
                MANIFEST_DIR / f"manifest_train_{pattern}.csv",
                OUTPUT_DIR / pattern / "train_predictions.csv",
                args.num_workers,
            )
            results[pattern]["train_mae"] = mae
        except FileNotFoundError as e:
            print(f"  train skip: {e}")

        try:
            mae = infer_one(
                pattern,
                MANIFEST_DIR / f"manifest_test_{pattern}.csv",
                OUTPUT_DIR / pattern / "test_predictions.csv",
                args.num_workers,
            )
            results[pattern]["test_mae"] = mae
        except FileNotFoundError as e:
            print(f"  test skip: {e}")

    print(f"\n{'='*64}\n  EXP029 MAE サマリ\n{'='*64}")
    for pattern, r in results.items():
        train_str = f"train_mae={r['train_mae']:.6f}" if "train_mae" in r else ""
        test_str = f"test_mae={r['test_mae']:.6f}" if "test_mae" in r else ""
        print(f"  Pattern {pattern}  {train_str}  {test_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
