"""EXP020 推論スクリプト: 訓練済み dual アンサンブルでテストセットを推論。

枝A=EXP019 X0（生PQ, raw_t）, 枝B=EXP019 X1/X2（P'Q', t_prime）を忠実に再現した
モデルを読み込み、各枝を生 t 空間へ逆変換してアンサンブルする。出力は生 t。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent.parent.parent
SCR_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCR_DIR))

from eyeliner.data.dataset import load_manifest  # noqa: E402
from eyeliner.data.dataset_dual import EyelinerPatchDualDataset  # noqa: E402
from eyeliner.data.weights import compute_eye_meta  # noqa: E402

from dual_branch_utils import build_branch_configs, fold_ensemble_raw_t  # noqa: E402
from dual_model_io import load_dual_fold_models  # noqa: E402


def infer_dual(
    model_dir: Path,
    manifest_path: Path,
    output_path: Path,
    fusion_mode: str,
    rescale_b: str = "eye_scale",
    batch_size: int = 32,
    num_workers: int = 0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    records = load_manifest(manifest_path)
    compute_eye_meta(records)
    print(f"サンプル数: {len(records)}")

    cfg_a, cfg_b = build_branch_configs(rescale_b)
    ds = EyelinerPatchDualDataset(records, cfg_a, cfg_b, augment_data=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    models = [m.to(device) for m in load_dual_fold_models(model_dir, fusion_mode)]
    print(f"fold モデル数: {len(models)}")

    all_preds: list[float] = []   # 生 t 予測
    label_range_a = cfg_a.label_t_max - cfg_a.label_t_min
    with torch.no_grad():
        for batch in loader:
            x_a = batch[0].to(device)
            x_b = batch[1].to(device)
            is_upper = batch[2].to(device)
            pq_dist = batch[12].to(device)
            pq_prime = batch[13].to(device)
            t_ens = fold_ensemble_raw_t(
                models, x_a, x_b, is_upper, fusion_mode, cfg_a, cfg_b, pq_dist, pq_prime,
            )
            all_preds.extend(t_ens.squeeze(-1).cpu().tolist())

    # 真値は manifest の生 t を使う。batch[4]（y_a）は学習ロス用に枝A の label 範囲
    # [label_t_min, label_t_max] へクランプ済みで、範囲外の t を持つサンプルの真値が頭打ちされてしまう。
    all_trues = [float(rec["t"]) for rec in records]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["idx", "image_path", "part_name", "sub_key", "point_index", "t_true", "t_pred"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec, t_true, t_pred in zip(records, all_trues, all_preds):
            writer.writerow({
                "idx":         rec.get("idx", ""),
                "image_path":  rec["image_path"],
                "part_name":   rec.get("part_name", ""),
                "sub_key":     rec.get("sub_key", ""),
                "point_index": rec.get("point_index", ""),
                "t_true":      f"{t_true:.6f}",
                "t_pred":      f"{t_pred:.6f}",
            })

    mae_raw = sum(abs(pt - tt) for pt, tt in zip(all_preds, all_trues)) / max(len(all_preds), 1)
    print(f"推論完了: MAE={mae_raw:.6f}（生t） / {mae_raw / label_range_a:.6f}（正規化）")
    print(f"保存: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP020 dual 推論")
    parser.add_argument("--model-dir", type=Path, required=True, help="訓練済み models ディレクトリ")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="推論対象 manifest（省略時は train/test 両方を推論し、"
                             "train_predictions.csv / test_predictions.csv を出力）")
    parser.add_argument("--output", type=Path, default=None, help="出力 CSV パス（--manifest 指定時のみ有効）")
    parser.add_argument("--fusion-mode", default="independent",
                        choices=["independent", "crossfuse", "attention"])
    parser.add_argument("--rescale-b", default="eye_scale", choices=["eye_scale", "eye_height"])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.manifest is not None:
        output = args.output or args.model_dir / "test_predictions.csv"
        infer_dual(
            args.model_dir, args.manifest, output,
            args.fusion_mode, args.rescale_b,
            args.batch, args.num_workers,
        )
        return

    manifest_dir = args.model_dir.parent.parent / "manifests"
    infer_dual(
        args.model_dir, manifest_dir / "manifest_train.csv", args.model_dir / "train_predictions.csv",
        args.fusion_mode, args.rescale_b,
        args.batch, args.num_workers,
    )
    infer_dual(
        args.model_dir, manifest_dir / "manifest_test.csv", args.model_dir / "test_predictions.csv",
        args.fusion_mode, args.rescale_b,
        args.batch, args.num_workers,
    )


def create_predictor(models_dir: Path):
    """WebUI server 向け dual predictor を生成する。

    server.py の _load_exp_predictor() が呼び出す。
    experiment.json に fusion_mode がない場合（X0 単一ブランチ）は ValueError を送出して
    server 側をデフォルトの単一ブランチ読み込みにフォールバックさせる。

    Args:
        models_dir: fold モデル (.pt) が格納されたディレクトリ。
                    同ディレクトリの experiment.json から fusion_mode と rescale_b を読む。

    Returns:
        callable: (image_rgb, px, py, qx, qy, is_upper, pq_prime=None) -> t_norm
                  t_norm は WebUI の [T_MIN, T_MAX] 正規化 float。
                  pq_prime は branch B の |P'Q'|（server が eye_scale から計算して渡す）。
    """
    import json as _json
    import math as _math
    import logging as _logging
    import torch
    import numpy as np  # noqa: F401

    from eyeliner.data.patch_utils import (
        IMAGENET_MEAN, IMAGENET_STD, T_MIN, T_MAX, extract_patch,
    )
    from dual_branch_utils import build_branch_configs, model_pred_raw_t
    from dual_model_io import load_dual_fold_models

    _logger = _logging.getLogger(__name__)

    exp_json = models_dir / "experiment.json"
    if not exp_json.exists():
        raise FileNotFoundError(f"experiment.json が見つかりません: {exp_json}")
    with open(exp_json, encoding="utf-8") as f:
        data = _json.load(f)
    hp = data.get("hyperparameters", {})

    if "fusion_mode" not in hp:
        raise ValueError(
            f"デュアルブランチ実験ではありません (fusion_mode なし): {models_dir}"
        )

    fusion_mode = hp["fusion_mode"]
    rescale_b = hp.get("rescale_b", "eye_scale")

    cfg_a, cfg_b = build_branch_configs(rescale_b)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [m.to(device).eval() for m in load_dual_fold_models(models_dir, fusion_mode)]
    _logger.info(
        "dual predictor 初期化: fusion_mode=%s rescale_b=%s folds=%d device=%s",
        fusion_mode, rescale_b, len(models), device,
    )

    mean_t = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(device)
    std_t  = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(device)

    def _to_tensor(patch):
        x = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        return (x - mean_t) / std_t

    def predictor(image_rgb, px, py, qx, qy, is_upper, pq_prime=None):
        try:
            orig_pq = _math.sqrt((qx - px) ** 2 + (qy - py) ** 2)

            # Branch A: raw PQ 座標系
            patch_a = extract_patch(
                image_rgb, px, py, qx, qy,
                patch_size=cfg_a.patch_size,
                t_min=cfg_a.patch_t_min, t_max=cfg_a.patch_t_max,
                perp_scale=cfg_a.patch_perp_scale,
            )
            x_a = _to_tensor(patch_a)

            # Branch B: P'Q' 座標系 (pq_prime = |P'Q'|)
            pq_prime_val = float(pq_prime) if (pq_prime is not None and float(pq_prime) > 1e-6) else orig_pq
            if orig_pq > 1e-6:
                ox = (px + qx) / 2.0
                oy = (py + qy) / 2.0
                ux = (qx - px) / orig_pq
                uy = (qy - py) / orig_pq
                px_b = ox - (pq_prime_val / 2.0) * ux
                py_b = oy - (pq_prime_val / 2.0) * uy
                qx_b = ox + (pq_prime_val / 2.0) * ux
                qy_b = oy + (pq_prime_val / 2.0) * uy
            else:
                px_b, py_b, qx_b, qy_b = px, py, qx, qy

            patch_b = extract_patch(
                image_rgb, px_b, py_b, qx_b, qy_b,
                patch_size=cfg_b.patch_size,
                t_min=cfg_b.patch_t_min, t_max=cfg_b.patch_t_max,
                perp_scale=cfg_b.patch_perp_scale,
            )
            x_b = _to_tensor(patch_b)

            is_upper_t  = torch.tensor([[is_upper]],      dtype=torch.float32, device=device)
            pq_dist_t   = torch.tensor([[orig_pq]],       dtype=torch.float32, device=device)
            pq_prime_t  = torch.tensor([[pq_prime_val]],  dtype=torch.float32, device=device)

            with torch.no_grad():
                raw_preds = [
                    model_pred_raw_t(
                        m, x_a, x_b, is_upper_t,
                        fusion_mode, cfg_a, cfg_b,
                        pq_dist_t, pq_prime_t,
                    )
                    for m in models
                ]
            t_raw = float(torch.stack(raw_preds).mean(0).item())
            return (t_raw - T_MIN) / (T_MAX - T_MIN)
        except Exception as exc:
            _logger.warning("dual predictor 推論失敗: %s", exc)
            return None

    return predictor


if __name__ == "__main__":
    main()
