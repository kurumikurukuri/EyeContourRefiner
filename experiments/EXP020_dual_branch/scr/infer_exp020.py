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

from eyeliner.training.dual_branch import build_branch_configs, fold_ensemble_raw_t  # noqa: E402
from eyeliner.models.dual_ensemble import load_dual_fold_models  # noqa: E402


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
    """WebUI server 向け dual predictor を生成する（共有実装に委譲）。

    server.py の _load_exp_predictor() が呼び出す。実体は
    eyeliner.models.dual_predictor.build_dual_predictor — EXP020/EXP021以降の
    dual 実験すべてで共通利用するため切り出し済み。
    experiment.json に fusion_mode がない場合（X0 単一ブランチ）は ValueError を送出して
    server 側をデフォルトの単一ブランチ読み込みにフォールバックさせる。
    """
    from eyeliner.models.dual_predictor import build_dual_predictor
    return build_dual_predictor(models_dir)


if __name__ == "__main__":
    main()
