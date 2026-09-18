"""EXP022 推論: デュアルブランチ fold アンサンブルで train/test を推論し
train_predictions.csv / test_predictions.csv を出力。

学習済み model_dir（base_channel/pool_width はモデル自体に保存済みのため CLI 引数不要）に対して
eyeliner.training.dual_inference.infer_dual を呼び出す薄いラッパー。exp022_compare.py がこの
test_predictions.csv（生 t）を使ってベースライン比較を行う。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
CONFIG_DIR = EXP_DIR / "config"

sys.path.insert(0, str(ROOT))
from eyeliner.training.dual_inference import infer_dual  # noqa: E402


def infer_single(
    model_dir: Path,
    manifest_path: Path,
    output_path: Path,
    cfg: dict,
    num_workers: int = 0,
) -> Path:
    """指定 model_dir のデュアルブランチアンサンブルで推論し CSV を保存する。"""
    infer_dual(
        model_dir, manifest_path, output_path,
        fusion_mode=cfg["fusion_mode"],
        rescale_b=cfg.get("pq_rescale_mode_b", "eye_scale"),
        batch_size=cfg["batch"],
        num_workers=num_workers,
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="EXP022 デュアルブランチ推論")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--config", default=None,
                        help="config YAML 名（拡張子なし）。省略時は --model-dir 名と同名の config を使う")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="省略時は train/test 両方を推論し、"
                             "train_predictions.csv / test_predictions.csv を出力")
    parser.add_argument("--output", type=Path, default=None, help="出力 CSV パス（--manifest 指定時のみ有効）")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    config_name = args.config or args.model_dir.name
    with open(CONFIG_DIR / f"{config_name}.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.manifest is not None:
        output = args.output or args.model_dir / "test_predictions.csv"
        infer_single(args.model_dir, args.manifest, output, cfg, args.num_workers)
        return 0

    manifest_dir = EXP_DIR / "manifests"
    infer_single(args.model_dir, manifest_dir / "manifest_train.csv",
                 args.model_dir / "train_predictions.csv", cfg, args.num_workers)
    infer_single(args.model_dir, manifest_dir / "manifest_test.csv",
                 args.model_dir / "test_predictions.csv", cfg, args.num_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
