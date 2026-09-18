"""EXP020 推論スクリプト: 訓練済み dual アンサンブルでテストセットを推論。

枝A=EXP019 X0（生PQ, raw_t）, 枝B=EXP019 X1/X2（P'Q', t_prime）を忠実に再現した
モデルを読み込み、各枝を生 t 空間へ逆変換してアンサンブルする。出力は生 t。

推論本体は eyeliner.training.dual_inference.infer_dual（EXP021/EXP022/EXP024 と共有）。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
SCR_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCR_DIR))

from eyeliner.training.dual_inference import infer_dual  # noqa: E402


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

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

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
