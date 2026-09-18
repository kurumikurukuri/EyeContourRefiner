"""EXP001: 損失関数・補助入力 sweep - 推論スクリプト

全条件（config/*.yaml）のモデルに対して manifest_train.csv / manifest_test.csv を推論し、
train_predictions.csv / test_predictions.csv を outputs/{config_name}/{config_name}/ に保存する。

使い方:
    # 全条件を推論
    python experiments/EXP001_loss_aux_sweep/scr/infer_exp001.py

    # 特定条件のみ推論
    python experiments/EXP001_loss_aux_sweep/scr/infer_exp001.py --configs A1_baseline A2_t_loss
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent  # eyelinerCNN/
EXP_DIR = Path(__file__).parent.parent             # EXP001_loss_aux_sweep/
CONFIG_DIR = EXP_DIR / "config"
MANIFEST_TRAIN = EXP_DIR / "manifests" / "manifest_train.csv"
MANIFEST_TEST  = EXP_DIR / "manifests" / "manifest_test.csv"
OUTPUT_DIR = EXP_DIR / "outputs"
LOG_PATH = EXP_DIR / "log.txt"

sys.path.insert(0, str(ROOT))
from scripts.exp_utils import run_with_log, find_model_dir  # noqa: E402


def load_experiment_params(config_name: str) -> dict:
    """experiment.json からハイパーパラメータを読み込む。"""
    exp_json = OUTPUT_DIR / config_name / "experiment.json"
    if not exp_json.exists():
        raise FileNotFoundError(
            f"experiment.json が見つかりません: {exp_json}\n"
            f"先に train_exp001.py で {config_name} を訓練してください。"
        )
    with open(exp_json, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("hyperparameters", {})


def build_infer_cmd(config_name: str, params: dict, manifest: Path, output_csv: Path) -> list[str]:
    """hyperparameters dict から infer.py 呼び出しコマンドを構築する。"""
    models_dir = OUTPUT_DIR / config_name

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "infer.py"),
        "--models-dir", str(models_dir),
        "--manifest",   str(manifest),
        "--output",     str(output_csv),
        "--patch_t_min", str(params.get("patch_t_min", -0.5)),
        "--patch_t_max", str(params.get("patch_t_max",  1.5)),
        "--patch_h",     str(params.get("patch_h",  64)),
        "--patch_w",     str(params.get("patch_w", 128)),
    ]

    label_t_min = params.get("label_t_min")
    label_t_max = params.get("label_t_max")
    if label_t_min is not None:
        cmd += ["--label_t_min", str(label_t_min)]
    if label_t_max is not None:
        cmd += ["--label_t_max", str(label_t_max)]

    return cmd


def run_infer(config_name: str) -> int:
    print(f"\n{'='*60}")
    print(f"  推論: {config_name}")
    print(f"{'='*60}")

    try:
        params = load_experiment_params(config_name)
        find_model_dir(OUTPUT_DIR / config_name)  # モデル存在確認
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return 1

    models_dir = OUTPUT_DIR / config_name

    # train 推論
    train_cmd = build_infer_cmd(config_name, params, MANIFEST_TRAIN, models_dir / "train_predictions.csv")
    print(f"  [train] コマンド: {' '.join(train_cmd)}")
    rc = run_with_log(train_cmd, LOG_PATH, cwd=ROOT)
    if rc != 0:
        return rc

    # test 推論
    test_cmd = build_infer_cmd(config_name, params, MANIFEST_TEST, models_dir / "test_predictions.csv")
    print(f"  [test]  コマンド: {' '.join(test_cmd)}")
    return run_with_log(test_cmd, LOG_PATH, cwd=ROOT)


def main() -> None:
    all_configs = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))

    parser = argparse.ArgumentParser(description="EXP001 推論")
    parser.add_argument(
        "--configs", nargs="+", default=all_configs,
        help=f"推論する設定名（デフォルト: 全条件 {all_configs}）",
    )
    args = parser.parse_args()

    print(f"EXP001 推論開始: {len(args.configs)} 条件")
    print(f"  manifest_test: {MANIFEST_TEST}")
    print(f"  output_dir:    {OUTPUT_DIR}")

    for manifest, label in [(MANIFEST_TRAIN, "manifest_train.csv"), (MANIFEST_TEST, "manifest_test.csv")]:
        if not manifest.exists():
            print(f"\n[ERROR] {label} が見つかりません: {manifest}")
            print("先に以下を実行してください:")
            print("  python scripts/split_holdout.py --method random --ratio 0.2 --seed 42 --exp 001")
            sys.exit(1)

    failed = []
    for config_name in args.configs:
        rc = run_infer(config_name)
        if rc != 0:
            print(f"\n[WARN] {config_name} が終了コード {rc} で失敗")
            failed.append(config_name)

    print(f"\n{'='*60}")
    print(f"  EXP001 推論完了: {len(args.configs) - len(failed)}/{len(args.configs)} 成功")
    if failed:
        print(f"  失敗: {failed}")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
