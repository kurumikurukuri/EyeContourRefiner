"""EXP002: 全条件の推論スクリプト

各条件の outputs/{config_name}/{config_name}/ にある fold モデルを
アンサンブルして manifest_train / manifest_test に対して推論し、
train_predictions.csv / test_predictions.csv を出力する。

使い方:
    # 全条件を実行
    python experiments/EXP002_sig_tloss_sampler_sweep/scr/infer_exp002.py

    # 特定条件のみ実行
    python experiments/EXP002_sig_tloss_sampler_sweep/scr/infer_exp002.py --configs sig_tloss sampler_a3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent  # EyelinerDetecter/
EXP_DIR = Path(__file__).parent.parent             # EXP002_sig_tloss_sampler_sweep/
MANIFEST_TRAIN = EXP_DIR / "manifests" / "manifest_train.csv"
MANIFEST_TEST  = EXP_DIR / "manifests" / "manifest_test.csv"
OUTPUT_DIR     = EXP_DIR / "outputs"
LOG_PATH       = EXP_DIR / "log.txt"

sys.path.insert(0, str(ROOT))
from scripts.exp_utils import run_with_log  # noqa: E402


def load_patch_params(config_name: str) -> dict:
    """experiment.json からパッチパラメータを取得する。"""
    exp_json = OUTPUT_DIR / config_name / config_name / "experiment.json"
    if not exp_json.exists():
        return {"patch_t_min": -0.5, "patch_t_max": 1.5, "label_t_min": None, "label_t_max": None}
    with open(exp_json, encoding="utf-8") as f:
        data = json.load(f)
    hp = data.get("hyperparameters", {})
    return {
        "patch_t_min":  hp.get("patch_t_min",  -0.5),
        "patch_t_max":  hp.get("patch_t_max",   1.5),
        "label_t_min":  hp.get("label_t_min",   None),
        "label_t_max":  hp.get("label_t_max",   None),
    }


def build_infer_cmd(config_name: str, manifest: Path, output_path: Path) -> list[str]:
    models_dir = OUTPUT_DIR / config_name / config_name
    params     = load_patch_params(config_name)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "infer.py"),
        "--models-dir", str(models_dir),
        "--manifest",   str(manifest),
        "--output",     str(output_path),
        "--patch_t_min", str(params["patch_t_min"]),
        "--patch_t_max", str(params["patch_t_max"]),
    ]
    if params["label_t_min"] is not None:
        cmd += ["--label_t_min", str(params["label_t_min"])]
    if params["label_t_max"] is not None:
        cmd += ["--label_t_max", str(params["label_t_max"])]
    return cmd


def run_infer(config_name: str) -> int:
    print(f"\n{'='*60}")
    print(f"  推論: {config_name}")
    print(f"{'='*60}")

    models_dir = OUTPUT_DIR / config_name / config_name

    # train 推論
    train_cmd = build_infer_cmd(config_name, MANIFEST_TRAIN, models_dir / "train_predictions.csv")
    print(f"  [train] コマンド: {' '.join(train_cmd)}")
    rc = run_with_log(train_cmd, LOG_PATH, cwd=ROOT)
    if rc != 0:
        return rc

    # test 推論
    test_cmd = build_infer_cmd(config_name, MANIFEST_TEST, models_dir / "test_predictions.csv")
    print(f"  [test]  コマンド: {' '.join(test_cmd)}")
    return run_with_log(test_cmd, LOG_PATH, cwd=ROOT)


def main() -> None:
    all_configs = sorted(
        p.name for p in OUTPUT_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )

    parser = argparse.ArgumentParser(description="EXP002 推論")
    parser.add_argument(
        "--configs", nargs="+", default=all_configs,
        help=f"実行する条件名（デフォルト: 全条件 {all_configs}）",
    )
    args = parser.parse_args()

    for manifest, label in [(MANIFEST_TRAIN, "manifest_train.csv"), (MANIFEST_TEST, "manifest_test.csv")]:
        if not manifest.exists():
            print(f"\n[ERROR] {label} が見つかりません: {manifest}")
            sys.exit(1)

    print(f"EXP002 infer開始: {len(args.configs)} 条件")
    print(f"  manifest_test: {MANIFEST_TEST}")
    print(f"  output_dir:    {OUTPUT_DIR}")

    failed = []
    for config_name in args.configs:
        rc = run_infer(config_name)
        if rc != 0:
            print(f"\n[WARN] {config_name} が終了コード {rc} で失敗")
            failed.append(config_name)

    print(f"\n{'='*60}")
    print(f"  EXP002 infer完了: {len(args.configs) - len(failed)}/{len(args.configs)} 成功")
    if failed:
        print(f"  失敗: {failed}")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
