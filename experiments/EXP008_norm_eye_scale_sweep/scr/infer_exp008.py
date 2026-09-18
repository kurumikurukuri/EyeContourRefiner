"""EXP008: norm_eye_scale 特徴量 sweep 推論スクリプト

各条件の fold モデルをアンサンブルして manifest_train / manifest_test に推論し、
train_predictions.csv / test_predictions.csv を出力する。

EXP008 は EXP006 と同一の manifests を共有しているため、
manifest のパスは EXP006 ディレクトリを参照する。

使い方:
    python experiments/EXP008_norm_eye_scale_sweep/scr/infer_exp008.py
    python experiments/EXP008_norm_eye_scale_sweep/scr/infer_exp008.py --configs J1_baseline_nes J2_wide_patch_nes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
CONFIG_DIR = EXP_DIR / "config"
# EXP006 の manifests を共有（同一 train/test 分割で公平な比較）
EXP006_DIR = ROOT / "experiments" / "EXP006_data_increase_sweep"
MANIFEST_TRAIN = EXP006_DIR / "manifests" / "manifest_train.csv"
MANIFEST_TEST  = EXP006_DIR / "manifests" / "manifest_test.csv"
OUTPUT_DIR     = EXP_DIR / "outputs"
LOG_PATH       = EXP_DIR / "log.txt"

sys.path.insert(0, str(ROOT))
from scripts.exp_utils import run_with_log  # noqa: E402


def load_infer_params(config_name: str) -> dict:
    """config.yaml からパッチ抽出・ラベル正規化に必要なパラメータを取得する。"""
    path = CONFIG_DIR / f"{config_name}.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {
        "patch_t_min":      cfg["patch_t_min"],
        "patch_t_max":      cfg["patch_t_max"],
        "label_t_min":      cfg.get("label_t_min"),
        "label_t_max":      cfg.get("label_t_max"),
        "patch_perp_scale": cfg.get("patch_perp_scale", 1.0),
        "patch_h":          cfg.get("patch_h", 64),
        "patch_w":          cfg.get("patch_w", 128),
    }


def build_infer_cmd(config_name: str, manifest: Path, output_path: Path) -> list[str]:
    p = load_infer_params(config_name)
    models_dir = OUTPUT_DIR / config_name
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "infer.py"),
        "--models-dir",       str(models_dir),
        "--manifest",         str(manifest),
        "--output",           str(output_path),
        "--patch_t_min",      str(p["patch_t_min"]),
        "--patch_t_max",      str(p["patch_t_max"]),
        "--patch_perp_scale", str(p["patch_perp_scale"]),
        "--patch_h",          str(p["patch_h"]),
        "--patch_w",          str(p["patch_w"]),
    ]
    if p["label_t_min"] is not None:
        cmd += ["--label_t_min", str(p["label_t_min"])]
    if p["label_t_max"] is not None:
        cmd += ["--label_t_max", str(p["label_t_max"])]
    return cmd


def run_infer(config_name: str) -> int:
    print(f"\n{'='*60}")
    print(f"  推論: {config_name}")
    print(f"{'='*60}")

    models_dir = OUTPUT_DIR / config_name
    if not models_dir.exists():
        print(f"  [SKIP] モデルディレクトリが存在しません: {models_dir}")
        return 1

    # train 推論
    train_cmd = build_infer_cmd(config_name, MANIFEST_TRAIN,
                                models_dir / "train_predictions.csv")
    print(f"  [train] {' '.join(train_cmd)}")
    rc = run_with_log(train_cmd, LOG_PATH, cwd=ROOT)
    if rc != 0:
        return rc

    # test 推論
    test_cmd = build_infer_cmd(config_name, MANIFEST_TEST,
                               models_dir / "test_predictions.csv")
    print(f"  [test]  {' '.join(test_cmd)}")
    return run_with_log(test_cmd, LOG_PATH, cwd=ROOT)


def main() -> None:
    all_configs = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))

    parser = argparse.ArgumentParser(description="EXP008 推論")
    parser.add_argument(
        "--configs", nargs="+", default=all_configs,
        help=f"実行する条件名（デフォルト: 全条件 {all_configs}）",
    )
    args = parser.parse_args()

    for manifest, label in [(MANIFEST_TRAIN, "manifest_train.csv"),
                            (MANIFEST_TEST,  "manifest_test.csv")]:
        if not manifest.exists():
            print(f"\n[ERROR] {label} が見つかりません: {manifest}")
            print("EXP006 の manifests を先に生成してください:")
            print("  python scripts/split_holdout.py --method random --ratio 0.2 --seed 42 --exp 006")
            sys.exit(1)

    print(f"EXP008 infer開始: {len(args.configs)} 条件")
    print(f"  manifest_train: {MANIFEST_TRAIN}")
    print(f"  manifest_test:  {MANIFEST_TEST}")

    failed = []
    for config_name in args.configs:
        rc = run_infer(config_name)
        if rc != 0:
            print(f"\n[WARN] {config_name} が終了コード {rc} で失敗")
            failed.append(config_name)

    print(f"\n{'='*60}")
    print(f"  EXP008 infer完了: {len(args.configs) - len(failed)}/{len(args.configs)} 成功")
    if failed:
        print(f"  失敗: {failed}")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
