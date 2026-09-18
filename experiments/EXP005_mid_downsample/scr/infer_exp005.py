"""EXP005: 全条件の推論スクリプト

各条件の fold モデルをアンサンブルして manifest_train / manifest_test に推論し、
train_predictions.csv / test_predictions.csv を出力する。

EXP005 は use_sigmoid=True, use_t_norm=True（EXP002ベース）なので
--no_t_norm は不要。t_pred は生の物理 t 値として出力される。

使い方:
    python experiments/EXP005_mid_downsample/scr/infer_exp005.py
    python experiments/EXP005_mid_downsample/scr/infer_exp005.py --configs F1_mid1500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
CONFIG_DIR = EXP_DIR / "config"
MANIFEST_TRAIN = EXP_DIR / "manifests" / "manifest_train.csv"
MANIFEST_TEST  = EXP_DIR / "manifests" / "manifest_test.csv"
OUTPUT_DIR     = EXP_DIR / "outputs"
LOG_PATH       = EXP_DIR / "log.txt"

sys.path.insert(0, str(ROOT))
from scripts.exp_utils import run_with_log  # noqa: E402

# EXP005はEXP002ベース設定（label_t=patch_t=[-0.5,1.5]）を継承
PATCH_T_MIN = -0.5
PATCH_T_MAX =  1.5


def build_infer_cmd(config_name: str, manifest: Path, output_path: Path) -> list[str]:
    models_dir = OUTPUT_DIR / config_name
    return [
        sys.executable,
        str(ROOT / "scripts" / "infer.py"),
        "--models-dir", str(models_dir),
        "--manifest",   str(manifest),
        "--output",     str(output_path),
        "--patch_t_min", str(PATCH_T_MIN),
        "--patch_t_max", str(PATCH_T_MAX),
    ]


def run_infer(config_name: str) -> int:
    print(f"\n{'='*60}")
    print(f"  推論: {config_name}")
    print(f"{'='*60}")

    models_dir = OUTPUT_DIR / config_name
    if not models_dir.exists():
        print(f"  [SKIP] モデルディレクトリが存在しません: {models_dir}")
        return 1

    # train推論（全サンプル manifest を使用 — ダウンサンプル版ではない）
    train_cmd = build_infer_cmd(config_name, MANIFEST_TRAIN,
                                models_dir / "train_predictions.csv")
    print(f"  [train] {' '.join(train_cmd)}")
    rc = run_with_log(train_cmd, LOG_PATH, cwd=ROOT)
    if rc != 0:
        return rc

    # test推論
    test_cmd = build_infer_cmd(config_name, MANIFEST_TEST,
                               models_dir / "test_predictions.csv")
    print(f"  [test]  {' '.join(test_cmd)}")
    return run_with_log(test_cmd, LOG_PATH, cwd=ROOT)


def main() -> None:
    all_configs = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))

    parser = argparse.ArgumentParser(description="EXP005 推論")
    parser.add_argument(
        "--configs", nargs="+", default=all_configs,
        help=f"実行する条件名（デフォルト: 全条件 {all_configs}）",
    )
    args = parser.parse_args()

    for manifest, label in [(MANIFEST_TRAIN, "manifest_train.csv"),
                            (MANIFEST_TEST,  "manifest_test.csv")]:
        if not manifest.exists():
            print(f"\n[ERROR] {label} が見つかりません: {manifest}")
            sys.exit(1)

    print(f"EXP005 infer開始: {len(args.configs)} 条件")

    failed = []
    for config_name in args.configs:
        rc = run_infer(config_name)
        if rc != 0:
            print(f"\n[WARN] {config_name} が終了コード {rc} で失敗")
            failed.append(config_name)

    print(f"\n{'='*60}")
    print(f"  EXP005 infer完了: {len(args.configs) - len(failed)}/{len(args.configs)} 成功")
    if failed:
        print(f"  失敗: {failed}")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
