"""EXP016: 学習曲線実験 推論スクリプト

各条件の fold モデル（k=3）をアンサンブルして
manifest_train / manifest_test に推論し、
train_predictions.csv / test_predictions.csv を outputs/{条件名}/{条件名}/ 以下に出力する。

使い方:
    # outputs/ にモデルが存在する全条件
    python experiments/EXP016_learning_curve/scr/infer_exp016.py

    # 特定条件のみ
    python experiments/EXP016_learning_curve/scr/infer_exp016.py --configs S08 S09
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
CONFIG_DIR = EXP_DIR / "config"
MANIFEST_DIR = EXP_DIR / "manifests"
OUTPUT_DIR = EXP_DIR / "outputs"
LOG_PATH = EXP_DIR / "log.txt"

MANIFEST_TRAIN = MANIFEST_DIR / "manifest_train.csv"
MANIFEST_TEST = MANIFEST_DIR / "manifest_test.csv"

sys.path.insert(0, str(ROOT))
from scripts.exp_utils import run_with_log  # noqa: E402


def load_infer_params(config_name: str) -> dict:
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
        "patch_w":          cfg.get("patch_w", 64),
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
        print(f"         先に train_exp016.py --configs {config_name} を実行してください")
        return 1

    train_out = models_dir / "train_predictions.csv"
    train_cmd = build_infer_cmd(config_name, MANIFEST_TRAIN, train_out)
    print(f"  [train] manifest: {MANIFEST_TRAIN.name} → {train_out.name}")
    rc = run_with_log(train_cmd, LOG_PATH, cwd=ROOT)
    if rc != 0:
        print(f"  [ERROR] train 推論失敗: rc={rc}")
        return rc

    test_out = models_dir / "test_predictions.csv"
    test_cmd = build_infer_cmd(config_name, MANIFEST_TEST, test_out)
    print(f"  [test]  manifest: {MANIFEST_TEST.name} → {test_out.name}")
    return run_with_log(test_cmd, LOG_PATH, cwd=ROOT)


def main() -> None:
    all_configs = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))

    parser = argparse.ArgumentParser(description="EXP016 学習曲線実験 推論")
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="実行する条件名（デフォルト: outputs/ 以下に存在する全条件）",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="outputs/ の存在チェックをスキップして全条件を対象にする",
    )
    args = parser.parse_args()

    if args.configs is not None:
        target_configs = args.configs
    elif args.all:
        target_configs = all_configs
    else:
        target_configs = [
            c for c in all_configs
            if (OUTPUT_DIR / c / c).exists()
        ]
        if not target_configs:
            print("[INFO] 推論対象のモデルが見つかりません。")
            print("       先に train_exp016.py を実行してください。")
            sys.exit(0)

    if not target_configs:
        print("[ERROR] 推論対象の条件が見つかりません。")
        sys.exit(1)

    print(f"\nEXP016 infer 開始: {len(target_configs)} 条件")

    failed = []
    skipped = []
    for config_name in target_configs:
        rc = run_infer(config_name)
        if rc != 0:
            print(f"\n[WARN] {config_name} が終了コード {rc} で失敗/スキップ")
            if rc == 1:
                skipped.append(config_name)
            else:
                failed.append(config_name)

    success = len(target_configs) - len(failed) - len(skipped)
    print(f"\n{'='*60}")
    print(f"  EXP016 infer 完了: {success}/{len(target_configs)} 成功")
    if skipped:
        print(f"  スキップ（モデル未生成）: {skipped}")
    if failed:
        print(f"  失敗: {failed}")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
