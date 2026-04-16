"""EXP011: ハイパーパラメータ sweep 推論スクリプト

各条件の fold モデル（k=1 なので fold1.pt のみ）をアンサンブルして
manifest_train / manifest_test に推論し、
train_predictions.csv / test_predictions.csv を outputs/{条件名}/{条件名}/ 以下に出力する。

出力される CSV:
    - test_predictions.csv  : holdout セットの予測
    - train_predictions.csv : 訓練セットの予測（過学習チェック用）

使い方:
    # 全条件
    python experiments/EXP011_hparam_sweep/scr/infer_exp011.py

    # 特定条件のみ
    python experiments/EXP011_hparam_sweep/scr/infer_exp011.py --configs A1_twl05 B3_tws2

    # グループまとめて
    python experiments/EXP011_hparam_sweep/scr/infer_exp011.py --group A B
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
CONFIG_DIR = EXP_DIR / "config"
MANIFEST_TRAIN = EXP_DIR / "manifests" / "manifest_train.csv"
MANIFEST_TEST  = EXP_DIR / "manifests" / "manifest_test.csv"
OUTPUT_DIR     = EXP_DIR / "outputs"
LOG_PATH       = EXP_DIR / "log.txt"

_GROUP_PREFIXES = ["M", "A", "B", "C", "D", "E", "F", "G", "H", "J", "L"]

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
    models_dir = OUTPUT_DIR / config_name / config_name
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

    models_dir = OUTPUT_DIR / config_name / config_name
    if not models_dir.exists():
        print(f"  [SKIP] モデルディレクトリが存在しません: {models_dir}")
        print(f"         先に train_exp011.py --configs {config_name} を実行してください")
        return 1

    # train 推論（過学習チェック用）
    train_out = models_dir / "train_predictions.csv"
    train_cmd = build_infer_cmd(config_name, MANIFEST_TRAIN, train_out)
    print(f"  [train] → {train_out.name}")
    rc = run_with_log(train_cmd, LOG_PATH, cwd=ROOT)
    if rc != 0:
        print(f"  [ERROR] train 推論失敗: rc={rc}")
        return rc

    # test 推論（holdout 評価）
    test_out = models_dir / "test_predictions.csv"
    test_cmd = build_infer_cmd(config_name, MANIFEST_TEST, test_out)
    print(f"  [test]  → {test_out.name}")
    return run_with_log(test_cmd, LOG_PATH, cwd=ROOT)


def filter_by_group(configs: list[str], group: str) -> list[str]:
    prefix = group.upper()
    return [c for c in configs if c.upper().startswith(prefix)]


def main() -> None:
    all_configs = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))

    parser = argparse.ArgumentParser(description="EXP011 推論")
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="実行する条件名（デフォルト: outputs/ 以下に存在する全条件）",
    )
    parser.add_argument(
        "--group", nargs="+", default=None,
        metavar="GROUP",
        help=f"グループ文字で絞り込み（例: --group A B C）。選択肢: {_GROUP_PREFIXES}",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="outputs/ の存在チェックをスキップして全条件を対象にする",
    )
    args = parser.parse_args()

    # 実行対象の決定
    if args.configs is not None:
        target_configs = args.configs
    elif args.group is not None:
        target_configs = []
        for g in args.group:
            target_configs += filter_by_group(all_configs, g)
        target_configs = sorted(set(target_configs))
    elif args.all:
        target_configs = all_configs
    else:
        # デフォルト: outputs/ にモデルが存在する条件だけ推論
        target_configs = [
            c for c in all_configs
            if (OUTPUT_DIR / c / c).exists()
        ]
        if not target_configs:
            print("[INFO] 推論対象のモデルが見つかりません。")
            print("       先に train_exp011.py を実行してください。")
            sys.exit(0)

    if not target_configs:
        print("[ERROR] 推論対象の条件が見つかりません。")
        sys.exit(1)

    for manifest, label in [(MANIFEST_TRAIN, "manifest_train.csv"),
                            (MANIFEST_TEST,  "manifest_test.csv")]:
        if not manifest.exists():
            print(f"\n[ERROR] {label} が見つかりません: {manifest}")
            sys.exit(1)

    print(f"\nEXP011 infer 開始: {len(target_configs)} 条件")
    print(f"  manifest_train : {MANIFEST_TRAIN}")
    print(f"  manifest_test  : {MANIFEST_TEST}")

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
    print(f"  EXP011 infer 完了: {success}/{len(target_configs)} 成功")
    if skipped:
        print(f"  スキップ（モデル未生成）: {skipped}")
    if failed:
        print(f"  失敗: {failed}")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
