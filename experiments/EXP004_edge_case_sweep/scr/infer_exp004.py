"""EXP004: 全条件の推論スクリプト

各条件の outputs/{config_name}/{config_name}/ にある fold モデルを
アンサンブルして manifest_train / manifest_test に対して推論し、
train_predictions.csv / test_predictions.csv を出力する。

EXP004 は use_t_norm=False で学習しているため、--no_t_norm を自動的に付与する。
t_pred は生の物理 t 値として出力される。

使い方:
    # 全条件を実行
    python experiments/EXP004_edge_case_sweep/scr/infer_exp004.py

    # 特定条件のみ実行
    python experiments/EXP004_edge_case_sweep/scr/infer_exp004.py --configs E1_no_norm E2_no_norm_dfc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent  # EyelinerDetecter/
EXP_DIR = Path(__file__).parent.parent             # EXP004_edge_case_sweep/
CONFIG_DIR = EXP_DIR / "config"
MANIFEST_TRAIN = EXP_DIR / "manifests" / "manifest_train.csv"
MANIFEST_TEST  = EXP_DIR / "manifests" / "manifest_test.csv"
OUTPUT_DIR     = EXP_DIR / "outputs"
LOG_PATH       = EXP_DIR / "log.txt"

sys.path.insert(0, str(ROOT))
from scripts.exp_utils import run_with_log  # noqa: E402


def load_config(config_name: str) -> dict:
    path = CONFIG_DIR / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_infer_cmd(config_name: str, manifest: Path, output_path: Path) -> list[str]:
    models_dir = OUTPUT_DIR / config_name / config_name
    cfg = load_config(config_name)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "infer.py"),
        "--models-dir", str(models_dir),
        "--manifest",   str(manifest),
        "--output",     str(output_path),
        "--patch_t_min", str(cfg.get("patch_t_min", -0.5)),
        "--patch_t_max", str(cfg.get("patch_t_max",  1.5)),
        "--patch_perp_scale", str(cfg.get("patch_perp_scale", 1.0)),
        "--patch_h",    str(cfg.get("patch_h", 64)),
        "--patch_w",    str(cfg.get("patch_w", 128)),
    ]

    label_t_min = cfg.get("label_t_min")
    label_t_max = cfg.get("label_t_max")
    if label_t_min is not None:
        cmd += ["--label_t_min", str(label_t_min)]
    if label_t_max is not None:
        cmd += ["--label_t_max", str(label_t_max)]

    # EXP004 固有: t正規化なしで学習したモデルには --no_t_norm を付与
    if not cfg.get("use_t_norm", True):
        cmd.append("--no_t_norm")

    return cmd


def run_infer(config_name: str) -> int:
    print(f"\n{'='*60}")
    print(f"  推論: {config_name}")
    print(f"{'='*60}")

    models_dir = OUTPUT_DIR / config_name / config_name
    if not models_dir.exists():
        print(f"  [SKIP] モデルディレクトリが存在しません: {models_dir}")
        return 1

    # train 推論
    train_out = models_dir / "train_predictions.csv"
    train_cmd = build_infer_cmd(config_name, MANIFEST_TRAIN, train_out)
    print(f"  [train] コマンド: {' '.join(train_cmd)}")
    rc = run_with_log(train_cmd, LOG_PATH, cwd=ROOT)
    if rc != 0:
        return rc

    # test 推論
    test_out = models_dir / "test_predictions.csv"
    test_cmd = build_infer_cmd(config_name, MANIFEST_TEST, test_out)
    print(f"  [test]  コマンド: {' '.join(test_cmd)}")
    return run_with_log(test_cmd, LOG_PATH, cwd=ROOT)


def main() -> None:
    all_configs = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))

    parser = argparse.ArgumentParser(description="EXP004 推論")
    parser.add_argument(
        "--configs", nargs="+", default=all_configs,
        help=f"実行する条件名（デフォルト: 全条件 {all_configs}）",
    )
    args = parser.parse_args()

    for manifest, label in [(MANIFEST_TRAIN, "manifest_train.csv"), (MANIFEST_TEST, "manifest_test.csv")]:
        if not manifest.exists():
            print(f"\n[ERROR] {label} が見つかりません: {manifest}")
            sys.exit(1)

    print(f"EXP004 infer開始: {len(args.configs)} 条件")
    print(f"  manifest_train: {MANIFEST_TRAIN}")
    print(f"  manifest_test:  {MANIFEST_TEST}")
    print(f"  output_dir:     {OUTPUT_DIR}")

    failed = []
    for config_name in args.configs:
        rc = run_infer(config_name)
        if rc != 0:
            print(f"\n[WARN] {config_name} が終了コード {rc} で失敗")
            failed.append(config_name)

    print(f"\n{'='*60}")
    print(f"  EXP004 infer完了: {len(args.configs) - len(failed)}/{len(args.configs)} 成功")
    if failed:
        print(f"  失敗: {failed}")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
