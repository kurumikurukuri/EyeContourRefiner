"""EXP009: small_eye_filter sweep

eye_scale（目の点群バウンディングボックス対角長）が下位20%の目を
manifest から除外したデータセットで学習・推論する。

フィルタリング方針:
    - 閾値は訓練データの eye_scale ユニーク値（眼単位）の20パーセンタイルで決定
    - 同じ閾値を holdout にも適用
    - 眼単位でフィルタリング（同一 image_path + part_name の全点を一括除外）

条件:
    K1_baseline   ← H1_baseline 条件（フィルタあり）
    K2_wide_patch ← H2_wide_patch 条件（フィルタあり）

比較ベースライン:
    H1 → EXP006 H1_baseline
    H2 → EXP006 H2_wide_patch

manifests: EXP006 の manifests をベースにフィルタリングして
           EXP009/manifests/ に保存する

使い方:
    # 全条件を実行（manifests 生成も自動）
    python scripts/sweep.py --exp 009

    # 特定条件のみ実行
    python experiments/EXP009_small_eye_filter_sweep/scr/train_exp009.py --configs K1_baseline
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent.parent.parent  # eyelinerCNN/
EXP_DIR = Path(__file__).parent.parent             # EXP009_small_eye_filter_sweep/
CONFIG_DIR = EXP_DIR / "config"
MANIFEST_DIR = EXP_DIR / "manifests"
OUTPUT_DIR = EXP_DIR / "outputs"
LOG_PATH = EXP_DIR / "log.txt"

EXP006_DIR = ROOT / "experiments" / "EXP006_data_increase_sweep"
EXP006_TRAIN = EXP006_DIR / "manifests" / "manifest_train.csv"
EXP006_TEST  = EXP006_DIR / "manifests" / "manifest_test.csv"

MANIFEST_TRAIN = MANIFEST_DIR / "manifest_train.csv"
MANIFEST_TEST  = MANIFEST_DIR / "manifest_test.csv"

FILTER_PERCENTILE = 20.0  # 下位 N% を除外

sys.path.insert(0, str(ROOT))
from scripts.exp_utils import run_with_log          # noqa: E402
from eyeliner.data.dataset import load_manifest     # noqa: E402
from eyeliner.data.weights import compute_eye_meta  # noqa: E402

# manifest の保存時に使うカラム順（元 CSV と揃える）
MANIFEST_COLUMNS = ["idx", "image_path", "face_index", "part_name", "sub_key", "point_index",
                    "px", "py", "qx", "qy", "t"]


# ---------------------------------------------------------------------------
# manifest フィルタリング
# ---------------------------------------------------------------------------

def compute_eye_scale_threshold(records: list[dict], percentile: float) -> float:
    """訓練レコードから eye_scale の閾値を算出する。

    眼単位（image_path + face_index + part_name）でユニークな eye_scale 値を集め、
    その percentile を閾値とする。
    """
    seen: dict[tuple, float] = {}
    for rec in records:
        key = (rec["image_path"], rec.get("face_index", 0), rec["part_name"])
        if key not in seen:
            seen[key] = float(rec["eye_scale"])
    unique_scales = list(seen.values())
    threshold = float(np.percentile(unique_scales, percentile))
    return threshold


def filter_by_eye_scale(records: list[dict], threshold: float) -> list[dict]:
    """eye_scale >= threshold のレコードのみ残す（眼単位フィルタリング）。"""
    return [rec for rec in records if float(rec["eye_scale"]) >= threshold]


def save_manifest(records: list[dict], path: Path) -> None:
    """レコードリストを manifest CSV として保存する（元のカラムのみ）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def create_filtered_manifests() -> tuple[int, int, float]:
    """EXP006 manifests を読み込み、eye_scale 下位 N% を除外して保存する。

    Returns:
        (n_train_after, n_test_after, threshold)
    """
    print(f"\n[フィルタリング] EXP006 manifests を読み込み中...")
    train_records = load_manifest(EXP006_TRAIN)
    test_records  = load_manifest(EXP006_TEST)

    # 訓練データで eye_scale を計算（閾値決定用）
    compute_eye_meta(train_records)
    threshold = compute_eye_scale_threshold(train_records, FILTER_PERCENTILE)

    n_train_before = len(train_records)
    n_test_before  = len(test_records)

    # 訓練データをフィルタリング
    train_filtered = filter_by_eye_scale(train_records, threshold)

    # holdout も同じ閾値でフィルタリング（eye_scale を計算してから）
    compute_eye_meta(test_records)
    test_filtered = filter_by_eye_scale(test_records, threshold)

    # 保存
    save_manifest(train_filtered, MANIFEST_TRAIN)
    save_manifest(test_filtered,  MANIFEST_TEST)

    n_train_after = len(train_filtered)
    n_test_after  = len(test_filtered)

    print(f"  eye_scale 閾値 (p{FILTER_PERCENTILE:.0f}): {threshold:.2f} px")
    print(f"  train: {n_train_before} → {n_train_after} "
          f"({n_train_before - n_train_after} 点除外, "
          f"{100*(n_train_before - n_train_after)/n_train_before:.1f}%)")
    print(f"  test:  {n_test_before} → {n_test_after} "
          f"({n_test_before - n_test_after} 点除外, "
          f"{100*(n_test_before - n_test_after)/n_test_before:.1f}%)")

    return n_train_after, n_test_after, threshold


# ---------------------------------------------------------------------------
# 実験実行
# ---------------------------------------------------------------------------

def load_config(config_name: str) -> dict:
    path = CONFIG_DIR / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_cmd(config_name: str, cfg: dict) -> list[str]:
    """config dict から train.py 呼び出しコマンドを構築する。"""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--manifest",        str(MANIFEST_TRAIN),
        "--holdout",         str(MANIFEST_TEST),
        "--output_dir",      str(OUTPUT_DIR),
        "--exp_name",        config_name,
        "--backbone",        cfg["backbone"],
        "--loss_fn",         cfg["loss_fn"],
        "--lr",              str(cfg["lr"]),
        "--weight_decay",    str(cfg["weight_decay"]),
        "--epochs",          str(cfg["epochs"]),
        "--batch",           str(cfg["batch"]),
        "--k",               str(cfg["k"]),
        "--patience",        str(cfg["patience"]),
        "--seed",            str(cfg["seed"]),
        "--num_conv_layers", str(cfg.get("num_conv_layers", 3)),
        "--freeze_epochs",   str(cfg.get("freeze_epochs", 0)),
        "--aug_flags",       str(cfg.get("aug_flags", 79)),
        "--patch_t_min",     str(cfg["patch_t_min"]),
        "--patch_t_max",     str(cfg["patch_t_max"]),
        "--patch_perp_scale", str(cfg["patch_perp_scale"]),
        "--patch_h",         str(cfg["patch_h"]),
        "--patch_w",         str(cfg["patch_w"]),
    ]

    if cfg["loss_fn"] == "dead_zone":
        cmd += ["--loss_epsilon", str(cfg.get("loss_epsilon", 0.01))]
        cmd += ["--loss_delta",   str(cfg.get("loss_delta",   0.05))]

    label_t_min = cfg.get("label_t_min")
    label_t_max = cfg.get("label_t_max")
    if label_t_min is not None:
        cmd += ["--label_t_min", str(label_t_min)]
    if label_t_max is not None:
        cmd += ["--label_t_max", str(label_t_max)]

    if cfg.get("use_pq_ratio"):
        cmd.append("--use_pq_ratio")
    if cfg.get("use_norm_eye_scale"):
        cmd.append("--use_norm_eye_scale")
    if cfg.get("use_sigmoid"):
        cmd.append("--use_sigmoid")
    if not cfg.get("use_batchnorm", True):
        cmd.append("--no_batchnorm")
    if cfg.get("use_t_weighted_sampler"):
        cmd.append("--use_t_weighted_sampler")
        cmd += ["--t_weight_alpha", str(cfg.get("t_weight_alpha", 3.0))]
        cmd += ["--t_loss_alpha",   str(cfg.get("t_loss_alpha",   3.0))]

    return cmd


def run_experiment(config_name: str) -> int:
    print(f"\n{'='*60}")
    print(f"  実行: {config_name}")
    print(f"{'='*60}")

    cfg = load_config(config_name)
    cmd = build_train_cmd(config_name, cfg)

    output_dir = OUTPUT_DIR / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  コマンド: {' '.join(cmd)}")
    return run_with_log(cmd, LOG_PATH, cwd=ROOT)


def main() -> None:
    all_configs = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))

    parser = argparse.ArgumentParser(description="EXP009 small_eye_filter sweep")
    parser.add_argument(
        "--configs", nargs="+", default=all_configs,
        help=f"実行する設定名（デフォルト: 全条件 {all_configs}）",
    )
    args = parser.parse_args()

    # 前提チェック
    if not EXP006_TRAIN.exists():
        print(f"\n[ERROR] EXP006 manifest_train.csv が見つかりません: {EXP006_TRAIN}")
        print("先に EXP006 の分割を実行してください:")
        print("  python scripts/split_holdout.py --method random --ratio 0.2 --seed 42 --exp 006")
        sys.exit(1)

    # フィルタリング済み manifests が未生成なら作成
    if not MANIFEST_TRAIN.exists() or not MANIFEST_TEST.exists():
        create_filtered_manifests()
    else:
        print(f"\n[スキップ] フィルタリング済み manifests は既に存在します: {MANIFEST_DIR}")

    print(f"\nEXP009 sweep開始: {len(args.configs)} 条件")
    print(f"  manifest_train: {MANIFEST_TRAIN}")
    print(f"  manifest_test:  {MANIFEST_TEST}")
    print(f"  output_dir:     {OUTPUT_DIR}")
    print(f"  log:            {LOG_PATH}")

    failed = []
    for config_name in args.configs:
        rc = run_experiment(config_name)
        if rc != 0:
            print(f"\n[WARN] {config_name} が終了コード {rc} で失敗")
            failed.append(config_name)

    print(f"\n{'='*60}")
    print(f"  EXP009 sweep完了: {len(args.configs) - len(failed)}/{len(args.configs)} 成功")
    if failed:
        print(f"  失敗: {failed}")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
