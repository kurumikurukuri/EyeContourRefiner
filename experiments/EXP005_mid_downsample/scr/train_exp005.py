"""EXP005: mid領域ダウンサンプリング sweep

訓練データのmid帯（デフォルト 0.2 ≤ t ≤ 0.8）を指定数までランダムサンプリングし、
エッジ帯（t<0.2, t>0.8）の訓練比率を高めてエッジケース学習を改善する。

EXP004の分析から:
  現状: edge_low=3.1% / mid=71.9% / edge_high=25.0% (total 3150)
  midを減らして edge への exposure を増やす

sweep 軸: mid_max (mid帯の最大サンプル数)
    F1_mid1500: mid=1500 → edge_low=4.1%, mid=62.9%, edge_high=33.0%
    F2_mid1000: mid=1000 → edge_low=5.2%, mid=53.0%, edge_high=41.7%
    F3_mid700:  mid=700  → edge_low=6.2%, mid=44.1%, edge_high=49.6%

ベース設定: EXP002 sig_tloss (CV=0.064264 / Holdout raw=0.1285)
テストセット: EXP004と共通のmanifest_test.csv（seed=42, ratio=0.2）

使い方:
    python experiments/EXP005_mid_downsample/scr/train_exp005.py
    python experiments/EXP005_mid_downsample/scr/train_exp005.py --configs F1_mid1500 F2_mid1000
    python scripts/sweep.py --exp 005
"""
from __future__ import annotations

import argparse
import csv
import random
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

sys.path.insert(0, str(ROOT))
from scripts.exp_utils import run_with_log  # noqa: E402


def load_config(config_name: str) -> dict:
    path = CONFIG_DIR / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_downsampled_manifest(
    src_path: Path,
    dst_path: Path,
    mid_max: int,
    mid_t_low: float,
    mid_t_high: float,
    seed: int,
    t_filter_min: float | None = None,
    t_filter_max: float | None = None,
) -> tuple[int, int, int]:
    """mid帯をダウンサンプリングした訓練manifest を dst_path に書き出す。

    t_filter_min / t_filter_max が指定された場合、その範囲外のサンプルを除外する。

    Returns:
        (n_edge_low, n_mid, n_edge_high) — 書き出したサンプル数
    """
    edge_low, mid, edge_high = [], [], []
    with open(src_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            try:
                t = float(row["t"])
            except (ValueError, KeyError):
                continue
            # 外れ値フィルタ
            if t_filter_min is not None and t < t_filter_min:
                continue
            if t_filter_max is not None and t > t_filter_max:
                continue
            if t < mid_t_low:
                edge_low.append(row)
            elif t > mid_t_high:
                edge_high.append(row)
            else:
                mid.append(row)

    rng = random.Random(seed)
    if len(mid) > mid_max:
        mid = rng.sample(mid, mid_max)

    rows = edge_low + mid + edge_high
    rng.shuffle(rows)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(edge_low), len(mid), len(edge_high)


def build_train_cmd(config_name: str, cfg: dict, manifest_path: Path) -> list[str]:
    output_dir = OUTPUT_DIR / config_name
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--manifest",        str(manifest_path),
        "--holdout",         str(MANIFEST_TEST),
        "--output_dir",      str(output_dir),
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
    ]
    if cfg.get("use_pq_ratio"):
        cmd.append("--use_pq_ratio")
    if cfg.get("use_sigmoid"):
        cmd.append("--use_sigmoid")
    if not cfg.get("use_batchnorm", True):
        cmd.append("--no_batchnorm")
    return cmd


def run_experiment(config_name: str) -> int:
    print(f"\n{'='*60}")
    print(f"  実行: {config_name}")
    print(f"{'='*60}")

    cfg = load_config(config_name)
    mid_max      = cfg.get("mid_max", 99999)
    mid_t_low    = cfg.get("mid_t_low",     0.2)
    mid_t_high   = cfg.get("mid_t_high",    0.8)
    seed         = cfg.get("seed",          42)
    t_filter_min = cfg.get("t_filter_min",  None)
    t_filter_max = cfg.get("t_filter_max",  None)

    # ダウンサンプル済みmanifestをconfig別に保存
    ds_manifest = EXP_DIR / "manifests" / f"manifest_train_{config_name}.csv"
    n_low, n_mid, n_high = make_downsampled_manifest(
        MANIFEST_TRAIN, ds_manifest, mid_max, mid_t_low, mid_t_high, seed,
        t_filter_min=t_filter_min, t_filter_max=t_filter_max,
    )
    total = n_low + n_mid + n_high
    filter_desc = ""
    if t_filter_min is not None:
        filter_desc += f" t>={t_filter_min}"
    if t_filter_max is not None:
        filter_desc += f" t<={t_filter_max}"
    print(f"  フィルタ:{filter_desc or ' なし'}  ダウンサンプル後: "
          f"edge_low={n_low}({100*n_low/total:.1f}%) "
          f"mid={n_mid}({100*n_mid/total:.1f}%) "
          f"edge_high={n_high}({100*n_high/total:.1f}%) total={total}")

    output_dir = OUTPUT_DIR / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_train_cmd(config_name, cfg, ds_manifest)
    print(f"  コマンド: {' '.join(cmd)}")
    return run_with_log(cmd, LOG_PATH, cwd=ROOT)


def main() -> None:
    all_configs = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))

    parser = argparse.ArgumentParser(description="EXP005 mid領域ダウンサンプリング sweep")
    parser.add_argument(
        "--configs", nargs="+", default=all_configs,
        help=f"実行する設定名（デフォルト: 全条件 {all_configs}）",
    )
    args = parser.parse_args()

    print(f"EXP005 sweep開始: {len(args.configs)} 条件")
    print(f"  manifest_train: {MANIFEST_TRAIN}")
    print(f"  manifest_test:  {MANIFEST_TEST}")
    print(f"  output_dir:     {OUTPUT_DIR}")
    print(f"  log:            {LOG_PATH}")

    if not MANIFEST_TRAIN.exists():
        print(f"\n[ERROR] manifest_train.csv が見つかりません: {MANIFEST_TRAIN}")
        print("先に以下を実行してください:")
        print("  python scripts/split_holdout.py --method random --ratio 0.2 --seed 42 --exp 005")
        sys.exit(1)

    failed = []
    for config_name in args.configs:
        rc = run_experiment(config_name)
        if rc != 0:
            print(f"\n[WARN] {config_name} が終了コード {rc} で失敗")
            failed.append(config_name)

    print(f"\n{'='*60}")
    print(f"  EXP005 sweep完了: {len(args.configs) - len(failed)}/{len(args.configs)} 成功")
    if failed:
        print(f"  失敗: {failed}")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
