"""EXP021: 損失 sweep 訓練ドライバ（単一ブランチ、6損失 × 2系統）。

設計（Phase 1 = 空間 × カーネルの 2軸直積）:

    空間＼カーネル | L1          | Huber             | Wing
    --------------|-------------|-------------------|--------
    正規化t       | t_l1        | t                 | wing
    目正規化px    | error_ratio | error_ratio_huber | wing_px

系統（base config）:
    none      = 生PQ座標系   (base_none.yaml,      EXP020 X0_none_baseline 相当)
    eye_scale = P'Q'座標系   (base_eye_scale.yaml, EXP020 X0_eye_scale_baseline 相当)

各 (loss, system) を scripts/train.py で k=3 学習し、infer_exp021 で生 t の
test_predictions.csv を生成する。aug は EXP020 baseline(127) 固定（loss 効果の分離）。

使い方:
    # 全 12 条件
    python experiments/EXP021_loss_aug_sweep/scr/train_exp021.py

    # 一部のみ
    python experiments/EXP021_loss_aug_sweep/scr/train_exp021.py --losses error_ratio t --systems none

    # スモークテスト（短縮）
    python experiments/EXP021_loss_aug_sweep/scr/train_exp021.py \
        --losses error_ratio --systems none --epochs 2 --k 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent
SCR_DIR = Path(__file__).parent
EXP_DIR = Path(__file__).parent.parent
CONFIG_DIR = EXP_DIR / "config"
MANIFEST_DIR = EXP_DIR / "manifests"
OUTPUT_DIR = EXP_DIR / "outputs"
LOG_PATH = EXP_DIR / "log.txt"
MANIFEST_TRAIN = MANIFEST_DIR / "manifest_train.csv"
MANIFEST_TEST = MANIFEST_DIR / "manifest_test.csv"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCR_DIR))
from scripts.exp_utils import run_with_log  # noqa: E402
from infer_exp021 import infer_single  # noqa: E402

LOSSES = ["t", "t_l1", "wing", "error_ratio", "error_ratio_huber", "wing_px"]
SYSTEMS = {"none": "base_none", "eye_scale": "base_eye_scale"}


def load_config(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_cmd(cfg: dict, exp_name: str, loss_fn: str, epochs: int, k: int, num_workers: int,
              max_train_samples: int | None = None) -> list[str]:
    cmd = [
        sys.executable, str(ROOT / "scripts" / "train.py"),
        "--manifest",         str(MANIFEST_TRAIN),
        "--holdout",          str(MANIFEST_TEST),
        "--output_dir",       str(OUTPUT_DIR),
        "--exp_name",         exp_name,
        "--backbone",         "custom",
        "--loss_fn",          loss_fn,
        "--lr",               str(cfg["lr"]),
        "--weight_decay",     str(cfg["weight_decay"]),
        "--epochs",           str(epochs),
        "--batch",            str(cfg["batch"]),
        "--k",                str(k),
        "--patience",         str(cfg["patience"]),
        "--seed",             str(cfg["seed"]),
        "--num_conv_layers",  str(cfg.get("num_conv_layers", 4)),
        "--dropout_rate",     str(cfg.get("dropout_rate", 0.3)),
        "--aug_flags",        str(cfg.get("aug_flags", 127)),
        "--patch_t_min",      str(cfg["patch_t_min"]),
        "--patch_t_max",      str(cfg["patch_t_max"]),
        "--patch_perp_scale", str(cfg["patch_perp_scale"]),
        "--patch_h",          str(cfg["patch_h"]),
        "--patch_w",          str(cfg["patch_w"]),
        "--label_t_min",      str(cfg["label_t_min"]),
        "--label_t_max",      str(cfg["label_t_max"]),
        "--use_sigmoid",
    ]
    if not cfg.get("use_batchnorm", True):
        cmd.append("--no_batchnorm")
    if not cfg.get("use_is_upper", True):
        cmd.append("--no_is_upper")
    mode = cfg.get("pq_rescale_mode", "none")
    if mode != "none":
        cmd += ["--pq_rescale_mode", mode,
                "--pq_rescale_factor", str(cfg.get("pq_rescale_factor", 1.0))]
    if num_workers > 0:
        cmd += ["--num_workers", str(num_workers)]
    if max_train_samples is not None:
        cmd += ["--max_train_samples", str(max_train_samples)]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="EXP021 損失 sweep")
    parser.add_argument("--losses", nargs="+", default=LOSSES, choices=LOSSES)
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    parser.add_argument("--epochs", type=int, default=None, help="config 既定を上書き（スモーク用）")
    parser.add_argument("--k", type=int, default=None, help="config 既定を上書き（スモーク用）")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=None, help="スモーク用サブサンプル上限")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="test_predictions.csv が既にある条件はスキップ（再開用）")
    args = parser.parse_args()

    if not MANIFEST_TRAIN.exists() or not MANIFEST_TEST.exists():
        print(f"[ERROR] manifest が見つかりません: {MANIFEST_DIR}")
        return 1

    combos = [(s, l) for s in args.systems for l in args.losses]
    print(f"EXP021 sweep: {len(combos)} 条件 = {args.systems} × {args.losses}")

    for system, loss in combos:
        cfg = load_config(SYSTEMS[system])
        epochs = args.epochs if args.epochs is not None else cfg["epochs"]
        k = args.k if args.k is not None else cfg["k"]
        exp_name = f"{loss}__{system}"
        model_dir = OUTPUT_DIR / exp_name

        if args.skip_existing and (model_dir / "test_predictions.csv").exists():
            print(f"  [skip] 既存: {exp_name}")
            continue

        print(f"\n{'='*64}\n  {exp_name}  (loss={loss}, system={system}, epochs={epochs}, k={k})\n{'='*64}")

        if not args.skip_train:
            cmd = build_cmd(cfg, exp_name, loss, epochs, k, args.num_workers, args.max_train_samples)
            rc = run_with_log(cmd, LOG_PATH, cwd=ROOT)
            if rc != 0:
                print(f"  [ERROR] 訓練失敗 (rc={rc}): {exp_name}")
                return rc

        if not args.skip_infer:
            try:
                infer_single(
                    model_dir, MANIFEST_TEST, model_dir / "test_predictions.csv", cfg,
                    num_workers=args.num_workers,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [ERROR] 推論失敗: {exp_name}: {exc}")
                return 1

    print("\nEXP021 sweep 完了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
