"""EXP021 Phase2: aug sweep（損失 = t_l1 固定、2系統）。

Phase1 で勝った t_l1（素のL1損失）に固定し、aug を sweep する。

variants（base = flags127 + EBR0.25 = Phase1 の t_l1__{system}、再学習不要）:
    woflip  : FLIP off (flags=63)               — EXP010「wo_flip 有利」の再ablation
    ebr000  : ENDPOINT_BLEND off (range=0.0)     — EBR の有無
    ebr015  : ENDPOINT_BLEND_RANGE=0.15
    ebr035  : ENDPOINT_BLEND_RANGE=0.35
    ebr050  : ENDPOINT_BLEND_RANGE=0.50

systems: none / eye_scale（Phase1 と同 base config）。

使い方:
    python experiments/EXP021_loss_aug_sweep/scr/train_aug_exp021.py --num_workers 3
    python .../train_aug_exp021.py --variants woflip --systems none --epochs 2 --k 2 --max_train_samples 600
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

LOSS = "t_l1"  # Phase1 勝者で固定
SYSTEMS = {"none": "base_none", "eye_scale": "base_eye_scale"}
VARIANTS = {
    "woflip": {"aug_flags": 63,  "ebr": 0.25},
    "ebr000": {"aug_flags": 127, "ebr": 0.00},
    "ebr015": {"aug_flags": 127, "ebr": 0.15},
    "ebr035": {"aug_flags": 127, "ebr": 0.35},
    "ebr050": {"aug_flags": 127, "ebr": 0.50},
}


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_cmd(cfg, exp_name, aug_flags, ebr, epochs, k, num_workers, max_train_samples):
    cmd = [
        sys.executable, str(ROOT / "scripts" / "train.py"),
        "--manifest", str(MANIFEST_TRAIN), "--holdout", str(MANIFEST_TEST),
        "--output_dir", str(OUTPUT_DIR), "--exp_name", exp_name,
        "--backbone", "custom", "--loss_fn", LOSS,
        "--lr", str(cfg["lr"]), "--weight_decay", str(cfg["weight_decay"]),
        "--epochs", str(epochs), "--batch", str(cfg["batch"]), "--k", str(k),
        "--patience", str(cfg["patience"]), "--seed", str(cfg["seed"]),
        "--num_conv_layers", str(cfg.get("num_conv_layers", 4)),
        "--dropout_rate", str(cfg.get("dropout_rate", 0.3)),
        "--aug_flags", str(aug_flags),
        "--endpoint_blend_range", str(ebr),
        "--patch_t_min", str(cfg["patch_t_min"]), "--patch_t_max", str(cfg["patch_t_max"]),
        "--patch_perp_scale", str(cfg["patch_perp_scale"]),
        "--patch_h", str(cfg["patch_h"]), "--patch_w", str(cfg["patch_w"]),
        "--label_t_min", str(cfg["label_t_min"]), "--label_t_max", str(cfg["label_t_max"]),
        "--use_sigmoid",
    ]
    if not cfg.get("use_batchnorm", True):
        cmd.append("--no_batchnorm")
    if not cfg.get("use_is_upper", True):
        cmd.append("--no_is_upper")
    mode = cfg.get("pq_rescale_mode", "none")
    if mode != "none":
        cmd += ["--pq_rescale_mode", mode, "--pq_rescale_factor", str(cfg.get("pq_rescale_factor", 1.0))]
    if num_workers > 0:
        cmd += ["--num_workers", str(num_workers)]
    if max_train_samples is not None:
        cmd += ["--max_train_samples", str(max_train_samples)]
    return cmd


def main() -> int:
    p = argparse.ArgumentParser(description="EXP021 Phase2 aug sweep (loss=t_l1)")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--skip-infer", action="store_true")
    args = p.parse_args()

    if not MANIFEST_TRAIN.exists() or not MANIFEST_TEST.exists():
        print(f"[ERROR] manifest 不在: {MANIFEST_DIR}")
        return 1

    combos = [(s, v) for s in args.systems for v in args.variants]
    print(f"EXP021 Phase2 aug sweep (loss={LOSS}): {len(combos)} 条件 = {args.systems} × {args.variants}")

    for system, variant in combos:
        cfg = load_config(SYSTEMS[system])
        epochs = args.epochs if args.epochs is not None else cfg["epochs"]
        k = args.k if args.k is not None else cfg["k"]
        v = VARIANTS[variant]
        exp_name = f"{variant}__{system}"
        model_dir = OUTPUT_DIR / exp_name

        if args.skip_existing and (model_dir / "test_predictions.csv").exists():
            print(f"  [skip] 既存: {exp_name}")
            continue

        print(f"\n{'='*64}\n  {exp_name}  (loss={LOSS}, flags={v['aug_flags']}, ebr={v['ebr']}, system={system})\n{'='*64}")
        cmd = build_cmd(cfg, exp_name, v["aug_flags"], v["ebr"], epochs, k, args.num_workers, args.max_train_samples)
        rc = run_with_log(cmd, LOG_PATH, cwd=ROOT)
        if rc != 0:
            print(f"  [ERROR] 訓練失敗 (rc={rc}): {exp_name}")
            return rc

        if not args.skip_infer:
            try:
                infer_single(model_dir, MANIFEST_TEST, model_dir / "test_predictions.csv", cfg,
                             num_workers=args.num_workers)
            except Exception as exc:  # noqa: BLE001
                print(f"  [ERROR] 推論失敗: {exp_name}: {exc}")
                return 1

    print("\nEXP021 Phase2 aug sweep 完了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
