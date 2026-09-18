"""EXP021 Phase 3: t_l1 + woflip → dual-branch X1_independent 移植・評価。

eyeliner.training.dual_kfold.run_dual_kfold() を使い、損失=t_l1, aug=woflip(flags=63) で
dual-branch X1_independent (none+eye_scale) を学習。EXP020 ベスト 0.0854 との
paired test 比較を行う。

使い方:
    python experiments/EXP021_loss_aug_sweep/scr/train_phase3.py --num_workers 2
    python .../train_phase3.py --skip-train  # 推論・分析のみ
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

ROOT = Path(__file__).parent.parent.parent.parent
SCR_DIR = Path(__file__).parent
EXP_DIR = Path(__file__).parent.parent
CONFIG_DIR = EXP_DIR / "config"
OUTPUT_DIR = EXP_DIR / "outputs"
LOG_PATH = EXP_DIR / "log_phase3.txt"

EXP020_DIR = ROOT / "experiments" / "EXP020_dual_branch"
MANIFEST_DIR = EXP020_DIR / "manifests"
MANIFEST_TRAIN = MANIFEST_DIR / "manifest_train.csv"
MANIFEST_TEST = MANIFEST_DIR / "manifest_test.csv"

# EXP020 ベスト（比較対象）
EXP020_BEST_DIR = EXP020_DIR / "outputs" / "X1_independent_none_eye_scale"

sys.path.insert(0, str(ROOT))
from eyeliner.training.dual_inference import infer_dual  # noqa: E402
from eyeliner.training.dual_kfold import (  # noqa: E402
    DualTrainConfig,
    build_dual_experiment_record,
    run_dual_kfold,
)


DEFAULT_CONFIG_NAME = "phase3_X1_t_l1_woflip"

# Phase3 baseline（EBR=0.25）。EBR variant 等との比較に使う
PHASE3_BASELINE_DIR = OUTPUT_DIR / "phase3_X1_t_l1_woflip"


def load_predictions(csv_path: Path) -> dict[int, dict]:
    """test_predictions.csv を {idx: {t_true, t_pred}} で読み込む。"""
    data: dict[int, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int(row["idx"])
            data[idx] = {
                "t_true": float(row["t_true"]),
                "t_pred": float(row["t_pred"]),
            }
    return data


def paired_test(
    ours: dict[int, dict], baseline: dict[int, dict],
) -> tuple[float, float, float]:
    """paired t-test。共通 idx のみ使用。"""
    common = sorted(set(ours) & set(baseline))
    if not common:
        raise ValueError("共通サンプルなし")
    err_ours = np.array([abs(ours[i]["t_pred"] - ours[i]["t_true"]) for i in common])
    err_base = np.array([abs(baseline[i]["t_pred"] - baseline[i]["t_true"]) for i in common])
    delta = err_ours - err_base
    t_stat, p_t = stats.ttest_1samp(delta, 0.0)
    mae_ours = float(err_ours.mean())
    mae_base = float(err_base.mean())
    return mae_ours, mae_base, p_t


def sig(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def main() -> int:
    p = argparse.ArgumentParser(description="EXP021 Phase 3: dual t_l1+woflip")
    p.add_argument("--config", default=DEFAULT_CONFIG_NAME, help="config YAML 名（拡張子なし）")
    p.add_argument("--exp-name", default=None, help="出力サブディレクトリ名（省略時は --config と同名）")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-infer", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    CONFIG_NAME = args.config
    EXP_NAME = args.exp_name or args.config

    if not MANIFEST_TRAIN.exists() or not MANIFEST_TEST.exists():
        print(f"[ERROR] manifest 不在: {MANIFEST_DIR}")
        print("  先に以下を実行: python scripts/split_holdout.py --exp 020")
        return 1

    with open(CONFIG_DIR / f"{CONFIG_NAME}.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_dir = OUTPUT_DIR / EXP_NAME

    # --- Train ---
    if not args.skip_train:
        print(f"\n{'='*60}")
        print(f"  EXP021 Phase 3: {EXP_NAME}")
        print(f"  loss={cfg['loss_fn']}, aug_flags={cfg['aug_flags']}, "
              f"ebr={cfg.get('endpoint_blend_range', 0.25)}")
        print(f"  output → {model_dir}")
        print(f"{'='*60}")

        config = DualTrainConfig(
            fusion_mode=cfg["fusion_mode"],
            pq_rescale_mode_b=cfg.get("pq_rescale_mode_b", "eye_scale"),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            epochs=cfg["epochs"],
            batch_size=cfg["batch"],
            k=cfg["k"],
            patience=cfg["patience"],
            seed=cfg["seed"],
            num_conv_layers=cfg.get("num_conv_layers", 4),
            use_sigmoid=cfg.get("use_sigmoid", True),
            use_batchnorm=cfg.get("use_batchnorm", True),
            dropout_rate=cfg.get("dropout_rate", 0.3),
            aug_flags=cfg.get("aug_flags", 127),
            endpoint_blend_range=float(cfg.get("endpoint_blend_range", 0.25)),
            loss_fn=cfg.get("loss_fn", "t"),
            num_workers=args.num_workers,
        )
        result = run_dual_kfold(MANIFEST_TRAIN, MANIFEST_TEST, model_dir, config)
        exp_record = build_dual_experiment_record(EXP_NAME, config, result)
        (model_dir / "experiment.json").write_text(
            json.dumps(exp_record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  CV MAE: {exp_record['results']['cv_mae']:.6f}")
        if result.holdout_mae is not None:
            print(f"  Hold-out MAE: {result.holdout_mae:.6f}（正規化） / "
                  f"{result.holdout_mae_raw:.6f}（生t）")
    else:
        print(f"[skip] 訓練: {EXP_NAME}")

    # --- Infer ---
    pred_csv = model_dir / "test_predictions.csv"
    if not args.skip_infer:
        print(f"\n推論: {EXP_NAME}")
        infer_dual(
            model_dir, MANIFEST_TEST, pred_csv,
            fusion_mode=cfg["fusion_mode"],
            rescale_b=cfg.get("pq_rescale_mode_b", "eye_scale"),
            batch_size=cfg["batch"],
            num_workers=args.num_workers,
        )
    else:
        print(f"[skip] 推論: {EXP_NAME}")

    # --- Compare ---
    ours_csv = model_dir / "test_predictions.csv"
    if not ours_csv.exists():
        print(f"[ERROR] 自モデル test_predictions.csv 不在: {ours_csv}")
        return 1
    ours = load_predictions(ours_csv)

    baselines = [
        ("EXP020 X1_independent_none_eye_scale", EXP020_BEST_DIR),
        ("Phase3 baseline (t_l1+woflip, EBR=0.25)", PHASE3_BASELINE_DIR),
    ]
    any_compared = False
    for label, baseline_dir in baselines:
        if baseline_dir.resolve() == model_dir.resolve():
            continue  # 自分自身との比較はスキップ
        baseline_csv = baseline_dir / "test_predictions.csv"
        print(f"\n{'='*60}")
        print(f"  Phase 3 結果: {EXP_NAME} vs {label}")
        print(f"{'='*60}")
        if not baseline_csv.exists():
            print(f"[WARN] baseline test_predictions.csv 不在: {baseline_csv}")
            continue
        baseline = load_predictions(baseline_csv)
        mae_ours, mae_base, p_t = paired_test(ours, baseline)
        print(f"  {EXP_NAME:<38} raw-t MAE: {mae_ours:.6f}")
        print(f"  {label:<38} raw-t MAE: {mae_base:.6f}")
        delta = mae_ours - mae_base
        direction = "改善" if delta < 0 else "悪化"
        print(f"  ΔMAE: {delta:+.6f} ({direction}, {abs(delta)/mae_base*100:.1f}%)")
        print(f"  Paired t-test:  p={p_t:.2e}  {sig(p_t)}")
        print(f"  n={len(set(ours) & set(baseline))}")
        any_compared = True

    if not any_compared:
        errs = [abs(v["t_pred"] - v["t_true"]) for v in ours.values()]
        print(f"\n  {EXP_NAME} raw-t MAE: {np.mean(errs):.6f}  (n={len(errs)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
