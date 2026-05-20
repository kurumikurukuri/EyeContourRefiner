"""EXP018: pq_ratio 困難サンプル対策 & アーキテクチャ強化 sweep 推論スクリプト

各条件の fold モデル（k=3）をアンサンブルして
manifest_train / manifest_test に推論し、
train_predictions.csv / test_predictions.csv を出力する。

標準条件 (X0〜Y1): scripts/infer.py 経由でコマンド実行
Y2_aux_face_shape:  カスタム Python パス（補助特徴量 11次元 concat）

使い方:
    # outputs/ にモデルが存在する全条件
    python experiments/EXP018_pq_and_arch/scr/infer_exp018.py

    # 特定条件のみ
    python experiments/EXP018_pq_and_arch/scr/infer_exp018.py --configs X0_baseline Y2_aux_face_shape
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
CONFIG_DIR = EXP_DIR / "config"
MANIFEST_DIR = EXP_DIR / "manifests"
OUTPUT_DIR = EXP_DIR / "outputs"
LOG_PATH = EXP_DIR / "log.txt"
DB_PATH = ROOT / "data" / "samples.db"

MANIFEST_TRAIN = MANIFEST_DIR / "manifest_train.csv"
MANIFEST_TEST = MANIFEST_DIR / "manifest_test.csv"

sys.path.insert(0, str(ROOT))
from scripts.exp_utils import run_with_log  # noqa: E402

logger = logging.getLogger(__name__)

STANDARD_CONFIGS = [
    "X0_baseline", "X1_pq_sampler_a3", "X2_pq_sampler_a5",
    "X3_pq_loss_weighted", "Y1_custom_conv5_d4",
]
Y2_CONFIG = "Y2_aux_face_shape"


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


# ---------------------------------------------------------------------------
# 標準 config（X0-X3, Y1）: scripts/infer.py を subprocess で実行
# ---------------------------------------------------------------------------

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


def run_infer_standard(config_name: str) -> int:
    print(f"\n{'='*60}")
    print(f"  推論: {config_name}")
    print(f"{'='*60}")

    models_dir = OUTPUT_DIR / config_name
    if not models_dir.exists():
        print(f"  [SKIP] モデルディレクトリが存在しません: {models_dir}")
        print(f"         先に train_exp018.py --configs {config_name} を実行してください")
        return 1

    train_out = models_dir / "train_predictions.csv"
    train_cmd = build_infer_cmd(config_name, MANIFEST_TRAIN, train_out)
    print(f"  [train] manifest: {MANIFEST_TRAIN.name} -> {train_out.name}")
    rc = run_with_log(train_cmd, LOG_PATH, cwd=ROOT)
    if rc != 0:
        print(f"  [ERROR] train 推論失敗: rc={rc}")
        return rc

    test_out = models_dir / "test_predictions.csv"
    test_cmd = build_infer_cmd(config_name, MANIFEST_TEST, test_out)
    print(f"  [test]  manifest: {MANIFEST_TEST.name} -> {test_out.name}")
    return run_with_log(test_cmd, LOG_PATH, cwd=ROOT)


# ---------------------------------------------------------------------------
# Y2_aux_face_shape: 独自 Python パス（aux 特徴）
# ---------------------------------------------------------------------------

def _run_y2_manifest(
    models_dir: Path,
    manifest_path: Path,
    output_path: Path,
    cfg: dict,
) -> None:
    """Y2 の manifest 推論を実行して CSV を書き出す。"""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from eyeliner.data.dataset import EyelinerPatchDataset, load_manifest
    from eyeliner.data.weights import compute_aux_face_shape_meta, compute_eye_meta
    from eyeliner.models.ensemble import create_ensemble_from_dir

    ensemble, _ = create_ensemble_from_dir(models_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble = ensemble.to(device).eval()

    records = load_manifest(manifest_path)
    compute_eye_meta(records)
    db_path = DB_PATH if DB_PATH.exists() else None
    compute_aux_face_shape_meta(records, db_path)

    p = load_infer_params(Y2_CONFIG)
    ds = EyelinerPatchDataset(
        records,
        augment_data=False,
        use_aux_face_shape=True,
        patch_t_min=p["patch_t_min"],
        patch_t_max=p["patch_t_max"],
        patch_perp_scale=p["patch_perp_scale"],
        patch_size=(p["patch_h"], p["patch_w"]),
        label_t_min=p["label_t_min"],
        label_t_max=p["label_t_max"],
    )
    loader = DataLoader(ds, batch_size=cfg["batch"], shuffle=False)

    eff_label_min = p["label_t_min"] if p["label_t_min"] is not None else p["patch_t_min"]
    eff_label_max = p["label_t_max"] if p["label_t_max"] is not None else p["patch_t_max"]
    label_range = eff_label_max - eff_label_min

    out_fieldnames = [
        "idx", "image_path", "part_name", "sub_key", "point_index",
        "px", "py", "qx", "qy",
        "true_t_raw", "pred_t_raw", "pred_t_norm",
        "rx_pred", "ry_pred",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_preds = []
    all_true = []

    sample_idx = 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            is_upper = batch[1].to(device)
            y = batch[3]
            aux_features = batch[8].to(device)

            pred = ensemble(x, is_upper, aux_features=aux_features)
            pred_norm = torch.clamp(pred, 0.0, 1.0).cpu()

            for i in range(pred_norm.shape[0]):
                rec = records[sample_idx]
                sample_idx += 1

                pred_t_norm = float(pred_norm[i].item())
                pred_t_raw = pred_t_norm * label_range + eff_label_min
                true_t_raw = float(rec["t"])
                true_t_norm = (true_t_raw - eff_label_min) / label_range if label_range != 0 else 0.0

                px_val, py_val = float(rec["px"]), float(rec["py"])
                qx_val, qy_val = float(rec["qx"]), float(rec["qy"])
                pq_dist = float(np.sqrt((qx_val - px_val) ** 2 + (qy_val - py_val) ** 2))
                rx_pred = px_val + pred_t_raw * (qx_val - px_val) / max(pq_dist, 1e-6)
                ry_pred = py_val + pred_t_raw * (qy_val - py_val) / max(pq_dist, 1e-6)

                all_preds.append(pred_t_raw)
                all_true.append(true_t_raw)

                with open(output_path, "w" if sample_idx == 1 else "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=out_fieldnames)
                    if sample_idx == 1:
                        writer.writeheader()
                    writer.writerow({
                        "idx":              rec.get("idx", ""),
                        "image_path":       rec["image_path"],
                        "part_name":        rec.get("part_name", ""),
                        "sub_key":          rec.get("sub_key", ""),
                        "point_index":      rec.get("point_index", ""),
                        "px": px_val, "py": py_val, "qx": qx_val, "qy": qy_val,
                        "true_t_raw":       f"{true_t_raw:.6f}",
                        "pred_t_raw":       f"{pred_t_raw:.6f}",
                        "pred_t_norm":      f"{pred_t_norm:.6f}",
                        "rx_pred":          f"{rx_pred:.2f}",
                        "ry_pred":          f"{ry_pred:.2f}",
                    })

    if all_preds:
        mae = float(np.mean([abs(p - t) for p, t in zip(all_preds, all_true)]))
        _update_experiment_json(output_path, mae)
        logger.info("MAE (raw t): %.4f  (n=%d)", mae, len(all_preds))


def _update_experiment_json(predictions_path: Path, mae_raw: float) -> None:
    exp_json = predictions_path.parent / "experiment.json"
    if not exp_json.exists():
        return
    name = predictions_path.name
    prefix = "test" if "test" in name else "train" if "train" in name else None
    if prefix is None:
        return
    try:
        data = json.loads(exp_json.read_text(encoding="utf-8"))
        data.setdefault("results", {})[f"{prefix}_mae"] = mae_raw
        exp_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("experiment.json に %s_mae=%.6f を記録", prefix, mae_raw)
    except Exception as exc:
        logger.warning("experiment.json の更新に失敗: %s", exc)


def run_infer_y2() -> int:
    print(f"\n{'='*60}")
    print(f"  推論: {Y2_CONFIG}  (カスタム Python パス)")
    print(f"{'='*60}")

    models_dir = OUTPUT_DIR / Y2_CONFIG
    if not models_dir.exists():
        print(f"  [SKIP] モデルディレクトリが存在しません: {models_dir}")
        return 1

    cfg = load_infer_params(Y2_CONFIG)
    full_cfg = yaml.safe_load((CONFIG_DIR / f"{Y2_CONFIG}.yaml").open(encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(LOG_PATH), encoding="utf-8"),
        ],
    )

    train_out = models_dir / "train_predictions.csv"
    print(f"  [train] manifest: {MANIFEST_TRAIN.name} -> {train_out.name}")
    _run_y2_manifest(models_dir, MANIFEST_TRAIN, train_out, full_cfg)

    test_out = models_dir / "test_predictions.csv"
    print(f"  [test]  manifest: {MANIFEST_TEST.name} -> {test_out.name}")
    _run_y2_manifest(models_dir, MANIFEST_TEST, test_out, full_cfg)

    return 0


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

def main() -> None:
    all_configs = STANDARD_CONFIGS + [Y2_CONFIG]

    parser = argparse.ArgumentParser(description="EXP018 pq & arch sweep 推論")
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
        target_configs = []
        for c in all_configs:
            models_dir = OUTPUT_DIR / c
            if models_dir.exists():
                target_configs.append(c)
        if not target_configs:
            print("[INFO] 推論対象のモデルが見つかりません。")
            print("       先に train_exp018.py を実行してください。")
            sys.exit(0)

    print(f"\nEXP018 infer 開始: {len(target_configs)} 条件")

    failed: list[str] = []
    skipped: list[str] = []
    for config_name in target_configs:
        if config_name == Y2_CONFIG:
            rc = run_infer_y2()
        else:
            rc = run_infer_standard(config_name)
        if rc != 0:
            print(f"\n[WARN] {config_name} が終了コード {rc} で失敗/スキップ")
            if rc == 1:
                skipped.append(config_name)
            else:
                failed.append(config_name)

    success = len(target_configs) - len(failed) - len(skipped)
    print(f"\n{'='*60}")
    print(f"  EXP018 infer 完了: {success}/{len(target_configs)} 成功")
    if skipped:
        print(f"  スキップ（モデル未生成）: {skipped}")
    if failed:
        print(f"  失敗: {failed}")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
