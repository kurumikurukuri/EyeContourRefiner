"""EXP014: R2 / R7 でunlabeledデータを推論し、差異から二重幅候補を抽出する。

Usage:
    python analyses/double_fold_candidates/scr/infer_unlabeled_r2_r7.py

出力:
    analyses/double_fold_candidates/outputs/predictions/unlabeled_R2.csv
    analyses/double_fold_candidates/outputs/predictions/unlabeled_R7.csv
    analyses/double_fold_candidates/outputs/unlabeled_r2_r7_diff.csv
    analyses/double_fold_candidates/outputs/unlabeled_r2_r7_diff_by_face.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = ANALYSIS_DIR / "outputs" / "predictions"

EXP014_DIR = ROOT / "experiments" / "EXP014_t_resolution_sweep"
CONFIG_DIR = EXP014_DIR / "config"
LOG_PATH = EXP014_DIR / "log.txt"

UNLABELED_MANIFEST = ANALYSIS_DIR / "manifests" / "manifest_unlabeled_upper.csv"

CONFIGS = ["R2_baseline_t_perturbation", "R7_baseline_patch_t_narrow"]
CONFIG_SHORT = {
    "R2_baseline_t_perturbation": "R2",
    "R7_baseline_patch_t_narrow": "R7",
}

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
        "patch_w":          cfg.get("patch_w", 128),
        "t_quant_step":     cfg.get("t_quant_step"),
    }


def build_infer_cmd(config_name: str, manifest: Path, output_path: Path) -> list[str]:
    p = load_infer_params(config_name)
    models_dir = EXP014_DIR / "outputs" / config_name / config_name
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
    if p["t_quant_step"] is not None:
        cmd += ["--t_quant_step", str(p["t_quant_step"])]
    return cmd


def run_unlabeled_infer(config_name: str) -> Path:
    models_dir = EXP014_DIR / "outputs" / config_name / config_name
    if not models_dir.exists():
        print(f"[ERROR] モデルなし: {models_dir}")
        sys.exit(1)

    short = CONFIG_SHORT[config_name]
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"unlabeled_{short}.csv"
    if out_path.exists():
        print(f"[SKIP] 既存ファイルあり: {out_path}")
        return out_path

    print(f"\n推論中: {config_name} → {out_path.name}")
    cmd = build_infer_cmd(config_name, UNLABELED_MANIFEST, out_path)
    rc = run_with_log(cmd, LOG_PATH, cwd=ROOT)
    if rc != 0:
        print(f"[ERROR] 推論失敗: rc={rc}")
        sys.exit(1)
    return out_path


def compute_diff(r2_path: Path, r7_path: Path) -> pd.DataFrame:
    r2 = pd.read_csv(r2_path)
    r7 = pd.read_csv(r7_path)

    r2 = r2.rename(columns={"pred_t_raw": "pred_t_raw_R2", "pred_t_norm": "pred_t_norm_R2"})
    r7 = r7.rename(columns={"pred_t_raw": "pred_t_raw_R7", "pred_t_norm": "pred_t_norm_R7"})

    merge_keys = ["idx", "image_path", "part_name", "sub_key", "point_index", "px", "py", "qx", "qy"]
    df = pd.merge(
        r2[merge_keys + ["pred_t_raw_R2", "pred_t_norm_R2"]],
        r7[merge_keys + ["pred_t_raw_R7", "pred_t_norm_R7"]],
        on=merge_keys,
    )

    df["diff_t_raw"] = (df["pred_t_raw_R2"] - df["pred_t_raw_R7"]).abs()
    df["diff_t_norm"] = (df["pred_t_norm_R2"] - df["pred_t_norm_R7"]).abs()
    df = df.sort_values("diff_t_raw", ascending=False).reset_index(drop=True)
    return df


def summarize_by_face(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["image_path", "part_name"]).agg(
        n_points=("diff_t_raw", "count"),
        max_diff_t_raw=("diff_t_raw", "max"),
        mean_diff_t_raw=("diff_t_raw", "mean"),
        mean_pred_R2=("pred_t_raw_R2", "mean"),
        mean_pred_R7=("pred_t_raw_R7", "mean"),
    ).reset_index()
    grp = grp.sort_values("mean_diff_t_raw", ascending=False).reset_index(drop=True)
    return grp


def main() -> None:
    if not UNLABELED_MANIFEST.exists():
        print(f"[ERROR] unlabeled manifest が見つかりません: {UNLABELED_MANIFEST}")
        print("       build_unlabeled_manifest.py を先に実行してください")
        sys.exit(1)

    print("=== EXP014 unlabeled 推論 (R2 / R7) ===")
    print(f"manifest: {UNLABELED_MANIFEST}")

    pred_paths: dict[str, Path] = {}
    for cfg in CONFIGS:
        pred_paths[cfg] = run_unlabeled_infer(cfg)

    print("\n=== 差異計算 ===")
    df_diff = compute_diff(pred_paths["R2_baseline_t_perturbation"], pred_paths["R7_baseline_patch_t_narrow"])

    diff_out = ANALYSIS_DIR / "outputs" / "unlabeled_r2_r7_diff.csv"
    diff_out.parent.mkdir(parents=True, exist_ok=True)
    df_diff.to_csv(diff_out, index=False)
    print(f"点レベル差異: {diff_out}  ({len(df_diff)} 行)")

    df_face = summarize_by_face(df_diff)
    face_out = ANALYSIS_DIR / "outputs" / "unlabeled_r2_r7_diff_by_face.csv"
    df_face.to_csv(face_out, index=False)
    print(f"face レベル集計: {face_out}  ({len(df_face)} 行)")

    print("\n--- Top 20 二重幅候補 (face 単位・mean_diff 降順) ---")
    print(df_face.head(20).to_string(index=False))

    top1_pct = (df_diff["diff_t_raw"] > df_diff["diff_t_raw"].quantile(0.99)).sum()
    print(f"\ndiff_t_raw > 99th percentile: {top1_pct} 点")
    print(f"diff_t_raw 分布: mean={df_diff['diff_t_raw'].mean():.4f}, "
          f"median={df_diff['diff_t_raw'].median():.4f}, "
          f"max={df_diff['diff_t_raw'].max():.4f}")


if __name__ == "__main__":
    main()
