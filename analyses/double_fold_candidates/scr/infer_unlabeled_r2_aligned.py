"""R2_baseline_t_perturbation vs aligned_wide_large の unlabeled 比較。

manifest は samples.db から最新状態で生成する（samples.csv ベースとは別）。

Usage:
    python analyses/double_fold_candidates/scr/infer_unlabeled_r2_aligned.py

出力:
    analyses/double_fold_candidates/manifests/manifest_unlabeled_upper_db.csv
    analyses/double_fold_candidates/outputs/predictions/unlabeled_R2_db.csv
    analyses/double_fold_candidates/outputs/predictions/unlabeled_aligned_wide_large_db.csv
    analyses/double_fold_candidates/outputs/unlabeled_r2_aligned_diff.csv
    analyses/double_fold_candidates/outputs/unlabeled_r2_aligned_diff_by_face.csv
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = ANALYSIS_DIR / "outputs" / "predictions"

EXP014_DIR = ROOT / "experiments" / "EXP014_t_resolution_sweep"
EXP003_DIR = ROOT / "experiments" / "EXP003_input_sweep"
LOG_PATH = EXP014_DIR / "log.txt"

DB_PATH = ROOT / "data" / "samples.db"
MANIFEST_OUT = ANALYSIS_DIR / "manifests" / "manifest_unlabeled_upper_db.csv"

DUMMY_T = 0.5

CONFIGS: dict[str, dict] = {
    "R2_baseline_t_perturbation": {
        "models_dir": EXP014_DIR / "outputs" / "R2_baseline_t_perturbation" / "R2_baseline_t_perturbation",
        "patch_t_min": -2.0,
        "patch_t_max": 3.0,
        "patch_perp_scale": 2.0,
        "patch_h": 64,
        "patch_w": 64,
        "label_t_min": -1.0,
        "label_t_max": 2.0,
        "t_quant_step": None,
        "out_name": "unlabeled_R2_db.csv",
    },
    "aligned_wide_large": {
        "models_dir": EXP003_DIR / "outputs" / "aligned_wide_large" / "aligned_wide_large",
        "patch_t_min": -1.0,
        "patch_t_max": 2.0,
        "patch_perp_scale": 1.0,
        "patch_h": 96,
        "patch_w": 192,
        "label_t_min": None,
        "label_t_max": None,
        "t_quant_step": None,
        "out_name": "unlabeled_aligned_wide_large_db.csv",
    },
}

sys.path.insert(0, str(ROOT))
from scripts.exp_utils import run_with_log  # noqa: E402


def build_manifest_from_db() -> Path:
    print(f"[manifest] samples.db から unlabeled upper を抽出: {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        "SELECT idx, image_path, face_index, part_name, sub_key, point_index, "
        "px, py, qx, qy, skip_flag "
        "FROM samples "
        "WHERE t IS NULL AND sub_key = 'upper' AND skip_flag = 0",
        conn,
    )
    conn.close()

    df["t"] = DUMMY_T
    cols = ["idx", "image_path", "face_index", "part_name", "sub_key", "point_index",
            "px", "py", "qx", "qy", "t", "skip_flag"]
    out = df[cols]

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(MANIFEST_OUT, index=False)
    print(f"[manifest] 書き出し完了: {MANIFEST_OUT} ({len(out)} 行)")
    return MANIFEST_OUT


def build_infer_cmd(config_name: str, manifest: Path, output_path: Path) -> list[str]:
    cfg = CONFIGS[config_name]
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "infer.py"),
        "--models-dir",       str(cfg["models_dir"]),
        "--manifest",         str(manifest),
        "--output",           str(output_path),
        "--patch_t_min",      str(cfg["patch_t_min"]),
        "--patch_t_max",      str(cfg["patch_t_max"]),
        "--patch_perp_scale", str(cfg["patch_perp_scale"]),
        "--patch_h",          str(cfg["patch_h"]),
        "--patch_w",          str(cfg["patch_w"]),
    ]
    if cfg["label_t_min"] is not None:
        cmd += ["--label_t_min", str(cfg["label_t_min"])]
    if cfg["label_t_max"] is not None:
        cmd += ["--label_t_max", str(cfg["label_t_max"])]
    if cfg["t_quant_step"] is not None:
        cmd += ["--t_quant_step", str(cfg["t_quant_step"])]
    return cmd


def run_infer(config_name: str, manifest: Path, force: bool = False) -> Path:
    cfg = CONFIGS[config_name]
    if not cfg["models_dir"].exists():
        print(f"[ERROR] モデルなし: {cfg['models_dir']}")
        sys.exit(1)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / cfg["out_name"]
    if out_path.exists() and not force:
        print(f"[SKIP] 既存ファイルあり: {out_path}")
        return out_path

    print(f"\n推論中: {config_name} → {out_path.name}")
    cmd = build_infer_cmd(config_name, manifest, out_path)
    rc = run_with_log(cmd, LOG_PATH, cwd=ROOT)
    if rc != 0:
        print(f"[ERROR] 推論失敗: rc={rc}")
        sys.exit(1)
    return out_path


def compute_diff(r2_path: Path, aligned_path: Path) -> pd.DataFrame:
    r2 = pd.read_csv(r2_path)
    al = pd.read_csv(aligned_path)

    r2 = r2.rename(columns={"pred_t_raw": "pred_t_raw_R2", "pred_t_norm": "pred_t_norm_R2"})
    al = al.rename(columns={"pred_t_raw": "pred_t_raw_aligned", "pred_t_norm": "pred_t_norm_aligned"})

    merge_keys = ["idx", "image_path", "part_name", "sub_key", "point_index", "px", "py", "qx", "qy"]
    df = pd.merge(
        r2[merge_keys + ["pred_t_raw_R2", "pred_t_norm_R2"]],
        al[merge_keys + ["pred_t_raw_aligned", "pred_t_norm_aligned"]],
        on=merge_keys,
    )

    df["diff_t_raw"] = (df["pred_t_raw_R2"] - df["pred_t_raw_aligned"]).abs()
    df["diff_t_norm"] = (df["pred_t_norm_R2"] - df["pred_t_norm_aligned"]).abs()
    df = df.sort_values("diff_t_raw", ascending=False).reset_index(drop=True)
    return df


def summarize_by_face(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["image_path", "part_name"]).agg(
        n_points=("diff_t_raw", "count"),
        max_diff_t_raw=("diff_t_raw", "max"),
        mean_diff_t_raw=("diff_t_raw", "mean"),
        mean_pred_R2=("pred_t_raw_R2", "mean"),
        mean_pred_aligned=("pred_t_raw_aligned", "mean"),
    ).reset_index()
    grp = grp.sort_values("mean_diff_t_raw", ascending=False).reset_index(drop=True)
    return grp


def main() -> None:
    print("=== unlabeled 比較: R2 vs aligned_wide_large (DB manifest) ===")

    manifest = build_manifest_from_db()

    out_paths: dict[str, Path] = {}
    for cfg_name in CONFIGS:
        out_paths[cfg_name] = run_infer(cfg_name, manifest, force=True)

    print("\n=== 差異計算 ===")
    df_diff = compute_diff(out_paths["R2_baseline_t_perturbation"], out_paths["aligned_wide_large"])

    diff_out = ANALYSIS_DIR / "outputs" / "unlabeled_r2_aligned_diff.csv"
    diff_out.parent.mkdir(parents=True, exist_ok=True)
    df_diff.to_csv(diff_out, index=False)
    print(f"点レベル差異: {diff_out}  ({len(df_diff)} 行)")

    df_face = summarize_by_face(df_diff)
    face_out = ANALYSIS_DIR / "outputs" / "unlabeled_r2_aligned_diff_by_face.csv"
    df_face.to_csv(face_out, index=False)
    print(f"face レベル集計: {face_out}  ({len(df_face)} 行)")

    print("\n--- Top 20 差異 face (mean_diff_t_raw 降順) ---")
    print(df_face.head(20).to_string(index=False))

    top1_pct = (df_diff["diff_t_raw"] > df_diff["diff_t_raw"].quantile(0.99)).sum()
    print(f"\ndiff_t_raw > 99th percentile: {top1_pct} 点")
    print(f"diff_t_raw 分布: mean={df_diff['diff_t_raw'].mean():.4f}, "
          f"median={df_diff['diff_t_raw'].median():.4f}, "
          f"max={df_diff['diff_t_raw'].max():.4f}")

    print("\n--- 各モデルの予測分布 ---")
    for col, label in [("pred_t_raw_R2", "R2"), ("pred_t_raw_aligned", "aligned_wide_large")]:
        s = df_diff[col]
        print(f"{label}: mean={s.mean():.4f}, std={s.std():.4f}, "
              f"q10={s.quantile(0.10):.4f}, q50={s.median():.4f}, q90={s.quantile(0.90):.4f}")


if __name__ == "__main__":
    main()
