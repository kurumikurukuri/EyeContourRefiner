"""unlabeled.csv の lower[0] 行に XGBoost 推論を実行する。

select_candidates_xgb.py でモデルを学習し、
manifest/unlabeled.csv に含まれる lower[0] パッチ（12k 件）の
pred_score を全件出力する。

Usage:
    python analyses/xgboost_shortpqlens/infer_unlabeled.py
    python analyses/xgboost_shortpqlens/infer_unlabeled.py \\
        --manifest analyses/xgboost_shortpqlens/manifest/unlabeled.csv \\
        --output   analyses/xgboost_shortpqlens/outputs/predictions_unlabeled_lower0.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from analyses.xgboost_shortpqlens.select_candidates_xgb import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    build_features,
    load_from_db,
    load_oi_labels,
    load_skip_data,
    predict_skip_prob,
    t_extremeness,
    train_skip_model,
    train_xgb,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="unlabeled.csv lower[0] への XGBoost 推論")
    parser.add_argument("--db",       default="data/samples.db")
    parser.add_argument(
        "--manifest",
        default="analyses/xgboost_shortpqlens/manifest/unlabeled.csv",
    )
    parser.add_argument(
        "--output",
        default="analyses/xgboost_shortpqlens/outputs/predictions_unlabeled_lower0.csv",
    )
    parser.add_argument("--cache", default=DEFAULT_CACHE_DIR, type=Path)
    args = parser.parse_args()

    # ---- 0. OI ラベル -------------------------------------------------------
    print("[0] loading OI labels ...")
    oi_labels = load_oi_labels(args.cache)
    print(f"  {len(oi_labels)} 画像" if oi_labels else "  OI ラベルなし")

    # ---- 1. 学習データ (labeled lower[0] from DB) ---------------------------
    print("\n[1] loading labeled lower[0] from DB ...")
    train_df = load_from_db(args.db, labeled=True)
    train_df["t_extremeness"] = t_extremeness(train_df["t"])
    print(f"  rows: {len(train_df):,}")
    print(f"  t_extremeness  mean={train_df['t_extremeness'].mean():.4f}"
          f"  max={train_df['t_extremeness'].max():.4f}")

    X_train = build_features(train_df)
    y_train  = train_df["t_extremeness"]

    # ---- 2. XGBoost 学習 ----------------------------------------------------
    print("\n[2] training XGBoost ...")
    model = train_xgb(X_train, y_train)

    import pandas as _pd
    imp = (
        _pd.Series(model.feature_importances_, index=X_train.columns)
        .sort_values(ascending=False)
    )
    print("  Feature importances:")
    for feat, v in imp.items():
        print(f"    {feat:<25} {v:.4f}")

    # ---- 3. manifest から lower[0] idx を取得 --------------------------------
    print(f"\n[3] reading manifest: {args.manifest}")
    manifest = pd.read_csv(args.manifest)
    lower0_rows = manifest[
        (manifest["sub_key"] == "lower") & (manifest["point_index"] == 0)
    ]
    target_idx = set(lower0_rows["idx"].astype(int).tolist())
    print(f"  manifest 内 lower[0]: {len(target_idx):,} 件")

    # ---- 4. DB から unlabeled lower[0] を読み込み ---------------------------
    labeled_images = set(train_df["image_path"].unique())
    print("\n[4] loading unlabeled lower[0] from DB ...")
    unlabeled_db = load_from_db(args.db, labeled=False)
    before = len(unlabeled_db)
    unlabeled = unlabeled_db[
        ~unlabeled_db["image_path"].isin(labeled_images)
        & unlabeled_db["idx"].isin(target_idx)
    ].reset_index(drop=True)
    print(f"  DB lower[0] (unlabeled): {before:,}")
    print(f"  manifest フィルタ後:      {len(unlabeled):,}")

    if unlabeled.empty:
        print("推論対象が見つかりませんでした。終了します。")
        return

    # ---- 4.5. skip 予測モデル -----------------------------------------------
    print("\n[4.5] training skip predictor ...")
    skip_df = load_skip_data(args.db)
    skip_model = train_skip_model(skip_df, oi_labels)

    # ---- 5. スコアリング ----------------------------------------------------
    print("\n[5] scoring ...")
    X_infer = build_features(unlabeled)
    result = unlabeled.copy()
    result["pred_score"] = model.predict(X_infer)
    result["pq_ratio"] = (
        np.sqrt(
            (result["px"] - result["qx"]) ** 2
            + (result["py"] - result["qy"]) ** 2
        )
        / result["eye_scale"].clip(lower=1e-6)
    )

    if skip_model is not None:
        result["skip_prob"] = predict_skip_prob(result, oi_labels, skip_model)
    else:
        result["skip_prob"] = 0.0

    result_sorted = result.sort_values("pred_score", ascending=False).reset_index(drop=True)

    print(f"  pred_score  mean={result_sorted['pred_score'].mean():.4f}"
          f"  max={result_sorted['pred_score'].max():.4f}"
          f"  min={result_sorted['pred_score'].min():.4f}")
    print(f"  |pitch|>20°: {(result_sorted['face_pitch'].abs() > 20).sum()} / {len(result_sorted)}"
          f"  ({100*(result_sorted['face_pitch'].abs() > 20).mean():.1f}%)")

    # ---- 6. 保存 ------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "idx", "image_path", "face_index", "part_name", "sub_key", "point_index",
        "pred_score", "skip_prob",
        "pq_len", "pq_ratio", "pq_ratio_rel",
        "eye_opening", "eye_opening_ratio",
        "face_pitch", "face_yaw", "face_roll",
        "eye_scale", "norm_eye_scale", "lap_var",
        "px", "py", "qx", "qy",
    ]
    result_sorted[cols].to_csv(out_path, index=False)
    print(f"\n[6] saved → {out_path}  ({len(result_sorted):,} rows)")


if __name__ == "__main__":
    main()
