"""
lower[0]（下まぶた目尻）に絞った XGBoost 難易度スコアで候補選定。

データソースは samples.db のみ。
難しさの代替指標: t_extremeness = max(0, -t) + max(0, t - 1.0)
  → t が [0,1] の外に出るほど大きい（PQ 線分の外挿量）。
  → 分析で確認した regression-to-mean バイアスと対応。

DB が更新されたら再実行するだけで最新候補を出力できる。

Usage:
    python analyses/select_candidates_xgb.py
    python analyses/select_candidates_xgb.py --n 500 --output analyses/candidates_lower0.csv
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

PART_MAP = {"eyeliner_left": 0, "eyeliner_right": 1}


def t_extremeness(t: pd.Series) -> pd.Series:
    """t が [0,1] の外に出た量。[0,1] 内では 0。"""
    return np.maximum(0.0, -t) + np.maximum(0.0, t - 1.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    pq_len = df["pq_len"] if "pq_len" in df.columns else (
        np.sqrt((df["px"] - df["qx"]) ** 2 + (df["py"] - df["qy"]) ** 2)
    )
    part_id = df["part_name"].map(PART_MAP).fillna(-1).astype(int)
    abs_pitch = df["face_pitch"].abs()
    return pd.DataFrame(
        {
            "face_pitch": df["face_pitch"],
            "abs_pitch": abs_pitch,
            "face_yaw": df["face_yaw"],
            "abs_yaw": df["face_yaw"].abs(),
            "face_roll": df["face_roll"],
            "abs_roll": df["face_roll"].abs(),
            "eye_scale": df["eye_scale"],
            "norm_eye_scale": df["norm_eye_scale"],
            "lap_var": df["lap_var"],
            "pq_len": pq_len,
            "pq_ratio": pq_len / df["eye_scale"].clip(lower=1e-6),
            "pq_ratio_rel": df["pq_ratio_rel"],  # lower[0]/max(lower[0-6])
            "part_id": part_id,
            "pitch_x_part": abs_pitch * part_id,
        },
        index=df.index,
    )


def train_xgb(X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
    )
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
    print(f"  CV-MAE : {-scores.mean():.5f}  (std={scores.std():.5f})")
    model.fit(X, y)
    print(f"  Train-MAE: {mean_absolute_error(y, model.predict(X)):.5f}")
    return model


def load_from_db(db_path: str, labeled: bool) -> pd.DataFrame:
    """lower[0] を返す。pq_ratio_rel の計算のため lower[0-6] 全点を経由する。"""
    t_cond = "t IS NOT NULL" if labeled else "t IS NULL"
    conn = sqlite3.connect(db_path)
    # lower 全点（0-6）を取得して max pq_len を集計
    all_lower = pd.read_sql(
        f"""
        SELECT idx, image_path, face_index, part_name, sub_key, point_index,
               px, py, qx, qy, t,
               face_yaw, face_pitch, face_roll, eye_scale, norm_eye_scale, lap_var
        FROM samples
        WHERE {t_cond}
          AND skip_flag = 0
          AND sub_key = 'lower'
        """,
        conn,
    )
    conn.close()

    all_lower["pq_len"] = np.sqrt(
        (all_lower["px"] - all_lower["qx"]) ** 2 + (all_lower["py"] - all_lower["qy"]) ** 2
    )
    # 同じ目（image_path, face_index, part_name）内の max pq_len
    max_pq = (
        all_lower.groupby(["image_path", "face_index", "part_name"])["pq_len"]
        .max()
        .rename("max_pq_len")
        .reset_index()
    )
    all_lower = all_lower.merge(max_pq, on=["image_path", "face_index", "part_name"], how="left")
    # pq_ratio_rel: lower[0] の pq_len が同じ目の最大 pq_len の何割か
    all_lower["pq_ratio_rel"] = all_lower["pq_len"] / all_lower["max_pq_len"].clip(lower=1e-6)

    return all_lower[all_lower["point_index"] == 0].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/samples.db")
    parser.add_argument("--output", default="analyses/candidates_lower0.csv")
    parser.add_argument("--n", type=int, default=300, help="出力するパッチ数")
    args = parser.parse_args()

    # ---- 1. 学習データ（DB の labeled lower[0]） -----------------------------
    print("[1] loading labeled lower[0] from DB ...")
    train_df = load_from_db(args.db, labeled=True)
    train_df["t_extremeness"] = t_extremeness(train_df["t"])
    print(f"  rows: {len(train_df):,}")
    print(f"  t_extremeness  mean={train_df['t_extremeness'].mean():.4f}"
          f"  median={train_df['t_extremeness'].median():.4f}"
          f"  max={train_df['t_extremeness'].max():.4f}")
    print(f"  t range: [{train_df['t'].min():.3f}, {train_df['t'].max():.3f}]")

    X_train = build_features(train_df)
    y_train = train_df["t_extremeness"]

    # ---- 2. XGBoost 学習 -----------------------------------------------------
    print("\n[2] training XGBoost ...")
    model = train_xgb(X_train, y_train)

    imp = (
        pd.Series(model.feature_importances_, index=X_train.columns)
        .sort_values(ascending=False)
    )
    print("\n  Feature importances:")
    for feat, v in imp.items():
        print(f"    {feat:<25} {v:.4f}")

    # ---- 3. 未ラベル lower[0] 読み込み --------------------------------------
    labeled_images = set(train_df["image_path"].unique())
    print(f"\n[3] loading unlabeled lower[0] from DB ...")
    unlabeled = load_from_db(args.db, labeled=False)
    before = len(unlabeled)
    unlabeled = unlabeled[~unlabeled["image_path"].isin(labeled_images)].reset_index(drop=True)
    print(f"  rows: {before:,} → {len(unlabeled):,} (labeled images 除外後)")

    if unlabeled.empty:
        print("候補が見つかりませんでした。")
        return

    # ---- 4. スコアリング -----------------------------------------------------
    print("\n[4] scoring ...")
    X_infer = build_features(unlabeled)
    unlabeled = unlabeled.copy()
    unlabeled["pred_score"] = model.predict(X_infer)
    unlabeled["pq_ratio"] = (
        np.sqrt((unlabeled["px"] - unlabeled["qx"]) ** 2 + (unlabeled["py"] - unlabeled["qy"]) ** 2)
        / unlabeled["eye_scale"].clip(lower=1e-6)
    )

    top = unlabeled.sort_values("pred_score", ascending=False).head(args.n).reset_index(drop=True)

    # ---- 5. サマリー ---------------------------------------------------------
    print(f"\n[5] top {args.n} summary:")
    print(f"  pred_score     mean={top['pred_score'].mean():.4f}"
          f"  min={top['pred_score'].min():.4f}"
          f"  max={top['pred_score'].max():.4f}")
    print(f"  |pitch|>20°:  {(top['face_pitch'].abs() > 20).sum()} / {len(top)}"
          f"  ({100*(top['face_pitch'].abs() > 20).mean():.1f}%)")
    print(f"  pq_ratio<0.10: {(top['pq_ratio'] < 0.10).sum()} / {len(top)}"
          f"  ({100*(top['pq_ratio'] < 0.10).mean():.1f}%)")

    print(f"\n  [ref] all unlabeled lower[0] ({len(unlabeled):,} patches):")
    print(f"  |pitch|>20°:  {(unlabeled['face_pitch'].abs() > 20).sum()} / {len(unlabeled)}"
          f"  ({100*(unlabeled['face_pitch'].abs() > 20).mean():.1f}%)")
    print(f"  pq_ratio<0.10: {(unlabeled['pq_ratio'] < 0.10).sum()} / {len(unlabeled)}"
          f"  ({100*(unlabeled['pq_ratio'] < 0.10).mean():.1f}%)")

    # ---- 6. 保存 -------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "idx", "image_path", "face_index", "part_name", "sub_key", "point_index",
        "pred_score", "pq_ratio", "pq_ratio_rel", "face_pitch", "face_yaw", "face_roll",
        "eye_scale", "norm_eye_scale", "lap_var",
        "px", "py", "qx", "qy",
    ]
    top[cols].to_csv(out_path, index=False)
    print(f"\n[6] saved → {out_path}  ({len(top)} rows)")


if __name__ == "__main__":
    main()
