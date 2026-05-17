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
import csv
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import cross_val_score

PART_MAP = {"eyeliner_left": 0, "eyeliner_right": 1}

# Open Images ラベル → 特徴量名
OI_LABEL_FEATURES: Dict[str, str] = {
    "/m/01bl7v": "has_boy",
    "/m/05r655": "has_girl",
    "/m/017ftj": "has_sunglasses",
    "/m/0jyfg":  "has_glasses",
    "/m/015h_t": "has_beard",
    "/m/04yx4":  "has_man",
    "/m/02_n6y": "has_goggles",
    "/m/061hd_": "has_infant_bed",
}

DEFAULT_CACHE_DIR = Path("data/.open_images_cache")
OI_LABELS_FILE = "oi_image_labels.csv"


# ---------------------------------------------------------------------------
# OI ラベル
# ---------------------------------------------------------------------------

def load_oi_labels(cache_dir: Path) -> Dict[str, Set[str]]:
    """image_id → set of label MIDs を返す。ファイルがなければ空 dict。"""
    path = cache_dir / OI_LABELS_FILE
    if not path.exists():
        return {}
    labels: Dict[str, Set[str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.setdefault(row["image_id"], set()).add(row["label_name"])
    return labels


def _attach_oi_features(df: pd.DataFrame, oi_labels: Dict[str, Set[str]]) -> pd.DataFrame:
    """image_path から image_id を取り OI ラベル bool 列を追加する（in-place 不可）。"""
    df = df.copy()
    image_ids = df["image_path"].apply(lambda p: Path(p).stem)
    for mid, col in OI_LABEL_FEATURES.items():
        df[col] = image_ids.map(lambda iid, m=mid: int(m in oi_labels.get(iid, set())))
    return df


# ---------------------------------------------------------------------------
# skip 予測
# ---------------------------------------------------------------------------

def load_skip_data(db_path: str) -> pd.DataFrame:
    """アノテーター対応済み画像（skip 含む）を画像単位で集約して返す。"""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        """
        SELECT image_path,
               MAX(skip_flag)        AS skipped,
               AVG(ABS(face_pitch))  AS avg_abs_pitch,
               AVG(ABS(face_yaw))    AS avg_abs_yaw,
               AVG(ABS(face_roll))   AS avg_abs_roll,
               AVG(eye_scale)        AS avg_eye_scale,
               AVG(lap_var)          AS avg_lap_var
        FROM samples
        WHERE skip_flag = 1
           OR (skip_flag = 0 AND t IS NOT NULL)
        GROUP BY image_path
        """,
        conn,
    )
    conn.close()
    return df


def _skip_feature_cols() -> list[str]:
    return [
        "avg_abs_pitch", "avg_abs_yaw", "avg_abs_roll",
        "avg_eye_scale", "avg_lap_var",
    ] + list(OI_LABEL_FEATURES.values())


def build_skip_features(df: pd.DataFrame, oi_labels: Dict[str, Set[str]]) -> pd.DataFrame:
    df = _attach_oi_features(df, oi_labels)
    return df[_skip_feature_cols()].fillna(0.0)


def train_skip_model(
    df: pd.DataFrame, oi_labels: Dict[str, Set[str]]
) -> Optional[xgb.XGBClassifier]:
    if df.empty:
        return None
    X = build_skip_features(df, oi_labels)
    y = df["skipped"].astype(int)
    n_skip = int(y.sum())
    n_ok = len(y) - n_skip
    print(f"  学習データ: {len(df)} 画像  (skipped={n_skip}, ok={n_ok})")
    if n_skip < 10 or n_ok < 10:
        print("  データ不足のため skip 予測モデルをスキップ")
        return None

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=n_ok / max(n_skip, 1),
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    cv_auc = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    print(f"  CV-AUC: {cv_auc.mean():.4f}  (std={cv_auc.std():.4f})")
    model.fit(X, y)

    imp = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
    )
    print("  Feature importances (skip model):")
    for feat, v in imp.items():
        print(f"    {feat:<25} {v:.4f}")

    return model


def predict_skip_prob(
    df: pd.DataFrame,
    oi_labels: Dict[str, Set[str]],
    model: xgb.XGBClassifier,
) -> pd.Series:
    """df の各行に skip_prob を返す（image_path 単位で集約→推論し patch にブロードキャスト）。"""
    img_df = df.groupby("image_path", as_index=False).agg(
        avg_abs_pitch=("face_pitch", lambda x: x.abs().mean()),
        avg_abs_yaw=("face_yaw",    lambda x: x.abs().mean()),
        avg_abs_roll=("face_roll",  lambda x: x.abs().mean()),
        avg_eye_scale=("eye_scale", "mean"),
        avg_lap_var=("lap_var",     "mean"),
    )
    X = build_skip_features(img_df, oi_labels)
    img_df["skip_prob"] = model.predict_proba(X)[:, 1]
    return df["image_path"].map(img_df.set_index("image_path")["skip_prob"]).fillna(0.0)


# ---------------------------------------------------------------------------
# t_extremeness
# ---------------------------------------------------------------------------

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
            "eye_opening_ratio": df["eye_opening_ratio"],  # upper[3]-lower[3] 距離 / eye_scale
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
    """lower[0] を返す。pq_ratio_rel・eye_opening の計算のため lower[0-6]/upper[3] も経由する。"""
    t_cond = "t IS NOT NULL" if labeled else "t IS NULL"
    conn = sqlite3.connect(db_path)
    all_pts = pd.read_sql(
        f"""
        SELECT idx, image_path, face_index, part_name, sub_key, point_index,
               px, py, qx, qy, t,
               face_yaw, face_pitch, face_roll, eye_scale, norm_eye_scale, lap_var
        FROM samples
        WHERE {t_cond}
          AND skip_flag = 0
          AND (sub_key = 'lower' OR (sub_key = 'upper' AND point_index = 3))
        """,
        conn,
    )
    conn.close()

    all_pts["pq_mid_x"] = (all_pts["px"] + all_pts["qx"]) / 2
    all_pts["pq_mid_y"] = (all_pts["py"] + all_pts["qy"]) / 2
    all_pts["pq_len"] = np.sqrt(
        (all_pts["px"] - all_pts["qx"]) ** 2 + (all_pts["py"] - all_pts["qy"]) ** 2
    )

    group = ["image_path", "face_index", "part_name"]

    # lower[0-6] の max pq_len → pq_ratio_rel
    all_lower = all_pts[all_pts["sub_key"] == "lower"].copy()
    max_pq = (
        all_lower.groupby(group)["pq_len"].max().rename("max_pq_len").reset_index()
    )
    all_lower = all_lower.merge(max_pq, on=group, how="left")
    all_lower["pq_ratio_rel"] = all_lower["pq_len"] / all_lower["max_pq_len"].clip(lower=1e-6)

    # upper[3] の中点
    upper3 = (
        all_pts[(all_pts["sub_key"] == "upper") & (all_pts["point_index"] == 3)]
        [group + ["pq_mid_x", "pq_mid_y"]]
        .rename(columns={"pq_mid_x": "u3_mid_x", "pq_mid_y": "u3_mid_y"})
    )
    # lower[3] の中点
    lower3 = (
        all_lower[all_lower["point_index"] == 3]
        [group + ["pq_mid_x", "pq_mid_y"]]
        .rename(columns={"pq_mid_x": "l3_mid_x", "pq_mid_y": "l3_mid_y"})
    )

    # eye_opening: upper[3] と lower[3] の中点間距離
    eye_open = upper3.merge(lower3, on=group, how="inner")
    eye_open["eye_opening"] = np.sqrt(
        (eye_open["u3_mid_x"] - eye_open["l3_mid_x"]) ** 2
        + (eye_open["u3_mid_y"] - eye_open["l3_mid_y"]) ** 2
    )
    eye_open = eye_open[group + ["eye_opening"]]

    # lower[0] に結合
    lower0 = all_lower[all_lower["point_index"] == 0].reset_index(drop=True)
    lower0 = lower0.merge(eye_open, on=group, how="left")
    lower0["eye_opening_ratio"] = lower0["eye_opening"] / lower0["eye_scale"].clip(lower=1e-6)

    return lower0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/samples.db")
    parser.add_argument("--output", default="analyses/xgboost_shortpqlens/outputs/candidates_lower0.csv")
    parser.add_argument("--n", type=int, default=1000, help="出力するパッチ数")
    parser.add_argument("--cache", default=DEFAULT_CACHE_DIR, type=Path,
                        help="OI ラベルキャッシュディレクトリ")
    args = parser.parse_args()

    # ---- 0. OI ラベル読み込み ------------------------------------------------
    print("[0] loading OI image labels ...")
    oi_labels = load_oi_labels(args.cache)
    if oi_labels:
        print(f"  OI ラベルあり: {len(oi_labels)} 画像")
    else:
        print("  OI ラベルなし（tools/fetch_oi_image_labels.py を先に実行してください）")

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

    # ---- 3.5. skip 予測モデル ------------------------------------------------
    print("\n[3.5] training skip predictor ...")
    skip_df = load_skip_data(args.db)
    skip_model = train_skip_model(skip_df, oi_labels)

    # ---- 4. スコアリング -----------------------------------------------------
    print("\n[4] scoring ...")
    X_infer = build_features(unlabeled)
    unlabeled = unlabeled.copy()
    unlabeled["pred_score"] = model.predict(X_infer)
    unlabeled["pq_ratio"] = (
        np.sqrt((unlabeled["px"] - unlabeled["qx"]) ** 2 + (unlabeled["py"] - unlabeled["qy"]) ** 2)
        / unlabeled["eye_scale"].clip(lower=1e-6)
    )

    if skip_model is not None:
        unlabeled["skip_prob"] = predict_skip_prob(unlabeled, oi_labels, skip_model)
        print(f"  skip_prob  mean={unlabeled['skip_prob'].mean():.3f}"
              f"  median={unlabeled['skip_prob'].median():.3f}"
              f"  >0.5: {(unlabeled['skip_prob'] > 0.5).sum():,}")
    else:
        unlabeled["skip_prob"] = 0.0

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
        "pred_score", "skip_prob",
        "pq_len", "pq_ratio", "pq_ratio_rel",
        "eye_opening", "eye_opening_ratio",
        "face_pitch", "face_yaw", "face_roll",
        "eye_scale", "norm_eye_scale", "lap_var",
        "px", "py", "qx", "qy",
    ]
    top[cols].to_csv(out_path, index=False)
    print(f"\n[6] saved → {out_path}  ({len(top)} rows)")


if __name__ == "__main__":
    main()
