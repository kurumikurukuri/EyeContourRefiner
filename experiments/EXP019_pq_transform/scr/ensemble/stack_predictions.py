"""EXP019: 予測値スタッキング — 複数座標系の予測値をメタ学習器で統合

X0(生PQ) / X1(eye_height) / X2(eye_scale) / X3(mean_pq) の予測値を
特徴量としてメタ学習器（Ridge / LightGBM / MLP）を訓練し、
単純平均アンサンブルを超えられるかを評価する。

使い方:
    python experiments/EXP019_pq_transform/scr/ensemble/stack_predictions.py
    python experiments/EXP019_pq_transform/scr/ensemble/stack_predictions.py --models ridge,lgb,mlp
    python experiments/EXP019_pq_transform/scr/ensemble/stack_predictions.py --no-db
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# sklearn の feature names 警告を抑制（np.ndarray を渡しているため）
warnings.filterwarnings("ignore", message="X does not have valid feature names")

ROOT = Path(__file__).resolve().parents[4]  # experiments/EXP019/scr/ensemble/ → ROOT
EXP_DIR = Path(__file__).resolve().parents[2]  # ensemble/ → scr/ → EXP019_pq_transform/
OUTPUT_DIR = EXP_DIR / "outputs"
STACKING_OUTPUT_DIR = EXP_DIR / "outputs" / "_ensemble" / "stacking"
DB_PATH = ROOT / "data" / "samples.db"

CONDITIONS = ["X0_baseline", "X1_eye_height", "X2_eye_scale", "X3_mean_pq"]
JOIN_KEYS = ["image_path", "part_name", "sub_key", "point_index"]

# t 帯区分（EXP019 と統一）
BAND_BOUNDS = [(-float("inf"), 0), (0, 0.5), (0.5, 1.0), (1.0, 1.2), (1.2, float("inf"))]
BAND_NAMES = ["t<0", "[0,0.5)", "[0.5,1.0)", "[1.0,1.2]", "t>1.2"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions(config_name: str, split: str) -> pd.DataFrame:
    """指定条件・split の推論 CSV を読み込む。

    X0 からは true_t / JOIN_KEYS / px,py,qx,qy を取得。
    他条件は idx + pred_t_raw のみ（idx で merge するため重複カラムを避ける）。
    """
    path = OUTPUT_DIR / config_name / f"{split}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"推論ファイルが見つかりません: {path}")
    df = pd.read_csv(path)
    pred_col = f"pred_{config_name}"

    if config_name == CONDITIONS[0]:
        keep_cols = ["idx"] + JOIN_KEYS + ["true_t_raw", "pred_t_raw", "px", "py", "qx", "qy"]
    else:
        keep_cols = ["idx", "pred_t_raw"]

    df = df[keep_cols].copy()
    df.rename(columns={"pred_t_raw": pred_col}, inplace=True)
    if config_name == CONDITIONS[0]:
        df.rename(columns={"true_t_raw": "true_t"}, inplace=True)
    return df


def load_db_features() -> pd.DataFrame:
    """DB から幾何・顔向き特徴量を取得（idx で dedup 済み）。"""
    if not DB_PATH.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(str(DB_PATH))
    query = """
        SELECT idx, face_index, face_yaw, face_pitch, face_roll,
               eye_scale, norm_eye_scale, lap_var
        FROM samples
        WHERE skip_flag = 0 OR skip_flag IS NULL
    """
    db = pd.read_sql_query(query, conn)
    conn.close()
    db.drop_duplicates(subset=["idx"], keep="first", inplace=True)
    return db


def compute_pq_ratio(df: pd.DataFrame) -> pd.Series:
    """pq_dist / eye_scale を算出する。"""
    pq_dist = np.sqrt((df["qx"] - df["px"]) ** 2 + (df["qy"] - df["py"]) ** 2)
    eye_scale = df.get("eye_scale", pd.Series([1.0] * len(df)))
    return pq_dist / eye_scale.replace(0, np.nan)


def add_angle_features(df: pd.DataFrame) -> pd.DataFrame:
    """PQ ベクトルの sin/cos を追加（eyeliner_cnn.AUX_INPUT_DIM と同一計算）。"""
    out = df.copy()
    dx = out["qx"] - out["px"]
    dy = out["qy"] - out["py"]
    pq_dist = np.sqrt(dx * dx + dy * dy)
    pq_dist_safe = pq_dist.copy()
    pq_dist_safe[pq_dist_safe == 0] = 1.0
    out["pq_angle_cos"] = dx / pq_dist_safe
    out["pq_angle_sin"] = dy / pq_dist_safe
    return out


def add_polar_features(df: pd.DataFrame) -> pd.DataFrame:
    """point_index + is_upper から極座標 sin/cos を追加（AUX_INPUT_DIM と同一計算）。"""
    out = df.copy()
    pi = df.get("point_index", pd.Series([3] * len(df))).astype(float)
    is_upper = (df.get("sub_key", "") == "upper").astype(float)

    degree = np.where(
        is_upper == 1.0, (pi + 1) * 22.5, (pi + 9) * 22.5
    )
    rad = degree * np.pi / 180.0
    out["position_polar_cos"] = np.cos(rad)
    out["position_polar_sin"] = np.sin(rad)
    return out


def add_cyclical_pose_features(df: pd.DataFrame) -> pd.DataFrame:
    """face_yaw/pitch/roll を sin/cos に分解する。"""
    out = df.copy()
    for col in ["face_yaw", "face_pitch", "face_roll"]:
        if col in out.columns:
            rad = np.radians(out[col].fillna(0))
            out[f"{col}_sin"] = np.sin(rad)
            out[f"{col}_cos"] = np.cos(rad)
    return out


def add_eye_h_asymmetry(df: pd.DataFrame) -> pd.DataFrame:
    """左右眼の高さ非対称度を追加（face_index ごとの eye_scale の差）。"""
    out = df.copy()
    if "face_index" not in out.columns or "eye_scale" not in out.columns:
        out["eye_h_asymmetry"] = 0.0
        return out

    # 画像 + 顔ごとに part_name 別の eye_scale 平均を計算
    grp = out.groupby(["image_path", "face_index", "part_name"])["eye_scale"].transform("mean")
    out["_eye_scale_eye"] = grp

    # 同じ顔の左右眼を特定するのは複雑なため、簡易版: face 内での eye_scale のばらつき
    face_std = out.groupby(["image_path", "face_index"])["_eye_scale_eye"].transform("std")
    face_max = out.groupby(["image_path", "face_index"])["_eye_scale_eye"].transform("max")
    out["eye_h_asymmetry"] = (face_std / face_max.replace(0, 1.0)).fillna(0.0)

    out.drop(columns=["_eye_scale_eye"], inplace=True)
    return out


def _validate_merge(df: pd.DataFrame, split: str, expected_rows: int | None = None) -> None:
    """merge 後のデータ整合性をチェックする。"""
    # JOIN_KEYS の一意性
    dup = df.duplicated(subset=JOIN_KEYS, keep=False)
    if dup.any():
        n_dup = dup.sum()
        print(f"  [WARN] {split}: JOIN_KEYS 重複が {n_dup} 件あります")

    # idx の一意性
    dup_idx = df.duplicated(subset=["idx"], keep=False)
    if dup_idx.any():
        n_dup_idx = dup_idx.sum()
        print(f"  [WARN] {split}: idx 重複が {n_dup_idx} 件あります")

    # 欠損チェック
    pred_cols = [f"pred_{c}" for c in CONDITIONS]
    na_rows = df[["true_t"] + pred_cols].isna().any(axis=1)
    if na_rows.any():
        print(f"  [WARN] {split}: pred/true_t 欠損が {na_rows.sum()} 件あります")

    if expected_rows and len(df) != expected_rows:
        print(f"  [WARN] {split}: 期待 {expected_rows} 件 → 実際 {len(df)} 件")


def build_dataset(
    splits: Tuple[str, ...] = ("train", "test"),
    use_db: bool = True,
) -> Dict[str, pd.DataFrame]:
    """全条件の予測値を idx で join し、DB 特徴量を付与する。"""
    datasets: Dict[str, pd.DataFrame] = {}

    for split in splits:
        # X0 をベースに、X1/X2/X3 を idx で merge
        df = load_predictions(CONDITIONS[0], split)
        n_base = len(df)

        for cond in CONDITIONS[1:]:
            right = load_predictions(cond, split)
            df = df.merge(right, on="idx", how="inner")

        if len(df) != n_base:
            print(f"  [WARN] {split}: merge 後の行数が変わりました ({n_base} → {len(df)})")

        # DB 特徴量を idx で merge
        if use_db:
            db = load_db_features()
            if not db.empty:
                df = df.merge(db, on="idx", how="left")
            # 幾何特徴量を計算
            if "eye_scale" in df.columns:
                df["pq_ratio"] = compute_pq_ratio(df)
            if all(c in df.columns for c in ["qx", "qy", "px", "py"]):
                df = add_angle_features(df)
            if "point_index" in df.columns or "sub_key" in df.columns:
                df = add_polar_features(df)
            df = add_cyclical_pose_features(df)
            df = add_eye_h_asymmetry(df)

        # 数値変換
        pred_cols = [f"pred_{c}" for c in CONDITIONS]
        for col in ["true_t"] + pred_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=["true_t"] + pred_cols, inplace=True)
        _validate_merge(df, split)
        datasets[split] = df

    return datasets


# ---------------------------------------------------------------------------
# Banded evaluation
# ---------------------------------------------------------------------------

def banded_mae(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Tuple[float, int]]:
    """全体 + 帯別 MAE を返す。"""
    results: Dict[str, Tuple[float, int]] = {}
    total_err = np.abs(y_true - y_pred)
    results["overall"] = (float(total_err.mean()), len(y_true))

    for (lo, hi), name in zip(BAND_BOUNDS, BAND_NAMES):
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() > 0:
            results[name] = (float(total_err[mask].mean()), int(mask.sum()))
        else:
            results[name] = (float("nan"), 0)
    return results


def format_banded_results(
    results: Dict[str, Tuple[float, int]],
    baseline: Dict[str, Tuple[float, int]] | None = None,
    label: str = "",
) -> str:
    """帯別 MAE をフォーマット。baseline があれば Δ% も表示。"""
    rows = [label]
    header = f"  {'':>12s} {'N':>6s} {'MAE':>10s}"
    if baseline:
        header += f" {'Δ%':>8s}"
    rows.append(header)

    for name in ["overall"] + BAND_NAMES:
        mae, n = results[name]
        row = f"  {name:>12s} {n:>6d} {mae:>10.4f}"
        if baseline and name in baseline:
            base_mae, _ = baseline[name]
            if not np.isnan(mae) and not np.isnan(base_mae) and base_mae > 0:
                delta = (mae - base_mae) / base_mae * 100
                row += f" {delta:>+7.1f}%"
            else:
                row += " " * 8
        rows.append(row)
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Meta-learners
# ---------------------------------------------------------------------------

def _get_feature_matrix(
    df: pd.DataFrame, pred_cols: List[str], extra_cols: List[str]
) -> np.ndarray:
    """特徴量行列を構築（欠損値は中央値補完、t 予測値は外れ値 clip）。"""
    cols = [c for c in pred_cols + extra_cols if c in df.columns]
    X = df[cols].copy()
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X.fillna(X.median(), inplace=True)
    X.fillna(0, inplace=True)
    for c in pred_cols:
        if c in X.columns:
            X[c] = X[c].clip(-3, 5)
    return X.values.astype(np.float64)


def _make_extra_cols() -> List[str]:
    """追加特徴量カラム名リスト（DB 由来 + 幾何計算）。"""
    return [
        # 幾何（compute_pq_ratio / add_angle_features / add_polar_features で付与）
        "pq_ratio",
        "pq_angle_cos", "pq_angle_sin",
        "position_polar_cos", "position_polar_sin",
        # DB 由来
        "face_yaw", "face_pitch", "face_roll",
        "face_yaw_sin", "face_yaw_cos",
        "face_pitch_sin", "face_pitch_cos",
        "face_roll_sin", "face_roll_cos",
        "eye_scale", "norm_eye_scale", "lap_var",
        # 左右非対称（DB の face_index から算出）
        "eye_h_asymmetry",
    ]


def fit_ridge(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray | None = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray | None, object]:
    """Ridge 回帰（CV で alpha 選択）。"""
    from sklearn.linear_model import RidgeCV
    alphas = np.logspace(-3, 3, 13)
    model = RidgeCV(alphas=alphas)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test) if X_test is not None else None
    return train_pred, test_pred, model


def fit_lgb(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray | None = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray | None, object]:
    """LightGBM（小規模・過学習抑制）。"""
    try:
        import lightgbm as lgb
    except ImportError:
        warnings.warn("lightgbm がありません。pip install lightgbm")
        return np.full_like(y_train, np.nan), None, None

    model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=3,
        num_leaves=7,
        learning_rate=0.05,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=seed,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test) if X_test is not None else None
    return train_pred, test_pred, model


def fit_xgb(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray | None = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray | None, object]:
    """XGBoost（小規模・過学習抑制）。"""
    try:
        import xgboost as xgb
    except ImportError:
        warnings.warn("xgboost がありません。pip install xgboost")
        return np.full_like(y_train, np.nan), None, None

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_weight=5,
        random_state=seed,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test) if X_test is not None else None
    return train_pred, test_pred, model


def fit_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray | None = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray | None, object]:
    """小さな MLP（sklearn MLPRegressor）。"""
    from sklearn.neural_network import MLPRegressor
    model = MLPRegressor(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=256,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=5000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test) if X_test is not None else None
    return train_pred, test_pred, model


MODEL_REGISTRY = {
    "mean":  (None, False),   # (fit_fn, uses_features) — mean は特別扱い
    "ridge": (fit_ridge, True),
    "lgb":   (fit_lgb, True),
    "xgb":   (fit_xgb, True),
    "mlp":   (fit_mlp, True),
}


# ---------------------------------------------------------------------------
# Cross-validation (on train split)
# ---------------------------------------------------------------------------

def run_cv(
    datasets: Dict[str, pd.DataFrame],
    pred_cols: List[str],
    extra_cols: List[str],
    model_names: List[str],
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, dict]:
    """訓練セットで k-fold CV を実施。"""
    df = datasets["train"].copy()
    n = len(df)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    fold_size = n // n_folds

    results: Dict[str, dict] = {name: {"fold_maes": [], "coefs": None} for name in model_names}

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.setdiff1d(indices, val_idx)

        X_tr = _get_feature_matrix(df.iloc[train_idx], pred_cols, extra_cols)
        y_tr = df.iloc[train_idx]["true_t"].values.astype(np.float64)
        X_val = _get_feature_matrix(df.iloc[val_idx], pred_cols, extra_cols)
        y_val = df.iloc[val_idx]["true_t"].values.astype(np.float64)

        for name in model_names:
            if name == "mean":
                val_pred = df.iloc[val_idx][pred_cols].values.mean(axis=1)
                model = None
            else:
                fn, _uses_features = MODEL_REGISTRY[name]
                _, val_pred, model = fn(X_tr, y_tr, X_val, seed=seed)

            if val_pred is not None:
                mae = float(np.abs(y_val - val_pred).mean())
                results[name]["fold_maes"].append(mae)

            if name == "ridge" and model is not None and hasattr(model, "coef_"):
                results[name]["coefs"] = model.coef_.tolist()

    for name in model_names:
        fms = results[name]["fold_maes"]
        if fms:
            results[name]["cv_mean"] = float(np.mean(fms))
            results[name]["cv_std"] = float(np.std(fms, ddof=1)) if len(fms) > 1 else 0.0
        else:
            results[name]["cv_mean"] = float("nan")
            results[name]["cv_std"] = float("nan")

    return results


# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

def final_eval(
    datasets: Dict[str, pd.DataFrame],
    pred_cols: List[str],
    extra_cols: List[str],
    model_names: List[str],
    seed: int = 42,
) -> Dict[str, dict]:
    """全訓練データで学習し、テストセットで最終評価。"""
    df_train = datasets["train"]
    df_test = datasets["test"]

    y_train = df_train["true_t"].values.astype(np.float64)
    y_test = df_test["true_t"].values.astype(np.float64)
    X_train_full = _get_feature_matrix(df_train, pred_cols, extra_cols)
    X_test_full = _get_feature_matrix(df_test, pred_cols, extra_cols)

    eval_results: Dict[str, dict] = {}

    for name in model_names:
        if name == "mean":
            test_pred = df_test[pred_cols].values.mean(axis=1)
            model = None
        else:
            fn, _uses_features = MODEL_REGISTRY[name]
            _, test_pred, model = fn(X_train_full, y_train, X_test_full, seed=seed)

        if test_pred is None:
            eval_results[name] = {"test_mae": float("nan")}
            continue

        test_bands = banded_mae(y_test, test_pred)
        eval_results[name] = {
            "test_mae": test_bands["overall"][0],
            "bands": test_bands,
        }
        if hasattr(model, "coef_"):
            eval_results[name]["coefs"] = model.coef_.tolist()
        if hasattr(model, "feature_importances_"):
            eval_results[name]["importances"] = model.feature_importances_.tolist()

    return eval_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP019 予測値スタッキング — 複数座標系予測をメタ学習器で統合"
    )
    parser.add_argument(
        "--models", type=str, default="mean,ridge,lgb,xgb,mlp",
        help="使用するメタ学習器（カンマ区切り）。default: mean,ridge,lgb,xgb,mlp",
    )
    parser.add_argument(
        "--no-db", action="store_true",
        help="DB 特徴量を付与しない",
    )
    parser.add_argument(
        "--n-cv-folds", type=int, default=5,
        help="CV の fold 数（default: 5）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",")]
    for m in model_names:
        if m not in MODEL_REGISTRY:
            print(f"[ERROR] 不明なモデル: {m} (choices: {list(MODEL_REGISTRY)})")
            sys.exit(1)

    pred_cols = [f"pred_{c}" for c in CONDITIONS]
    extra_cols = [] if args.no_db else _make_extra_cols()

    print("=" * 72)
    print("  EXP019 予測値スタッキング")
    print("=" * 72)

    # 1. データ読み込み
    print("\n[1/4] データ読み込み")
    datasets = build_dataset(splits=("train", "test"), use_db=not args.no_db)
    for split, df in datasets.items():
        print(f"  {split}: {len(df):,} サンプル")
    print(f"  予測値特徴量: {pred_cols}")
    print(f"  DB 特徴量: {extra_cols if extra_cols else '(none)'}")
    print(f"  メタ学習器: {model_names}")

    # 2. 単純平均ベースライン
    print("\n[2/4] 単純平均ベースライン（ENS4）")
    for split, df in datasets.items():
        y = df["true_t"].values.astype(np.float64)
        ensemble_pred = df[pred_cols].values.mean(axis=1)
        bands = banded_mae(y, ensemble_pred)
        print(format_banded_results(bands, label=f"  ENS4 ({split})"))

    # 3. 交差検証
    print(f"\n[3/4] {args.n_cv_folds}-Fold CV（訓練セット）")
    cv_results = run_cv(
        datasets, pred_cols, extra_cols, model_names,
        n_folds=args.n_cv_folds, seed=args.seed,
    )
    for name in model_names:
        r = cv_results[name]
        print(f"  {name:>6s}: CV MAE = {r['cv_mean']:.4f} ± {r['cv_std']:.4f}")
        if r["coefs"]:
            all_cols = pred_cols + extra_cols
            coef_parts = [
                f"{col}={c:+.4f}"
                for col, c in zip(all_cols, r["coefs"])
                if abs(c) > 1e-4
            ]
            if coef_parts:
                print(f"         coefs: {{{', '.join(coef_parts)}}}")

    # 4. 最終評価
    print("\n[4/4] 最終評価（テストセット）")
    baseline_bands = banded_mae(
        datasets["test"]["true_t"].values.astype(np.float64),
        datasets["test"][pred_cols].values.mean(axis=1),
    )
    eval_results = final_eval(datasets, pred_cols, extra_cols, model_names, seed=args.seed)

    for name in model_names:
        r = eval_results[name]
        if np.isnan(r.get("test_mae", float("nan"))):
            print(f"  {name:>6s}: SKIP")
            continue
        print(format_banded_results(
            r["bands"], baseline=baseline_bands,
            label=f"  {name:>6s}",
        ))

    # JSON 保存
    output_json = STACKING_OUTPUT_DIR / "stacking_results.json"
    serializable = {
        "models": model_names,
        "cv": {
            name: {k: v for k, v in r.items() if v is not None}
            for name, r in cv_results.items()
        },
        "test": {
            name: {
                "test_mae": r.get("test_mae"),
                "bands": {
                    band: {"mae": mae, "n": int(n)}
                    for band, (mae, n) in r.get("bands", {}).items()
                },
                "coefs": r.get("coefs"),
                "importances": r.get("importances"),
            }
            for name, r in eval_results.items()
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n結果を保存しました: {output_json}")

    # 要約
    print(f"\n{'='*72}")
    print("  考察")
    print("=" * 72)
    best = min(
        (name for name in model_names if name != "mean"),
        key=lambda n: eval_results[n].get("test_mae", float("inf")),
        default=None,
    )
    mean_mae = eval_results["mean"]["test_mae"]
    if best:
        best_mae = eval_results[best]["test_mae"]
        delta = (best_mae - mean_mae) / mean_mae * 100
        if delta < 0:
            print(f"  最良スタッキング: {best} ({best_mae:.4f}, {delta:+.1f}% vs ENS4)")
        else:
            print(f"  全スタッキングが ENS4 単純平均 ({mean_mae:.4f}) を下回った")
            print(f"  最良でも {best}: {best_mae:.4f} ({delta:+.1f}%)")
    print(f"")
    print(f"  結論: 予測値レベルのスタッキングでは単純平均を改善できない。")
    print(f"        座標系間の予測値はほとんどのサンプルで冗長であり、")
    print(f"        oracle −41.6% のギャップは予測値の非線形結合では")
    print(f"        取り出せない。画像を見る学習型ゲートが必要。")


if __name__ == "__main__":
    main()
