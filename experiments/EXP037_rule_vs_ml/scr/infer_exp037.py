"""EXP037 推論スクリプト: 固定テストセット（db_replica, 4,900 点）の全点を、ルールベース（face_makeup）と
ML（eye_contour_refiner.onnx）の両方で補正する。
ルールベースへの入力は face_makeup 本番と同じく整数に丸めた P,Q（to_image_coords 相当）、ML は小数の P,Q。

出力:
    outputs/rule_based/test_predictions.csv   評価ツール標準形式（idx, ..., t_true, t_pred）
    outputs/ml_onnx/test_predictions.csv      同上
    outputs/points_exp037.csv                 点ごとの両方式の t・補正点・誤差・fallback

ML は目（14 点）単位で 1 バッチ推論する。eye_scale はその目の 14 点から計算し、DB の値と一致することを確認する。

使い方:
    python experiments/EXP037_rule_vs_ml/scr/infer_exp037.py
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
MANIFEST_PATH = EXP_DIR / "manifests" / "manifest_test.csv"
OUTPUT_DIR = EXP_DIR / "outputs"
MODEL_PATH = ROOT / "models" / "eye_contour_refiner.onnx"
DB_PATH = ROOT / "data" / "samples.db"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from refiners_exp037 import (
    eye_scale_of,
    refine_ml,
    refine_rule,
    rule_point_to_t,
    to_production_coords,
)

EYE_KEY = ["image_path", "face_index", "part_name"]
REQUIRED_COLUMNS = {"idx", *EYE_KEY, "sub_key", "point_index", "px", "py", "qx", "qy", "t"}
PREDICTION_COLUMNS = ["idx", "image_path", "part_name", "sub_key", "point_index", "t_true", "t_pred"]
EYE_SCALE_TOLERANCE = 1e-3
# rule は本番と同じ整数座標を入力にした従来の出力（主結果）。以下は参考値:
#   rule_untrunc = rule の出力点を整数に切り捨てる前の検出位置
#   rule_float   = 小数の P,Q をそのまま入力した場合
#   ml_int       = ML にルールと同じ整数座標の P,Q（eye_scale も整数座標から計算）を入力した場合
ERROR_METHODS = ("rule", "rule_untrunc", "rule_float", "ml", "ml_int")


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest に必要な列がありません: {sorted(missing)} ({path})")
    eye_sizes = manifest.groupby(EYE_KEY).size()
    if not (eye_sizes == 14).all():
        raise ValueError(f"14 点揃っていない目があります: {eye_sizes[eye_sizes != 14].to_dict()}")
    return manifest


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"画像を読み込めません: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def refine_eye(session: ort.InferenceSession, image: np.ndarray, eye: pd.DataFrame) -> pd.DataFrame:
    """1 つの目（14 点）を両方式で補正し、点ごとの結果を返す。"""
    pq = eye[["px", "py", "qx", "qy"]].to_numpy(dtype=np.float64)
    is_upper = (eye["sub_key"] == "upper").to_numpy()
    eye_scale = eye_scale_of(pq)

    pq_production = to_production_coords(pq)

    rule = refine_rule(image, pq_production)
    rule_float = refine_rule(image, pq)
    ml = refine_ml(session, image, pq, is_upper, np.full(len(pq), eye_scale))
    ml_int = refine_ml(session, image, pq_production, is_upper, np.full(len(pq), eye_scale_of(pq_production)))

    # 切り捨て前の検出位置は丸め後の P,Q 上の点なので、点に戻してから正解の P-Q 直線に射影する
    t_u = rule.t_untruncated[:, None]
    untrunc = t_u * pq_production[:, :2] + (1.0 - t_u) * pq_production[:, 2:]
    # t はすべて正解を定義している元の（小数の）P-Q 直線への射影で求める
    return eye.assign(
        eye_scale=eye_scale,
        t_true=eye["t"],
        t_rule=rule_point_to_t(pq, rule.points),
        t_rule_untrunc=rule_point_to_t(pq, untrunc),
        t_rule_float=rule_point_to_t(pq, rule_float.points),
        t_ml=ml.t,
        t_ml_int=rule_point_to_t(pq, ml_int.points),
        rule_fallback=rule.fallback,
        rule_float_fallback=rule_float.fallback,
        rx_rule=rule.points[:, 0], ry_rule=rule.points[:, 1],
        rx_rule_untrunc=untrunc[:, 0], ry_rule_untrunc=untrunc[:, 1],
        rx_rule_float=rule_float.points[:, 0], ry_rule_float=rule_float.points[:, 1],
        rx_ml=ml.points[:, 0], ry_ml=ml.points[:, 1],
        rx_ml_int=ml_int.points[:, 0], ry_ml_int=ml_int.points[:, 1],
    ).drop(columns=["t"])


def add_errors(points: pd.DataFrame) -> pd.DataFrame:
    """正解点 R_true との誤差（t 誤差・ピクセル誤差・error_ratio）を追加する。"""
    t = points["t_true"]
    rx_true = t * points["px"] + (1 - t) * points["qx"]
    ry_true = t * points["py"] + (1 - t) * points["qy"]
    pq_dist = np.hypot(points["qx"] - points["px"], points["qy"] - points["py"])
    errors = {}
    for method in ERROR_METHODS:
        t_err = (points[f"t_{method}"] - t).abs()
        errors[f"abs_t_err_{method}"] = t_err
        errors[f"px_err_{method}"] = np.hypot(points[f"rx_{method}"] - rx_true, points[f"ry_{method}"] - ry_true)
        errors[f"error_ratio_{method}"] = t_err * pq_dist / points["eye_scale"]
    return points.assign(pq_dist=pq_dist, **errors)


def require_db(db_path: Path) -> None:
    """sqlite3.connect は存在しないパスに空の DB を作るので、開く前に存在を確認する。"""
    if not db_path.exists():
        raise FileNotFoundError(
            f"DB が見つかりません: {db_path}（tools/data_management/restore_dataset.py で復元する）"
        )


def check_eye_scale_against_db(points: pd.DataFrame, db_path: Path) -> float:
    """計算した eye_scale と DB の eye_scale（学習時の値）の最大差を返す。許容差を超えたら例外。"""
    require_db(db_path)
    with sqlite3.connect(db_path) as con:
        db = pd.read_sql("SELECT idx, eye_scale AS eye_scale_db FROM samples", con)
    joined = points.merge(db, on="idx", how="left")
    if joined["eye_scale_db"].isna().any():
        raise ValueError("DB に存在しない idx があります")
    max_diff = float((joined["eye_scale"] - joined["eye_scale_db"]).abs().max())
    if max_diff > EYE_SCALE_TOLERANCE:
        raise ValueError(f"eye_scale が DB と一致しません（最大差 {max_diff}）")
    return max_diff


def write_predictions(points: pd.DataFrame, method: str, condition: str) -> Path:
    out_dir = OUTPUT_DIR / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "test_predictions.csv"
    points.assign(t_pred=points[f"t_{method}"])[PREDICTION_COLUMNS].to_csv(path, index=False, float_format="%.6f")
    return path


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description="EXP037: ルールベースと ONNX で固定テストセットを補正する")
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    ap.add_argument("--db", type=Path, default=DB_PATH, help="eye_scale の照合に使う samples.db")
    args = ap.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"ONNX が見つかりません: {args.model}（EyeContourRefiner main の models/ からコピーする）")
    # 全点の推論を終えてから DB 不在で落ちないよう、先に確認する
    require_db(args.db)

    manifest = load_manifest(args.manifest)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])

    results = []
    for image_path, image_rows in manifest.groupby("image_path", sort=False):
        image = load_rgb(ROOT / image_path)
        for _, eye in image_rows.groupby(EYE_KEY, sort=False):
            results.append(refine_eye(session, image, eye))
    points = add_errors(pd.concat(results, ignore_index=True))

    max_diff = check_eye_scale_against_db(points, args.db)
    print(f"eye_scale と DB の最大差: {max_diff:.2e}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    points.to_csv(OUTPUT_DIR / "points_exp037.csv", index=False, float_format="%.6f")
    for method, condition in (("rule", "rule_based"), ("ml", "ml_onnx")):
        print(f"{condition}: {write_predictions(points, method, condition)}")
    for method in ERROR_METHODS:
        print(f"{method}: MAE(t)={points[f'abs_t_err_{method}'].mean():.6f}  "
              f"px_err={points[f'px_err_{method}'].mean():.3f}")
    print(f"rule fallback: {points['rule_fallback'].mean() * 100:.2f}%  (n={len(points)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
