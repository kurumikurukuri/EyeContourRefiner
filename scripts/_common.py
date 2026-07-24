"""scripts/ 共有ユーティリティ。

compute_passrate.py, paired_ttest.py, init_exp_summary.py,
generate_summary.py から import される共通シンボルを一元管理する。
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "samples.db"

# ---------------------------------------------------------------------------
# Thresholds (from experiments/Completion_Criteria.md)
# ---------------------------------------------------------------------------

THRESHOLD_STD = 0.01788    # p10
THRESHOLD_STRICT = 0.01604  # p5

# ---------------------------------------------------------------------------
# Standardised t-bands
# ---------------------------------------------------------------------------

T_BANDS: list[tuple[str, Callable[[float], bool]]] = [
    ("t<0",     lambda t: t < 0),
    ("0-0.3",   lambda t: (t >= 0) & (t < 0.3)),
    ("0.3-0.7", lambda t: (t >= 0.3) & (t < 0.7)),
    ("0.7-1.0", lambda t: (t >= 0.7) & (t < 1.0)),
    ("1.0-1.2", lambda t: (t >= 1.0) & (t < 1.2)),
    ("t>=1.2",  lambda t: t >= 1.2),
]

# backward-compat alias
BANDS = T_BANDS


# ---------------------------------------------------------------------------
# NaN helpers
# ---------------------------------------------------------------------------

def is_missing(v: object) -> bool:
    """Check if a value is missing (None or NaN). Safe for any type."""
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return False


def sanitize_for_json(obj: object) -> object:
    """Recursively replace float('nan') with None for valid JSON output."""
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}  # type: ignore[arg-type]
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def load_db_index(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    """samples.db から idx, px, py, qx, qy, eye_scale を読み込み idx を index にした DataFrame を返す。"""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"データベースが見つかりません: {db_path}")
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(
                "SELECT idx, px, py, qx, qy, eye_scale FROM samples", conn
            )
        return df.set_index("idx")
    except sqlite3.Error as e:
        raise sqlite3.Error(f"データベース読み込みエラー ({db_path}): {e}") from e


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_error_ratio(pred_df: pd.DataFrame, db_df: pd.DataFrame) -> pd.Series:
    """predictions.csv の DataFrame と DB index から error_ratio を計算する。

    error_ratio = |t_pred - t_true| * pq_dist / eye_scale

    DB に存在しない idx、eye_scale <= 0 の行は除外される。
    戻り値の Series は pred_df と同じ index を持ち、無効行は NaN。

    Args:
        pred_df: predictions.csv を読み込んだ DataFrame (columns: idx, t_true, t_pred)
        db_df:   load_db_index() の戻り値 (index: idx)

    Returns:
        error_ratio の Series (pred_df と同じ index, 無効行は NaN)
    """
    joined = pred_df.join(db_df, on="idx", rsuffix="_db")

    # px, py, qx, qy: CSV にあればそれを、なければ DB の値を使う
    for col in ("px", "py", "qx", "qy"):
        if col not in joined.columns:
            joined[col] = joined.get(f"{col}_db", np.nan)

    pq_dist = np.sqrt(
        (joined["px"] - joined["qx"]) ** 2 + (joined["py"] - joined["qy"]) ** 2
    )
    eye_scale = joined["eye_scale"]

    valid = eye_scale.notna() & (eye_scale > 0) & pq_dist.notna()
    result = pd.Series(np.nan, index=pred_df.index)
    result[valid] = (
        (joined["t_pred"] - joined["t_true"]).abs()[valid]
        * pq_dist[valid]
        / eye_scale[valid]
    )
    return result


# ---------------------------------------------------------------------------
# Experiment discovery
# ---------------------------------------------------------------------------

def find_exp_dir(exp_num: str) -> Path:
    """EXP番号から実験ディレクトリを探す（例: "026" or "EXP026" → experiments/EXP026_xxx/）"""
    num = exp_num.removeprefix("EXP") if exp_num.startswith("EXP") else exp_num
    pattern = f"EXP{num}_*"
    matches = sorted((ROOT / "experiments").glob(pattern))
    if not matches:
        raise FileNotFoundError(f"実験ディレクトリが見つかりません: experiments/{pattern}")
    if len(matches) > 1:
        raise ValueError(
            f"複数の実験ディレクトリが見つかりました: {[m.name for m in matches]}"
        )
    return matches[0]


# infer スクリプトによって出力ファイル名が異なる（predictions.csv or test_predictions.csv）。
# 両方存在する場合は predictions.csv を優先。train_predictions.csv は対象外（holdout 評価用ではないため）。
PREDICTIONS_CSV_NAMES: tuple[str, ...] = ("predictions.csv", "test_predictions.csv")


def find_predictions_csv(exp_dir: Path, config: str) -> Path:
    """実験ディレクトリ + config 名から predictions.csv（or test_predictions.csv）を探す。"""
    config_dir = exp_dir / "outputs" / config
    for name in PREDICTIONS_CSV_NAMES:
        csv_path = config_dir / name
        if csv_path.exists():
            return csv_path
    raise FileNotFoundError(
        f"{' または '.join(PREDICTIONS_CSV_NAMES)} が見つかりません: {config_dir}"
    )
