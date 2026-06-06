"""predictions_unlabeled.csv のフィルタリング。

除外条件:
  1. 0 < pred_t_raw < 1.2  （正常範囲内の予測）
  2. DB で t IS NOT NULL   （推論後に新規ラベル済みになったもの）
  3. DB で skip_flag = 1   （スキップ済み）

Usage:
    python analyses/X1_independent_none_eye_scale_pred/scr/filter_predictions.py
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = Path(__file__).resolve().parents[1]

PREDICTIONS_CSV = ANALYSIS_DIR / "predictions_unlabeled.csv"
DB_PATH = ROOT / "data" / "samples.db"


def load_db_status() -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query("SELECT idx, t, skip_flag FROM samples", conn)
    conn.close()
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    logger.info("読み込み: %s", PREDICTIONS_CSV)
    pred = pd.read_csv(PREDICTIONS_CSV)
    logger.info("元行数: %d", len(pred))

    # 1. 正常範囲フィルタ: 0 < pred_t_raw < 1.2 を除外
    mask_extreme = ~((pred["pred_t_raw"] > 0) & (pred["pred_t_raw"] < 1.2))
    pred = pred[mask_extreme]
    logger.info("pred フィルタ後 (pred<=0 or pred>=1.2): %d 行", len(pred))

    # 2. DB の最新状態で t IS NOT NULL / skip_flag=1 を除外
    db = load_db_status()
    pred = pred.merge(db, on="idx", how="left")

    before = len(pred)
    pred = pred[pred["t"].isna() & (pred["skip_flag"] == 0)]
    logger.info("DB フィルタ後 (t=NULL & skip=0): %d 行 (%d 件除外)",
                len(pred), before - len(pred))

    pred = pred.drop(columns=["t", "skip_flag"])
    pred.to_csv(PREDICTIONS_CSV, index=False)
    logger.info("保存: %s", PREDICTIONS_CSV)


if __name__ == "__main__":
    main()
