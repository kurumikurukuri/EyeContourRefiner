"""samples.csv から未ラベル upper サンプルを抽出し、infer.py 用の manifest を作成する。

infer.py は t カラムに有効値が入っていないと行をスキップする仕様なので、
ダミー値 (0.5) を埋めて推論させる。pred_t 側が本来の出力で、true_t_raw は
後段のフィルタリングでは参照しない。

Usage:
    python analyses/double_fold_candidates/build_unlabeled_manifest.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_CSV = PROJECT_ROOT / "data" / "samples.csv"
OUTPUT_DIR = PROJECT_ROOT / "analyses" / "double_fold_candidates"
OUTPUT_CSV = OUTPUT_DIR / "manifest_unlabeled_upper.csv"

DUMMY_T = 0.5


def build() -> Path:
    df = pd.read_csv(SAMPLES_CSV)
    logger.info("samples.csv loaded: %d rows", len(df))

    mask = (
        df["true_t_raw"].isna()
        & (df["sub_key"] == "upper")
        & (df["skip_flag"] == 0)
    )
    sub = df.loc[mask].copy()
    logger.info("filtered unlabeled upper (skip_flag=0): %d rows", len(sub))

    sub["t"] = DUMMY_T
    keep_cols = [
        "idx",
        "image_path",
        "face_index",
        "part_name",
        "sub_key",
        "point_index",
        "px",
        "py",
        "qx",
        "qy",
        "t",
        "skip_flag",
    ]
    out = sub[keep_cols]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    logger.info("wrote: %s (%d rows)", OUTPUT_CSV, len(out))
    return OUTPUT_CSV


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build()
