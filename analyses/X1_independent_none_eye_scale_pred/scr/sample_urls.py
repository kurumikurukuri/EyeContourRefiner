"""フィルタ後に idx をランダムサンプルして annotator URL を表示する。

Usage:
    python analyses/X1_independent_none_eye_scale_pred/scr/sample_urls.py
    python analyses/X1_independent_none_eye_scale_pred/scr/sample_urls.py --n 20
    python analyses/X1_independent_none_eye_scale_pred/scr/sample_urls.py --seed 42
"""
from __future__ import annotations

import argparse
import runpy
from pathlib import Path

import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS_CSV = ANALYSIS_DIR / "predictions_unlabeled.csv"
FILTER_SCRIPT = Path(__file__).resolve().parent / "filter_predictions.py"
ANNOTATOR_URL = "http://127.0.0.1:8000/?idx={idx}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="表示サンプル数")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード")
    parser.add_argument("--no-filter", action="store_true", help="フィルタをスキップ")
    args = parser.parse_args()

    if not args.no_filter:
        runpy.run_path(str(FILTER_SCRIPT), run_name="__main__")

    df = pd.read_csv(PREDICTIONS_CSV)
    idxs = df["idx"].drop_duplicates()
    n = min(args.n, len(idxs))
    sample = idxs.sample(n, random_state=args.seed).tolist()

    print(f"\n--- {n} サンプル ---")
    for idx in sample:
        print(ANNOTATOR_URL.format(idx=idx))


if __name__ == "__main__":
    main()
