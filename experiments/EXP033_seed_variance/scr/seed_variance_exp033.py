"""EXP033: seed 間ばらつきの集計。

3 seed（dual_s42/43/44）の test_predictions.csv から次を算出する。

  1. seed ごとの MAE(生t)・pass_std・pass_strict（全体 + t帯別）と、seed 間の mean / sd / range
  2. seed 同士の全 3 ペアの paired t-test（seed だけの違いで何回 p<0.05 が出るか
     ＝ 1 seed 実験の有意差判定がどれだけ偶然に左右されるか）
  3. 3 seed アンサンブル（t_pred の単純平均）の MAE・pass_std（参考値）
  4. 再現性チェック: dual_s42 vs EXP031 SD0_baseline（同一構成・同一 seed）の paired t-test。
     SD0_baseline の予測 CSV は読み取り専用で参照する（EXP032 の compare スクリプトと同じ先例。
     他 EXP のコード・manifest は import しない）。

pass_std は現在の samples.db の eye_scale で再計算するため、SD0_baseline も同じ DB で
評価し直した値を並べる（過去 summary の数値とは DB 復元差の分だけずれ得る）。

使い方:
    python experiments/EXP033_seed_variance/scr/seed_variance_exp033.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
OUTPUT_DIR = EXP_DIR / "outputs"
sys.path.insert(0, str(ROOT / "scripts"))

from _common import (  # noqa: E402
    T_BANDS,
    load_db_index,
    read_predictions_csv,
    sanitize_for_json,
)
from compute_passrate import compute_band_stats  # noqa: E402
from paired_ttest import paired_comparison  # noqa: E402

CONFIGS = ("dual_s42", "dual_s43", "dual_s44")
ENSEMBLE_NAME = "seed_ensemble"
REFERENCE_CSV = (
    ROOT / "experiments" / "EXP031_source_space_aug" / "outputs" / "SD0_baseline" / "test_predictions.csv"
)
METRICS = ("mae_raw_t", "pass_std", "pass_strict")


def prediction_path(config: str) -> Path:
    path = OUTPUT_DIR / config / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"test_predictions.csv が見つかりません（推論未実行？）: {path}")
    return path


def build_seed_ensemble(paths: list[Path]) -> pd.DataFrame:
    """各 seed の t_pred を idx で揃えて単純平均した予測 DataFrame を返す。"""
    frames = [read_predictions_csv(p).set_index("idx") for p in paths]
    common = sorted(set.intersection(*(set(f.index) for f in frames)))
    if len(common) != len(frames[0]):
        raise ValueError(f"seed 間で idx 集合が一致しません（共通 {len(common)} / {len(frames[0])}）")
    base = frames[0].loc[common]
    mean_pred = np.mean([f.loc[common, "t_pred"].to_numpy() for f in frames], axis=0)
    return base.assign(t_pred=mean_pred).reset_index()


def spread(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "range": float(arr.max() - arr.min()),
    }


def summarize_spread(stats: dict[str, dict]) -> dict:
    """seed ごとの stats から、全体・t帯別の各指標の mean/sd/range を作る。"""
    seeds = [stats[c] for c in CONFIGS]
    overall = {m: spread([s[m] for s in seeds]) for m in METRICS}
    bands = {
        bname: {m: spread([s["bands"][bname][m] for s in seeds]) for m in METRICS}
        for bname, _ in T_BANDS
    }
    return {"overall": overall, "bands": bands}


def pairwise_tests(db_df: pd.DataFrame) -> dict:
    results = {}
    for a, b in itertools.combinations(CONFIGS, 2):
        results[f"{b}_vs_{a}"] = paired_comparison(
            prediction_path(a), prediction_path(b), db_df, metric="raw_t")
    return results


def print_report(stats: dict, spread_summary: dict, pairs: dict, repro: dict | None) -> None:
    print(f"\n{'='*72}\n  seed ごとの結果（db_replica hold-out）\n{'='*72}")
    print(f"  {'config':<16} {'mae_raw_t':>10} {'pass_std%':>10} {'pass_strict%':>13}")
    for name, s in stats.items():
        print(f"  {name:<16} {s['mae_raw_t']:>10.6f} {s['pass_std']:>10.2f} {s['pass_strict']:>13.2f}")

    print(f"\n{'='*72}\n  seed 間ばらつき（3 seed, sd は不偏）\n{'='*72}")
    o = spread_summary["overall"]
    for m in METRICS:
        print(f"  {m:<12} mean={o[m]['mean']:.6f}  sd={o[m]['sd']:.6f}  range={o[m]['range']:.6f}")
    print("\n  t帯別 mae_raw_t: mean ± sd (range)")
    for bname, _ in T_BANDS:
        b = spread_summary["bands"][bname]["mae_raw_t"]
        print(f"    {bname:<8} {b['mean']:.6f} ± {b['sd']:.6f}  ({b['range']:.6f})")

    print(f"\n{'='*72}\n  seed 同士の paired t-test（raw_t）\n{'='*72}")
    for key, r in pairs.items():
        ov = r["overall"]
        sig_bands = [bn for bn, br in r["by_band"].items() if br["sig"] not in ("n.s.", "n/a")]
        print(f"  {key:<22} 改善率={ov['improvement_pct']:+.3f}%  p={ov['ttest_p']:.4g} {ov['sig']}"
              f"  有意な帯={sig_bands or 'なし'}")

    if repro is not None:
        ov = repro["overall"]
        print(f"\n{'='*72}\n  再現性: dual_s42 vs EXP031 SD0_baseline（同一構成・同一 seed）\n{'='*72}")
        print(f"  SD0={ov['mae_base']:.6f}  dual_s42={ov['mae_target']:.6f}  "
              f"差={ov['improvement_pct']:+.3f}%  p={ov['ttest_p']:.4g} {ov['sig']}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    db_df = load_db_index()
    paths = [prediction_path(c) for c in CONFIGS]

    stats = {c: compute_band_stats(read_predictions_csv(p), db_df) for c, p in zip(CONFIGS, paths)}
    ensemble_df = build_seed_ensemble(paths)
    ensemble_stats = compute_band_stats(ensemble_df, db_df)
    ensemble_csv = OUTPUT_DIR / "seed_ensemble_test_predictions.csv"
    ensemble_df.to_csv(ensemble_csv, index=False)

    reference_stats = None
    repro = None
    if REFERENCE_CSV.exists():
        reference_stats = compute_band_stats(read_predictions_csv(REFERENCE_CSV), db_df)
        repro = paired_comparison(REFERENCE_CSV, prediction_path("dual_s42"), db_df, metric="raw_t")
    else:
        print(f"[WARN] 再現性チェック用の参照 CSV が無いためスキップ: {REFERENCE_CSV}")

    spread_summary = summarize_spread(stats)
    pairs = pairwise_tests(db_df)

    table = {**stats, ENSEMBLE_NAME: ensemble_stats}
    if reference_stats is not None:
        table = {**table, "EXP031_SD0(ref)": reference_stats}
    print_report(table, spread_summary, pairs, repro)

    out = {
        "per_seed": stats,
        "spread": spread_summary,
        "seed_ensemble": ensemble_stats,
        "pairwise_ttest": pairs,
        "reference_EXP031_SD0": reference_stats,
        "reproducibility_s42_vs_EXP031_SD0": repro,
    }
    out_path = OUTPUT_DIR / "seed_variance.json"
    out_path.write_text(json.dumps(sanitize_for_json(out), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n保存: {out_path}\n      {ensemble_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
