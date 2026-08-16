"""EXP031 再実験（aug_flags ビット統合後）と旧結果の paired t-test 比較。

`source_jpeg` / `source_res_jitter` の bool キーを `aug_flags` のビット
（AUG_SOURCE_JPEG=0x080 / AUG_SOURCE_RES_JITTER=0x100）へ統合したリファクタが、
学習挙動を変えていないことを確認する。

集約 MAE の増減を「run-to-run 分散の範囲内」と目分量で判定すると主観が入るため、
同一 hold-out（n=4900）の **per-sample 対応差** を直接検定する。

注意（解釈上の前提）:
  paired t-test の帰無仮説は「差がない」であり、本比較は**棄却したくない側**である。
  さらに学習は `torch.use_deterministic_algorithms(True, warn_only=True)` 下でも
  CuBLAS / adaptive_avg_pool2d_backward_cuda が非決定的なため、コードが同一でも
  再学習すれば異なるモデルになる。したがって:
    - p 値だけで「同一」と結論しない
    - 効果量（平均対応差とその 95% CI）を passrate 閾値と比べて実用的に判断する

使い方:
    python analyses/exp031_rerun_check/scr/compare_rerun.py --old-dir <旧outputs退避先>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _common import THRESHOLD_STD, load_db_index  # noqa: E402
from paired_ttest import paired_comparison  # noqa: E402

EXP_DIR = ROOT / "experiments" / "EXP031_source_space_aug"
CONFIGS = ["SD1_jpeg", "SD2_resjitter", "SD3_both"]


def abs_errors(csv_path: Path) -> pd.Series:
    """predictions.csv から idx をキーにした |t_pred - t_true| を返す。"""
    df = pd.read_csv(csv_path)
    return (df["t_pred"] - df["t_true"]).abs().set_axis(df["idx"]).sort_index()


def effect_size(old_csv: Path, new_csv: Path) -> dict:
    """対応差の平均・95% CI・実用換算を返す（paired t の効果量側）。"""
    old_e, new_e = abs_errors(old_csv), abs_errors(new_csv)
    common = old_e.index.intersection(new_e.index)
    diff = (new_e.loc[common] - old_e.loc[common]).to_numpy()

    n = len(diff)
    mean = float(diff.mean())
    sem = float(stats.sem(diff)) if n > 1 else 0.0
    ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem) if sem > 0 else (mean, mean)
    return {
        "n": n,
        "mean_diff": mean,
        "ci95": [float(ci[0]), float(ci[1])],
        "sd_diff": float(diff.std(ddof=1)),
        # 実用スケール: passrate 判定閾値（error_ratio 0.01788）と同じ土俵ではないが、
        # 生 t の平均誤差 0.0843 に対する比率で「無視できる差か」を見る。
        "mean_diff_vs_mae_pct": mean / float(old_e.loc[common].mean()) * 100,
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-dir", type=Path, required=True,
                    help="旧 outputs を退避したディレクトリ（<dir>/<config>/test_predictions.csv）")
    ap.add_argument("--metric", choices=["raw_t", "error_ratio"], default="raw_t")
    ap.add_argument("--out", type=Path, default=None, help="結果 JSON の保存先")
    args = ap.parse_args()

    db_df = load_db_index(ROOT / "data" / "samples.db")
    results = []

    print("=" * 78)
    print("  EXP031 再実験 vs 旧結果  paired t-test（同一 hold-out・per-sample 対応差）")
    print("=" * 78)
    print(f"  metric={args.metric}  passrate 閾値(std)={THRESHOLD_STD}")
    print()
    header = f"  {'config':<15} {'MAE 旧':>10} {'MAE 新':>10} {'差':>10} {'t':>8} {'p':>10}  判定"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name in CONFIGS:
        old_csv = args.old_dir / name / "test_predictions.csv"
        new_csv = EXP_DIR / "outputs" / name / "test_predictions.csv"
        if not old_csv.exists() or not new_csv.exists():
            print(f"  {name:<15} スキップ（CSV なし: "
                  f"{'旧' if not old_csv.exists() else '新'}）")
            continue

        cmp = paired_comparison(old_csv, new_csv, db_df, metric=args.metric)
        eff = effect_size(old_csv, new_csv)
        ov = cmp["overall"]
        print(f"  {name:<15} {ov['mae_base']:>10.6f} {ov['mae_target']:>10.6f} "
              f"{ov['mae_target'] - ov['mae_base']:>+10.6f} "
              f"{ov['t_stat']:>8.3f} {ov['ttest_p']:>10.6f}  {ov['sig']}")
        results.append({"config": name, "paired": cmp, "effect": eff})

    print()
    print("  --- 効果量（新 − 旧の per-sample 対応差, 生t） ---")
    print(f"  {'config':<15} {'平均差':>12} {'95% CI':>26} {'MAE比':>8}")
    print("  " + "-" * 64)
    for r in results:
        e = r["effect"]
        ci = f"[{e['ci95'][0]:+.6f}, {e['ci95'][1]:+.6f}]"
        print(f"  {r['config']:<15} {e['mean_diff']:>+12.6f} {ci:>26} "
              f"{e['mean_diff_vs_mae_pct']:>+7.2f}%")

    print()
    print("  --- t帯別 p 値 ---")
    bands = list(results[0]["paired"]["by_band"]) if results else []
    print(f"  {'config':<15}" + "".join(f"{b:>12}" for b in bands))
    print("  " + "-" * (15 + 12 * len(bands)))
    for r in results:
        cells = ""
        for b in bands:
            p = r["paired"]["by_band"][b]["ttest_p"]
            cells += f"{'—' if p is None else format(p, '.4f'):>12}"
        print(f"  {r['config']:<15}{cells}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float),
                            encoding="utf-8")
        print(f"\n  保存: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
