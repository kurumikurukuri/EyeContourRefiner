"""決定論的 passrate 計算ツール。

error_ratio = |t_pred - t_true| * pq_dist / eye_scale
pass_rate(threshold) = #{error_ratio <= threshold} / total

閾値は experiments/Completion_Criteria.md 準拠:
  Standard : 0.01788 (p10)
  Strict   : 0.01604 (p5)

使用例:
  python scripts/compute_passrate.py --exp EXP026
  python scripts/compute_passrate.py --exp EXP026 --configs k3_b32 k5_b64
  python scripts/compute_passrate.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (
    DB_PATH,
    PREDICTIONS_CSV_NAMES,
    ROOT,
    THRESHOLD_STD,
    THRESHOLD_STRICT,
    T_BANDS,
    compute_error_ratio,
    find_exp_dir,
    is_missing,
    load_db_index,
    sanitize_for_json,
)


# ---------------------------------------------------------------------------
# Core computation (display/logic layer — shared functions are in _common.py)
# ---------------------------------------------------------------------------


def compute_passrate(error_ratio: pd.Series, threshold: float) -> float:
    """error_ratio の Series から pass rate (%) を計算する。NaN は除外。"""
    valid = error_ratio.dropna()
    if len(valid) == 0:
        return float("nan")
    return float((valid <= threshold).mean() * 100)


def compute_band_stats(
    pred_df: pd.DataFrame,
    db_df: pd.DataFrame,
) -> dict:
    """1 config の predictions から overall + per-band の passrate / MAE を計算する。

    すべての数値は丸めずに返す（表示層でフォーマットする）。
    NaN はサンプル不足時にそのまま返る。

    Args:
        pred_df: predictions.csv を読み込んだ DataFrame
        db_df:   load_db_index() の戻り値

    Returns:
        {
            "n_total": int,
            "n_valid": int,
            "n_missing": int,
            "pass_std": float (%),  "pass_strict": float (%),
            "mae_raw_t": float,
            "bands": { band_name: { "n": int, "pass_std": float, "pass_strict": float,
                                    "mae_raw_t": float } }
        }
    """
    missing_mask = ~pred_df["idx"].isin(db_df.index)
    n_missing = int(missing_mask.sum())

    error_ratio = compute_error_ratio(pred_df, db_df)
    valid_mask = error_ratio.notna()
    n_valid = int(valid_mask.sum())

    t_true = pred_df["t_true"]
    t_pred = pred_df["t_pred"]
    abs_err_raw = (t_pred - t_true).abs()

    result: dict = {
        "n_total": len(pred_df),
        "n_valid": n_valid,
        "n_missing": n_missing,
        "pass_std": compute_passrate(error_ratio, THRESHOLD_STD),
        "pass_strict": compute_passrate(error_ratio, THRESHOLD_STRICT),
        "mae_raw_t": float(abs_err_raw[valid_mask].mean()) if n_valid > 0 else float("nan"),
        "bands": {},
    }

    for bname, bfn in T_BANDS:
        band_mask = bfn(t_true) & valid_mask
        bn = int(band_mask.sum())
        if bn == 0:
            result["bands"][bname] = {
                "n": 0,
                "pass_std": float("nan"),
                "pass_strict": float("nan"),
                "mae_raw_t": float("nan"),
            }
        else:
            result["bands"][bname] = {
                "n": bn,
                "pass_std": float((error_ratio[band_mask] <= THRESHOLD_STD).mean() * 100),
                "pass_strict": float((error_ratio[band_mask] <= THRESHOLD_STRICT).mean() * 100),
                "mae_raw_t": float(abs_err_raw[band_mask].mean()),
            }

    return result


# ---------------------------------------------------------------------------
# Experiment discovery
# ---------------------------------------------------------------------------

def collect_prediction_files(exp_dir: Path, configs: list[str] | None = None) -> list[tuple[str, Path]]:
    """実験ディレクトリの outputs/ 以下から predictions.csv を収集する。

    Args:
        exp_dir: 実験ディレクトリ
        configs: 対象 config 名のリスト（None なら全 config）

    Returns:
        [(config_name, csv_path), ...] のリスト
    """
    outputs_dir = exp_dir / "outputs"
    if not outputs_dir.exists():
        return []

    result: list[tuple[str, Path]] = []
    for d in sorted(outputs_dir.iterdir()):
        if not d.is_dir():
            continue
        if configs is not None and d.name not in configs:
            continue
        for name in PREDICTIONS_CSV_NAMES:
            csv_path = d / name
            if csv_path.exists():
                result.append((d.name, csv_path))
                break
    return result


def collect_all_experiments() -> list[Path]:
    """全 experiments/EXP* ディレクトリを返す。"""
    return sorted(p for p in (ROOT / "experiments").glob("EXP*") if p.is_dir())


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_passrate_outputs(
    all_stats: list[dict],
    exp_dir: Path,
) -> dict:
    """passrate 計算結果を CSV + JSON で exp_dir/outputs/ に書き出す。

    Args:
        all_stats: [(config_name, stats_dict), ...] のリスト
        exp_dir:   実験ディレクトリ

    Returns:
        書き出した summary dict
    """
    out_dir = exp_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # overall CSV
    overall_rows = []
    for config_name, stats in all_stats:
        overall_rows.append({
            "config": config_name,
            "n_total": stats["n_total"],
            "n_valid": stats["n_valid"],
            "n_missing": stats["n_missing"],
            "pass_std_%": stats["pass_std"],
            "pass_strict_%": stats["pass_strict"],
            "mae_raw_t": stats["mae_raw_t"],
        })
    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(out_dir / "passrate_overall.csv", index=False)

    # by-band CSV
    band_rows = []
    for config_name, stats in all_stats:
        for bname, _ in T_BANDS:
            bs = stats["bands"][bname]
            band_rows.append({
                "config": config_name,
                "band": bname,
                "n": bs["n"],
                "pass_std_%": bs["pass_std"],
                "pass_strict_%": bs["pass_strict"],
                "mae_raw_t": bs["mae_raw_t"],
            })
    band_df = pd.DataFrame(band_rows)
    band_df.to_csv(out_dir / "passrate_by_band.csv", index=False)

    # summary JSON
    summary = {
        "thresholds": {
            "std": THRESHOLD_STD,
            "strict": THRESHOLD_STRICT,
        },
        "overall": {r["config"]: {
            "n_total": r["n_total"],
            "n_valid": r["n_valid"],
            "n_missing": r["n_missing"],
            "pass_std_%": r["pass_std_%"],
            "pass_strict_%": r["pass_strict_%"],
            "mae_raw_t": r["mae_raw_t"],
        } for r in overall_rows},
        "by_band": {
            config_name: stats["bands"]
            for config_name, stats in all_stats
        },
    }
    summary_path = out_dir / "passrate_summary.json"
    summary_path.write_text(
        json.dumps(sanitize_for_json(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return summary


def print_passrate_table(all_stats: list[dict]) -> None:
    """passrate 結果を標準出力に表形式で表示する。"""
    if not all_stats:
        print("  (no data)")
        return

    band_names = [b for b, _ in T_BANDS]

    print(f"\n{'='*80}")
    print(f"  Pass Rate Summary  (std={THRESHOLD_STD}, strict={THRESHOLD_STRICT})")
    print(f"{'='*80}")

    # overall
    header = f"  {'config':<20} {'n_valid':>8} {'pass_std%':>10} {'pass_strict%':>12} {'mae_raw_t':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for config_name, stats in all_stats:
        print(
            f"  {config_name:<20} {stats['n_valid']:>8} "
            f"{stats['pass_std']:>10.2f} {stats['pass_strict']:>12.2f} "
            f"{stats['mae_raw_t']:>10.6f}"
        )

    # by band (pass_std)
    print("\n  --- pass_std% by t-band ---")
    header2 = f"  {'config':<20} " + " ".join(f"{b:>10}" for b in band_names)
    print(header2)
    print("  " + "-" * (len(header2) - 2))
    for config_name, stats in all_stats:
        line = f"  {config_name:<20} "
        for b in band_names:
            v = stats["bands"][b]["pass_std"]
            line += f"{v:>10.1f}" if not is_missing(v) else f"{'—':>10}"
        print(line)

    # by band (n)
    print("\n  --- n per band ---")
    first_stats = all_stats[0][1] if all_stats else None
    if first_stats:
        counts = "  " + " ".join(f"{b}={first_stats['bands'][b]['n']}" for b in band_names)
        print(counts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read_predictions_csv(csv_path: Path) -> pd.DataFrame | None:
    """predictions.csv を読み込む。エラー時はメッセージを表示して None を返す。"""
    try:
        return pd.read_csv(csv_path)
    except pd.errors.ParserError as e:
        print(f"[ERROR] CSV のパースに失敗しました: {csv_path}")
        print(f"  {e}")
        return None
    except FileNotFoundError:
        print(f"[ERROR] CSV ファイルが見つかりません: {csv_path}")
        return None


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(
        description="決定論的 passrate 計算ツール"
    )
    ap.add_argument(
        "--exp", type=str, default=None,
        help="実験番号（例: 026）。EXP026_xxx を自動検出",
    )
    ap.add_argument(
        "--configs", nargs="*", default=None,
        help="対象 config 名（省略時は全 config）",
    )
    ap.add_argument(
        "--all", action="store_true",
        help="全 experiments を横断的に処理",
    )
    ap.add_argument(
        "--db", type=Path, default=DB_PATH,
        help="samples.db のパス",
    )
    args = ap.parse_args()

    if not args.exp and not args.all:
        ap.error("--exp または --all のいずれかを指定してください")

    db_df = load_db_index(args.db)

    if args.all:
        exp_dirs = collect_all_experiments()
        print(f"全 {len(exp_dirs)} 実験を処理中...")
        for exp_dir in exp_dirs:
            exp_num = exp_dir.name[:6]  # "EXP026"
            pred_files = collect_prediction_files(exp_dir)
            if not pred_files:
                print(f"  {exp_num}: predictions.csv なし → skip")
                continue
            all_stats = []
            for config_name, csv_path in pred_files:
                pred_df = _read_predictions_csv(csv_path)
                if pred_df is None:
                    continue
                stats = compute_band_stats(pred_df, db_df)
                all_stats.append((config_name, stats))
            write_passrate_outputs(all_stats, exp_dir)
            print(f"  {exp_num}: {len(all_stats)} configs → {exp_dir / 'outputs' / 'passrate_summary.json'}")
        return 0

    # single experiment
    exp_dir = find_exp_dir(args.exp)
    print(f"実験ディレクトリ: {exp_dir}")

    config_list = args.configs if args.configs else None
    pred_files = collect_prediction_files(exp_dir, config_list)

    if not pred_files:
        names = " または ".join(PREDICTIONS_CSV_NAMES)
        print(f"[ERROR] {names} が見つかりません: {exp_dir / 'outputs' / '*'}")
        return 1

    print(f"対象 config: {len(pred_files)} 件")

    all_stats: list[dict] = []
    for config_name, csv_path in pred_files:
        pred_df = _read_predictions_csv(csv_path)
        if pred_df is None:
            continue
        stats = compute_band_stats(pred_df, db_df)
        all_stats.append((config_name, stats))

    write_passrate_outputs(all_stats, exp_dir)
    print_passrate_table(all_stats)

    out_dir = exp_dir / "outputs"
    print("\n保存先:")
    print(f"  {out_dir / 'passrate_overall.csv'}")
    print(f"  {out_dir / 'passrate_by_band.csv'}")
    print(f"  {out_dir / 'passrate_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
