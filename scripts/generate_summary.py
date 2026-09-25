"""決定論的 summary markdown 生成ツール。

experiment.json + predictions.csv から EXP{n}_summary.md を機械生成する。
定量部分（MAE, passrate, t帯別）は自動計算し、考察・結論セクションは
<!-- TODO --> プレースホルダとして残す。

使用例:
  python scripts/generate_summary.py --exp EXP027
  python scripts/generate_summary.py --exp EXP026 --baseline k3_b32
  python scripts/generate_summary.py --exp EXP026 --force
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from _common import (
    DB_PATH,
    THRESHOLD_STD,
    THRESHOLD_STRICT,
    T_BANDS,
    find_exp_dir,
    find_predictions_csv,
    is_missing,
    load_db_index,
    locate_config_file,
    read_predictions_csv,
)
from compute_passrate import (
    collect_prediction_files,
    compute_band_stats,
)
from paired_ttest import paired_comparison

def load_experiment_json(exp_dir: Path, config: str) -> dict | None:
    """experiment.json を読み込む（outputs/{config}/ と outputs/{config}/{config}/ の両構成に対応）。"""
    path = locate_config_file(exp_dir, config, ("experiment.json",))
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_val(v: float | None, precision: int = 4) -> str:
    """値をフォーマット。None なら '—'。"""
    if v is None:
        return "—"
    return f"{v:.{precision}f}"


def generate_summary_md(
    exp_dir: Path,
    baseline: str | None = None,
) -> str:
    """summary markdown を生成する。"""
    exp_name = exp_dir.name
    exp_num = exp_name[:6]  # "EXP027"

    db_df = load_db_index(DB_PATH)

    # 全 config の stats を収集
    pred_files = collect_prediction_files(exp_dir)
    if not pred_files:
        raise FileNotFoundError(f"predictions.csv が見つかりません: {exp_dir / 'outputs'}")

    all_configs: dict[str, dict] = {}
    for config_name, csv_path in pred_files:
        pred_df = read_predictions_csv(csv_path)
        stats = compute_band_stats(pred_df, db_df)
        exp_json = load_experiment_json(exp_dir, config_name)
        all_configs[config_name] = {
            "stats": stats,
            "exp_json": exp_json,
        }

    config_names = sorted(all_configs.keys())
    band_names = [b for b, _ in T_BANDS]

    # --- 定量評価表 ---
    quant_header = "| config | n_valid | cv_mae | holdout_mae(raw t) | pass_std% | pass_strict% |"
    quant_sep = "|---|---|---|---|---|---|"
    quant_rows = []
    for cn in config_names:
        info = all_configs[cn]
        stats = info["stats"]
        exp = info["exp_json"] or {}
        results = exp.get("results", {})
        cv_mae = results.get("cv_mae")
        holdout_raw = results.get("holdout_mae_raw_t")
        quant_rows.append(
            f"| {cn} | {stats['n_valid']} | {fmt_val(cv_mae, 6)} | "
            f"{fmt_val(holdout_raw, 6)} | {fmt_val(stats['pass_std'], 2)} | "
            f"{fmt_val(stats['pass_strict'], 2)} |"
        )

    # --- t帯別 pass_std 表 ---
    band_header = f"| config | " + " | ".join(band_names) + " | ALL |"
    band_sep = "|" + "|".join(["---"] * (len(band_names) + 2)) + "|"
    band_rows = []
    for cn in config_names:
        stats = all_configs[cn]["stats"]
        cells = []
        for b in band_names:
            v = stats["bands"][b]["pass_std"]
            cells.append(f"{v:.1f}" if not is_missing(v) else "—")
        cells.append(f"{stats['pass_std']:.1f}")
        band_rows.append(f"| {cn} | " + " | ".join(cells) + " |")

    # --- t帯別 mae_raw_t 表 ---
    mae_header = f"| config | " + " | ".join(band_names) + " | ALL |"
    mae_sep = "|" + "|".join(["---"] * (len(band_names) + 2)) + "|"
    mae_rows = []
    for cn in config_names:
        stats = all_configs[cn]["stats"]
        cells = []
        for b in band_names:
            v = stats["bands"][b]["mae_raw_t"]
            cells.append(f"{v:.6f}" if not is_missing(v) else "—")
        cells.append(f"{stats['mae_raw_t']:.6f}")
        mae_rows.append(f"| {cn} | " + " | ".join(cells) + " |")

    # --- n per band ---
    if all_configs:
        first_stats = list(all_configs.values())[0]["stats"]
        n_info = "  n: " + " ".join(f"{b}={first_stats['bands'][b]['n']}" for b in band_names)
    else:
        n_info = ""

    # --- assemble ---
    lines = [
        f"# {exp_num}: <!-- TODO: 実験タイトル -->",
        "",
        "## 目的・仮説",
        "",
        "<!-- TODO: 目的と仮説を記入 -->",
        "",
        "## 実験設計・条件",
        "",
        f"- **実験ディレクトリ**: `{exp_dir}`",
        f"- **config 数**: {len(config_names)}",
        f"- **config 一覧**: {', '.join(config_names)}",
        f"- **評価指標**: cv_mae, holdout_mae(raw t), passrate(std/strict)",
        f"- **passrate 閾値**: std={THRESHOLD_STD} (p10), strict={THRESHOLD_STRICT} (p5)",
        "- **検定方法**: scipy paired t-test（per-sample abs error）",
        "",
        "<!-- TODO: 固定条件・変動軸を記入 -->",
        "",
        "## 結果（hold-out）",
        "",
        "### 定量評価",
        "",
        quant_header,
        quant_sep,
        *quant_rows,
        "",
        "### t帯別 pass_std (%)",
        "",
        band_header,
        band_sep,
        *band_rows,
        "",
        n_info,
        "",
        "### t帯別 mae_raw_t",
        "",
        mae_header,
        mae_sep,
        *mae_rows,
        "",
        "## 考察",
        "",
        "<!-- TODO: 全体MAE・passrateで見た優劣、軸ごとの傾向を記入 -->",
        "",
        "## t帯別結果考察",
        "",
        "<!-- TODO: 極端t帯と多数派帯でのトレードオフの有無を記入 -->",
        "",
    ]

    # --- paired t-test section (if baseline specified) ---
    if baseline and baseline in all_configs:
        lines.extend([
            "## ペア比較 (paired t-test)",
            "",
            f"ベースライン: **{baseline}**  \\* = paired t-test p<0.05",
            "",
        ])

        for cn in config_names:
            if cn == baseline:
                continue
            try:
                base_csv = find_predictions_csv(exp_dir, baseline)
                tgt_csv = find_predictions_csv(exp_dir, cn)
                result = paired_comparison(base_csv, tgt_csv, db_df, metric="raw_t")
                if "error" in result:
                    continue

                ov = result["overall"]
                lines.extend([
                    f"### {cn} vs {baseline}",
                    "",
                    f"| band | n | MAE({cn}) | MAE({baseline}) | diff% | t_p | sig |",
                    "|---|---|---|---|---|---|---|",
                    f"| **ALL** | {result['n_common']} | {ov['mae_target']:.6f} | "
                    f"{ov['mae_base']:.6f} | {ov['improvement_pct']:+.2f}% | "
                    f"{ov['ttest_p']:.6f} | {ov['sig']} |",
                ])
                for b in band_names:
                    bd = result["by_band"][b]
                    if bd["n"] < 2:
                        continue
                    mae_t = bd["mae_target"] if bd["mae_target"] is not None else float('nan')
                    mae_b = bd["mae_base"] if bd["mae_base"] is not None else float('nan')
                    impr = bd["improvement_pct"] if bd["improvement_pct"] is not None else float('nan')
                    pval = bd["ttest_p"] if bd["ttest_p"] is not None else float('nan')
                    lines.append(
                        f"| {b} | {bd['n']} | {mae_t:.6f} | {mae_b:.6f} | "
                        f"{impr:+.2f}% | {pval:.6f} | {bd['sig']} |"
                    )
                lines.append("")
            except FileNotFoundError:
                lines.append(f"<!-- {cn}: predictions.csv なし → skip -->")
                lines.append("")

    lines.extend([
        "## ペア比較考察",
        "",
        "<!-- TODO: 帯ごとの有意差／非有意の解釈を記入 -->",
        "",
        "## 結論",
        "",
        "<!-- TODO: 採用/見送りの判断と根拠を記入 -->",
        "",
        "## 今後の課題",
        "",
        "<!-- TODO: 未検証の軸・交互作用、次実験の提案を記入 -->",
        "",
    ])

    return "\n".join(lines)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(
        description="決定論的 summary markdown 生成ツール"
    )
    ap.add_argument("--exp", type=str, required=True, help="実験番号（例: 027）")
    ap.add_argument("--baseline", type=str, default=None, help="ベースライン config 名")
    ap.add_argument("--force", action="store_true", help="既存の summary.md を強制上書き")
    ap.add_argument("--dry-run", action="store_true", help="ファイルに書き込まず標準出力のみ")
    args = ap.parse_args()

    exp_dir = find_exp_dir(args.exp)
    out_path = exp_dir / f"{exp_dir.name}_summary.md"  # EXP027_xxx_summary.md

    # EXP{n}_summary.md を探す（テンプレート名と異なる場合がある）
    existing = sorted(exp_dir.glob("EXP*_summary.md"))
    if existing and not args.force:
        # テンプレート名と一致しなければ警告
        print(f"[WARNING] 既存の summary があります: {existing[0].name}")
        print(f"  上書きする場合は --force を指定してください")
        if not args.dry_run:
            return 1

    # 適切なファイル名を決定
    if existing:
        out_path = existing[0]
    else:
        # EXP{n}_summary.md パターンを試す
        exp_num = exp_dir.name[:6]
        candidate = exp_dir / f"{exp_num}_summary.md"
        if candidate.exists() and not args.force:
            print(f"[WARNING] 既存: {candidate.name}")
            print(f"  上書きする場合は --force を指定してください")
            if not args.dry_run:
                return 1
        out_path = candidate

    print(f"生成中: {exp_dir.name}")
    md = generate_summary_md(exp_dir, baseline=args.baseline)

    if args.dry_run:
        print(md)
        print(f"\n[dry-run] 書き込み先: {out_path}")
    else:
        out_path.write_text(md, encoding="utf-8")
        print(f"保存: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
