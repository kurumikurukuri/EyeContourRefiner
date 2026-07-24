"""EXP024 比較: EXP024 predictions.csv を EXP022 baseline と paired test + t-band別MAE で比較。

EXP022 の t_band_compare.py に相当する。
推論済みの outputs/<config>/predictions.csv（生 t）を入力とし:
  - 全体 MAE の paired t-test（新しいラベルで両モデルを採点）
  - t-band 別 MAE breakdown + 帯域内 paired test
結果を標準出力し、experiment.json の comparison_vs_exp022 ブロックに永続化する。

使い方:
    python experiments/EXP024_retrain_bc32_pw8/scr/compare_exp024_vs_exp022.py
    python experiments/EXP024_retrain_bc32_pw8/scr/compare_exp024_vs_exp022.py --config bc32_pw8
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from scipy.stats import ttest_rel

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
MANIFEST_DIR = EXP_DIR / "manifests"
OUTPUT_DIR = EXP_DIR / "outputs"

EXP022_DIR = ROOT / "experiments" / "EXP022_channel_pool_sweep"

# 正準 t-band 定義（scripts/_common.py と同一）
sys.path.insert(0, str(ROOT / "scripts"))
from _common import T_BANDS  # noqa: E402


def load_manifest_t(path: Path) -> dict[int, float]:
    """manifest から idx→t_true 辞書を読み込む。"""
    with open(path, newline="", encoding="utf-8") as f:
        return {int(r["idx"]): float(r["t"]) for r in csv.DictReader(f)}


def load_old_predictions(path: Path) -> dict[int, dict]:
    """EXP022 predictions.csv を idx→{t_true, t_pred} で読み込む。"""
    with open(path, newline="", encoding="utf-8") as f:
        return {int(r["idx"]): {"t_true": float(r["t_true"]), "t_pred": float(r["t_pred"])}
                for r in csv.DictReader(f)}


def compute_mae(rows: list[dict], key: str = "abs_err") -> float:
    return sum(r[key] for r in rows) / len(rows)


def compare_with_exp022(
    new_pred_path: Path,
    new_manifest_test: Path,
    old_manifest_test: Path,
    old_predictions_path: Path,
) -> dict:
    """EXP022 と EXP024 の予測を比較し、paired test + t-band breakdown を返す。

    両モデルとも新しいラベル（new_manifest_test の t_true）で採点し、
    ラベル修正効果とモデル変更効果を分離する。
    """
    # 新しい予測
    new_pred: dict[int, float] = {}
    with open(new_pred_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            new_pred[int(r["idx"])] = float(r["t_pred"])

    # EXP022 予測
    old_pred = load_old_predictions(old_predictions_path)

    # 現在の test manifest の t_true（修正後）
    new_t = load_manifest_t(new_manifest_test)
    # EXP022 の test manifest の t_true（修正前）
    old_t = load_manifest_t(old_manifest_test)

    common = sorted(set(new_pred) & set(old_pred) & set(new_t) & set(old_t))
    if len(common) == 0:
        return {"error": "no common idx"}

    # per-sample 比較
    rows = []
    for idx in common:
        t_new = new_t[idx]
        t_old = old_t[idx]
        p_new = new_pred[idx]
        p_old = old_pred[idx]["t_pred"]
        rows.append({
            "idx": idx,
            "t_true_new": t_new,
            "t_true_old": t_old,
            "t_pred_new": p_new,
            "t_pred_old": p_old,
            "abs_err_new": abs(p_new - t_new),
            "abs_err_old": abs(p_old - t_new),  # 新しいラベルで統一
        })

    # 全体 MAE
    mae_new = compute_mae(rows, "abs_err_new")
    mae_old = compute_mae(rows, "abs_err_old")

    # Paired t-test（per-sample の絶対誤差を全ペアで比較、零差ペアも含む）
    diffs_new = [r["abs_err_new"] for r in rows]
    diffs_old = [r["abs_err_old"] for r in rows]
    if len(rows) >= 10:
        stat, pval = ttest_rel(diffs_new, diffs_old)
    else:
        stat, pval = None, None

    # t-band breakdown（正準 T_BANDS を使用）
    band_summary = {}
    for bname, bfn in T_BANDS:
        mask = [bfn(r["t_true_new"]) for r in rows]
        band_new = [rows[i]["abs_err_new"] for i, m in enumerate(mask) if m]
        band_old = [rows[i]["abs_err_old"] for i, m in enumerate(mask) if m]
        bn = len(band_new)
        if bn == 0:
            band_summary[bname] = {"n": 0, "mae_old": None, "mae_new": None, "ttest_p": None}
            continue
        new_mae = sum(band_new) / bn
        old_mae = sum(band_old) / bn
        bp = None
        if bn >= 10:
            _, bp = ttest_rel(band_new, band_old)
        band_summary[bname] = {"n": bn,
                               "mae_old": round(old_mae, 6),
                               "mae_new": round(new_mae, 6),
                               "ttest_p": bp}

    return {
        "n": len(rows),
        "mae_old": round(mae_old, 6),
        "mae_new": round(mae_new, 6),
        "improvement": round((mae_old - mae_new) / mae_old * 100, 2),
        "t_stat": round(float(stat), 4) if stat is not None else None,
        "ttest_p": round(float(pval), 6) if pval is not None else None,
        "bands": band_summary,
    }


def print_comparison(comp: dict) -> None:
    if "error" in comp:
        print(f"[ERROR] {comp['error']}")
        return
    print(f"  test件数: {comp['n']}")
    print(f"  EXP022 MAE (old model vs new labels): {comp['mae_old']:.6f}")
    print(f"  EXP024 MAE (new model vs new labels): {comp['mae_new']:.6f}")
    print(f"  改善率: {comp['improvement']:+.2f}%")
    if comp["ttest_p"] is not None:
        print(f"  Paired t-test p-value: {comp['ttest_p']:.6f} (t={comp['t_stat']:.3f}) "
              f"({'**有意' if comp['ttest_p'] < 0.01 else '*有意' if comp['ttest_p'] < 0.05 else 'n.s.'})")

    print(f"\n  t-band breakdown:")
    print(f"  {'band':<12} {'n':>6} {'MAE_old':>10} {'MAE_new':>10} {'改善%':>8} {'p':>8}")
    print(f"  {'-'*54}")
    for bname, _ in T_BANDS:
        info = comp["bands"].get(bname)
        if info is None or info["n"] == 0:
            continue
        imp = (info["mae_old"] - info["mae_new"]) / info["mae_old"] * 100
        p_str = f"{info['ttest_p']:.4f}" if info["ttest_p"] is not None else "n/a"
        sig = "**" if info["ttest_p"] is not None and info["ttest_p"] < 0.01 else \
              "*" if info["ttest_p"] is not None and info["ttest_p"] < 0.05 else ""
        print(f"  {bname:<12} {info['n']:>6} {info['mae_old']:>10.6f} {info['mae_new']:>10.6f} {imp:>+7.2f} {p_str:>8}{sig}")


def main() -> int:
    parser = argparse.ArgumentParser(description="EXP024 と EXP022 baseline の比較")
    parser.add_argument("--config", default="bc32_pw8", help="config 名（拡張子なし）")
    parser.add_argument("--predictions", type=Path, default=None,
                        help="EXP024 predictions.csv（省略時は outputs/<config>/predictions.csv）")
    parser.add_argument("--exp-json", type=Path, default=None,
                        help="experiment.json（省略時は outputs/<config>/experiment.json）")
    parser.add_argument("--new-manifest", type=Path, default=None,
                        help="修正後 test manifest（省略時は manifests/manifest_test.csv）")
    parser.add_argument("--old-manifest", type=Path, default=None,
                        help="EXP022 test manifest（省略時は EXP022 の manifests/manifest_test.csv）")
    parser.add_argument("--old-predictions", type=Path, default=None,
                        help="EXP022 predictions.csv（省略時は EXP022 の outputs/<config>/predictions.csv）")
    parser.add_argument("--no-write", action="store_true",
                        help="experiment.json への comparison_vs_exp022 追記をスキップ")
    args = parser.parse_args()

    cfg_name = args.config
    predictions = args.predictions or OUTPUT_DIR / cfg_name / "predictions.csv"
    exp_json = args.exp_json or OUTPUT_DIR / cfg_name / "experiment.json"
    new_manifest = args.new_manifest or MANIFEST_DIR / "manifest_test.csv"
    old_manifest = args.old_manifest or EXP022_DIR / "manifests" / "manifest_test.csv"
    old_predictions = args.old_predictions or EXP022_DIR / "outputs" / cfg_name / "predictions.csv"

    for label, p in [("predictions", predictions), ("new_manifest", new_manifest),
                     ("old_manifest", old_manifest), ("old_predictions", old_predictions)]:
        if not p.exists():
            print(f"[ERROR] {label} が見つかりません: {p}")
            return 1

    print(f"\n--- EXP022 との比較 ---")
    comp = compare_with_exp022(predictions, new_manifest, old_manifest, old_predictions)
    print_comparison(comp)
    if "error" in comp:
        return 1

    # comparison 結果を experiment.json に追記して永続化
    if not args.no_write:
        if exp_json.exists():
            exp_record = json.loads(exp_json.read_text(encoding="utf-8"))
        else:
            print(f"[WARN] experiment.json がないため新規作成: {exp_json}")
            exp_record = {"exp_name": cfg_name}
        exp_record["comparison_vs_exp022"] = comp
        exp_json.write_text(json.dumps(exp_record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nexperiment.json に comparison_vs_exp022 を追記: {exp_json}")
    else:
        print("\n(--no-write 指定のため experiment.json は更新しません)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
