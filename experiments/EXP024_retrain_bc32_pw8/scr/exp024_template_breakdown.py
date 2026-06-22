"""EXP024 テンプレート帯域別 breakdown + passrate / norm MAE 再集計。

EXP024_summary.md を EXPn_summary_template.md の帯域区分
(t<0, 0-0.3, 0.3-0.7, 0.7-1.0, 1.0-1.2, t>=1.2) で埋めるための数値を生成する。
compare_exp024.py が隔離する「label fix 効果」と整合させるため、
両モデルとも新ラベル(修正後 manifest_test の t)で採点し、
EXP024(new model) と EXP022(old model) を paired t-test で比較する。

使い方:
    python experiments/EXP024_retrain_bc32_pw8/scr/exp024_template_breakdown.py
"""
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from scipy.stats import ttest_rel

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
DB = ROOT / "data" / "samples.db"

NEW_PRED = EXP_DIR / "outputs" / "bc32_pw8" / "test_predictions.csv"
OLD_PRED = ROOT / "experiments" / "EXP022_channel_pool_sweep" / "outputs" / "bc32_pw8" / "test_predictions.csv"
NEW_MANI = EXP_DIR / "manifests" / "manifest_test.csv"
EXP_JSON = EXP_DIR / "outputs" / "bc32_pw8" / "experiment.json"

# experiments/Completion_Criteria.md 準拠
PASS_STD = 0.01788  # p10
PASS_STRICT = 0.01604  # p5

BANDS = ["t<0", "0-0.3", "0.3-0.7", "0.7-1.0", "1.0-1.2", "t>=1.2"]


def band_of(t: float) -> str:
    if t < 0:
        return "t<0"
    if t < 0.3:
        return "0-0.3"
    if t < 0.7:
        return "0.3-0.7"
    if t < 1.0:
        return "0.7-1.0"
    if t < 1.2:
        return "1.0-1.2"
    return "t>=1.2"


def load_pred(path: Path) -> dict[int, float]:
    with open(path, newline="", encoding="utf-8") as f:
        return {int(r["idx"]): float(r["t_pred"]) for r in csv.DictReader(f)}


def load_manifest(path: Path) -> dict[int, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {int(r["idx"]): r for r in csv.DictReader(f)}


def load_eye_scale(db: Path) -> dict[int, float]:
    con = sqlite3.connect(db)
    try:
        cur = con.execute("SELECT idx, eye_scale FROM samples")
        return {int(i): float(e) for i, e in cur if e is not None}
    finally:
        con.close()


def pq_dist(r: dict) -> float:
    return ((float(r["px"]) - float(r["qx"])) ** 2 + (float(r["py"]) - float(r["qy"])) ** 2) ** 0.5


def paired_p(a: list[float], b: list[float]) -> float | None:
    if len(a) >= 10:
        _, p = ttest_rel(a, b)
        return float(p)
    return None


def main() -> None:
    new_pred = load_pred(NEW_PRED)
    old_pred = load_pred(OLD_PRED)
    mani = load_manifest(NEW_MANI)
    eye = load_eye_scale(DB)

    common = sorted(set(new_pred) & set(old_pred) & set(mani))
    rows = []
    miss_eye = 0
    for idx in common:
        r = mani[idx]
        t_new = float(r["t"])
        es = eye.get(idx)
        if es is None or es <= 0:
            miss_eye += 1
            continue
        pq = pq_dist(r)
        dt_new = abs(new_pred[idx] - t_new)
        dt_old = abs(old_pred[idx] - t_new)
        rows.append({
            "idx": idx, "t": t_new, "band": band_of(t_new),
            "dt_new": dt_new, "dt_old": dt_old,
            "ratio_new": dt_new * pq / es, "ratio_old": dt_old * pq / es,
        })
    if miss_eye:
        print(f"[WARN] eye_scale 欠損/非正で {miss_eye} 件スキップ")

    n = len(rows)
    mae_new = sum(r["dt_new"] for r in rows) / n
    mae_old = sum(r["dt_old"] for r in rows) / n
    # t_norm = (t - label_t_min)/(label_t_max - label_t_min), label_t_min=-1/max=2 → range=3
    # よって norm 空間の MAE = raw-t MAE / 3（EXP022 summary と同定義）
    norm_new = mae_new / 3.0
    norm_old = mae_old / 3.0
    ps_std = sum(1 for r in rows if r["ratio_new"] <= PASS_STD) / n * 100
    ps_strict = sum(1 for r in rows if r["ratio_new"] <= PASS_STRICT) / n * 100
    po_std = sum(1 for r in rows if r["ratio_old"] <= PASS_STD) / n * 100
    po_strict = sum(1 for r in rows if r["ratio_old"] <= PASS_STRICT) / n * 100

    cv: float | None = None
    if EXP_JSON.exists():
        try:
            ej = json.loads(EXP_JSON.read_text(encoding="utf-8"))
            cv = ej.get("cv_mae")
            if cv is None and isinstance(ej.get("results"), dict):
                cv = ej["results"].get("cv_mae")
        except Exception:
            pass

    print("## 定量評価 (n=%d)\n" % n)
    print("| config | cv_mae | holdout_mae(norm) | holdout_mae(raw t) | passrate(std) | passrate(strict) |")
    print("|---|---|---|---|---|---|")
    print(f"| EXP022 bc32_pw8 (old model vs new labels) | n/a | {norm_old:.6f} | {mae_old:.6f} | {po_std:.2f} | {po_strict:.2f} |")
    cv_s = f"{cv:.6f}" if isinstance(cv, (int, float)) else "0.029459(doc)"
    print(f"| **EXP024 bc32_pw8 (new model vs new labels)** | {cv_s} | {norm_new:.6f} | {mae_new:.6f} | {ps_std:.2f} | {ps_strict:.2f} |")
    print()

    p_all = paired_p([r["dt_new"] for r in rows], [r["dt_old"] for r in rows])

    print("## t帯別 + ペア比較 (EXP024 new vs EXP022 old, ともに新ラベルで採点, paired t-test)")
    print("\\* = paired t-test で p<0.05\n")
    print("| band | n | MAE(EXP024) | MAE(EXP022) | diff | t_p |")
    print("|---|---|---|---|---|---|")
    for b in BANDS:
        sub = [r for r in rows if r["band"] == b]
        if not sub:
            continue
        mn = sum(r["dt_new"] for r in sub) / len(sub)
        mo = sum(r["dt_old"] for r in sub) / len(sub)
        pb = paired_p([r["dt_new"] for r in sub], [r["dt_old"] for r in sub])
        ps = f"{pb:.4f}" + (" *" if pb is not None and pb < 0.05 else "") if pb is not None else "n/a"
        print(f"| {b} | {len(sub)} | {mn:.6f} | {mo:.6f} | {mn-mo:+.6f} | {ps} |")
    ps_all = f"{p_all:.4f}" + (" *" if p_all is not None and p_all < 0.05 else "") if p_all is not None else "n/a"
    print(f"| **ALL** | {n} | {mae_new:.6f} | {mae_old:.6f} | {mae_new-mae_old:+.6f} | {ps_all} |")


if __name__ == "__main__":
    main()
