"""EXP021: 生 t バンド別 MAE + 直積の paired 検定。

全条件の test_predictions.csv（生 t）を読み込み、
  1. バンド別 MAE テーブル
  2. paired t-test vs 基準 t__none（= EXP020 X0_none_baseline 相当）
  3. 直積の核仮説対比（空間効果・カーネル効果）を per-system で paired 検定
を出力する。
"""
import csv
import math
import sys
from pathlib import Path

from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

OUT = Path(__file__).parent.parent / "outputs"

LOSSES = ["t", "t_l1", "wing", "error_ratio", "error_ratio_huber", "wing_px"]
SYSTEMS = ["none", "eye_scale"]
CONDITIONS = [f"{loss}__{sys}" for sys in SYSTEMS for loss in LOSSES]
REFERENCE = "t__none"

BANDS = [
    ("t<0",     lambda t: t < 0),
    ("0-0.3",   lambda t: 0 <= t < 0.3),
    ("0.3-0.7", lambda t: 0.3 <= t < 0.7),
    ("0.7-1.0", lambda t: 0.7 <= t < 1.0),
    ("1.0-1.2", lambda t: 1.0 <= t < 1.2),
    ("t>=1.2",  lambda t: t >= 1.2),
]


def load_data(cname: str):
    """(t_true, abs_error) のリスト。CSV がなければ None。"""
    csv_path = OUT / cname / "test_predictions.csv"
    if not csv_path.exists():
        return None
    data = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t_true = float(row["t_true"])
            t_pred = float(row["t_pred"])
            data.append((t_true, abs(t_true - t_pred)))
    return data


def sig(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def paired(a, b):
    """a, b: (t_true, err) リスト（同順）。paired t-test, delta=mean(a)-mean(b)。"""
    ea = [e for _, e in a]
    eb = [e for _, e in b]
    if len(ea) != len(eb):
        return None
    t_stat, p_t = stats.ttest_rel(ea, eb)
    delta = sum(ea) / len(ea) - sum(eb) / len(eb)
    return t_stat, p_t, delta


def main() -> None:
    all_data = {c: d for c in CONDITIONS if (d := load_data(c)) is not None}

    # --- Band MAE テーブル ---
    header = f"{'condition':<24}" + "".join(f"{b:>9}" for b, _ in BANDS) + f"{'ALL':>9}{'n':>7}"
    print(header)
    print("-" * len(header))
    for cname in CONDITIONS:
        if cname not in all_data:
            print(f"{cname:<24} NO CSV")
            continue
        data = all_data[cname]
        band_sum = {b: 0.0 for b, _ in BANDS}
        band_n = {b: 0 for b, _ in BANDS}
        total = 0.0
        for t_true, err in data:
            total += err
            for bname, bfn in BANDS:
                if bfn(t_true):
                    band_sum[bname] += err
                    band_n[bname] += 1
                    break
        row = f"{cname:<24}"
        for bname, _ in BANDS:
            n = band_n[bname]
            row += f"{(band_sum[bname]/n if n else math.nan):>9.4f}"
        row += f"{total/len(data):>9.4f}{len(data):>7}"
        print(row)

    # --- paired vs REFERENCE ---
    print(f"\nPaired test vs {REFERENCE}  (delta = cond - ref; negative = cond wins)")
    print(f"{'condition':<24}{'t-stat':>9}{'p(t)':>11}{'sig':>6}{'dMAE':>9}")
    print("-" * 59)
    if REFERENCE in all_data:
        for cname in CONDITIONS:
            if cname == REFERENCE or cname not in all_data:
                continue
            res = paired(all_data[cname], all_data[REFERENCE])
            if res is None:
                print(f"{cname:<24}  n mismatch")
                continue
            t_stat, p_t, delta = res
            print(f"{cname:<24}{t_stat:>9.3f}{p_t:>11.2e}{sig(p_t):>6}{delta:>9.4f}")

    # --- 直積の核仮説対比（per system）---
    print("\n核仮説対比 (per system, delta = A - B; negative = A wins)")
    contrasts = [
        ("空間効果(L1):  error_ratio vs t_l1",          "error_ratio",       "t_l1"),
        ("空間効果(Huber): error_ratio_huber vs t",     "error_ratio_huber", "t"),
        ("カーネル(t空間): t_l1 vs t",                    "t_l1",              "t"),
        ("カーネル(t空間): wing vs t",                    "wing",              "t"),
        ("カーネル(px空間): error_ratio vs error_ratio_huber", "error_ratio",  "error_ratio_huber"),
        ("カーネル(px空間): wing_px vs error_ratio",      "wing_px",           "error_ratio"),
    ]
    for system in SYSTEMS:
        print(f"\n[{system}]")
        print(f"{'contrast':<44}{'p(t)':>11}{'sig':>6}{'dMAE':>9}")
        print("-" * 70)
        for label, a, b in contrasts:
            ca, cb = f"{a}__{system}", f"{b}__{system}"
            if ca not in all_data or cb not in all_data:
                print(f"{label:<44}  (欠損)")
                continue
            res = paired(all_data[ca], all_data[cb])
            if res is None:
                print(f"{label:<44}  n mismatch")
                continue
            _, p_t, delta = res
            print(f"{label:<44}{p_t:>11.2e}{sig(p_t):>6}{delta:>9.4f}")


if __name__ == "__main__":
    main()
