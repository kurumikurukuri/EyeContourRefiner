"""EXP020: 生 t バンド別 MAE 比較 + paired t-test (vs X0_none_baseline)"""
import csv
import math
from pathlib import Path

from scipy import stats

OUT = Path(__file__).parent.parent / "outputs"
REFERENCE = "X0_none_baseline"
CONDITIONS = [
    "X0_none_baseline",
    "X0_eye_scale_baseline",
    "X0_eye_height_baseline",
    "X1_independent_none_eye_scale",
    "X1_independent_none_eye_height",
    "X2_crossfuse_none_eye_scale",
    "X2_crossfuse_none_eye_height",
    "X3_attention_none_eye_scale",
    "X3_attention_none_eye_height",
]

BANDS = [
    ("t<0",     lambda t: t < 0),
    ("0-0.3",   lambda t: 0 <= t < 0.3),
    ("0.3-0.7", lambda t: 0.3 <= t < 0.7),
    ("0.7-1.0", lambda t: 0.7 <= t < 1.0),
    ("1.0-1.2", lambda t: 1.0 <= t < 1.2),
    ("t>=1.2",  lambda t: t >= 1.2),
]


def _get_col(row: dict, *names: str, path: Path) -> float:
    for name in names:
        v = row.get(name)
        if v is not None and v != "":
            return float(v)
    raise KeyError(f"None of {names} found in {path}: {list(row.keys())}")


def load_data(cname: str) -> list[tuple[float, float]] | None:
    """(t_true, abs_error) のリストを返す。CSV がなければ None。"""
    csv_path = OUT / cname / "test_predictions.csv"
    if not csv_path.exists():
        return None
    data = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t_true = _get_col(row, "t_true", "true_t_raw", path=csv_path)
            t_pred = _get_col(row, "t_pred", "pred_t_raw", path=csv_path)
            data.append((t_true, abs(t_true - t_pred)))
    return data


# --- データ読み込み ---
all_data: dict[str, list[tuple[float, float]]] = {}
for cname in CONDITIONS:
    d = load_data(cname)
    if d is not None:
        all_data[cname] = d

# --- Band MAE テーブル ---
header = (
    f"{'condition':<38}"
    + "".join(f"{b:>10}" for b, _ in BANDS)
    + f"{'ALL':>10}"
    + f"{'n':>7}"
)
print(header)
print("-" * len(header))

for cname in CONDITIONS:
    if cname not in all_data:
        print(f"{cname:<38} NO CSV")
        continue
    data = all_data[cname]

    band_sum = {b: 0.0 for b, _ in BANDS}
    band_n = {b: 0 for b, _ in BANDS}
    total_err = 0.0

    for t_true, err in data:
        total_err += err
        for bname, bfn in BANDS:
            if bfn(t_true):
                band_sum[bname] += err
                band_n[bname] += 1
                break

    row_str = f"{cname:<38}"
    for bname, _ in BANDS:
        n = band_n[bname]
        mae = band_sum[bname] / n if n > 0 else math.nan
        row_str += f"{mae:>10.4f}"
    n_total = len(data)
    overall = total_err / n_total if n_total > 0 else math.nan
    row_str += f"{overall:>10.4f}"
    row_str += f"{n_total:>7}"
    print(row_str)

# サンプル数フッター（REFERENCE の t_true 分布から算出）
ref_data = all_data.get(REFERENCE)
if ref_data:
    band_n_ref = {b: 0 for b, _ in BANDS}
    for t_true, _ in ref_data:
        for bname, bfn in BANDS:
            if bfn(t_true):
                band_n_ref[bname] += 1
                break
    print()
    print(f"{'n_samples':<38}" + "".join(f"{band_n_ref[b]:>10}" for b, _ in BANDS))

# --- Paired t-test (vs REFERENCE) ---
print()
print(f"Paired t-test vs {REFERENCE}")
ref_errors = [err for _, err in all_data.get(REFERENCE, [])]
if not ref_errors:
    print("  REFERENCE CSV not found — skipping")
else:
    print(
        f"{'condition':<38} {'t-stat':>9} {'p-value':>12} {'sig':>5} {'delta_MAE':>10}"
    )
    print("-" * 77)
    for cname in CONDITIONS:
        if cname == REFERENCE or cname not in all_data:
            continue
        cond_errors = [err for _, err in all_data[cname]]
        if len(cond_errors) != len(ref_errors):
            print(f"{cname:<38}  n mismatch ({len(ref_errors)} vs {len(cond_errors)})")
            continue
        t_stat, p_val = stats.ttest_rel(ref_errors, cond_errors)
        delta = sum(cond_errors) / len(cond_errors) - sum(ref_errors) / len(ref_errors)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        print(f"{cname:<38} {t_stat:>9.3f} {p_val:>12.2e} {sig:>5} {delta:>10.4f}")
    print("  delta_MAE = cond - ref  (negative = cond wins)")
