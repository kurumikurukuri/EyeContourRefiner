"""EXP021 Phase2: aug sweep のバンド別 MAE + paired 検定。

各系統で baseline = t_l1__{system}（flags127 + EBR0.25, Phase1 結果）を基準に、
aug variant を paired t-test で比較する。
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).parent))
from exp021_band_mae import load_data, paired, sig, BANDS  # noqa: E402

SYSTEMS = ["none", "eye_scale"]
# baseline(再掲) + aug variants
VARIANTS = ["t_l1", "woflip", "ebr000", "ebr015", "ebr035", "ebr050"]
BASELINE = "t_l1"  # flags127 + EBR0.25


def band_row(cname, data):
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
    row = f"{cname:<22}"
    for bname, _ in BANDS:
        n = band_n[bname]
        row += f"{(band_sum[bname]/n if n else float('nan')):>9.4f}"
    row += f"{total/len(data):>9.4f}{len(data):>7}"
    return row


def main() -> None:
    header = f"{'condition':<22}" + "".join(f"{b:>9}" for b, _ in BANDS) + f"{'ALL':>9}{'n':>7}"
    for system in SYSTEMS:
        print(f"\n===== system = {system}  (baseline = {BASELINE}__{system}) =====")
        print(header)
        print("-" * len(header))
        data_map = {}
        for v in VARIANTS:
            c = f"{v}__{system}"
            d = load_data(c)
            if d is None:
                print(f"{c:<22} NO CSV")
                continue
            data_map[v] = d
            print(band_row(c, d))

        base = data_map.get(BASELINE)
        if not base:
            print("  baseline CSV 不在 — 検定スキップ")
            continue
        print(f"\nPaired vs {BASELINE}__{system}  (delta = variant - baseline; negative = variant wins)")
        print(f"{'variant':<22}{'p(t)':>11}{'sig':>6}{'dMAE':>9}")
        print("-" * 48)
        for v in VARIANTS:
            if v == BASELINE or v not in data_map:
                continue
            res = paired(data_map[v], base)
            if res is None:
                print(f"{v:<22}  n mismatch")
                continue
            _, p_t, delta = res
            print(f"{v:<22}{p_t:>11.2e}{sig(p_t):>6}{delta:>9.4f}")


if __name__ == "__main__":
    main()
