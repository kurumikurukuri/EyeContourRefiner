"""EXP023: OOF残差ベースの誤ラベル候補抽出。

多数派かつ OOF 残差（|t_pred-t_true|、生t単位）が大きいサンプルを誤ラベル候補としてランキングする。

「多数派」は t_true を bin-width 単位（既定 0.05）でビニングしたヒストグラムの
半値全幅（FWHM: ピーク bin のカウントの半分以上を持つ、ピークを含む連続 bin 群）で
定義する。固定の正常範囲 [0,1.2] は分布の疎な肩部分（モデルが苦手で残差が出やすい領域）
まで含んでしまい絞り込みが弱いため、密度の高いコア部分だけを「多数派」とする方が
誤ラベル候補のノイズを減らせる。

使い方:
    python experiments/EXP023_oof_mislabel/scr/rank_candidates.py \
        --oof-csv outputs/bc32_pw8_nested/oof_predictions.csv
    python .../rank_candidates.py --oof-csv outputs/bc16_pw1_nested/oof_predictions.csv \
        --top 100 --bin-width 0.05
    python .../rank_candidates.py --oof-csv <path> --majority-method range --t-min 0.0 --t-max 1.2

備考: bc16_pw1/bc32_pw8 の複数モデルが存在し単一デフォルトを置けないため、
      --oof-csv は必須。
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

EXP_DIR = Path(__file__).parent.parent
DEFAULT_OUTPUT = EXP_DIR / "outputs" / "candidates_majority_high_residual.csv"

ANNOTATOR_URL = "http://127.0.0.1:8000/?idx={idx}"

OUT_FIELDNAMES = [
    "rank", "idx", "image_path", "part_name", "sub_key", "point_index",
    "fold_idx", "t_true", "t_pred", "residual", "error_rate", "url",
]


def load_oof(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fwhm_range(values: list[float], bin_width: float) -> tuple[float, float]:
    """t_true 分布を bin_width でビニングし、ピーク bin を含む半値全幅の範囲を返す。

    ピーク bin から左右に、カウントが半値（ピークカウントの1/2）以上である限り
    連続的に bin を広げる。戻り値は [下限, 上限) の t 範囲（bin 境界）。
    """
    counts = Counter(math.floor(v / bin_width) for v in values)
    peak_bin = max(counts, key=lambda b: counts[b])
    half_max = counts[peak_bin] / 2.0

    left = peak_bin
    while counts.get(left - 1, 0) >= half_max:
        left -= 1
    right = peak_bin
    while counts.get(right + 1, 0) >= half_max:
        right += 1

    return left * bin_width, (right + 1) * bin_width


def rank_candidates(rows: list[dict], t_min: float, t_max: float, sort_by: str) -> list[dict]:
    """多数派 (t_min<=t_true<t_max) かつ sort_by（residual または error_rate）降順でランキングする。

    error_rate = residual / |t_true|（t_true に対する相対誤差率）。
    多数派フィルタ後は t_true が 0 近傍に来ないことを前提とする。
    """
    majority = [r for r in rows if t_min <= float(r["t_true"]) < t_max]
    for r in majority:
        t_true = float(r["t_true"])
        residual = abs(float(r["t_pred"]) - t_true)
        r["residual"] = residual
        r["error_rate"] = residual / abs(t_true) if t_true != 0 else float("inf")
    majority.sort(key=lambda r: r[sort_by], reverse=True)
    return majority


def main() -> int:
    p = argparse.ArgumentParser(description="EXP023 誤ラベル候補抽出（多数派×高残差）")
    p.add_argument("--oof-csv", type=Path, required=True,
                   help="OOF predictions CSV（例: outputs/bc32_pw8_nested/oof_predictions.csv）")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--majority-method", choices=["fwhm", "range"], default="fwhm")
    p.add_argument("--bin-width", type=float, default=0.05, help="fwhm 法のビン幅")
    p.add_argument("--t-min", type=float, default=0.0, help="range 法での多数派下限")
    p.add_argument("--t-max", type=float, default=1.2, help="range 法での多数派上限")
    p.add_argument("--sort-by", choices=["residual", "error_rate"], default="residual",
                   help="residual=|t_pred-t_true|（生t単位の絶対残差）, "
                        "error_rate=residual/|t_true|（t_trueに対する相対誤差率）")
    p.add_argument("--point-index", type=int, nargs="+", default=None,
                   help="対象 point_index を絞る（例: 目の中心 2 3 4）。省略時は全点")
    p.add_argument("--top", type=int, default=50, help="標準出力に表示する件数")
    args = p.parse_args()

    if not args.oof_csv.exists():
        print(f"[ERROR] OOF predictions 不在: {args.oof_csv}")
        return 1

    rows = load_oof(args.oof_csv)
    n_total = len(rows)

    if args.point_index is not None:
        allowed = set(args.point_index)
        rows = [r for r in rows if int(r["point_index"]) in allowed]
        print(f"point_index∈{sorted(allowed)} に絞り込み: {n_total} → {len(rows)} 件")

    if args.majority_method == "fwhm":
        t_values = [float(r["t_true"]) for r in rows]
        t_min, t_max = fwhm_range(t_values, args.bin_width)
        print(f"FWHM法（bin幅={args.bin_width}）で決定した多数派範囲: t∈[{t_min:.3f}, {t_max:.3f})")
    else:
        t_min, t_max = args.t_min, args.t_max

    ranked = rank_candidates(rows, t_min, t_max, args.sort_by)
    n_excluded = len(rows) - len(ranked)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDNAMES)
        writer.writeheader()
        for rank, r in enumerate(ranked, start=1):
            writer.writerow({
                "rank": rank,
                "idx": r["idx"],
                "image_path": r["image_path"],
                "part_name": r["part_name"],
                "sub_key": r["sub_key"],
                "point_index": r["point_index"],
                "fold_idx": r["fold_idx"],
                "t_true": f"{float(r['t_true']):.6f}",
                "t_pred": f"{float(r['t_pred']):.6f}",
                "residual": f"{r['residual']:.6f}",
                "error_rate": f"{r['error_rate']:.6f}",
                "url": ANNOTATOR_URL.format(idx=r["idx"]),
            })

    print(f"OOF全件: {n_total}  多数派(t∈[{t_min:.3f},{t_max:.3f})): {len(ranked)}  "
          f"範囲外除外: {n_excluded}  (sort_by={args.sort_by})")
    print(f"→ {args.output}\n")

    print(f"[TOP {min(args.top, len(ranked))} 誤ラベル候補（多数派×{args.sort_by}）]")
    for rank, r in enumerate(ranked[:args.top], start=1):
        print(
            f"  #{rank:<4} idx={r['idx']:<8} t_true={float(r['t_true']):.3f} "
            f"t_pred={float(r['t_pred']):.3f} resid={r['residual']:.3f} "
            f"err_rate={r['error_rate']:.3f} "
            f"{r['part_name']}/{r['sub_key']}[{r['point_index']}] "
            f"{ANNOTATOR_URL.format(idx=r['idx'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
