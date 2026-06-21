"""EXP023: 2モデル（bc16_pw1, bc32_pw8）OOF残差の突き合わせ。

両モデルで共通して残差が大きいサンプルを、誤ラベル候補として min(resid_16, resid_32)
でランキングする（モデル個別の癖による誤検出を減らす狙い。spec EXP023_spec.md 参照）。

誤ラベルの強いシグナル:
  - min_resid が大きい（両モデルとも t_true から大きく外す）
  - sign_agree=1（両モデルが同じ方向に外す = 真の値は t_true とは逆側にある）
  - pred_diff が小さい（両モデルが同じ予測値に収束 = 真の値で合意している）

使い方:
    python experiments/EXP023_oof_mislabel/scr/intersect_candidates.py \
        --oof-csv-16 outputs/bc16_pw1_nested/oof_predictions.csv \
        --oof-csv-32 outputs/bc32_pw8_nested/oof_predictions.csv \
        --output outputs/candidates_intersect.csv --top 50
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

# 同一 scr ディレクトリの rank_candidates から多数派範囲判定と URL 生成を再利用する。
sys.path.insert(0, str(Path(__file__).parent))
from rank_candidates import ANNOTATOR_URL, fwhm_range  # noqa: E402

EXP_DIR = Path(__file__).parent.parent

OUT_FIELDNAMES = [
    "rank", "idx", "image_path", "part_name", "sub_key", "point_index",
    "t_true", "t_pred_16", "t_pred_32", "resid_16", "resid_32",
    "min_resid", "pred_diff", "sign_agree", "url",
]


def load_oof(path: Path) -> dict[str, dict]:
    """OOF CSV を読み idx（DB主キー・ユニーク）→ row の辞書で返す。"""
    with open(path, newline="", encoding="utf-8") as f:
        return {r["idx"]: r for r in csv.DictReader(f)}


def join_residuals(rows16: dict[str, dict], rows32: dict[str, dict]) -> list[dict]:
    """両モデルのOOF予測を idx で結合し、resid_16 / resid_32 を付与した行リストを返す。

    両方に存在する idx のみ残す（同一 manifest から派生しているため基本全件一致）。
    t_true は両モデルで同一なので rows16 側を採用する。
    """
    joined: list[dict] = []
    for idx, r16 in rows16.items():
        r32 = rows32.get(idx)
        if r32 is None:
            continue
        t_true = float(r16["t_true"])
        t_pred_16 = float(r16["t_pred"])
        t_pred_32 = float(r32["t_pred"])
        joined.append({
            "idx": idx,
            "image_path": r16["image_path"],
            "part_name": r16["part_name"],
            "sub_key": r16["sub_key"],
            "point_index": r16["point_index"],
            "t_true": t_true,
            "t_pred_16": t_pred_16,
            "t_pred_32": t_pred_32,
            "resid_16": abs(t_pred_16 - t_true),
            "resid_32": abs(t_pred_32 - t_true),
        })
    return joined


def rank_intersection(joined: list[dict], t_min: float, t_max: float) -> list[dict]:
    """多数派 (t_min<=t_true<t_max) で絞り込み、min_resid 降順でランキングする。

    各行に min_resid / pred_diff / sign_agree を付与する。
      - min_resid = min(resid_16, resid_32): 両モデルとも少なくともこれだけ外している
      - pred_diff = |t_pred_16 - t_pred_32|: 両モデルの予測値の差（小さいほど合意）
      - sign_agree = 1 if (t_pred_16-t_true)*(t_pred_32-t_true) > 0 else 0
            両モデルが同方向に外す（誤ラベルで典型的なパターン）
    """
    rows = [r for r in joined if t_min <= r["t_true"] < t_max]
    for r in rows:
        r["min_resid"] = min(r["resid_16"], r["resid_32"])
        r["pred_diff"] = abs(r["t_pred_16"] - r["t_pred_32"])
        d16 = r["t_pred_16"] - r["t_true"]
        d32 = r["t_pred_32"] - r["t_true"]
        r["sign_agree"] = 1 if d16 * d32 > 0 else 0
    rows.sort(key=lambda r: r["min_resid"], reverse=True)
    return rows


def _fmt(x: float) -> str:
    return f"{x:.6f}"


def main() -> int:
    p = argparse.ArgumentParser(description="EXP023 両モデルOOF残差の突き合わせ（誤ラベル候補）")
    p.add_argument("--oof-csv-16", type=Path, required=True,
                   help="bc16_pw1（悪いモデル）の OOF predictions CSV")
    p.add_argument("--oof-csv-32", type=Path, required=True,
                   help="bc32_pw8（良いモデル）の OOF predictions CSV")
    p.add_argument("--output", type=Path,
                   default=EXP_DIR / "outputs" / "candidates_intersect.csv")
    p.add_argument("--majority-method", choices=["fwhm", "range"], default="fwhm")
    p.add_argument("--bin-width", type=float, default=0.05, help="fwhm 法のビン幅")
    p.add_argument("--t-min", type=float, default=0.0, help="range 法での多数派下限")
    p.add_argument("--t-max", type=float, default=1.2, help="range 法での多数派上限")
    p.add_argument("--point-index", type=int, nargs="+", default=None,
                   help="対象 point_index を絞る（例: 2 3 4）。省略時は全点")
    p.add_argument("--top", type=int, default=50, help="標準出力に表示する件数")
    args = p.parse_args()

    for label, path in (("16", args.oof_csv_16), ("32", args.oof_csv_32)):
        if not path.exists():
            print(f"[ERROR] OOF predictions({label}) 不在: {path}")
            return 1

    rows16 = load_oof(args.oof_csv_16)
    rows32 = load_oof(args.oof_csv_32)
    n16, n32 = len(rows16), len(rows32)

    joined = join_residuals(rows16, rows32)
    n_total = len(joined)
    n_only16 = len(rows16) - n_total
    n_only32 = len(rows32) - n_total
    print(f"OOF件数: bc16_pw1={n16}  bc32_pw8={n32}  共通idx={n_total}  "
          f"(16のみ={n_only16}  32のみ={n_only32})")

    if args.point_index is not None:
        allowed = set(args.point_index)
        joined = [r for r in joined if int(r["point_index"]) in allowed]
        print(f"point_index∈{sorted(allowed)} に絞り込み: {n_total} → {len(joined)} 件")

    if args.majority_method == "fwhm":
        t_values = [r["t_true"] for r in joined]
        t_min, t_max = fwhm_range(t_values, args.bin_width)
        print(f"FWHM法（bin幅={args.bin_width}）で決定した多数派範囲: t∈[{t_min:.3f}, {t_max:.3f})")
    else:
        t_min, t_max = args.t_min, args.t_max

    ranked = rank_intersection(joined, t_min, t_max)
    n_excluded = len(joined) - len(ranked)

    # min_resid の分布サマリ（レビュー可能規模の目安）
    if ranked:
        min_resids = sorted(r["min_resid"] for r in ranked)
        print(f"min_resid: mean={statistics.mean(min_resids):.4f} "
              f"median={min_resids[len(min_resids) // 2]:.4f} "
              f"max={min_resids[-1]:.4f}")
        for thr in (0.20, 0.25, 0.30):
            n = sum(1 for v in min_resids if v >= thr)
            n_agree = sum(1 for r in ranked if r["min_resid"] >= thr and r["sign_agree"] == 1)
            print(f"  min_resid>={thr:.2f}: {n} 件 (うち sign_agree=1: {n_agree} 件)")

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
                "t_true": _fmt(r["t_true"]),
                "t_pred_16": _fmt(r["t_pred_16"]),
                "t_pred_32": _fmt(r["t_pred_32"]),
                "resid_16": _fmt(r["resid_16"]),
                "resid_32": _fmt(r["resid_32"]),
                "min_resid": _fmt(r["min_resid"]),
                "pred_diff": _fmt(r["pred_diff"]),
                "sign_agree": r["sign_agree"],
                "url": ANNOTATOR_URL.format(idx=r["idx"]),
            })

    print(f"多数派(t∈[{t_min:.3f},{t_max:.3f})): {len(ranked)}  範囲外除外: {n_excluded}")
    print(f"→ {args.output}\n")

    print(f"[TOP {min(args.top, len(ranked))} 誤ラベル候補（多数派×min_resid）]")
    for rank, r in enumerate(ranked[:args.top], start=1):
        flag = "★同方向" if r["sign_agree"] == 1 else " 逆方向"
        print(
            f"  #{rank:<4} idx={r['idx']:<8} t_true={r['t_true']:.3f} "
            f"p16={r['t_pred_16']:+.3f} p32={r['t_pred_32']:+.3f} "
            f"min={r['min_resid']:.3f} diff={r['pred_diff']:.3f} {flag} "
            f"{r['part_name']}/{r['sub_key']}[{r['point_index']}] "
            f"{ANNOTATOR_URL.format(idx=r['idx'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
