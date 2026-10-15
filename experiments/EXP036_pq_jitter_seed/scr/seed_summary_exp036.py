"""EXP036: 条件（J0 / J3 / J4）ごとの 3 seed 集計と 3 seed アンサンブル予測の生成。

- 各 config の test_predictions.csv から hold-out MAE（生 t）を読み、条件ごとに seed 平均・SD を出す
- passrate_overall.csv（compute_passrate.py の出力）があれば pass_std / pass_strict も同様に集計する
- 3 seed の t_pred を平均したアンサンブル予測を outputs/{cond}_ens3/test_predictions.csv に保存する
  （passrate はその後 compute_passrate.py --configs J0_ens3 J3_ens3 J4_ens3 で計算する）

使い方:
    python experiments/EXP036_pq_jitter_seed/scr/seed_summary_exp036.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

EXP_DIR = Path(__file__).parent.parent
OUTPUT_DIR = EXP_DIR / "outputs"
CONDS = ("J0", "J3", "J4")
SEEDS = (42, 43, 44)
KEY_COLS = ["idx", "part_name", "sub_key", "point_index"]


def load_predictions(config: str) -> pd.DataFrame:
    path = OUTPUT_DIR / config / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"予測 CSV がありません: {path}")
    return pd.read_csv(path)


def build_ensemble(cond: str) -> pd.DataFrame:
    """同条件 3 seed の t_pred を平均する。サンプルの並びが一致しない場合はエラー。"""
    frames = [load_predictions(f"{cond}_s{s}") for s in SEEDS]
    base = frames[0]
    for s, df in zip(SEEDS[1:], frames[1:]):
        if not df[KEY_COLS + ["t_true"]].equals(base[KEY_COLS + ["t_true"]]):
            raise ValueError(f"{cond}_s{s} のサンプル順が {cond}_s{SEEDS[0]} と一致しません")
    t_pred = sum(df["t_pred"] for df in frames) / len(frames)
    return base.assign(t_pred=t_pred.round(6))


def seed_stats(values: list[float]) -> dict:
    s = pd.Series(values)
    return {"values": values, "mean": float(s.mean()), "sd": float(s.std(ddof=1))}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    passrate_path = OUTPUT_DIR / "passrate_overall.csv"
    passrate = pd.read_csv(passrate_path).set_index("config") if passrate_path.exists() else None

    summary: dict = {}
    for cond in CONDS:
        maes = []
        for s in SEEDS:
            df = load_predictions(f"{cond}_s{s}")
            maes.append(float((df["t_pred"] - df["t_true"]).abs().mean()))
        rec = {"mae_raw_t": seed_stats(maes)}
        if passrate is not None:
            for col in ("pass_std", "pass_strict"):
                match = [c for c in passrate.columns if c.startswith(col)]
                if match:
                    rec[col] = seed_stats(
                        [float(passrate.loc[f"{cond}_s{s}", match[0]]) for s in SEEDS])

        ens = build_ensemble(cond)
        ens_dir = OUTPUT_DIR / f"{cond}_ens3"
        ens_dir.mkdir(parents=True, exist_ok=True)
        ens.to_csv(ens_dir / "test_predictions.csv", index=False)
        rec["ens3_mae_raw_t"] = float((ens["t_pred"] - ens["t_true"]).abs().mean())
        summary[cond] = rec

    base_mean = summary["J0"]["mae_raw_t"]["mean"]
    print(f"{'cond':<5} {'MAE mean':>9} {'MAE sd':>8} {'vs J0':>7} {'pass_std':>14} {'ens3 MAE':>9}")
    for cond, rec in summary.items():
        m = rec["mae_raw_t"]
        ps = rec.get("pass_std")
        ps_txt = f"{ps['mean']:.2f}±{ps['sd']:.2f}" if ps else "-"
        print(f"{cond:<5} {m['mean']:9.6f} {m['sd']:8.6f} {(1 - m['mean'] / base_mean) * 100:+6.2f}% "
              f"{ps_txt:>14} {rec['ens3_mae_raw_t']:9.6f}")

    out = OUTPUT_DIR / "seed_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
