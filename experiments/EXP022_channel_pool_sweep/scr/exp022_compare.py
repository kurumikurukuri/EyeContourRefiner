"""EXP022: base_channel x pool_width 各条件のベースライン比較。

全条件の predictions.csv（生 t）を読み込み、CV MAE / hold-out MAE と、
ベースライン（bc32_pw32）に対する paired t-test / Wilcoxon を一覧表示する。

bc32_pw32（config/bc32_pw32.yaml）は EXP021 Phase3 ベスト構成と同一設定で
EXP022 内で学習し直した条件であり、EXP022 自身の outputs/ 内で完結する
（未実行の場合は比較なしで一覧のみ表示する）。

各 (base_channel, pool_width) 条件は config/bc{base_channel}_pw{pool_width}.yaml
として個別 yaml に定義され、discover_variants() がそのファイル名から
条件一覧を読み取る。
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

EXP_DIR = Path(__file__).parent.parent
CONFIG_DIR = EXP_DIR / "config"
OUTPUT_DIR = EXP_DIR / "outputs"

VARIANT_NAME_RE = re.compile(r"^bc(\d+)_pw(\d+)$")


def exp_name_for(base_channel: int, pool_width: int) -> str:
    return f"bc{base_channel}_pw{pool_width}"


def discover_variants(config_dir: Path = CONFIG_DIR) -> list[tuple[int, int]]:
    """config/bc{N}_pw{M}.yaml から (base_channel, pool_width) 条件一覧を読み取る。"""
    combos = []
    for path in sorted(config_dir.glob("bc*_pw*.yaml")):
        m = VARIANT_NAME_RE.match(path.stem)
        if m:
            combos.append((int(m.group(1)), int(m.group(2))))
    return sorted(combos)


def load_predictions(csv_path: Path) -> dict[int, dict]:
    """predictions.csv を {idx: {t_true, t_pred}} で読み込む。"""
    data = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int(row["idx"])
            data[idx] = {"t_true": float(row["t_true"]), "t_pred": float(row["t_pred"])}
    return data


def paired_test(ours: dict[int, dict], baseline: dict[int, dict]) -> tuple[float, float, float, float]:
    """paired t-test + Wilcoxon。共通 idx のみ使用。"""
    common = sorted(set(ours) & set(baseline))
    if not common:
        raise ValueError("共通サンプルなし")
    err_ours = np.array([abs(ours[i]["t_pred"] - ours[i]["t_true"]) for i in common])
    err_base = np.array([abs(baseline[i]["t_pred"] - baseline[i]["t_true"]) for i in common])
    delta = err_ours - err_base
    t_stat, p_t = stats.ttest_1samp(delta, 0.0)
    try:
        _, p_w = stats.wilcoxon(err_ours, err_base, zero_method="zsplit")
    except ValueError:
        p_w = float("nan")
    mae_ours = float(err_ours.mean())
    mae_base = float(err_base.mean())
    return mae_ours, mae_base, p_t, p_w


def sig(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def summarize(combos: list[tuple[int, int]], baseline_dir: Path) -> None:
    baseline_csv = baseline_dir / "predictions.csv"
    baseline = load_predictions(baseline_csv) if baseline_csv.exists() else None
    if baseline is None:
        print(f"\n[WARN] 比較ベースラインの predictions.csv が見つかりません: {baseline_csv}")

    print(f"\n{'=' * 78}\n  EXP022 結果一覧（baseline: {baseline_dir.name}）\n{'=' * 78}")
    header = (f"  {'exp_name':<14}{'CV MAE':>10}{'holdout MAE(raw t)':>20}"
              f"{'p(t-test)':>12}{'sig':>6}")
    print(header)

    for base_channel, pool_width in combos:
        exp_name = exp_name_for(base_channel, pool_width)
        model_dir = OUTPUT_DIR / exp_name
        exp_json = model_dir / "experiment.json"
        pred_csv = model_dir / "predictions.csv"

        if not exp_json.exists():
            print(f"  {exp_name:<14}  [未実行]")
            continue

        record = json.loads(exp_json.read_text(encoding="utf-8"))
        cv_mae = record["results"]["cv_mae"]
        holdout_mae_raw = record["results"].get("holdout_mae_raw_t")

        p_str, sig_str = "-", "-"
        if (baseline is not None and pred_csv.exists()
                and model_dir.resolve() != baseline_dir.resolve()):
            ours = load_predictions(pred_csv)
            _, _, p_t, _ = paired_test(ours, baseline)
            p_str, sig_str = f"{p_t:.2e}", sig(p_t)

        holdout_str = f"{holdout_mae_raw:.6f}" if holdout_mae_raw is not None else "-"
        print(f"  {exp_name:<14}{cv_mae:>10.6f}{holdout_str:>20}{p_str:>12}{sig_str:>6}")


def main() -> int:
    combos = discover_variants()
    baseline_dir = OUTPUT_DIR / exp_name_for(32, 32)
    summarize(combos, baseline_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
