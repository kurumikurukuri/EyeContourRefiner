"""EXP035 ランドマーク誤差頑健性評価: テストの P, Q を R 中心に回転させた入力で MAE を測る。

テストレコードの P, Q を R 中心に θ ~ U(±rot_deg)（固定 seed）回転する。t と R は不変なので
正解ラベルは変わらず、クリーン入力からの MAE 劣化幅 = FaceMesh の PQ 向き誤差への脆さになる。
クリーン MAE に差が出なくても、頑健性が上がったかを判定できる。

  劣化幅 = MAE(摂動入力) - MAE(クリーン入力)
  per-sample 劣化 d_i = |err_pert_i| - |err_clean_i| を baseline と paired t-test で比較する。

使い方:
    python experiments/EXP035_pq_jitter/scr/robustness_exp035.py
    python experiments/EXP035_pq_jitter/scr/robustness_exp035.py --rot-deg 3 --seed 0 --baseline J0_base
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import stats
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
CONFIG_DIR = EXP_DIR / "config"
MANIFEST_DIR = EXP_DIR / "manifests"
OUTPUT_DIR = EXP_DIR / "outputs"
DB_PATH = ROOT / "data" / "samples.db"

sys.path.insert(0, str(ROOT))

from eyeliner.data.dataset import load_manifest  # noqa: E402
from eyeliner.data.dataset_dual import EyelinerPatchDualDataset  # noqa: E402
from eyeliner.data.pq_perturbation import perturb_records_pq_rotation  # noqa: E402
from eyeliner.data.weights import (  # noqa: E402
    compute_aux_face_shape_meta,
    compute_eye_meta,
)
from eyeliner.models.dual_ensemble import load_dual_model  # noqa: E402
from eyeliner.training.dual_branch import (  # noqa: E402
    build_branch_configs,
    fold_ensemble_raw_t,
)

CONFIGS = ("J0_base", "J1_weak", "J2_strong", "J3_xstrong", "J4_rot20", "J5_rot30")


def predict_raw_t(models, records: list[dict], cfg: dict, device: torch.device) -> np.ndarray:
    """fold アンサンブルの生 t 予測を返す（augment_data=False）。"""
    use_aux = bool(cfg.get("use_aux_face_shape", False))
    fusion_mode = cfg.get("fusion_mode", "independent")
    cfg_a, cfg_b = build_branch_configs(cfg.get("pq_rescale_mode_b", "eye_scale"))
    ds = EyelinerPatchDualDataset(
        records, cfg_a, cfg_b, augment_data=False, preload_images=False,
        use_aux_face_shape=use_aux,
    )
    loader = DataLoader(ds, batch_size=cfg.get("batch", 32), shuffle=False, num_workers=0)
    preds: list[float] = []
    with torch.no_grad():
        for batch in loader:
            x_a, x_b, is_upper = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            pq_dist, pq_prime = batch[12].to(device), batch[13].to(device)
            aux = batch[14].to(device) if use_aux else None
            t_ens = fold_ensemble_raw_t(
                models, x_a, x_b, is_upper, fusion_mode, cfg_a, cfg_b, pq_dist, pq_prime, aux,
            )
            preds.extend(t_ens.squeeze(-1).cpu().tolist())
    return np.asarray(preds, dtype=np.float64)


def prepare_records(records: list[dict], use_aux: bool) -> list[dict]:
    """eye_scale / pq_ratio 等のメタを付与する（PQ 摂動の前に呼ぶこと）。"""
    compute_eye_meta(records)
    if use_aux:
        compute_aux_face_shape_meta(records, str(DB_PATH))
    return records


def evaluate_config(name: str, base_records: list[dict], rot_deg: float, seed: int,
                    device: torch.device) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    use_aux = bool(cfg.get("use_aux_face_shape", False))

    fold_paths = sorted((OUTPUT_DIR / name).glob("eyeliner_detection_fold*.pt"))
    if not fold_paths:
        raise FileNotFoundError(f"fold モデルが見つかりません: {OUTPUT_DIR / name}")
    models = [load_dual_model(p).to(device) for p in fold_paths]

    # eye_scale / pq_ratio 等のメタはクリーンな P, Q から先に計算し、摂動レコードへ引き継ぐ。
    # 学習時の PQ 回転も compute_eye_meta の後に適用されメタはクリーン値のままなので、
    # 同じ条件にそろえる（摂動後に再計算すると eye_scale の変化が混入する）。
    clean = prepare_records([dict(r) for r in base_records], use_aux)
    perturbed = perturb_records_pq_rotation(clean, rot_deg, seed)
    t_true = np.asarray([float(r["t"]) for r in clean], dtype=np.float64)

    err_clean = np.abs(predict_raw_t(models, clean, cfg, device) - t_true)
    err_pert = np.abs(predict_raw_t(models, perturbed, cfg, device) - t_true)
    return {
        "mae_clean": float(err_clean.mean()),
        "mae_perturbed": float(err_pert.mean()),
        "degradation": float((err_pert - err_clean).mean()),
        "per_sample_degradation": (err_pert - err_clean),
        "train_pq_jitter_rot_deg": float(cfg.get("pq_jitter_rot_deg", 0.0)),
    }


def compare_to_baseline(results: dict[str, dict], baseline: str) -> dict[str, dict]:
    """per-sample 劣化幅を baseline と paired t-test する（負 = baseline より頑健）。"""
    base = results[baseline]["per_sample_degradation"]
    out: dict[str, dict] = {}
    for name, res in results.items():
        if name == baseline:
            continue
        diff = res["per_sample_degradation"] - base
        t_stat, p = stats.ttest_1samp(diff, 0.0)
        out[name] = {
            "mean_diff_degradation": float(diff.mean()),
            "rel_diff_pct": float(100.0 * diff.mean() / base.mean()) if base.mean() else float("nan"),
            "t": float(t_stat), "p": float(p),
        }
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description="EXP035 PQ 回転ランドマーク誤差への頑健性評価")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--rot-deg", type=float, default=3.0, help="評価時の回転幅 ±deg（既定 3）")
    ap.add_argument("--seed", type=int, default=0, help="摂動の固定 seed（全 config で同一）")
    ap.add_argument("--baseline", default="J0_base", choices=list(CONFIGS))
    ap.add_argument("--manifest", type=Path, default=MANIFEST_DIR / "manifest_test.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_records = load_manifest(args.manifest)
    print(f"デバイス: {device}  サンプル数: {len(base_records)}  回転: ±{args.rot_deg}°  seed={args.seed}")

    results = {n: evaluate_config(n, base_records, args.rot_deg, args.seed, device) for n in args.configs}

    print(f"\n{'='*78}\n  EXP035 頑健性（テスト P,Q を R 中心に ±{args.rot_deg}° 回転）\n{'='*78}")
    print(f"  {'config':<10} {'train_rot':>9} {'MAE_clean':>10} {'MAE_pert':>10} {'劣化幅':>10} {'劣化率':>8}")
    for n, r in results.items():
        pct = 100.0 * r["degradation"] / r["mae_clean"]
        print(f"  {n:<10} {r['train_pq_jitter_rot_deg']:>8.1f}° {r['mae_clean']:>10.5f} "
              f"{r['mae_perturbed']:>10.5f} {r['degradation']:>10.5f} {pct:>7.2f}%")

    comparison: dict[str, dict] = {}
    if args.baseline in results:
        comparison = compare_to_baseline(results, args.baseline)
        print(f"\n  劣化幅の paired t-test（vs {args.baseline}、負 = より頑健）")
        for n, c in comparison.items():
            print(f"  {n:<10} Δ劣化={c['mean_diff_degradation']:+.5f} ({c['rel_diff_pct']:+.1f}%)  "
                  f"t={c['t']:.2f}  p={c['p']:.4f}")

    out = {
        "rot_deg": args.rot_deg, "seed": args.seed, "baseline": args.baseline,
        "configs": {n: {k: v for k, v in r.items() if k != "per_sample_degradation"}
                    for n, r in results.items()},
        "vs_baseline": comparison,
    }
    path = OUTPUT_DIR / "EXP035_robustness_summary.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  保存: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
