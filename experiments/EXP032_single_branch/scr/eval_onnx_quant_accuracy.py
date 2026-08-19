"""単独ブランチ ONNX(fp32) vs ONNX INT8(動的量子化) の精度比較。

`tools/model_benchmark/cli.py --runtimes onnx onnx_int8` はレイテンシ/サイズのみを
合成入力で測定する（精度は見ない）。本スクリプトは実際のdb_replicaテストマニフェスト
（4,900点）を使い、`tools/model_benchmark/results/{config}_ensemble.onnx` と
`{config}_ensemble_int8.onnx`（3fold平均を1グラフに焼き込んだアンサンブルONNX、
infer_exp032.pyのPyTorchアンサンブルと同一の前処理・後処理）で推論し、
t_true/t_predを比較してMAE・passrateを算出する。

事前に `tools/model_benchmark/cli.py --models-dir .../SB_none --runtimes onnx onnx_int8`
等を実行し、`tools/model_benchmark/results/{config}_ensemble[_int8].onnx` が
存在している必要がある。

使い方:
    python experiments/EXP032_single_branch/scr/eval_onnx_quant_accuracy.py
    python experiments/EXP032_single_branch/scr/eval_onnx_quant_accuracy.py --configs SB_none
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
MANIFEST_DIR = EXP_DIR / "manifests"
OUTPUT_DIR = EXP_DIR / "outputs"
BENCH_RESULTS_DIR = ROOT / "tools" / "model_benchmark" / "results"

sys.path.insert(0, str(ROOT))

from eyeliner.data.dataset import EyelinerPatchDataset, load_manifest  # noqa: E402
from eyeliner.data.patch_utils import get_pq_prime_dist  # noqa: E402
from eyeliner.data.weights import compute_eye_meta  # noqa: E402
from eyeliner.training.dual_branch import (  # noqa: E402
    branch_a_x0,
    branch_b_eye_scale,
    branch_pred_to_raw_t,
)

CONFIGS = {
    "SB_none": "none",
    "SB_eyescale": "eye_scale",
}
_BRANCH_BUILDERS = {
    "none": branch_a_x0,
    "eye_scale": branch_b_eye_scale,
}


def run_onnx_ensemble(
    onnx_path: Path,
    records: list[dict],
    branch,
    batch_size: int = 32,
) -> np.ndarray:
    """アンサンブルONNXグラフでrecordsを推論し、生tの配列を返す。"""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]

    orig_pq_dists = [
        np.sqrt((float(r["qx"]) - float(r["px"])) ** 2 + (float(r["qy"]) - float(r["py"])) ** 2)
        for r in records
    ]
    pq_prime_dists = [
        get_pq_prime_dist(r, d, branch.pq_rescale_mode, branch.pq_rescale_factor)
        for r, d in zip(records, orig_pq_dists)
    ]

    ds = EyelinerPatchDataset(
        records, augment_data=False, preload_images=True,
        patch_t_min=branch.patch_t_min, patch_t_max=branch.patch_t_max,
        patch_perp_scale=branch.patch_perp_scale, patch_size=branch.patch_size,
        use_t_norm=True,
        label_t_min=branch.label_t_min, label_t_max=branch.label_t_max,
        pq_rescale_mode=branch.pq_rescale_mode, pq_rescale_factor=branch.pq_rescale_factor,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds_norm: list[float] = []
    for batch in loader:
        x = batch[0].numpy().astype(np.float32)
        is_upper = batch[1].numpy().astype(np.float32)
        feed = {input_names[0]: x, input_names[1]: is_upper}
        out = session.run(None, feed)[0]
        preds_norm.extend(out.reshape(-1).tolist())

    preds_norm_t = torch.tensor(preds_norm, dtype=torch.float32).unsqueeze(-1)
    pq_dist_t = torch.tensor(orig_pq_dists, dtype=torch.float32).unsqueeze(-1)
    pq_prime_t = torch.tensor(pq_prime_dists, dtype=torch.float32).unsqueeze(-1)
    t_raw = branch_pred_to_raw_t(preds_norm_t, branch, pq_dist_t, pq_prime_t).squeeze(-1).numpy()
    return t_raw


def evaluate(config_name: str, pq_rescale_mode: str) -> dict:
    branch = _BRANCH_BUILDERS[pq_rescale_mode]()
    records = load_manifest(MANIFEST_DIR / "manifest_test.csv")
    compute_eye_meta(records)
    t_true = np.array([float(r["t"]) for r in records])

    fp32_path = BENCH_RESULTS_DIR / f"{config_name}_ensemble.onnx"
    int8_path = BENCH_RESULTS_DIR / f"{config_name}_ensemble_int8.onnx"
    if not fp32_path.exists() or not int8_path.exists():
        raise FileNotFoundError(
            f"{fp32_path} / {int8_path} が見つかりません。先に "
            f"`python tools/model_benchmark/cli.py --models-dir "
            f"experiments/EXP032_single_branch/outputs/{config_name} "
            f"--runtimes onnx onnx_int8` を実行してください。"
        )

    t_pred_fp32 = run_onnx_ensemble(fp32_path, records, branch)
    t_pred_int8 = run_onnx_ensemble(int8_path, records, branch)

    mae_fp32 = float(np.mean(np.abs(t_pred_fp32 - t_true)))
    mae_int8 = float(np.mean(np.abs(t_pred_int8 - t_true)))
    max_abs_diff = float(np.max(np.abs(t_pred_int8 - t_pred_fp32)))
    mean_abs_diff = float(np.mean(np.abs(t_pred_int8 - t_pred_fp32)))

    # 参考: PyTorchでの既存test_predictions.csv（infer_exp032.py出力）のMAEと比較
    pt_csv = OUTPUT_DIR / config_name / "test_predictions.csv"
    mae_pytorch = None
    if pt_csv.exists():
        with open(pt_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        diffs = [abs(float(r["t_pred"]) - float(r["t_true"])) for r in rows]
        mae_pytorch = float(np.mean(diffs))

    return {
        "config": config_name,
        "n": len(records),
        "mae_pytorch_fp32": mae_pytorch,
        "mae_onnx_fp32": mae_fp32,
        "mae_onnx_int8": mae_int8,
        "int8_vs_fp32_pct": (mae_int8 - mae_fp32) / mae_fp32 * 100 if mae_fp32 > 0 else None,
        "pred_diff_mean_abs": mean_abs_diff,
        "pred_diff_max_abs": max_abs_diff,
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description="ONNX INT8量子化の精度影響評価")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    args = ap.parse_args()

    print(f"\n{'='*78}\n  ONNX INT8量子化 精度比較（db_replica test, n=4900）\n{'='*78}")
    for config_name in args.configs:
        result = evaluate(config_name, CONFIGS[config_name])
        print(f"\n--- {config_name} ---")
        print(f"  n                    = {result['n']}")
        if result["mae_pytorch_fp32"] is not None:
            print(f"  MAE PyTorch(fp32)    = {result['mae_pytorch_fp32']:.6f}  (参考: infer_exp032.py出力)")
        print(f"  MAE ONNX(fp32)       = {result['mae_onnx_fp32']:.6f}")
        print(f"  MAE ONNX(int8)       = {result['mae_onnx_int8']:.6f}")
        print(f"  int8 vs fp32 diff%   = {result['int8_vs_fp32_pct']:+.3f}%")
        print(f"  pred差分 mean|Δt|    = {result['pred_diff_mean_abs']:.6f}")
        print(f"  pred差分 max|Δt|     = {result['pred_diff_max_abs']:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
