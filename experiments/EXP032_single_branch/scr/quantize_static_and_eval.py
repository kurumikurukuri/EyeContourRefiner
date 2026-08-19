"""単独ブランチ ONNX の静的量子化（QDQ, INT8, Conv含む）を試し、速度・サイズ・精度を評価する。

`eval_onnx_quant_accuracy.py`（動的量子化=MatMulのみ）の続き。Conv 層を含めて量子化するには
静的量子化（QDQ形式、キャリブレーションデータで活性化レンジを事前計算）が必要である。
動的量子化のConvIntegerがCPUExecutionProviderで未実装だった問題を、
QDQ形式は回避できる（ConvInteger等の専用整数opを使わずQuantizeLinear/DequantizeLinearの
標準opで表現するため）。

キャリブレーションデータは実際のtrain manifestからランダムサンプリングした
`--calib-samples`件（既定200）。

使い方:
    python experiments/EXP032_single_branch/scr/quantize_static_and_eval.py
    python experiments/EXP032_single_branch/scr/quantize_static_and_eval.py --configs SB_none
"""
from __future__ import annotations

import argparse
import sys
import time
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
sys.path.insert(0, str(Path(__file__).parent))

from eyeliner.data.dataset import EyelinerPatchDataset, load_manifest  # noqa: E402
from eyeliner.data.weights import compute_eye_meta  # noqa: E402
from eyeliner.training.dual_branch import branch_a_x0, branch_b_eye_scale  # noqa: E402
from tools.model_benchmark import onnx_utils  # noqa: E402
from tools.model_benchmark.metrics import file_size_bytes  # noqa: E402

from eval_onnx_quant_accuracy import run_onnx_ensemble  # noqa: E402

CONFIGS = {
    "SB_none": "none",
    "SB_eyescale": "eye_scale",
}
_BRANCH_BUILDERS = {
    "none": branch_a_x0,
    "eye_scale": branch_b_eye_scale,
}


class PatchCalibrationDataReader:
    """実データ（train manifestからのランダムサンプル）でキャリブレーションを与える。"""

    def __init__(self, records: list[dict], branch, n_samples: int, seed: int = 42) -> None:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(records), size=min(n_samples, len(records)), replace=False)
        sampled = [records[i] for i in idx]
        ds = EyelinerPatchDataset(
            sampled, augment_data=False, preload_images=True,
            patch_t_min=branch.patch_t_min, patch_t_max=branch.patch_t_max,
            patch_perp_scale=branch.patch_perp_scale, patch_size=branch.patch_size,
            use_t_norm=True,
            label_t_min=branch.label_t_min, label_t_max=branch.label_t_max,
            pq_rescale_mode=branch.pq_rescale_mode, pq_rescale_factor=branch.pq_rescale_factor,
        )
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        self._batches = [
            {"patch": b[0].numpy().astype(np.float32), "is_upper": b[1].numpy().astype(np.float32)}
            for b in loader
        ]
        self._iter = iter(self._batches)

    def get_next(self) -> dict | None:
        return next(self._iter, None)

    def __len__(self) -> int:
        return len(self._batches)


def benchmark_latency(onnx_path: Path, batch_sizes=(1, 8, 32), iterations: int = 100, warmup: int = 10) -> dict:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]
    input_shapes = {i.name: i.shape for i in session.get_inputs()}
    h, w = input_shapes["patch"][2], input_shapes["patch"][3]
    if not isinstance(h, int):
        h, w = 64, 64

    results = {}
    for bs in batch_sizes:
        rng = np.random.RandomState(42)
        feed = {
            "patch": rng.randn(bs, 3, h, w).astype(np.float32),
            "is_upper": rng.randint(0, 2, size=(bs, 1)).astype(np.float32),
        }
        for _ in range(warmup):
            session.run(None, feed)
        lat = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            session.run(None, feed)
            lat.append((time.perf_counter() - t0) * 1000.0)
        arr = np.asarray(lat)
        results[bs] = {
            "mean_ms": float(arr.mean()), "p50_ms": float(np.percentile(arr, 50)),
            "thr_per_s": bs / (arr.mean() / 1000.0),
        }
    return results


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description="ONNX 静的量子化(QDQ, Conv含む)の速度・サイズ・精度評価")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--calib-samples", type=int, default=200)
    args = ap.parse_args()

    train_records = load_manifest(MANIFEST_DIR / "manifest_train.csv")
    compute_eye_meta(train_records)
    test_records = load_manifest(MANIFEST_DIR / "manifest_test.csv")
    compute_eye_meta(test_records)
    t_true = np.array([float(r["t"]) for r in test_records])

    print(f"\n{'='*78}\n  ONNX 静的量子化(QDQ, Conv含む) 評価  calib_samples={args.calib_samples}\n{'='*78}")

    for config_name in args.configs:
        branch = _BRANCH_BUILDERS[CONFIGS[config_name]]()
        fp32_path = BENCH_RESULTS_DIR / f"{config_name}_ensemble.onnx"
        if not fp32_path.exists():
            print(f"[SKIP] {fp32_path} が見つかりません。先にtools/model_benchmark/cli.pyを実行してください。")
            continue

        calib_reader = PatchCalibrationDataReader(train_records, branch, args.calib_samples)
        static_path = BENCH_RESULTS_DIR / f"{config_name}_ensemble_static_int8.onnx"
        print(f"\n--- {config_name}: 静的量子化中 (calib n={len(calib_reader)}) ---")
        onnx_utils.quantize_static_onnx(fp32_path, static_path, calib_reader)

        fp32_size = file_size_bytes(fp32_path)
        static_size = file_size_bytes(static_path)

        lat_fp32 = benchmark_latency(fp32_path)
        lat_static = benchmark_latency(static_path)

        t_pred_fp32 = run_onnx_ensemble(fp32_path, test_records, branch)
        t_pred_static = run_onnx_ensemble(static_path, test_records, branch)
        mae_fp32 = float(np.mean(np.abs(t_pred_fp32 - t_true)))
        mae_static = float(np.mean(np.abs(t_pred_static - t_true)))

        print(f"\n=== {config_name} 結果 ===")
        print(f"  ファイルサイズ: fp32={fp32_size/1e6:.2f}MB  static_int8={static_size/1e6:.2f}MB"
              f"  ({(static_size-fp32_size)/fp32_size*100:+.1f}%)")
        for bs in (1, 8, 32):
            f, s = lat_fp32[bs], lat_static[bs]
            print(f"  bs={bs:<3} p50: fp32={f['p50_ms']:.3f}ms static={s['p50_ms']:.3f}ms "
                  f"({(s['p50_ms']-f['p50_ms'])/f['p50_ms']*100:+.1f}%)  "
                  f"thr: fp32={f['thr_per_s']:.1f}/s static={s['thr_per_s']:.1f}/s")
        print(f"  MAE: fp32={mae_fp32:.6f}  static_int8={mae_static:.6f}  "
              f"({(mae_static-mae_fp32)/mae_fp32*100:+.3f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
