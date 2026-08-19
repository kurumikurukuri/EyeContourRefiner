"""tools/model_benchmark — CNN 推論性能ベンチマークツール。

レイテンシ・スループット・パラメータ数・MACs・ファイルサイズを測定し、
PyTorch と ONNX Runtime を同一条件（スレッド・入力・バッチ）で比較する。

使い方:
    # EXP031 の dual fold + アンサンブルを PyTorch/ONNX で比較
    python tools/model_benchmark/cli.py \\
        --models-dir experiments/EXP031_source_space_aug/outputs/SD0_baseline

    # 単一 .pt と既存 .onnx を測定
    python tools/model_benchmark/cli.py \\
        --model experiments/EXP031_source_space_aug/outputs/SD0_baseline/eyeliner_detection_fold1.pt \\
        --onnx models/eyeliner_detection.onnx

    # スレッド固定で再現性重視の比較
    python tools/model_benchmark/cli.py --models-dir .../SD0_baseline --threads 4 --batch-sizes 1 8

出力: コンソールテーブル + results/benchmark_YYYYMMDD_HHMMSS.{csv,json}
終了コード: 0=成功 / 1=ターゲット解決失敗・環境エラー / 2=実行時エラー
"""
from __future__ import annotations

import argparse
import logging
import os
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "PyTorch が見つかりません。`pip install torch` でインストールしてください。"
    ) from exc

from tools.model_benchmark import loader, metrics, onnx_utils, report, runner  # noqa: E402
from tools.model_benchmark.inputs import DEFAULT_PATCH_SIZE, build_inputs  # noqa: E402
from tools.model_benchmark.runner import RunResult  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class TorchTarget:
    """PyTorch で測定する対象。"""

    model_id: str
    source_path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI 引数をパース・検証する。"""
    parser = argparse.ArgumentParser(
        description="CNN 推論性能ベンチマーク（PyTorch vs ONNX Runtime）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=None,
                        help="単一チェックポイント (.pt) を直接指定")
    parser.add_argument("--models-dir", type=Path, default=None,
                        help="outputs の config ディレクトリ（eyeliner_detection_fold*.pt を自動発見）")
    parser.add_argument("--onnx", type=Path, default=None,
                        help="既存 .onnx を直接測定（.pt と併用可）")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32],
                        help="測定するバッチサイズ")
    parser.add_argument("--warmup", type=int, default=10,
                        help="非計測のウォームアップ回数")
    parser.add_argument("--iterations", type=int, default=100,
                        help="計測イテレーション数")
    parser.add_argument("--runtimes", nargs="+", choices=["pytorch", "onnx", "onnx_int8"],
                        default=["pytorch", "onnx"],
                        help="測定するランタイム（onnx_int8=動的量子化INT8、キャリブレーション不要）")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="測定デバイス")
    parser.add_argument("--threads", type=int, default=None,
                        help="torch/ORT のスレッド数を固定（再現性向上）")
    parser.add_argument("--skip-ensemble", action="store_true",
                        help="--models-dir 時に fold 個別のみ測定（アンサンブルは除外）")
    parser.add_argument("--seed", type=int, default=42, help="合成入力のシード")
    parser.add_argument("--patch-size", type=int, nargs=2, default=None, metavar=("H", "W"),
                        help=f"ダミー入力の空間サイズ (既定: 現行dual標準/EXP032と同一の "
                             f"{DEFAULT_PATCH_SIZE[0]}x{DEFAULT_PATCH_SIZE[1]}。"
                             f"旧世代モデル測定時など明示指定したい場合に使用)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="結果の出力ディレクトリ")
    args = parser.parse_args(argv)

    if args.model is None and args.models_dir is None and args.onnx is None:
        parser.error("--model / --models-dir / --onnx のいずれかを指定してください")
    if args.model is not None and args.models_dir is not None:
        parser.error("--model と --models-dir は併用できません")
    if args.warmup < 0:
        parser.error(f"--warmup は 0 以上にしてください: {args.warmup}")
    if args.iterations < 1:
        parser.error(f"--iterations は 1 以上にしてください: {args.iterations}")
    if not args.batch_sizes or any(b < 1 for b in args.batch_sizes):
        parser.error(f"--batch-sizes は 1 以上の値にしてください: {args.batch_sizes}")
    if args.threads is not None and args.threads < 1:
        parser.error(f"--threads は 1 以上にしてください: {args.threads}")
    if args.patch_size is not None and any(v < 1 for v in args.patch_size):
        parser.error(f"--patch-size は 1 以上にしてください: {args.patch_size}")
    return args


def resolve_device(name: str) -> torch.device:
    """デバイス引数を torch.device に解決する。"""
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA が利用できません。--device cpu / auto を指定してください")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _fold_suffix(path: Path) -> str:
    return path.stem.replace("eyeliner_detection_", "")


def resolve_torch_targets(args: argparse.Namespace) -> list[TorchTarget]:
    """--model / --models-dir を測定対象の列に解決する。"""
    if args.model is not None:
        if not args.model.is_file():
            raise FileNotFoundError(f"チェックポイントが見つかりません: {args.model}")
        return [TorchTarget(model_id=args.model.stem, source_path=args.model)]
    if args.models_dir is not None:
        fold_paths = loader.discover_fold_models(args.models_dir)
        return [
            TorchTarget(
                model_id=f"{args.models_dir.name}/{_fold_suffix(p)}",
                source_path=p,
            )
            for p in fold_paths
        ]
    return []


def _result_to_row(result: RunResult, artifact_path: Path, artifact_size: int) -> dict[str, Any]:
    """RunResult を flat な測定行に変換する。"""
    return {
        "runtime": result.runtime,
        "device": result.device,
        "batch_size": result.batch_size,
        "iterations": result.iterations,
        "latency_mean_ms": result.stats.mean_ms,
        "latency_p50_ms": result.stats.p50_ms,
        "latency_p95_ms": result.stats.p95_ms,
        "latency_p99_ms": result.stats.p99_ms,
        "latency_min_ms": result.stats.min_ms,
        "latency_std_ms": result.stats.std_ms,
        "latency_cv": result.stats.cv,
        "throughput_samples_per_s": result.throughput_samples_per_s,
        "artifact_path": str(artifact_path),
        "artifact_size_bytes": artifact_size,
    }


def benchmark_torch_model(
    model_id: str,
    model: torch.nn.Module,
    family: loader.ModelFamily,
    source_label: str,
    pt_size_bytes: int,
    cfg: runner.BenchmarkConfig,
    device: torch.device,
    runtimes: Sequence[str],
    threads: int | None,
    output_dir: Path,
) -> dict[str, Any]:
    """1モデルを PyTorch / ONNX Runtime で測定し、モデルレポートを返す。"""
    sample = build_inputs(model, family, 1, torch.device("cpu"), cfg.seed, cfg.patch_size)
    params = metrics.count_parameters(model)
    macs = metrics.estimate_macs(model, sample.torch_args)
    file_sizes: dict[str, int | None] = {"pytorch": pt_size_bytes, "onnx": None, "onnx_int8": None}
    rows: list[dict[str, Any]] = []
    notes = _notes_for(family)

    if "pytorch" in runtimes:
        for r in runner.run_torch_benchmark(model, family, device, cfg):
            rows.append(_result_to_row(r, Path(source_label), pt_size_bytes))

    if "onnx" in runtimes:
        try:
            onnx_path = output_dir / f"{_sanitize(model_id)}.onnx"
            onnx_utils.export_onnx(model, family, onnx_path, cfg.patch_size)
            session, ort_device = onnx_utils.create_session(
                onnx_path, prefer_cuda=device.type == "cuda", threads=threads
            )
            onnx_size = metrics.file_size_bytes(onnx_path)
            file_sizes["onnx"] = onnx_size
            for r in runner.run_onnx_benchmark(session, ort_device, cfg):
                rows.append(_result_to_row(r, onnx_path, onnx_size))
        except Exception as exc:
            # 例: AdaptiveAvgPool2d の出力サイズ構成によっては ONNX export 不可。
            # モデル構成の制約なので警告して PyTorch 測定のみで続行する。
            logger.warning("ONNX 測定をスキップしました (%s): %s", model_id, exc)
            reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            skip_note = f"ONNX 測定スキップ: {reason}"
            notes = f"{notes} / {skip_note}" if notes else skip_note

    if "onnx_int8" in runtimes:
        try:
            fp32_path = output_dir / f"{_sanitize(model_id)}.onnx"
            if not fp32_path.exists():
                onnx_utils.export_onnx(model, family, fp32_path, cfg.patch_size)
            int8_path = output_dir / f"{_sanitize(model_id)}_int8.onnx"
            onnx_utils.quantize_dynamic_onnx(fp32_path, int8_path)
            session, ort_device = onnx_utils.create_session(
                int8_path, prefer_cuda=device.type == "cuda", threads=threads
            )
            int8_size = metrics.file_size_bytes(int8_path)
            file_sizes["onnx_int8"] = int8_size
            for r in runner.run_onnx_benchmark(session, ort_device, cfg, runtime_label="onnx_int8"):
                rows.append(_result_to_row(r, int8_path, int8_size))
        except Exception as exc:
            logger.warning("ONNX INT8量子化測定をスキップしました (%s): %s", model_id, exc)
            reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            skip_note = f"ONNX INT8量子化スキップ: {reason}"
            notes = f"{notes} / {skip_note}" if notes else skip_note

    return {
        "model_id": model_id,
        "family": family,
        "source_path": source_label,
        "params": params,
        "macs": macs,
        "macs_estimator": "thop" if macs is not None else None,
        "notes": notes,
        "file_size_bytes": file_sizes,
        "results": rows,
    }


def _notes_for(family: loader.ModelFamily) -> str:
    if family == "dual":
        return (
            "dual は正規化空間平均の fusion で測定。"
            "実運用（fold_ensemble_raw_t）の生 t 合成と差分はスカラー演算のみ"
        )
    return ""


def _sanitize(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name)


def benchmark_standalone_onnx(
    model_id: str,
    onnx_path: Path,
    cfg: runner.BenchmarkConfig,
    device: torch.device,
    threads: int | None,
) -> dict[str, Any]:
    """既存 .onnx を単体で測定する（params/Macs は未知）。"""
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX ファイルが見つかりません: {onnx_path}")
    session, ort_device = onnx_utils.create_session(
        onnx_path, prefer_cuda=device.type == "cuda", threads=threads
    )
    onnx_size = metrics.file_size_bytes(onnx_path)
    rows = [
        _result_to_row(r, onnx_path, onnx_size)
        for r in runner.run_onnx_benchmark(session, ort_device, cfg)
    ]
    return {
        "model_id": model_id,
        "family": "-",
        "source_path": str(onnx_path),
        "params": None,
        "macs": None,
        "macs_estimator": None,
        "notes": "既存 ONNX の単体測定（params/macs は未知）",
        "file_size_bytes": {"pytorch": None, "onnx": onnx_size},
        "results": rows,
    }


def build_environment(device: torch.device, threads: int | None, cfg: runner.BenchmarkConfig) -> dict[str, Any]:
    """結果の解釈可能性のための環境メタデータを構築する。"""
    env: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor()
        ),
        "cpu_count": os.cpu_count(),
        "threads": threads,
        "seed": cfg.seed,
        "patch_size": list(cfg.patch_size),
        "batch_sizes": list(cfg.batch_sizes),
        "warmup": cfg.warmup,
        "iterations": cfg.iterations,
    }
    try:
        import onnxruntime

        env["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        env["onnxruntime"] = None
    return env


def main(argv: Sequence[str] | None = None) -> int:
    """エントリポイント。"""
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    if args.threads is not None:
        torch.set_num_threads(args.threads)
        logger.info("torch スレッド数を %d に固定しました", args.threads)
    device = resolve_device(args.device)
    logger.info("デバイス: %s", device)
    cfg = runner.BenchmarkConfig(
        batch_sizes=tuple(args.batch_sizes),
        warmup=args.warmup,
        iterations=args.iterations,
        seed=args.seed,
        patch_size=tuple(args.patch_size) if args.patch_size else DEFAULT_PATCH_SIZE,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    targets = resolve_torch_targets(args)
    loaded = [(t, loader.load_benchmark_model(t.source_path)) for t in targets]
    model_reports: list[dict[str, Any]] = []
    for t, lc in loaded:
        logger.info("測定中: %s (kind=%s)", t.model_id, lc.kind)
        model_reports.append(
            benchmark_torch_model(
                t.model_id, lc.model, loader.family_of(lc.kind),
                str(t.source_path), metrics.file_size_bytes(t.source_path),
                cfg, device, args.runtimes, args.threads, args.output_dir,
            )
        )

    if args.models_dir is not None and not args.skip_ensemble and len(loaded) >= 2:
        ensemble = loader.build_ensemble_module(
            [lc.model for _, lc in loaded],
            [lc.kind for _, lc in loaded],
            [lc.backbone for _, lc in loaded],
        )
        pt_total = sum(metrics.file_size_bytes(t.source_path) for t, _ in loaded)
        model_id = f"{args.models_dir.name}/ensemble"
        logger.info("測定中: %s (%d folds)", model_id, len(loaded))
        model_reports.append(
            benchmark_torch_model(
                model_id, ensemble, loader.family_of(loaded[0][1].kind),
                str(args.models_dir), pt_total,
                cfg, device, args.runtimes, args.threads, args.output_dir,
            )
        )

    if args.onnx is not None:
        model_reports.append(
            benchmark_standalone_onnx(args.onnx.stem, args.onnx, cfg, device, args.threads)
        )

    if not model_reports:
        raise FileNotFoundError("測定対象が解決できませんでした")

    csv_path = args.output_dir / f"benchmark_{stamp}.csv"
    json_path = args.output_dir / f"benchmark_{stamp}.json"
    report.write_csv(report.rows_from_models(model_reports), csv_path)
    report.write_json(
        {"environment": build_environment(device, args.threads, cfg), "models": model_reports},
        json_path,
    )
    print(report.format_console_table(model_reports))
    logger.info("CSV : %s", csv_path)
    logger.info("JSON: %s", json_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ImportError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        logging.getLogger(__name__).exception("ベンチマーク実行中にエラーが発生しました: %s", exc)
        sys.exit(2)
