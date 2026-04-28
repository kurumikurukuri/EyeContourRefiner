"""
スイープ結果のポストプロセス

各スイープディレクトリに対して以下を実行する:
  1. アンサンブル作成 (fold .pt -> ensemble.pt)
  2. ONNX エクスポート
  3. 推論 (predictions.csv)
  4. 誤差分析 (error_analysis.csv, error_report.html)

使い方:
    python scripts/postprocess_sweeps.py --manifest experiments/EXP014_xxx/manifests/manifest_test.csv
    python scripts/postprocess_sweeps.py --models_dir models --manifest path/to/manifest.csv
    python scripts/postprocess_sweeps.py --manifest <csv> --skip_existing
    python scripts/postprocess_sweeps.py --manifest <csv> --filter sweep1
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def count_folds(model_dir: Path) -> int:
    """フォールドの数を数える。"""
    folds = sorted(model_dir.glob("eyeliner_detection_fold*.pt"))
    return len(folds)


def run(cmd: list[str], log_path: Path) -> bool:
    """コマンドを実行し、成功/失敗を返す。ログはファイルに追記する。"""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
        )
        f.write(f"[exit={result.returncode}]\n")
    return result.returncode == 0


def postprocess_dir(
    model_dir: Path,
    manifest: Path,
    skip_existing: bool,
    log_path: Path,
) -> dict:
    """1つのスイープディレクトリをポストプロセスし、結果を返す。"""
    k = count_folds(model_dir)
    if k == 0:
        logger.warning("  フォールドなし、スキップ: %s", model_dir)
        return {"status": "no_folds"}

    ensemble_pt  = model_dir / "eyeliner_detection_ensemble.pt"
    ensemble_onnx = model_dir / "eyeliner_detection_ensemble.onnx"
    predictions  = model_dir / "predictions.csv"
    error_csv    = model_dir / "error_analysis.csv"
    error_html   = model_dir / "error_report.html"

    result = {"k": k, "steps": {}}

    # ---- 1. アンサンブル作成 ----------------------------------------
    if skip_existing and ensemble_pt.exists():
        logger.info("  [1/4] アンサンブル: スキップ (既存)")
        result["steps"]["ensemble"] = "skipped"
    else:
        logger.info("  [1/4] アンサンブル作成 (k=%d)...", k)
        ok = run(
            [
                sys.executable, "scripts/train.py",
                "--mode", "ensemble",
                "--output_dir", str(model_dir),
                "--models_dir", str(model_dir),
                "--k", str(k),
            ],
            log_path,
        )
        result["steps"]["ensemble"] = "ok" if ok else "failed"
        if not ok:
            logger.error("  [1/4] アンサンブル失敗 -> 以降スキップ")
            return result

    # ---- 2. ONNX エクスポート ----------------------------------------
    if skip_existing and ensemble_onnx.exists():
        logger.info("  [2/4] ONNX: スキップ (既存)")
        result["steps"]["onnx"] = "skipped"
    else:
        logger.info("  [2/4] ONNX エクスポート...")
        ok = run(
            [
                sys.executable, "scripts/export_onnx.py",
                "--model", str(ensemble_pt),
                "--output", str(ensemble_onnx),
            ],
            log_path,
        )
        result["steps"]["onnx"] = "ok" if ok else "warn"
        if not ok:
            logger.warning("  [2/4] ONNX エクスポート失敗 (続行)")

    # ---- 3. 推論 -------------------------------------------------------
    if skip_existing and predictions.exists():
        logger.info("  [3/4] 推論: スキップ (既存)")
        result["steps"]["infer"] = "skipped"
    else:
        logger.info("  [3/4] 推論実行...")
        ok = run(
            [
                sys.executable, "scripts/infer_eyeliner_detection.py",
                "--model", str(ensemble_pt),
                "--manifest", str(manifest),
                "--output", str(predictions),
            ],
            log_path,
        )
        result["steps"]["infer"] = "ok" if ok else "failed"
        if not ok:
            logger.error("  [3/4] 推論失敗 -> 誤差分析スキップ")
            return result

    # ---- 4. 誤差分析 ---------------------------------------------------
    if skip_existing and error_csv.exists():
        logger.info("  [4/4] 誤差分析: スキップ (既存)")
        result["steps"]["errors"] = "skipped"
    else:
        logger.info("  [4/4] 誤差分析...")
        ok = run(
            [
                sys.executable, "scripts/analyze_errors.py",
                "--predictions", str(predictions),
                "--output", str(error_csv),
                "--report", str(error_html),
                "--top", "50",
            ],
            log_path,
        )
        result["steps"]["errors"] = "ok" if ok else "warn"
        if not ok:
            logger.warning("  [4/4] 誤差分析失敗")

    return result


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="スイープ結果のポストプロセス")
    parser.add_argument("--models_dir", type=Path, default=Path("models"),
                        help="スイープモデルが入った親ディレクトリ (default: models)")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="manifest.csv のパス（例: experiments/EXP014_xxx/manifests/manifest_test.csv）")
    parser.add_argument("--skip_existing", action="store_true",
                        help="既存の出力ファイルがあればそのステップをスキップする")
    parser.add_argument("--filter", type=str, default="",
                        help="ディレクトリ名にこの文字列を含むものだけ処理")
    args = parser.parse_args()

    models_dir: Path = args.models_dir
    manifest: Path   = args.manifest

    if not models_dir.exists():
        logger.error("models_dir が見つかりません: %s", models_dir)
        sys.exit(1)
    if not manifest.exists():
        logger.error("manifest が見つかりません: %s", manifest)
        sys.exit(1)

    # サブディレクトリ一覧（フォールドファイルがある or 処理予定）
    dirs = sorted(
        d for d in models_dir.iterdir()
        if d.is_dir() and (not args.filter or args.filter in d.name)
    )

    if not dirs:
        logger.warning("対象ディレクトリが見つかりません: %s", models_dir)
        sys.exit(0)

    log_path = models_dir / "postprocess.log"
    logger.info("ログ出力先: %s", log_path)
    logger.info("対象: %d ディレクトリ", len(dirs))

    summary: list[tuple[str, dict]] = []

    for i, d in enumerate(dirs, 1):
        logger.info("")
        logger.info("=== [%d/%d] %s ===", i, len(dirs), d.name)
        res = postprocess_dir(d, manifest, args.skip_existing, log_path)
        summary.append((d.name, res))

    # ---- サマリー出力 -----------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("  ポストプロセス サマリー")
    logger.info("=" * 60)

    ok_count  = 0
    err_count = 0
    skip_count = 0

    for name, res in summary:
        status = res.get("status", "")
        steps  = res.get("steps", {})

        if status == "no_folds":
            logger.info("  SKIP  %s  (フォールドなし)", name)
            skip_count += 1
            continue

        failed = [k for k, v in steps.items() if v == "failed"]
        if failed:
            logger.info("  FAIL  %s  (失敗: %s)", name, ", ".join(failed))
            err_count += 1
        else:
            logger.info("  OK    %s  %s", name, steps)
            ok_count += 1

    logger.info("")
    logger.info("  完了: %d / 失敗: %d / スキップ: %d (フォールドなし)",
                ok_count, err_count, skip_count)
    logger.info("  詳細ログ: %s", log_path)


if __name__ == "__main__":
    main()
