"""
dual-branch モデル（fold群）を ONNX にエクスポートする。

生t空間での正しい枝融合（eye_contour.training.dual_branch.fold_ensemble_raw_t）を
グラフに焼き込むため、単純に (pred_a+pred_b)/2 を平均する
（tools/model_benchmark の DualBenchmarkEnsemble、eye_contour.models.dual_ensemble の
DualEnsembleModel=[非推奨]）とは異なり、本番推論と数値的に一致する。

branch B（既定でeye_scale座標系）のt'→生t変換にpq_dist/pq_prime_distが必要なため、
入力はパッチ画像だけでなく px/py/qx/qy（元のPQ座標）と eye_scale も含む。

使い方:
    python scripts/export_dual_onnx.py \\
        --models-dir experiments/EXP031_source_space_aug/outputs/SD0_baseline \\
        --output models/eye_contour_refiner.onnx

出力後、--verify-manifest を指定すると実データで
「エクスポート前PyTorch」対「エクスポート後ONNX」の予測値を突き合わせ、数値一致を確認する。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def export(
    models_dir: Path,
    output_path: Path,
    pq_rescale_mode_b: str,
    fusion_mode: str,
) -> None:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch が見つかりません。`pip install torch` でインストールしてください。"
        ) from exc
    try:
        import onnx  # noqa: F401  エクスポート前に可用性だけ確認する（使用は finalize_onnx）
    except ImportError as exc:
        raise ImportError(
            "onnx が見つかりません。`pip install onnx` でインストールしてください。"
        ) from exc

    from eye_contour.models.dual_ensemble import DualRawTOnnxModel, load_dual_fold_models
    from eye_contour.training.dual_branch import build_branch_configs

    logger.info("dual fold モデルを読み込み中: %s", models_dir)
    models = load_dual_fold_models(models_dir, fusion_mode)
    for m in models:
        m.eval()
    logger.info("読み込み完了: %d folds", len(models))

    cfg_a, cfg_b = build_branch_configs(pq_rescale_mode_b)
    if cfg_a.patch_size != cfg_b.patch_size:
        raise ValueError(
            f"branch A/B のpatch_sizeが一致しません: {cfg_a.patch_size} vs {cfg_b.patch_size}"
        )
    h, w = cfg_a.patch_size

    wrapper = DualRawTOnnxModel(
        models, fusion_mode, cfg_a, cfg_b, pq_rescale_factor_b=cfg_b.pq_rescale_factor,
    )
    wrapper.eval()

    dummy = (
        torch.zeros(1, 3, h, w),   # patch_a
        torch.zeros(1, 3, h, w),   # patch_b
        torch.zeros(1, 1),          # is_upper
        torch.zeros(1, 1),          # px
        torch.zeros(1, 1),          # py
        torch.full((1, 1), 10.0),   # qx（px=qx=qy=0だとpq_dist=0でwhereガードに落ちるため非退化値にする）
        torch.zeros(1, 1),          # qy
        torch.full((1, 1), 20.0),   # eye_scale
    )
    input_names = ["patch_a", "patch_b", "is_upper", "px", "py", "qx", "qy", "eye_scale"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=input_names,
        output_names=["t"],
        dynamic_axes={name: {0: "batch"} for name in [*input_names, "t"]},
        opset_version=17,
        do_constant_folding=True,
    )
    finalize_onnx(output_path)
    logger.info("ONNX モデルをエクスポートしました: %s", output_path)

    total_params = sum(p.numel() for p in wrapper.parameters())
    logger.info("パラメータ数: %d (folds=%d)", total_params, len(models))


def finalize_onnx(output_path: Path) -> None:
    """torch.onnx.export 直後の .onnx を配布用の形に整えて上書き保存する。

    torch 2.9+ の既定エクスポータは重みを外部データファイル（<output>.data）に
    分離する。配布は単一 .onnx 前提（package-data も *.onnx のみ）のため
    initializer をインライン化する。あわせて、エクスポータが埋め込むスタックトレース
    （エクスポート環境の絶対パスを含む）等のデバッグ情報を除去する。
    """
    import onnx

    from eye_contour.models.onnx_sanitize import strip_debug_metadata

    data_file = output_path.with_name(output_path.name + ".data")
    onnx_model = onnx.load(str(output_path), load_external_data=True)
    if data_file.exists():
        for tensor in onnx_model.graph.initializer:
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                tensor.ClearField("data_location")
                tensor.ClearField("external_data")
    onnx_model = strip_debug_metadata(onnx_model)
    onnx.save(onnx_model, str(output_path))
    if data_file.exists():
        data_file.unlink()
        logger.info("外部データ %s を .onnx にインライン化しました", data_file.name)
    logger.info("デバッグ情報（スタックトレース等）を除去しました")
    onnx.checker.check_model(onnx_model)


def verify_against_pytorch(
    onnx_path: Path,
    models_dir: Path,
    manifest_path: Path,
    pq_rescale_mode_b: str,
    fusion_mode: str,
    n_samples: int,
) -> None:
    """実データでexport前(PyTorch fold_ensemble_raw_t)とexport後(ONNX)の予測が一致するか検証する。"""
    import numpy as np
    import onnxruntime as ort
    import torch

    from eye_contour.data.dataset import load_manifest
    from eye_contour.data.weights import compute_eye_meta
    from eye_contour.models.dual_ensemble import load_dual_fold_models
    from eye_contour.training.dual_branch import build_branch_configs, fold_ensemble_raw_t
    from eye_contour.data.patch_utils import extract_patch, compute_pq_rescale, IMAGENET_MEAN, IMAGENET_STD

    norm_mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    norm_std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    records = load_manifest(manifest_path)
    compute_eye_meta(records)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(records), size=min(n_samples, len(records)), replace=False)
    sample = [records[i] for i in idx]

    import cv2
    cfg_a, cfg_b = build_branch_configs(pq_rescale_mode_b)
    models = load_dual_fold_models(models_dir, fusion_mode)
    for m in models:
        m.eval()

    pt_preds: list[float] = []
    onnx_preds: list[float] = []
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    for rec in sample:
        bgr = cv2.imread(rec["image_path"])
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        px, py = float(rec["px"]), float(rec["py"])
        qx, qy = float(rec["qx"]), float(rec["qy"])
        is_upper_val = 1.0 if rec.get("sub_key", "") == "upper" else 0.0

        pa_px, pa_py, pa_qx, pa_qy, _, _ = compute_pq_rescale(px, py, qx, qy, 0.0, rec, cfg_a.pq_rescale_mode, cfg_a.pq_rescale_factor)
        patch_a = extract_patch(image, pa_px, pa_py, pa_qx, pa_qy, patch_size=cfg_a.patch_size, t_min=cfg_a.patch_t_min, t_max=cfg_a.patch_t_max, perp_scale=cfg_a.patch_perp_scale)
        pb_px, pb_py, pb_qx, pb_qy, _, _ = compute_pq_rescale(px, py, qx, qy, 0.0, rec, cfg_b.pq_rescale_mode, cfg_b.pq_rescale_factor)
        patch_b = extract_patch(image, pb_px, pb_py, pb_qx, pb_qy, patch_size=cfg_b.patch_size, t_min=cfg_b.patch_t_min, t_max=cfg_b.patch_t_max, perp_scale=cfg_b.patch_perp_scale)

        x_a_raw = torch.from_numpy(patch_a.transpose(2, 0, 1)).float().unsqueeze(0)
        x_b_raw = torch.from_numpy(patch_b.transpose(2, 0, 1)).float().unsqueeze(0)
        is_upper_t = torch.tensor([[is_upper_val]], dtype=torch.float32)
        pq_dist = float(np.sqrt((qx - px) ** 2 + (qy - py) ** 2))
        eye_scale = float(rec.get("eye_scale", pq_dist))
        pq_prime_dist = max(eye_scale * cfg_b.pq_rescale_factor, 1.0)

        # PyTorch参照側は学習時と同じ正規化を明示的に適用する
        # （ONNXグラフは DualRawTOnnxModel.forward 内部で同じ正規化を行うため、
        # ここで揃えないと外側から見た「生[0,1]パッチ入力」という契約の下での
        # 数値一致検証にならない）。
        x_a = (x_a_raw - norm_mean) / norm_std
        x_b = (x_b_raw - norm_mean) / norm_std

        with torch.no_grad():
            t_pt = fold_ensemble_raw_t(
                models, x_a, x_b, is_upper_t, fusion_mode, cfg_a, cfg_b,
                torch.tensor([[pq_dist]]), torch.tensor([[pq_prime_dist]]),
            ).item()
        pt_preds.append(t_pt)

        feed = {
            "patch_a": x_a_raw.numpy(), "patch_b": x_b_raw.numpy(), "is_upper": is_upper_t.numpy(),
            "px": np.array([[px]], dtype=np.float32), "py": np.array([[py]], dtype=np.float32),
            "qx": np.array([[qx]], dtype=np.float32), "qy": np.array([[qy]], dtype=np.float32),
            "eye_scale": np.array([[eye_scale]], dtype=np.float32),
        }
        t_onnx = float(session.run(None, feed)[0].reshape(-1)[0])
        onnx_preds.append(t_onnx)

    pt_arr = np.array(pt_preds)
    onnx_arr = np.array(onnx_preds)
    max_diff = float(np.max(np.abs(pt_arr - onnx_arr)))
    mean_diff = float(np.mean(np.abs(pt_arr - onnx_arr)))
    logger.info("検証(n=%d): PyTorch vs ONNX 予測差分 mean=%.8f max=%.8f", len(sample), mean_diff, max_diff)
    if max_diff > 1e-4:
        raise RuntimeError(
            f"PyTorchとONNXの予測値が乖離しています（max_diff={max_diff:.6f} > 1e-4）。"
            "エクスポート実装を確認してください。"
        )
    logger.info("検証OK: PyTorchとONNXの予測値は数値的に一致しています。")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="dual-branch モデルを ONNX にエクスポート（生t空間で正しく融合）")
    parser.add_argument("--models-dir", required=True, type=Path, help="fold*.ptが入ったディレクトリ")
    parser.add_argument("--output", default=Path("models/eye_contour_refiner.onnx"), type=Path)
    parser.add_argument("--pq-rescale-mode-b", default="eye_scale", choices=["eye_scale", "eye_height"])
    parser.add_argument("--fusion-mode", default="independent", choices=["independent", "attention"])
    parser.add_argument("--verify-manifest", type=Path, default=None,
                        help="指定するとこのmanifestから実データをサンプリングしPyTorchとの数値一致を検証する")
    parser.add_argument("--verify-samples", type=int, default=30)
    args = parser.parse_args()

    export(args.models_dir, args.output, args.pq_rescale_mode_b, args.fusion_mode)

    if args.verify_manifest is not None:
        verify_against_pytorch(
            args.output, args.models_dir, args.verify_manifest,
            args.pq_rescale_mode_b, args.fusion_mode, args.verify_samples,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
