"""scripts/export_dual_onnx.py の golden regression test。

`export_dual_onnx.verify_against_pytorch` は PyTorch参照側・ONNX側の両方を
その場で同じ前処理で計算して比較するため、両者に共通する前処理の誤りは
検出できない自己整合的な検証に過ぎない。

本テストは実画像パッチと「一度だけ生成し固定した」期待t値
（tests/fixtures/dual_onnx_golden.json）を比較する。期待値は正規化込みの
正しいpreprocessingで一度だけ計算してコミットしてあるため、将来
preprocessing契約が壊れる回帰が起きれば必ず検出できる。

学習済みcheckpoint（models/ の eyeliner_detection_fold{1,2,3}.pt、
配布ONNXと同一の重み。.gitignore対象で非配布）と元画像が手元に無い環境（data/images/ は .gitignore対象）
では skip する。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

MODELS_DIR = ROOT / "models"  # 配布ONNXのエクスポート元 fold*.pt（非配布）
GOLDEN_PATH = Path(__file__).parent / "fixtures" / "dual_onnx_golden.json"
TOLERANCE = 1e-3


def _load_golden() -> list[dict]:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def golden_samples() -> list[dict]:
    samples = _load_golden()
    for sample in samples:
        if not (ROOT / sample["image_path"]).exists():
            pytest.skip(f"golden fixtureが参照する画像が見つかりません: {sample['image_path']}")
    return samples


@pytest.fixture(scope="module")
def exported_onnx_path(tmp_path_factory) -> Path:
    if not any(MODELS_DIR.glob("eyeliner_detection_fold*.pt")):
        pytest.skip(f"fold checkpointが見つかりません: {MODELS_DIR}")

    from export_dual_onnx import export  # noqa: E402

    out_path = tmp_path_factory.mktemp("onnx") / "dual_test.onnx"
    export(MODELS_DIR, out_path, pq_rescale_mode_b="eye_scale", fusion_mode="independent")
    return out_path


class TestExportDualOnnxGolden:
    """`scripts/export_dual_onnx.py` がexportするONNXモデルの契約テスト。"""

    def test_input_output_contract(self, exported_onnx_path):
        """入出力名・shapeがREADMEの契約通りであること。"""
        import onnxruntime as ort

        session = ort.InferenceSession(str(exported_onnx_path), providers=["CPUExecutionProvider"])
        input_names = [i.name for i in session.get_inputs()]
        output_names = [o.name for o in session.get_outputs()]
        assert input_names == [
            "patch_a", "patch_b", "is_upper", "px", "py", "qx", "qy", "eye_scale",
        ]
        assert output_names == ["t"]

    def test_golden_predictions_match(self, exported_onnx_path, golden_samples):
        """実画像パッチ + 生[0,1]入力contractでの推論値が固定fixtureと一致すること。

        DualRawTOnnxModel.forward内のImageNet正規化を一時的に
        コメントアウトして本テストが red になることを確認済み。
        """
        import onnxruntime as ort

        from eye_contour.training.dual_branch import build_branch_configs
        from eye_contour.data.patch_utils import extract_patch, compute_pq_rescale

        session = ort.InferenceSession(str(exported_onnx_path), providers=["CPUExecutionProvider"])
        cfg_a, cfg_b = build_branch_configs("eye_scale")
        image_cache: dict[str, np.ndarray] = {}

        max_diff = 0.0
        for sample in golden_samples:
            path = str(ROOT / sample["image_path"])
            if path not in image_cache:
                bgr = cv2.imread(path)
                image_cache[path] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = image_cache[path]

            px, py, qx, qy = sample["px"], sample["py"], sample["qx"], sample["qy"]
            rec = {"eye_scale": sample["eye_scale"]}

            pa_px, pa_py, pa_qx, pa_qy, _, _ = compute_pq_rescale(
                px, py, qx, qy, 0.0, rec, cfg_a.pq_rescale_mode, cfg_a.pq_rescale_factor,
            )
            patch_a = extract_patch(
                image, pa_px, pa_py, pa_qx, pa_qy, patch_size=cfg_a.patch_size,
                t_min=cfg_a.patch_t_min, t_max=cfg_a.patch_t_max, perp_scale=cfg_a.patch_perp_scale,
            )
            pb_px, pb_py, pb_qx, pb_qy, _, _ = compute_pq_rescale(
                px, py, qx, qy, 0.0, rec, cfg_b.pq_rescale_mode, cfg_b.pq_rescale_factor,
            )
            patch_b = extract_patch(
                image, pb_px, pb_py, pb_qx, pb_qy, patch_size=cfg_b.patch_size,
                t_min=cfg_b.patch_t_min, t_max=cfg_b.patch_t_max, perp_scale=cfg_b.patch_perp_scale,
            )

            feed = {
                "patch_a": patch_a.transpose(2, 0, 1)[None].astype(np.float32),
                "patch_b": patch_b.transpose(2, 0, 1)[None].astype(np.float32),
                "is_upper": np.array([[sample["is_upper"]]], dtype=np.float32),
                "px": np.array([[px]], dtype=np.float32),
                "py": np.array([[py]], dtype=np.float32),
                "qx": np.array([[qx]], dtype=np.float32),
                "qy": np.array([[qy]], dtype=np.float32),
                "eye_scale": np.array([[sample["eye_scale"]]], dtype=np.float32),
            }
            t_pred = float(session.run(None, feed)[0].reshape(-1)[0])
            diff = abs(t_pred - sample["t_expected"])
            max_diff = max(max_diff, diff)
            assert diff <= TOLERANCE, (
                f"idx={sample['idx']}: t_pred={t_pred:.6f} が golden 期待値 "
                f"{sample['t_expected']:.6f} と乖離（diff={diff:.6f} > {TOLERANCE}）。"
                "preprocessing契約（正規化）が変わっていないか確認してください。"
            )
