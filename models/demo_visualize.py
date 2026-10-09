"""画像を読み込み、MediaPipe Face Meshで目のランドマークを検出し、
eyeliner_detection_dual.onnxで各点を精密化して、目の輪郭（アイライン）を図示する。

**重要な注意**: 本番の学習データ生成（`annotator/preprocessor.py`）は、
P,Qペアの正確な組み合わせを外部パッケージ`face_makeup.constants.EYELINER_CORRECTION_MAP`
から取得している（本リポジトリには含まれない、face_makeup側で管理）。本スクリプトは
そのマップにアクセスできないため、代わりにMediaPipe公式のFACEMESH_LEFT_EYE /
FACEMESH_RIGHT_EYE（公開されている標準の目輪郭16点ループ）を使い、隣接点ペアを
P,Qとして自前で構成している。デモとして目のふち検出パイプライン全体の動作は正しく
示せるが、学習時に実際使われた厳密なP,Qペアとは一致しない可能性がある
（`point_index`はモデルの入力ではないため、モデル自体の推論結果には影響しない——
どの2点をP,Qとして渡すかだけの違い）。

依存: このデモのみ mediapipe が追加で必要（`pip install -e .[demo]` または
`pip install "mediapipe>=0.10.21,<0.10.30"`）。coreのONNX推論自体はmediapipe不要。

使い方:
    python models/demo_visualize.py --image path/to/face.jpg --output result.jpg
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from eyeliner.data.patch_utils import compute_eye_scale  # noqa: E402
from infer_example import DEFAULT_MODEL_PATH, predict_point  # noqa: E402

# MediaPipe FaceMesh 公式の目輪郭16点ループ（FACEMESH_LEFT_EYE/FACEMESH_RIGHT_EYE
# の接続グラフをトレースし、各アークを「外側(外眼角)→内側(内眼角)」の順に並べたもの）。
# "eyeliner_left"/"eyeliner_right" は被写体本人から見た左右（mediapipeの命名規則と同じ）
# と仮定している。
EYE_LANDMARKS = {
    "eyeliner_right": {  # 被写体の右目（画像上では向かって左側に写ることが多い）
        "outer_corner": 33,
        "inner_corner": 133,
        "lower_arc": [33, 7, 163, 144, 145, 153, 154, 155, 133],   # 外側→内側
        "upper_arc": [33, 246, 161, 160, 159, 158, 157, 173, 133],  # 外側→内側
    },
    "eyeliner_left": {  # 被写体の左目
        "outer_corner": 263,
        "inner_corner": 362,
        "lower_arc": [263, 249, 390, 373, 374, 380, 381, 382, 362],
        "upper_arc": [263, 466, 388, 387, 386, 385, 384, 398, 362],
    },
}
NOSE_TIP_INDEX = 4  # MediaPipe FaceMesh の鼻先ランドマーク（参考: 顔中心の目安）


def detect_face_landmarks(image_rgb: np.ndarray, max_num_faces: int = 1) -> list[np.ndarray]:
    """MediaPipe Face Meshで顔ランドマークを検出する。

    Returns: 検出した顔ごとの (478, 2) ピクセル座標配列のリスト
             （refine_landmarks=Trueなので虹彩10点込みで478点）。
    """
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "mediapipe が見つかりません。`pip install -e .[demo]` でインストールしてください。"
        ) from exc

    h, w = image_rgb.shape[:2]
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=max_num_faces, refine_landmarks=True,
    ) as face_mesh:
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return []
    return [
        np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark], dtype=np.float32)
        for face_landmarks in results.multi_face_landmarks
    ]


def refine_eye_contour(
    session: ort.InferenceSession,
    image_rgb: np.ndarray,
    landmarks: np.ndarray,
    part_name: str,
) -> tuple[dict[str, list[tuple[float, float]]], int, float]:
    """1つの目のupper/lowerアークをモデルで精密化する。

    Returns: (refined, n_points, inference_seconds)
        refined: {"upper": [(rx,ry), ...], "lower": [(rx,ry), ...]}（各アーク8点、
        アークに沿った隣接ランドマークペアをP,Qとして精密化した座標）。
        n_points: 推論した点数。inference_seconds: predict_point呼び出し合計時間（秒）。
    """
    spec = EYE_LANDMARKS[part_name]
    all_idx = set(spec["lower_arc"]) | set(spec["upper_arc"])
    eye_scale = compute_eye_scale([tuple(landmarks[i]) for i in all_idx])

    refined: dict[str, list[tuple[float, float]]] = {"upper": [], "lower": []}
    n_points = 0
    inference_seconds = 0.0
    for sub_key, is_upper in (("upper", True), ("lower", False)):
        arc = spec[f"{sub_key}_arc"]
        for i in range(len(arc) - 1):
            p_idx, q_idx = arc[i], arc[i + 1]
            px, py = (float(v) for v in landmarks[p_idx])
            qx, qy = (float(v) for v in landmarks[q_idx])
            t0 = time.perf_counter()
            _, rx, ry = predict_point(session, image_rgb, px, py, qx, qy, is_upper, eye_scale)
            inference_seconds += time.perf_counter() - t0
            n_points += 1
            refined[sub_key].append((rx, ry))
    return refined, n_points, inference_seconds


def draw_result(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    refined_by_eye: dict[str, dict[str, list[tuple[float, float]]]],
) -> np.ndarray:
    """元のMediaPipe輪郭（グレー）と精密化後のアイライン（色付き）を重ねて描画する。"""
    out = image_bgr.copy()
    colors = {"eyeliner_left": (0, 200, 255), "eyeliner_right": (255, 150, 0)}  # BGR

    for part_name, refined in refined_by_eye.items():
        spec = EYE_LANDMARKS[part_name]
        # 元のMediaPipeランドマーク輪郭（比較用、細いグレー線）
        for arc_key in ("upper_arc", "lower_arc"):
            pts = landmarks[spec[arc_key]].astype(int)
            cv2.polylines(out, [pts], isClosed=False, color=(160, 160, 160), thickness=1)

        # 精密化後のアイライン（太めのカラー線+点）
        color = colors[part_name]
        for sub_key in ("upper", "lower"):
            pts = np.array(refined[sub_key], dtype=np.int32)
            cv2.polylines(out, [pts], isClosed=False, color=color, thickness=2)
            for x, y in pts:
                cv2.circle(out, (int(x), int(y)), 2, color, -1)

    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description="MediaPipe + eyeliner_detection_dual.onnx デモ")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("eyeliner_result.jpg"))
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--max-faces", type=int, default=1)
    args = ap.parse_args()

    t_start = time.perf_counter()

    t0 = time.perf_counter()
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise FileNotFoundError(f"画像を読み込めません: {args.image}")
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t_load_image = time.perf_counter() - t0

    t0 = time.perf_counter()
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    t_session_init = time.perf_counter() - t0

    t0 = time.perf_counter()
    faces = detect_face_landmarks(image_rgb, max_num_faces=args.max_faces)
    t_detect = time.perf_counter() - t0
    if not faces:
        print("顔が検出されませんでした。", file=sys.stderr)
        return 1
    print(f"{len(faces)} 人の顔を検出しました。")

    out = bgr
    t_refine_total = 0.0
    t_inference_total = 0.0
    n_points_total = 0
    for face_i, landmarks in enumerate(faces):
        t0 = time.perf_counter()
        refined_by_eye: dict[str, dict[str, list[tuple[float, float]]]] = {}
        for part_name in EYE_LANDMARKS:
            refined, n_points, inference_seconds = refine_eye_contour(
                session, image_rgb, landmarks, part_name,
            )
            refined_by_eye[part_name] = refined
            n_points_total += n_points
            t_inference_total += inference_seconds
        t_refine_total += time.perf_counter() - t0
        out = draw_result(out, landmarks, refined_by_eye)
        print(f"  顔 {face_i}: 両目のアイライン輪郭を精密化しました "
              f"(upper 8点 + lower 8点 x 2目)")

    t0 = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), out)
    t_draw_save = time.perf_counter() - t0

    t_total = time.perf_counter() - t_start

    print(f"保存しました: {args.output}"
          "（グレー細線=MediaPipe生ランドマーク、カラー太線=モデル精密化後）")

    print(f"\n{'='*56}\n  処理時間内訳\n{'='*56}")
    print(f"  画像読み込み                : {t_load_image*1000:8.1f} ms")
    print(f"  ONNXセッション初期化         : {t_session_init*1000:8.1f} ms")
    print(f"  MediaPipe顔検出（初期化込み）: {t_detect*1000:8.1f} ms")
    print(f"  輪郭精密化（描画準備込み）    : {t_refine_total*1000:8.1f} ms"
          f"  内、ONNX推論のみ: {t_inference_total*1000:8.1f} ms"
          f"（{n_points_total}点、平均{t_inference_total/max(n_points_total,1)*1000:.2f} ms/点）")
    print(f"  描画・保存                  : {t_draw_save*1000:8.1f} ms")
    print(f"  {'-'*52}")
    print(f"  合計                        : {t_total*1000:8.1f} ms")
    print(f"{'='*56}")
    print("  ※ MediaPipe顔検出はモデル初期化（TFLite delegate構築等）を含み初回実行が"
          "特に重い。ONNX推論のみの純粋なレイテンシは「ONNX推論のみ」の行を参照。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
