"""EXP037 処理時間計測: 両目（28 点）がアノテーション済みの顔だけを対象に、ルールベース（face_makeup）と
ML（eye_contour_refiner.onnx）の補正時間を同じ入力で測る。

計測区間（両方式で共通）:
    入口 = 読み込み済み RGB 画像 + 28 点分の P,Q（メモリ上）
    出口 = 28 点の補正点 R
    ルール（2 条件。入力は本番と同じ整数座標の P,Q。毎回新しい corrector を使い前処理キャッシュを無効化）:
      rule_per_eye: 目ごとに preprocess（画像全体の bilateralFilter 等）+ find_edge × 14 を 2 回（主結果）。
                    face_makeup 本番は左右の EyelinerPass がそれぞれ corrector を持ち、前処理キャッシュが
                    corrector ごとなので 1 顔で前処理が 2 回走る。これと同じ構成
      rule        : preprocess 1 回 + find_edge × 28（前処理を左右で共有した場合。参考）
    ML    : eye_scale × 2 眼 + パッチ切り出し 56 枚 + session.run（28 点 1 バッチ）+ R 変換
    区間外: 画像の読み込み・デコード、MediaPipe、InferenceSession 生成（初回コストとして別記録）

ウォームアップ後、全顔を 1 周する計測を --repeats 回繰り返す（他プロセスの一時的な負荷を全顔に分散させる）。
実行順は顔と周回ごとに 3 方式を巡回させ、各方式が各順位に均等に来るようにする。

出力:
    outputs/timing_{threads}.csv    1 計測 1 行（顔・方式・繰り返し・合計・内訳）
    outputs/timing_{threads}.json   集計（顔ごとの中央値 → 顔間の中央値 / p95 / 平均）と実行環境

使い方:
    python experiments/EXP037_rule_vs_ml/scr/timing_exp037.py                    # 既定スレッド（主結果）
    python experiments/EXP037_rule_vs_ml/scr/timing_exp037.py --threads single   # 単一スレッド（参考）
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
MANIFEST_PATH = EXP_DIR / "manifests" / "manifest_test.csv"
OUTPUT_DIR = EXP_DIR / "outputs"
MODEL_PATH = ROOT / "models" / "eye_contour_refiner.onnx"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from infer_exp037 import load_manifest, load_rgb
from refiners_exp037 import eye_scale_of, refine_ml, refine_rule, to_production_coords

FACE_KEY = ["image_path", "face_index"]
POINTS_PER_FACE = 28
SESSION_LOAD_REPEATS = 3


def select_full_faces(manifest: pd.DataFrame) -> list[pd.DataFrame]:
    """両目 28 点が揃った顔を、左目 → 右目・upper → lower・point_index 順に並べて返す。"""
    faces = []
    for _, face in manifest.groupby(FACE_KEY, sort=True):
        if len(face) == POINTS_PER_FACE:
            faces.append(face.sort_values(["part_name", "sub_key", "point_index"], ascending=[True, False, True]))
    return faces


def make_session(model_path: Path, single_thread: bool) -> ort.InferenceSession:
    options = ort.SessionOptions()
    if single_thread:
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


def time_session_load(model_path: Path, single_thread: bool) -> list[float]:
    durations = []
    for _ in range(SESSION_LOAD_REPEATS):
        start = time.perf_counter()
        make_session(model_path, single_thread)
        durations.append(time.perf_counter() - start)
    return durations


def run_rule(image: np.ndarray, pq: np.ndarray) -> dict:
    result = refine_rule(image, pq)
    return {"total_s": result.preprocess_s + result.search_s, "part1_s": result.preprocess_s, "part2_s": result.search_s}


def run_rule_per_eye(image: np.ndarray, pq: np.ndarray) -> dict:
    """本番と同じく目ごとに別の corrector で補正する（前処理が 2 回走る）。内訳は 2 眼の合計。"""
    left = refine_rule(image, pq[: POINTS_PER_FACE // 2])
    right = refine_rule(image, pq[POINTS_PER_FACE // 2 :])
    preprocess_s = left.preprocess_s + right.preprocess_s
    search_s = left.search_s + right.search_s
    return {"total_s": preprocess_s + search_s, "part1_s": preprocess_s, "part2_s": search_s}


def run_ml(session: ort.InferenceSession, image: np.ndarray, pq: np.ndarray, is_upper: np.ndarray) -> dict:
    start = time.perf_counter()
    left, right = pq[: POINTS_PER_FACE // 2], pq[POINTS_PER_FACE // 2 :]
    eye_scale = np.repeat([eye_scale_of(left), eye_scale_of(right)], POINTS_PER_FACE // 2)
    result = refine_ml(session, image, pq, is_upper, eye_scale)
    total = time.perf_counter() - start
    return {"total_s": total, "part1_s": result.patch_s, "part2_s": result.infer_s}


def measure(faces: list[pd.DataFrame], session: ort.InferenceSession, warmup: int, repeats: int) -> pd.DataFrame:
    inputs = []
    for face in faces:
        pq = face[["px", "py", "qx", "qy"]].to_numpy(dtype=np.float64)
        inputs.append({
            "image_path": face["image_path"].iloc[0],
            "face_index": int(face["face_index"].iloc[0]),
            "image": load_rgb(ROOT / face["image_path"].iloc[0]),
            "pq": pq,
            # ルールベースには本番と同じ整数座標を渡す（丸めは本番でも両方式の外側で行われるため区間外）
            "pq_production": to_production_coords(pq),
            "is_upper": (face["sub_key"] == "upper").to_numpy(),
        })

    runners = {
        "rule_per_eye": lambda face: run_rule_per_eye(face["image"], face["pq_production"]),
        "rule": lambda face: run_rule(face["image"], face["pq_production"]),
        "ml": lambda face: run_ml(session, face["image"], face["pq"], face["is_upper"]),
    }
    methods = list(runners)
    for face in inputs[:warmup]:
        for runner in runners.values():
            runner(face)

    # 「全顔を 1 周」を repeats 回繰り返す（顔ごとに連続計測しない）。他プロセスの一時的な負荷が
    # 特定の顔に集中せず全顔に分散するので、顔ごとの中央値が負荷の偏りに左右されにくい。
    rows = []
    for rep in range(repeats):
        for position, face in enumerate(inputs):
            shift = (rep + position) % len(methods)
            for method in methods[shift:] + methods[:shift]:
                rows.append({
                    "image_path": face["image_path"], "face_index": face["face_index"],
                    "height": face["image"].shape[0], "width": face["image"].shape[1],
                    "method": method, "rep": rep, **runners[method](face),
                })
    return pd.DataFrame(rows)


def summarize(timings: pd.DataFrame) -> dict:
    """顔ごとに繰り返しの中央値を取り、それを顔間で集計する（単位 ms）。"""
    per_face = timings.groupby(["method", *FACE_KEY])[["total_s", "part1_s", "part2_s"]].median() * 1000
    summary = {}
    for method, group in per_face.groupby(level="method"):
        summary[method] = {
            "total_ms": {
                "median": float(group["total_s"].median()),
                "p95": float(group["total_s"].quantile(0.95)),
                "mean": float(group["total_s"].mean()),
                "min": float(group["total_s"].min()),
                "max": float(group["total_s"].max()),
            },
            "part1_ms_median": float(group["part1_s"].median()),
            "part2_ms_median": float(group["part2_s"].median()),
        }
    ml_median = summary["ml"]["total_ms"]["median"]
    summary["ml_over_rule_per_eye_ratio_median"] = ml_median / summary["rule_per_eye"]["total_ms"]["median"]
    summary["ml_over_rule_ratio_median"] = ml_median / summary["rule"]["total_ms"]["median"]
    return summary


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError as error:
            print(f"[warn] CPU 名を取得できません: {error}")
    return platform.processor()


def face_makeup_commit() -> str | None:
    """実際に導入されている face_makeup の commit（pip の direct_url.json）。git 以外から入れた場合は None。"""
    from importlib.metadata import distribution

    direct_url = distribution("face_makeup").read_text("direct_url.json")
    if direct_url is None:
        return None
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id")


def environment(single_thread: bool) -> dict:
    from importlib.metadata import version

    return {
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "onnxruntime": ort.__version__,
        "face_makeup": version("face_makeup"),
        "face_makeup_commit": face_makeup_commit(),
        "threads": "single" if single_thread else "default",
        "cv2_num_threads": cv2.getNumThreads(),
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description="EXP037: ルールベースと ONNX の補正時間を計測する（両目 28 点の顔のみ）")
    ap.add_argument("--threads", choices=["default", "single"], default="default")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    args = ap.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        ap.error("--warmup は 0 以上、--repeats は 1 以上")
    if not args.model.exists():
        raise FileNotFoundError(f"ONNX が見つかりません: {args.model}")

    single_thread = args.threads == "single"
    if single_thread:
        cv2.setNumThreads(1)

    faces = select_full_faces(load_manifest(MANIFEST_PATH))
    session_load_s = time_session_load(args.model, single_thread)
    session = make_session(args.model, single_thread)
    timings = measure(faces, session, args.warmup, args.repeats)

    result = {
        "n_faces": len(faces),
        "points_per_face": POINTS_PER_FACE,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "breakdown": {
            "rule_per_eye": "part1=preprocess×2, part2=find_edge×28（2 眼の合計）",
            "rule": "part1=preprocess, part2=find_edge×28",
            "ml": "part1=patch+feed, part2=session.run",
        },
        "session_load_ms": [s * 1000 for s in session_load_s],
        "image_pixels_median": float((timings["height"] * timings["width"]).median()),
        **summarize(timings),
        "environment": environment(single_thread),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timings.to_csv(OUTPUT_DIR / f"timing_{args.threads}.csv", index=False, float_format="%.9f")
    (OUTPUT_DIR / f"timing_{args.threads}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
