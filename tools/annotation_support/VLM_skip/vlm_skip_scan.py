"""VLM (qwen3.5:4b) を使って未ラベル画像から子供・赤ちゃんのスキップ候補を検出する。

結果は samples.db とは別の vlm_skip_hints.db に保存する。
Ctrl+C で中断しても処理済み分は保存されており、再実行で続きから再開できる。

使い方:
    python tools/annotation_support/VLM_skip/vlm_skip_scan.py
    python tools/annotation_support/VLM_skip/vlm_skip_scan.py --max-images 100 --dry-run
    python tools/annotation_support/VLM_skip/vlm_skip_scan.py --force   # 処理済みも再スキャン
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# プロジェクトルートを sys.path に追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    import urllib.request
    import urllib.error
except ImportError:
    pass

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from tools.annotation_support.VLM_skip.vlm_skip_hints_db import (
    DEFAULT_HINTS_DB,
    get_processed_paths,
    init_db,
    upsert_vlm_hint,
    count_stats,
)

logger = logging.getLogger(__name__)

_PROMPT = """\
Determine if the subject is a child or infant (approximately under 12 years old).
Respond with ONLY a valid JSON object — no markdown, no explanation, no extra text:
{"skip": true or false, "confidence": 0.0-1.0}

Set skip=true ONLY if the subject clearly appears to be a child or infant. /no_think\
"""


# ---------------------------------------------------------------------------
# samples.db から未ラベル画像パスを取得
# ---------------------------------------------------------------------------

def _get_unlabeled_image_paths(db_path: Path, sort_by_eye_size: bool = False) -> List[str]:
    """全サンプルが未アノテーション(t IS NULL)かつ skip_flag=0 の image_path リストを返す。"""
    order_clause = "MAX(eye_scale) DESC" if sort_by_eye_size else "MIN(idx)"
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            f"""
            SELECT image_path FROM samples
            GROUP BY image_path
            HAVING SUM(CASE WHEN t IS NOT NULL THEN 1 ELSE 0 END) = 0
               AND MAX(skip_flag) = 0
            ORDER BY {order_clause}
            """
        ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# 画像エンコード
# ---------------------------------------------------------------------------

def _encode_image(image_path: str, max_side: int = 512) -> Optional[str]:
    """画像を base64 JPEG に変換して返す。失敗時は None。"""
    try:
        import cv2
        import numpy as np

        bgr = cv2.imread(image_path)
        if bgr is None:
            return None
        h, w = bgr.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buf.tobytes()).decode()
    except Exception as exc:
        logger.warning("画像エンコード失敗 %s: %s", image_path, exc)
        return None


# ---------------------------------------------------------------------------
# llama.cpp / Ollama OpenAI 互換 API 呼び出し
# ---------------------------------------------------------------------------

def _call_vlm(
    image_b64: str,
    api_url: str,
    model: str,
    timeout: int = 60,
) -> str:
    """OpenAI 互換 /v1/chat/completions エンドポイントを呼び出して raw テキストを返す。"""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise assistant. Always respond with valid JSON only. /no_think",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            },
        ],
        "max_tokens": 512,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode("utf-8")
    endpoint = api_url.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# VLM 出力パース
# ---------------------------------------------------------------------------

def _parse_vlm_response(raw: str) -> Tuple[int, Optional[str], Optional[float]]:
    """VLM のレスポンステキストをパースして (vlm_skip, reason, confidence) を返す。

    JSON パース失敗時は (0, raw_text, None) を返す（保守的にスキップしない）。
    """
    text = raw.strip()
    # ```json ... ``` マークダウンを除去
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    # 最初の {...} を抽出
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        obj: Dict[str, Any] = json.loads(text)
        skip = 1 if bool(obj.get("skip", False)) else 0
        reason = str(obj["reason"]) if "reason" in obj else None
        confidence = float(obj["confidence"]) if "confidence" in obj else None
        return skip, reason, confidence
    except (json.JSONDecodeError, KeyError, ValueError):
        logger.debug("JSON パース失敗: %s", raw[:200])
        return 0, raw[:500], None


# ---------------------------------------------------------------------------
# 進捗バー
# ---------------------------------------------------------------------------

def _make_progress(total: int, desc: str = "Scanning"):
    if _HAS_TQDM:
        return tqdm(total=total, desc=desc, unit="img")
    return None


def _progress_update(bar, n: int = 1) -> None:
    if bar is not None:
        bar.update(n)


def _progress_close(bar) -> None:
    if bar is not None:
        bar.close()


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def scan(
    db_path: Path,
    hints_db_path: Path,
    api_url: str,
    model: str,
    max_images: Optional[int],
    force: bool,
    dry_run: bool,
    timeout: int,
    sort_by_eye_size: bool = False,
) -> None:
    if not db_path.exists():
        print(f"ERROR: samples.db が見つかりません: {db_path}", file=sys.stderr)
        sys.exit(1)

    if not dry_run:
        init_db(hints_db_path)

    all_paths = _get_unlabeled_image_paths(db_path, sort_by_eye_size=sort_by_eye_size)
    print(f"未ラベル画像: {len(all_paths)} 枚")

    if not force:
        processed = get_processed_paths(hints_db_path)
        targets = [p for p in all_paths if p not in processed]
        print(f"処理済みをスキップ: {len(processed)} 枚 → 残り {len(targets)} 枚")
    else:
        targets = all_paths
        print("--force: 全件再スキャン")

    if max_images is not None:
        targets = targets[:max_images]
        print(f"--max-images {max_images}: {len(targets)} 枚を処理")

    if not targets:
        print("処理対象がありません。")
        return

    processed_count = 0
    skip_count = 0
    error_count = 0
    bar = _make_progress(len(targets))

    try:
        for image_path in targets:
            image_b64 = _encode_image(image_path, max_side=384)
            if image_b64 is None:
                logger.warning("スキップ（読み込み失敗）: %s", image_path)
                error_count += 1
                _progress_update(bar)
                continue

            try:
                raw = _call_vlm(image_b64, api_url, model, timeout=timeout)
                vlm_skip, reason, confidence = _parse_vlm_response(raw)
            except Exception as exc:
                logger.warning("VLM 呼び出し失敗 %s: %s", image_path, exc)
                error_count += 1
                _progress_update(bar)
                continue  # upsert しない → 次回実行でリトライ対象になる

            if dry_run:
                status = "SKIP" if vlm_skip else "keep"
                conf_str = f"{confidence:.2f}" if confidence is not None else "N/A"
                print(f"[{status}] conf={conf_str} | {Path(image_path).name} | {reason}")
            else:
                upsert_vlm_hint(
                    image_path=image_path,
                    vlm_skip=vlm_skip,
                    vlm_reason=reason,
                    vlm_confidence=confidence,
                    vlm_model=model,
                    hints_db_path=hints_db_path,
                )

            processed_count += 1
            if vlm_skip:
                skip_count += 1
            _progress_update(bar)

    except KeyboardInterrupt:
        _progress_close(bar)
        print(f"\n--- 中断しました ---")
        print(f"処理済み: {processed_count} 枚 / スキップ候補: {skip_count} 枚 / エラー: {error_count} 枚")
        print("再実行すると続きから再開します。（--force で全件再スキャン）")
        return

    _progress_close(bar)
    print(f"\n--- 完了 ---")
    print(f"処理済み: {processed_count} 枚 / スキップ候補: {skip_count} 枚 / エラー: {error_count} 枚")

    if not dry_run:
        stats = count_stats(hints_db_path)
        print(f"\n[DB 累計] 処理: {stats['vlm_processed']} / VLM skip: {stats['vlm_skip']} / 人間確認済み: {stats['human_reviewed']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VLM で未ラベル画像のスキップ候補を検出する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", default="data/samples.db", help="samples.db パス")
    parser.add_argument("--hints-db", default=str(DEFAULT_HINTS_DB), help="vlm_skip_hints.db パス")
    parser.add_argument("--api-url", default="http://localhost:8080", help="llama.cpp エンドポイント")
    parser.add_argument("--model", default="qwen3.5:4b", help="モデル名")
    parser.add_argument("--max-images", type=int, default=None, help="処理上限（省略=全件）")
    parser.add_argument("--force", action="store_true", help="処理済み画像も再スキャン")
    parser.add_argument("--dry-run", action="store_true", help="DB に書かず stdout に出力")
    parser.add_argument("--timeout", type=int, default=60, help="VLM API タイムアウト秒")
    parser.add_argument("--sort-eye-size", action="store_true", help="eye_scale 大きい順に処理（目の大きい画像を優先）")
    parser.add_argument("--verbose", action="store_true", help="DEBUG ログを表示")
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    scan(
        db_path=Path(args.db),
        hints_db_path=Path(args.hints_db),
        api_url=args.api_url,
        model=args.model,
        max_images=args.max_images,
        force=args.force,
        dry_run=args.dry_run,
        timeout=args.timeout,
        sort_by_eye_size=args.sort_eye_size,
    )


if __name__ == "__main__":
    main()
