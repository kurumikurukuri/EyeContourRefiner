"""
アノテーション操作ログから工数・アクション集計を行うスクリプト。

使い方:
    python tools/annotation_support/analyze_annotator_log.py
    python tools/annotation_support/analyze_annotator_log.py --log-dir logs --gap 5
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

# 2026-05-12 19:03:21,412 INFO [annotate:add] idx=8821 t=0.7500
_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \w+ \[(\w+)(?::(\w+))?\](.*)"
)
_KV_RE = re.compile(r"(\w+)=(\S+)")

SESSION_TAIL_MINUTES = 2  # 最終操作後に加算する余裕時間


def _parse_file(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _LINE_RE.match(line.rstrip())
            if not m:
                continue
            ts_str, action, subaction, rest = m.groups()
            kv = {k: v for k, v in _KV_RE.findall(rest)}
            events.append({
                "ts": datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S"),
                "action": action,
                "subaction": subaction,  # "add" | "modify" | None
                "kv": kv,
            })
    return events


def _classify(event: dict[str, Any]) -> str:
    """操作を 追加/修正/削除/スキップ/AI適用/その他 に分類する。"""
    action, sub = event["action"], event["subaction"]
    if action == "annotate":
        if sub == "add":
            return "add"
        if sub == "modify":
            return "modify"
        return "add"          # 旧フォーマット（subaction なし）は追加扱い
    if action == "skip":
        return "skip"
    if action == "apply_pred":
        return "apply_pred"
    if action == "reset_eye":
        return "delete"
    return "other"


def _build_sessions(
    events: list[dict[str, Any]], gap_minutes: int
) -> list[dict[str, Any]]:
    """連続するイベントをセッションにまとめる。"""
    if not events:
        return []
    ordered = sorted(events, key=lambda e: e["ts"])
    sessions: list[dict[str, Any]] = []
    start = ordered[0]["ts"]
    bucket = [ordered[0]]

    for ev in ordered[1:]:
        if ev["ts"] - bucket[-1]["ts"] > timedelta(minutes=gap_minutes):
            sessions.append({"start": start, "last": bucket[-1]["ts"], "events": bucket})
            start = ev["ts"]
            bucket = [ev]
        else:
            bucket.append(ev)

    sessions.append({"start": start, "last": bucket[-1]["ts"], "events": bucket})
    return sessions


def _session_hours(session: dict[str, Any], tail: int) -> float:
    duration = (session["last"] - session["start"]) + timedelta(minutes=tail)
    return duration.total_seconds() / 3600


def _empty_day() -> dict[str, int | float]:
    return {
        "hours": 0.0,
        "add": 0, "modify": 0,
        "delete_ops": 0, "delete_samples": 0,
        "skip_ops": 0, "skip_samples": 0,
        "apply_ops": 0, "apply_samples": 0,
    }


def _accumulate(
    day: dict[str, Any],
    classified: str,
    kv: dict[str, str],
) -> None:
    if classified in ("add", "modify"):
        day[classified] += 1
    elif classified == "delete":
        day["delete_ops"] += 1
        day["delete_samples"] += int(kv.get("updated", 1))
    elif classified == "skip":
        day["skip_ops"] += 1
        day["skip_samples"] += int(kv.get("updated", 1))
    elif classified == "apply_pred":
        day["apply_ops"] += 1
        day["apply_samples"] += int(kv.get("updated", 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="アノテーション工数・操作集計")
    parser.add_argument("--log-dir", default="logs", help="ログディレクトリ (default: logs)")
    parser.add_argument("--gap", type=int, default=5, help="セッション区切り間隔 分 (default: 5)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"ログディレクトリが見つかりません: {log_dir}")
        return

    log_files = sorted(log_dir.glob("annotator.log*"))
    if not log_files:
        print("annotator.log* が見つかりません")
        return

    all_events: list[dict[str, Any]] = []
    for lf in log_files:
        all_events.extend(_parse_file(lf))

    annotation_events = [e for e in all_events if e["action"] in ("annotate", "skip", "apply_pred", "reset_eye")]
    if not annotation_events:
        print("アノテーションイベントが見つかりません")
        return

    annotation_events.sort(key=lambda e: e["ts"])
    sessions = _build_sessions(annotation_events, gap_minutes=args.gap)

    daily: dict[str, dict[str, Any]] = defaultdict(_empty_day)
    total_hours = 0.0

    for session in sessions:
        date_key = session["start"].strftime("%Y-%m-%d")
        h = _session_hours(session, tail=SESSION_TAIL_MINUTES)
        daily[date_key]["hours"] += h
        total_hours += h
        for ev in session["events"]:
            _accumulate(daily[date_key], _classify(ev), ev["kv"])

    # ---- 出力 ----
    first_ts = annotation_events[0]["ts"].strftime("%Y-%m-%d")
    last_ts = annotation_events[-1]["ts"].strftime("%Y-%m-%d")

    print("=" * 50)
    print("  アノテーション工数サマリー")
    print("=" * 50)
    print(f"  集計期間  : {first_ts} 〜 {last_ts}")
    print(f"  セッション数: {len(sessions)}  (区切り {args.gap} 分)")
    print(f"  推定総工数  : {total_hours:.1f}h")
    print()

    col = "{:<12}  {:>5}  {:>6} {:>6} {:>6} {:>8} {:>6}"
    print(col.format("日付", "工数", "追加", "修正", "削除", "スキップ", "AI適用"))
    print("-" * 60)
    for date_key in sorted(daily.keys(), reverse=True):
        d = daily[date_key]
        print(col.format(
            date_key,
            f"{d['hours']:.1f}h",
            d["add"],
            d["modify"],
            d["delete_samples"],
            d["skip_samples"],
            d["apply_samples"],
        ))

    # 累計集計
    total_add    = sum(d["add"]             for d in daily.values())
    total_modify = sum(d["modify"]          for d in daily.values())
    total_del    = sum(d["delete_samples"]  for d in daily.values())
    total_del_op = sum(d["delete_ops"]      for d in daily.values())
    total_skip   = sum(d["skip_samples"]    for d in daily.values())
    total_skip_op = sum(d["skip_ops"]       for d in daily.values())
    total_apply  = sum(d["apply_samples"]   for d in daily.values())
    total_apply_op = sum(d["apply_ops"]     for d in daily.values())

    print()
    print("--- アクション集計（累計）---")
    print(f"  追加 (annotate:add)    : {total_add:>6} サンプル")
    print(f"  修正 (annotate:modify) : {total_modify:>6} サンプル", end="")
    if total_add + total_modify > 0:
        ratio = total_modify / (total_add + total_modify) * 100
        print(f"  (修正率 {ratio:.1f}%)", end="")
    print()
    print(f"  削除 (reset_eye)       : {total_del:>6} サンプル  ({total_del_op} 操作)")
    print(f"  スキップ               : {total_skip:>6} サンプル  ({total_skip_op} 操作)")
    print(f"  AI予測一括適用         : {total_apply:>6} サンプル  ({total_apply_op} 操作)")


if __name__ == "__main__":
    main()
