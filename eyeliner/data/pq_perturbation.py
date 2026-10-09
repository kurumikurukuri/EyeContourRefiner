"""テストレコードの PQ 摂動（ランドマーク誤差に対する頑健性評価用）。"""
from __future__ import annotations

import random

from eyeliner.data.augmentation import AUG_PQ_JITTER, augment_pq_jitter


def perturb_records_pq_rotation(records: list[dict], rot_deg: float, seed: int) -> list[dict]:
    """各レコードの P, Q を R 中心に θ ~ U(±rot_deg) 回転した新しいレコード列を返す。

    t と R は不変なので正解ラベルは変わらない。固定 seed の専用 RNG を使い、
    グローバル乱数は消費しない。入力レコードは変更しない。
    """
    rng = random.Random(seed)
    out: list[dict] = []
    for rec in records:
        px, py, qx, qy, _ = augment_pq_jitter(
            float(rec["px"]), float(rec["py"]), float(rec["qx"]), float(rec["qy"]), float(rec["t"]),
            aug_flags=AUG_PQ_JITTER, rot_deg=rot_deg, prob=1.0, rng=rng,
        )
        out.append({**rec, "px": px, "py": py, "qx": qx, "qy": qy, "t": rec["t"]})
    return out
