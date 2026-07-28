"""全パターンの train 側 t 帯分布を分析する。"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np

T_BANDS = [
    ('t<0',     -float('inf'), 0.0),
    ('0-0.3',   0.0, 0.3),
    ('0.3-0.7', 0.3, 0.7),
    ('0.7-1.0', 0.7, 1.0),
    ('1.0-1.2', 1.0, 1.2),
    ('t>=1.2',  1.2, float('inf')),
]

def t_to_band(t):
    for i, (_, lo, hi) in enumerate(T_BANDS):
        if lo <= t < hi:
            return i
    return 5

def band_dist(df):
    bands = np.array([t_to_band(float(t)) for t in df['t'].values])
    counts = np.bincount(bands, minlength=6)
    return counts / counts.sum()

# DB を読み込む
from annotator import db
samples = db.get_annotated_samples()
rows = []
for s in samples:
    if s.get('skip_flag', 0):
        continue
    rows.append({
        'idx': int(s['idx']),
        't': float(s['t']) if s.get('t') is not None else None,
    })
db_df = pd.DataFrame(rows).dropna(subset=['t'])
db_dist = band_dist(db_df)

band_names = [b[0] for b in T_BANDS]
patterns = ["oof_error_hard", "db_replica", "random"]

# ヘッダ
header = f"{'t-band':<10} {'DB_all':>7}"
for p in patterns:
    header += f" {'train_'+p:>9} {'test_'+p:>9}"
print(header)
print("-" * len(header))

# 各帯の行
for bi, bname in enumerate(band_names):
    row = f"{bname:<10} {db_dist[bi]*100:>6.2f}%"
    for p in patterns:
        test = pd.read_csv(f"experiments/EXP029_fixed_test/manifests/manifest_test_{p}.csv")
        train = pd.read_csv(f"experiments/EXP029_fixed_test/manifests/manifest_train_{p}.csv")
        t_dist = band_dist(test)
        tr_dist = band_dist(train)
        delta = tr_dist[bi] - db_dist[bi]
        # 印: train 側で極端 t が 0.5% 超 減少している帯
        depleted = (bname in ('t<0', 't>=1.2') and delta < -0.005)
        flag = '!' if depleted else ' '
        row += f" {flag}{tr_dist[bi]*100:>7.2f}% {t_dist[bi]*100:>7.2f}%"
    print(row)

# 注釈
print()
print("! = train depleted by >0.5% vs DB (extreme-t drain, model won't learn these)")
print()

# train 側で欠落した極端 t の全体数
db_n = len(db_df)
for p in patterns:
    test = pd.read_csv(f"experiments/EXP029_fixed_test/manifests/manifest_test_{p}.csv")
    train = pd.read_csv(f"experiments/EXP029_fixed_test/manifests/manifest_train_{p}.csv")
    tr_t0 = len(train[train['t'] < 0])
    tr_t12 = len(train[train['t'] >= 1.2])
    db_t0 = len(db_df[db_df['t'] < 0])
    db_t12 = len(db_df[db_df['t'] >= 1.2])
    print(f"Pattern {p} train: t<0={tr_t0}/{db_t0} ({tr_t0/db_t0*100:.0f}% of DB), "
          f"t>=1.2={tr_t12}/{db_t12} ({tr_t12/db_t12*100:.0f}% of DB)")
