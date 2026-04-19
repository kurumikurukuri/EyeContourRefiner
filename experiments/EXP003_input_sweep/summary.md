# EXP003: 入力パラメータ sweep

## 目的

入力画像の切り出しパラメータ（t範囲・垂直範囲・解像度）が予測精度に与える影響を体系的に検証する。  
ベースラインは EXP002 のベスト設定（sig_tloss）。

## ベースライン（EXP002 sig_tloss）

| パラメータ | 値 |
|---|---|
| patch_t_min / max | -0.5 / 1.5 |
| label_t_min / max | -0.5 / 1.5 (patch_t と同一) |
| patch_perp_scale | 1.0 |
| patch_h × patch_w | 64 × 128 |
| CV MAE | 0.064264 |
| Holdout MAE | 0.064225 |

## 実験条件（13条件）

| 条件名 | patch_t | label_t | perp | H×W | 検証仮説 |
|---|---|---|---|---|---|
| `base` | (-0.5, 1.5) | 同左 | 1.0 | 64×128 | EXP002 ベースライン再現 |
| `ultra_tight` | (0.0, 1.0) | 同左 | 1.0 | 64×128 | P-Q 間のみに完全絞る |
| `tight` | (-0.25, 1.25) | 同左 | 1.0 | 64×128 | base より少し絞る |
| `aligned_wide` | (-1.0, 2.0) | 同左 | 1.0 | 64×128 | 広い context を揃えて確保 |
| `wide_label_base` | (-1.0, 2.0) | (-0.5, 1.5) | 1.0 | 64×128 | patch だけ広く、label は base（ミスマッチ） |
| `perp_narrow` | (-0.5, 1.5) | 同左 | 0.5 | 64×128 | 垂直を絞る |
| `perp_mid` | (-0.5, 1.5) | 同左 | 1.5 | 64×128 | 垂直を中間に広げる |
| `perp_wide` | (-0.5, 1.5) | 同左 | 2.0 | 64×128 | 垂直を大きく広げる |
| `size_small` | (-0.5, 1.5) | 同左 | 1.0 | 32×64 | 小サイズ（正則化・汎化） |
| `size_large` | (-0.5, 1.5) | 同左 | 1.0 | 96×192 | 大サイズ（高解像度） |
| `tight_wide_perp` | (-0.25, 1.25) | 同左 | 2.0 | 64×128 | t 絞り × 垂直拡張 |
| `tight_large` | (-0.25, 1.25) | 同左 | 1.0 | 96×192 | t 絞り × 高解像度（最有望） |
| `aligned_wide_large` | (-1.0, 2.0) | 同左 | 1.0 | 96×192 | 広い context × 高解像度 |

## 考察のポイント

- **MAEのスケール**: label_t の範囲が変わると正規化 MAE のスケールも変わる。  
  生 t 換算: `MAE_raw = MAE_norm × (label_t_max - label_t_min)`
  - `ultra_tight`: ×1.0
  - `tight`: ×1.5
  - `base`: ×2.0
  - `aligned_wide`: ×3.0
- **端点の平均回帰チェック**: `ultra_tight` や `tight` で t≈0, t≈1 付近の予測精度を個別確認すること
- **t_pred / t_true 相関**: 全条件で相関係数を比較する
- **perp_scale と解像度の交互作用**: perp を広げると縦方向の画素密度が下がるため、後続 EXP では size_large との複合も検討

## 実行手順

```bash
# 1. manifest 生成（初回のみ）
python scripts/split_holdout.py --method random --ratio 0.2 --seed 42 --exp 003

# 2. 全条件を実行
python experiments/EXP003_input_sweep/scr/train_exp003.py

# または特定条件のみ
python experiments/EXP003_input_sweep/scr/train_exp003.py --configs base tight size_large
```

## 結果（全13条件）

label_range = label_t_max - label_t_min。生t MAE = norm MAE × label_range。

| 条件名 | label_range | CV MAE (norm) | Holdout MAE (norm) | **Holdout MAE (raw)** | 相関係数 r |
|---|---|---|---|---|---|
| **wide_label_base** | 2.0 | 0.0636 | 0.0634 | **0.1268** | **0.484** |
| aligned_wide_large | 3.0 | 0.0444 | 0.0431 | 0.1293 | 0.474 |
| perp_mid | 2.0 | 0.0655 | 0.0645 | 0.1290 | 0.471 |
| perp_wide | 2.0 | 0.0636 | 0.0642 | 0.1285 | 0.471 |
| size_small | 2.0 | 0.0649 | 0.0649 | 0.1297 | 0.457 |
| aligned_wide | 3.0 | 0.0427 | 0.0432 | 0.1297 | 0.476 |
| base | 2.0 | 0.0651 | 0.0655 | 0.1310 | 0.449 |
| perp_narrow | 2.0 | 0.0653 | 0.0656 | 0.1312 | 0.460 |
| size_large | 2.0 | 0.0655 | 0.0663 | 0.1326 | 0.449 |
| tight_wide_perp | 1.5 | 0.0858 | 0.0875 | 0.1312 | 0.438 |
| tight | 1.5 | 0.0877 | 0.0894 | 0.1341 | 0.424 |
| tight_large | 1.5 | 0.0912 | 0.0922 | 0.1383 | 0.390 |
| ultra_tight | 1.0 | 0.1345 | 0.1319 | **0.1319** | 0.357 |

参考: EXP002 sig_tloss Holdout raw=0.1284 / r=0.479

## 考察（全条件）

### 1. wide_label_base が生t MAE ベスト（0.1268）

`wide_label_base`（patch=(-1.0,2.0)、label=(-0.5,1.5)）が raw Holdout MAE=0.1268 / r=0.484 で最良。
EXP002 ベスト(raw=0.1284, r=0.479)を上回る。
**patch 範囲を広く取って周辺コンテキストを与えつつ、label 正規化はデフォルト範囲に保つ**
のが最も効果的なパターン。

### 2. normalized MAE の misleading 問題

`aligned_wide`（label_range=3.0）は norm MAE=0.043 と見かけ上は突出して低いが、
raw 換算では 0.1297 と base より悪化するケースすらある。
`aligned_wide_large` も同様（raw=0.1293）。
**label 範囲を広げても学習対象のスケールが薄まるだけで、実質的な精度向上には
寄与しない**ことが確認された。以降は生t MAE で比較統一する。

### 3. ultra_tight が最悪（0.1319）

`ultra_tight`（patch=(0,1)、label=同）は r=0.357 と相関も最低で、mid 外のサンプルに
対する識別力が崩壊している。**patch 範囲を label と同じ [0,1] に絞ることは、
境界外サンプルを学習から排除して CV を悪化させる**。

### 4. tight 系は生tスケールで見ても劣化

- tight: raw 0.1341 (-0.007 vs wide_label_base)
- tight_wide_perp: 0.1312 (-0.004)
- tight_large: 0.1383 (-0.012 最悪クラス)
- tight_large で解像度を上げても改善せず、逆に悪化

**patch_t を絞るのは相関を下げるだけで、tight × size_large の組み合わせで
さらに悪化する**（解像度を上げても情報量が見かけ上増えるだけで有用な信号は減る）。

### 5. perp_scale（垂直範囲）は広げても効果限定的

| perp_scale | raw MAE | r |
|---|---|---|
| 0.5 (narrow) | 0.1312 | 0.460 |
| 1.0 (base) | 0.1310 | 0.449 |
| 1.5 (mid) | 0.1290 | 0.471 |
| 2.0 (wide) | 0.1285 | 0.471 |

perp を広げると微改善するが、wide_label_base（+0.0017改善）ほどの効果はない。
垂直方向はアイライン位置特定には比較的冗長な情報を含む。

### 6. 解像度の効果は patch 範囲に依存

- size_small (32×64): 0.1297（base より改善）
- base (64×128): 0.1310
- size_large (96×192): 0.1326（base より悪化）

**解像度を上げるより、むしろデフォルトや縮小した方が汎化には良い**。
64×128 は現データ量では十分な解像度であり、大解像度は過学習を招く。

### 7. エッジケース平均回帰は全条件で解消されない

全13条件でバイアスパターン（t<0.2 で過大、t>1 で過小）は一貫して存在し、
patch 構成変更では根本解決にならない（EXP004-005 の動機と整合）。

## 結論

| 採用/不採用 | 項目 | 理由 |
|---|---|---|
| ✅ | patch (-1.0, 2.0) + label (-0.5, 1.5) | raw MAE ベスト、以降のベースライン |
| ❓ | perp_scale=2.0 | 微改善、単独では弱いが wide_patch と組み合わせ候補 |
| ❌ | label_range 拡大 (aligned_wide) | 見かけの norm 改善にすぎない |
| ❌ | tight 系 (patch 絞り) | 全て悪化 |
| ❌ | size_large | 過学習で逆効果 |
| ❌ | ultra_tight | 相関崩壊 |

**次実験への示唆:**
- wide_label_base を標準化して入力 sweep を卒業し、EXP004 で use_sigmoid=False や
  dfc（distance_from_center）など補助特徴量でエッジケース改善を試みる
- wide_label_base × perp_wide の複合は EXP006 H2_wide_patch で検証
