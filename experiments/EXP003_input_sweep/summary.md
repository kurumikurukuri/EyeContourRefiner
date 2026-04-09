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

## 結果（第1回: 8条件）

label_range = label_t_max - label_t_min。raw = norm × label_range。

| 条件名 | label_range | CV MAE (norm) | Holdout MAE (norm) | Holdout MAE (raw) | CV folds (norm) |
|---|---|---|---|---|---|
| aligned_wide | 3.0 | 0.042683 | 0.043247 | **0.1297** | [0.0412, 0.0458, 0.0443, 0.0431, 0.0389] |
| wide_label_base | 2.0 | 0.063566 | 0.063387 | **0.1268** | [0.0620, 0.0680, 0.0667, 0.0643, 0.0568] |
| perp_wide | 2.0 | 0.063607 | 0.064237 | 0.1285 | [0.0620, 0.0670, 0.0661, 0.0651, 0.0579] |
| base | 2.0 | 0.065144 | 0.065531 | 0.1311 | [0.0642, 0.0698, 0.0680, 0.0654, 0.0584] |
| perp_narrow | 2.0 | 0.065331 | 0.065597 | 0.1312 | [0.0651, 0.0692, 0.0683, 0.0657, 0.0584] |
| size_large | 2.0 | 0.065506 | 0.066313 | 0.1326 | [0.0644, 0.0691, 0.0677, 0.0661, 0.0602] |
| tight_wide_perp | 1.5 | 0.085831 | 0.087498 | 0.1312 | [0.0854, 0.0883, 0.0890, 0.0876, 0.0789] |
| tight | 1.5 | 0.087684 | 0.089424 | 0.1341 | [0.0871, 0.0909, 0.0914, 0.0885, 0.0805] |

参考: EXP002 ベスト(sig_tloss) Holdout 0.064225 (norm, label_range=2.0 → raw=0.1285)

## 考察（第1回）

### 1. aligned_wide の normalized MAE は misleading
`aligned_wide` は norm MAE=0.043 と突出して低いが、label_range=3.0 なので raw=0.130 となり、
`wide_label_base`(raw=0.127) より **悪い**。wide patch が有効なのは確かだが、
label 範囲を広げることは予測対象のスケールを薄めるだけで精度向上には寄与しない。

### 2. wide_label_base がベスト（raw MAE 基準）
`wide_label_base`（patch=(-1.0,2.0)、label=(-0.5,1.5)）が raw Holdout MAE=0.1268 で最良。
EXP002 ベスト(0.1285)を上回った。**patch 範囲を広く取って周辺コンテキストを与えつつ、
label 正規化はデフォルト範囲に保つ**のが効果的。

### 3. t 範囲を絞るのは逆効果
`tight` と `tight_wide_perp` は raw でも最悪クラス（0.134, 0.131）。
(-0.25, 1.25) への絞り込みはデータの多様性を削り過学習を招く可能性がある。

### 4. perp_wide は有効、perp_narrow は無効
`perp_wide`(perp=2.0) は base より若干改善(raw: 0.1285 vs 0.1311)。
`perp_narrow`(perp=0.5) は base とほぼ同じ。垂直方向は広い方が有益だが、
`wide_label_base` ほどの効果はない。

### 5. size_large は効果なし
96×192 は base(64×128) より悪化(raw: 0.1326 vs 0.1311)。
解像度増加よりも適切な patch 範囲の方が重要。

### 次の実験への示唆
- **patch_t をさらに広げる**: (-1.5, 2.5) や (-2.0, 3.0) でどこまで有効か確認
- **wide_label_base × perp_wide の複合**: 現在未検証の有望な組み合わせ
- **wide_label_base × size_large の複合**: tight_large で検証済みだが wide_label_base 版は未実施
