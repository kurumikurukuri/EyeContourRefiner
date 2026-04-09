# EXP001 考察まとめ: 損失関数・補助入力 Sweep

## 実験概要

**目的**: 損失関数・補助入力（pq_ratio）・出力活性化（sigmoid）・ネットワーク深さの効果を検証する。
バックボーンは全条件 `custom`（3〜4層CNN）固定。

## 実験条件

| 条件 | loss_fn | pq_ratio | sigmoid | conv_layers | 比較対象 |
|------|---------|----------|---------|-------------|----------|
| A1_baseline | normalized_pixel | - | - | 3 | ベースライン |
| A2_t_loss | t | - | - | 3 | 損失関数の変更 |
| A3_pq_ratio | normalized_pixel | ✓ | - | 3 | pq_ratio 補助入力の効果 |
| B1_sigmoid | normalized_pixel | - | ✓ | 3 | sigmoid 出力の効果 |
| B2_pq_sigmoid | normalized_pixel | ✓ | ✓ | 3 | pq_ratio + sigmoid 複合 |
| B3_deep | normalized_pixel | - | - | 4 | 深さの効果 |

## 結果

| 実験            | CV MAE       | Holdout MAE  | 特徴           |
| ------------- | ------------ | ------------ | ------------ |
| A1_baseline   | 0.070714     | 0.069345     | 基準           |
| A2_t_loss     | 0.070251     | **0.066788** | t-loss       |
| A3_pq_ratio   | 0.070096     | 0.069727     | pq_ratio     |
| B1_sigmoid    | 0.065844     | 0.066649     | sigmoid      |
| B2_pq_sigmoid | **0.065811** | 0.067053     | pq + sigmoid |
| B3_deep       | 0.069531     | 0.069207     | conv増加       |


## 結果（JSON実測値）

| 実験 | CV MAE | Holdout MAE |
|------|--------|-------------|
| A1_baseline | 0.07033 | 0.07002 |
| A2_t_loss | 0.07051 | 0.06786 |
| A3_pq_ratio | 0.07090 | 0.06909 |
| B1_sigmoid | **0.06596** | **0.06672** |
| B2_pq_sigmoid | 0.06585 | 0.06724 |
| B3_deep | 0.06946 | 0.06938 |

> 上記は `experiment.json` の実測値。summary表の値と微差があるが、こちらが正。

## 考察

### tの物理単位

MAEはt空間（PQ線上の比率）の生誤差。t=0がP点、t=1がQ点。
MAE=0.07 はPQ距離100pxなら約7px、50pxなら3.5pxの誤差に相当。

### sigmoid（B1）: 最大効果 -4.7% Holdout

CV・Holdout両方で安定した改善。
- sigmoid出力は(0,1)に制限される
- アイライン位置は物理的にt∈[0,1]に収まるため出力範囲が実態に整合
- sigmoid無しでは外れ予測（t<0やt>1.5）が誤差を押し上げていた可能性が高い
- 暗黙の正則化として機能している

### pq_ratio（A3）: 効果なし

CV MAEは微増（0.07033→0.07090）。Holdout改善-1.3%はノイズ範囲。
sigmoid組み合わせ(B2)でもsigmoid単体(B1)に劣る（Holdout 0.06724 vs 0.06672）。
**以降の実験から除外推奨。**

### t_loss（A2）: CV/Holdoutギャップに注意

CVは微悪化なのにHoldout -3.1%改善。KFold分布とHoldout分布の乖離か、t_lossが汎化に効く正則化として機能している可能性。**sigmoid+t_loss の組み合わせは未検証で有望候補。**

### deeper network（B3）: 効果なし

現データ量（3150サンプル）では4層は過剰。

### Fold分散パターン

全条件でFold 4が常に最低MAE、Fold 3が最高MAE。モデルではなくデータ分布の偏り。
改善幅 ±0.003未満の条件は統計的有意性が低い。

## 結論

| 採用 | 理由 |
|------|------|
| ✅ sigmoid | 安定した効果、物理的に合理的 |
| ❌ pq_ratio | 一貫した効果なし |
| ❓ t_loss | sigmoid との組み合わせ未検証 |
| ❌ conv_layers=4 | 効果なし |

**次実験の優先検討**: `sigmoid + t_loss` の組み合わせ
