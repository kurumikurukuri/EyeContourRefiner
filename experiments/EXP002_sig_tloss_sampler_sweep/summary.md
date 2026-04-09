# EXP002 考察まとめ: sigmoid + t_loss 組み合わせ & サンプラー sweep

## 実験概要

**目的**: EXP001 の知見（sigmoid と t_loss が独立して有効）を踏まえ、両者を組み合わせた場合の効果と、
エッジケース対策としての重み付きサンプラー（t_weighted_sampler）の有効性を検証する。

EXP001 のベストが B1_sigmoid (CV: 0.065844) と A2_t_loss (Holdout: 0.066788) に分かれており、
その組み合わせ C1_sig_tloss が主軸となる。

バックボーンは全条件 `custom`（3層CNN）固定。

## 実験条件

| 条件 | loss_fn | sigmoid | t_weighted_sampler | t_weight_alpha | 比較対象 |
|------|---------|---------|-------------------|----------------|----------|
| C1_sig_tloss | t | ✓ | - | - | 主軸：B1+A2の組み合わせ |
| C2_sampler_a1 | t | ✓ | ✓ | 1.0 | サンプラー弱め |
| C3_sampler_a3 | t | ✓ | ✓ | 3.0 | サンプラー標準 |
| C4_sampler_a5 | t | ✓ | ✓ | 5.0 | サンプラー強め |
| C5_sampler_nosig | t | - | ✓ | 3.0 | サンプラー単独効果の単離 |

EXP001 参照ベースライン:
- A1_baseline: CV 0.070714 / Holdout 0.069345
- A2_t_loss:   CV 0.070251 / Holdout **0.066788**
- B1_sigmoid:  CV **0.065844** / Holdout 0.066649

## 結果

| 実験 | CV MAE | Holdout MAE | 特徴 |
|------|--------|-------------|------|
| sig_tloss | **0.064264** | **0.064225** | sigmoid + t_loss |
| sampler_a1 | 0.064591 | 0.065579 | + sampler α=1.0 |
| sampler_a3 | 0.064740 | 0.065347 | + sampler α=3.0 |
| sampler_a5 | 0.065281 | 0.066775 | + sampler α=5.0 |
| sampler_nosig | 0.069471 | 0.068051 | sampler のみ（sigmoid無し） |

EXP001 参照ベースライン:

| 実験 | CV MAE | Holdout MAE |
|------|--------|-------------|
| A1_baseline | 0.070714 | 0.069345 |
| A2_t_loss | 0.070251 | 0.066788 |
| B1_sigmoid | 0.065844 | 0.066649 |

## 考察

*(実験完了後に記入)*

### 確認チェックリスト

- [ ] MAEを生のtスケールに変換して考察（正規化MAEのまま比較しない）
- [ ] t_pred vs t_true の相関係数（r値）を条件間で比較
- [ ] エッジケース（t < 0.1、t > 0.9）のサブセットMAEを確認
- [ ] sig_tloss と sampler_a1〜a5 の比較でサンプラーの効果量を定量化
- [ ] sampler_nosig と A2_t_loss（EXP001）の比較でサンプラー単独効果を確認
