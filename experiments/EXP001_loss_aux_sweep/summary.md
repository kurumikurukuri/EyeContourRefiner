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

*(実験完了後に記入)*

## 考察

*(実験完了後に記入)*
