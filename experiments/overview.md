# 実験 Overview

## 実験一覧

| EXP | テーマ | 条件数 | 状況 | ベスト結果（CV MAE / Holdout MAE） |
|-----|--------|--------|------|----------------------------------|
| [EXP001](EXP001_loss_aux_sweep/summary.md) | 損失関数・補助入力 sweep | 6 | 完了・考察済 | B1_sigmoid: CV 0.06596 / Holdout 0.06672 |
| [EXP002](EXP002_sig_tloss_sampler_sweep/summary.md) | sigmoid+t_loss組み合わせ & サンプラー sweep | 5 | 完了 | sig_tloss: CV 0.064264 / Holdout 0.064225 |
| [EXP003](EXP003_input_sweep/summary.md) | 入力パラメータ sweep（t範囲・垂直範囲・解像度） | 13 | 8条件完了 | wide_label_base: Holdout raw 0.1268（norm 0.0634） |
| [EXP004](EXP004_edge_case_sweep/summary.md) | エッジケース学習改善（t正規化なし・dfc・sampler） | 4 | 未実施 | - |
