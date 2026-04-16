# 実験 Overview

## 実験一覧

| EXP | テーマ | 条件数 | 状況 | ベスト結果（CV MAE / Holdout MAE） |
|-----|--------|--------|------|----------------------------------|
| [EXP001](EXP001_loss_aux_sweep/summary.md) | 損失関数・補助入力 sweep | 6 | 完了・考察済 | B1_sigmoid: CV 0.06596 / Holdout 0.06672 |
| [EXP002](EXP002_sig_tloss_sampler_sweep/summary.md) | sigmoid+t_loss組み合わせ & サンプラー sweep | 5 | 完了 | sig_tloss: CV 0.064264 / Holdout 0.064225 |
| [EXP003](EXP003_input_sweep/summary.md) | 入力パラメータ sweep（t範囲・垂直範囲・解像度） | 13 | 8条件完了 | wide_label_base: Holdout raw 0.1268（norm 0.0634） |
| [EXP004](EXP004_edge_case_sweep/summary.md) | エッジケース学習改善（t正規化なし・dfc・sampler） | 4 | 完了・考察済 | E4_no_norm_dfc_sampler: CV 0.1349 / Holdout 0.1342（生tスケール、EXP002換算≈0.1284で劣る） |
| [EXP005](EXP005_mid_downsample/summary.md) | midダウンサンプル & t外れ値フィルタ sweep | 11 | 完了 | F1_mid1500: CV 0.0709 / Holdout 0.0656 |
| [EXP006](EXP006_data_increase_sweep/summary.md) | データ増加(1.5x) × 最良構成の組み合わせ最適化 | 8 | 完了・考察済 | H2_wide_patch: CV 0.05641 / Holdout 0.06332（生t MAE=0.1266） |
| [EXP007](EXP007_dead_zone_sweep/summary.md) | dead_zone_huber_loss × EXP006全条件比較 | 8 | 未実施 | - |
| [EXP008](EXP008_norm_eye_scale_sweep/summary.md) | norm_eye_scale スカラー補助入力 sweep | 2 | 未実施 | - |
| [EXP009](EXP009_small_eye_filter_sweep/summary.md) | small_eye_filter（eye_scale下位20%除外） sweep | 2 | 完了・考察済 | K2_wide_patch: CV 0.05634 / Holdout 0.06205（**共通評価ではH2に劣る**） |
| [EXP010](EXP010_data_v2_wide_patch/summary.md) | データv2（~6930件）× wide_patch 再検証 | 3 | 完了・考察済 | L1_wide_patch: CV 0.05768 / Holdout 0.05603（生t MAE=0.1121、**新ベスト**）/ L2(pq_ratio)・L3(wo_flip)はいずれも効果なし |
| [EXP012](EXP012_patch_grid_sweep/summary.md) | パッチサイズ格子 sweep（patch_t_min/max × perp_scale × h × w） | 108 | 未実施 | - |
