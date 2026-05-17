# 実験 Overview

## 実験一覧

| EXP | テーマ | 条件数 | 状況 | ベスト結果（CV MAE / Holdout MAE） |
|-----|--------|--------|------|----------------------------------|
| [EXP001](EXP001_loss_aux_sweep/summary.md) | 損失関数・補助入力 sweep | 6 | 完了・考察済 | B1_sigmoid: CV 0.06596 / Holdout 0.06672 |
| [EXP002](EXP002_sig_tloss_sampler_sweep/summary.md) | sigmoid+t_loss組み合わせ & サンプラー sweep | 5 | 完了・考察済 | sig_tloss: CV 0.06426 / Holdout 0.06422（生t 0.1284、r=0.479） |
| [EXP003](EXP003_input_sweep/summary.md) | 入力パラメータ sweep（t範囲・垂直範囲・解像度） | 13 | 完了・考察済 | wide_label_base: Holdout raw 0.1268 / r=0.484 |
| [EXP004](EXP004_edge_case_sweep/summary.md) | エッジケース学習改善（t正規化なし・dfc・sampler） | 4 | 完了・考察済 | E4_no_norm_dfc_sampler: CV 0.1349 / Holdout 0.1342（生tスケール、EXP002換算≈0.1284で劣る） |
| [EXP005](EXP005_mid_downsample/summary.md) | midダウンサンプル & t外れ値フィルタ sweep | 11 | 完了・考察済 | F1_mid1500: CV 0.0709 / Holdout 0.0656（生t 0.1312、EXP002に届かず） |
| [EXP006](EXP006_data_increase_sweep/summary.md) | データ増加(1.5x) × 最良構成の組み合わせ最適化 | 8 | 完了・考察済 | H2_wide_patch: CV 0.05641 / Holdout 0.06332（生t MAE=0.1266） |
| [EXP007](EXP007_dead_zone_sweep/summary.md) | dead_zone_huber_loss × EXP006全条件比較 | 2 | 完了・考察済 | I2_wide_patch: CV 0.05676 / Holdout 0.06356（H2 とほぼ同等、dead_zone 効果なし） |
| [EXP008](EXP008_norm_eye_scale_sweep/summary.md) | norm_eye_scale / pq_ratio スカラー補助入力 sweep | 4 | 完了・考察済 | J4_wide_patch_pq_ratio: Holdout 0.0640（H2 比 +0.0015 悪化、補助入力は無効） |
| [EXP009](EXP009_small_eye_filter_sweep/summary.md) | small_eye_filter（eye_scale下位20%除外） sweep | 2 | 完了・考察済 | K2_wide_patch: CV 0.05634 / Holdout 0.06205（**共通評価ではH2に劣る**） |
| [EXP010](EXP010_data_v2_wide_patch/summary.md) | データv2（~6930件）× wide_patch 再検証 | 8 | 完了・考察済 | **L4_wide_patch_wo_flip_deep4**: CV 0.05377 / Holdout 0.05544（生t MAE=0.1111、r=0.711、**新ベスト**）／L6(lr=1e-3) が Holdout で同等、L2/L3/L5/L8 は効果なし |
| [EXP011](EXP011_hparam_sweep/summary.md) | ハイパーパラメータ総合 sweep（lr/wd・conv層数・loss・aug等） | 63 | 完了・考察済 | E6_lr1e3_wd1e4: test MAE=0.1123（**lr=1e-3+conv5+tws1_twl3の組み合わせ推奨**） |
| [EXP012](EXP012_patch_grid_sweep/summary.md) | パッチサイズ格子 sweep（patch_t_min/max × perp_scale × h × w） | 108 | 完了・考察済 | **A_p10_h064_w064**: CV 0.056557 / Holdout **0.056123**（patch_w=64・perp_scale=1.0・t_min=-1.0/t_max=2.0 推奨） |
| [EXP013](EXP013_hparam_combo/summary.md) | ハイパーパラメータ組み合わせ検証（L7 ベース × lr/conv/twl/正則化 × label範囲） | 11 | 完了・考察済 | **Q10_lr3_label_wide_scale**: CV(生t) 0.1133 / Holdout(生t) 0.1019（Q1 baseline 0.1041 比 `-0.0022`、単体効果の加算性なし） |
| [EXP014](EXP014_t_resolution_sweep/summary.md) | 生 t 分解能 sweep（ラベル量子化 step=0.1 × aug 強化: GRAYSCALE / ENDPOINT_BLEND）＋R6 long-train | 7 | 完了・考察済 | **R2_baseline_t_perturbation**: CV(norm) 0.0348 / Holdout(norm) 0.0333 / **Test(生t) 0.1000**（ENDPOINT_BLEND 有効・量子化悪化。R6 long-train=test 0.1007 で改善なし） |
| [EXP015](EXP015_interior_corner_model/summary.md) | interior / corner 分離モデル（point_index 0,6 vs 1–5 を専用モデルで学習） | 8 | 完了・考察済 | **S1_baseline（全点一括）**: CV(norm) 0.0340 / Holdout(norm) 0.0329 / **Test(生t) 0.0991**（専用モデル組み合わせ S3+S5=0.0997 / S3+S8=0.0998 いずれも未達。**ポイント分割は無効**、corner はデータ希少で under-fit、interior の perp 縮小は逆効果、t 範囲縮小は中立で perp 単独要因と切り分け） |
| [EXP016](EXP016_learning_curve/EXP016_summary.md) | 学習曲線（8,107〜12,880 行 × 980 行ステップ、k=3） | 5 | 完了・考察済 | **S12（全量 12,880 行）**: CV(norm) 0.031987 / Holdout(norm) **0.030191** / Test(生t) **0.090859**（8,000〜13,000 行の範囲で学習曲線は極めて平坦、test_mae 差 ≈ 0.001、**データ飽和域に入っておりアーキテクチャ改善が先決**） |
| [EXP017](EXP017_rebalancing/EXP017_summary.md) | データ再バランス sweep（mid-t 削減 × t 極端値 weighted sampler、k=3） | 6 | 完了・考察済 | **W0_baseline（フルデータ）**: CV(norm) 0.031772 / Holdout(norm) **0.029906** / Test(生t) **0.0900**（mid-t 削減・weighted sampler はいずれも全体 MAE を悪化。ただし **t<0 は W5 で−12.7%・t>1.2 は W4 で−20.5%** 改善。pq<0.10 は改善なし。**困難サブグループ改善には全体 MAE との trade-off あり**） |