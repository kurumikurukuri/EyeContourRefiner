# 完了条件 (Completion Criteria)

MediaPipe の目ランドマーク座標を CNN で補正し、目の輪郭点 R の位置を推定するプロジェクトの完了を判定するための定量的基準。

---

## 1. 完了条件

テストセット上の **パッチ通過率（pass_rate）** が目標値を超えた場合、本プロジェクトは完了と判断する。

**pass_rate 90%を努力目標とする。**

---

## 2. 判定指標

### 2.1 error_ratio の定義

各 (サンプル, 点インデックス) ペアについて、予測誤差を eye_scale で正規化した誤差率を次式で定義する。

```text
error_ratio = |t_pred_raw - t_true_raw| × pq_dist / eye_scale
```

| 変数 | 定義 |
|------|------|
| `t_pred_raw` | CNN が予測した t 値（生座標） |
| `t_true_raw` | アノテーションされた t 値 |
| `pq_dist` | P-Q 間ユークリッド距離 [px] |
| `eye_scale` | 目のスケール [px] |

### 2.2 pass_rate の定義

```text
pass_rate(threshold)
=
#{error_ratio ≤ threshold}
/
全テストポイント数
```

### 2.3 採用閾値

閾値はアノテーター内変動のキャリブレーションから決定する。
[`tools/annotation_support/variation_tolerance/`](../tools/annotation_support/variation_tolerance/README.md) で取得した
ratio 分布の下位パーセンタイル値をそのまま採用する。

| レベル | 算出式 | 閾値 |
|----------|----------|----------|
| 標準 | p10 | 0.01788 |
| 厳格 | p5 | 0.01604 |

> 導出方法・キャリブレーションデータ・品質検証の詳細は
> [variation_tolerance の README](../tools/annotation_support/variation_tolerance/README.md) を参照。

---

## 3. 実測値

[`tools/annotation_support/variation_tolerance/passrate.py`](../tools/annotation_support/variation_tolerance/README.md#ツール2-pass_rate-評価-passratepy)
が全実験条件の予測 CSV を採点して生成する（`outputs/best_per_exp.md`）。各 EXP の pass_std 最良条件のみを掲載。

| exp | best_condition | n_patches | pass_std_0.01788 % | pass_strict_0.01604 % |
|---|---|---|---|---|
| EXP001_loss_aux_sweep | B1_sigmoid | 672 | 65.03 | 60.57 |
| EXP002_sig_tloss_sampler_sweep | sig_tloss | 672 | 66.07 | 60.57 |
| EXP003_input_sweep | wide_label_base | 672 | 65.62 | 62.05 |
| EXP004_edge_case_sweep | E3_no_norm_sampler | 672 | 62.80 | 59.67 |
| EXP005_mid_downsample | G3_fmax | 672 | 64.88 | 60.71 |
| EXP006_data_increase_sweep | H1_baseline | 1022 | 67.71 | 64.77 |
| EXP007_dead_zone_sweep | I2_wide_patch | 1022 | 68.10 | 64.19 |
| EXP008_norm_eye_scale_sweep | J1_baseline_nes | 1022 | 67.42 | 63.99 |
| EXP009_small_eye_filter_sweep | K1_baseline | 868 | 63.25 | 59.45 |
| EXP010_data_v2_wide_patch | L4_wide_patch_wo_flip_deep4 | 1274 | 72.06 | 69.07 |
| EXP011_hparam_sweep | E1_lr1e4_wd5e4 | 1274 | 69.78 | 67.11 |
| EXP012_patch_grid_sweep | A_p05_h128_w064 | 1274 | 72.29 | 68.92 |
| EXP013_hparam_combo | Q3_conv5 | 2002 | 75.27 | 71.68 |
| EXP014_t_resolution_sweep | R6_baseline_t_perturbation_long | 2058 | 71.33 | 68.22 |
| EXP015_interior_corner_model | S8_interior_narrow_v3 | 1440 | 73.12 | 69.44 |
| EXP016_learning_curve | S12 | 3080 | 85.19 | 81.49 |
| EXP017_rebalancing | W0_baseline | 3262 | 84.92 | 81.42 |
| EXP018_pq_and_arch | Y2_aux_face_shape | 3262 | 85.50 | 82.10 |
| EXP019_pq_transform | X2_eye_scale | 3430 | 86.27 | 82.48 |
| EXP020_dual_branch | X1_independent_none_eye_scale | 7308 | 88.04 | 84.59 |
| EXP021_loss_aug_sweep | phase3_X1_t_l1_woflip | 7308 | 89.20 | 85.95 |
| EXP022_channel_pool_sweep | bc16_pw16 | 8372 | 88.27 | 85.08 |
| EXP024_retrain_bc32_pw8 | bc32_pw8 | 8372 | 88.70 | 85.40 |
| EXP025_aux_redesign | A2 | 8372 | 88.44 | 85.24 |
| EXP026_k_batch_sweep | k5_b64 | 8372 | 88.72 | 85.73 |
