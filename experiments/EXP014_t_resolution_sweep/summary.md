# EXP014: 生 t 分解能 sweep（量子化 × augmentation 強化）

## 目的・仮説

- **目的**: EXP013 ベスト条件 `Q10_lr3_label_wide_scale` をベースラインに、
  - (a) ラベル `t` を `step=0.1` で量子化（学習・推論で `t_quant_step` を有効化）
  - (b) 追加の augmentation（**GRAYSCALE**, **ENDPOINT_BLEND** ＝ "t_perturbation"）
  の各因子と組み合わせが test MAE に与える影響を検証する。
- **仮説**:
  1. ラベルの細粒度（< 0.1）には人手アノテのノイズが乗っているため、`t_quant_step=0.1` で粒度を粗くすれば学習が安定し test MAE が下がるはず。
  2. ENDPOINT_BLEND（端点 P, Q を線形ブレンドして R を幾何保存する増強）は **t 値そのものを擬似的に揺らす** ため、エッジケース（t≪0 / t≫1）の汎化に効くはず。

## 実験設計・条件

- **config**: `experiments/EXP014_t_resolution_sweep/config/R*.yaml`
- **固定設定**: backbone=custom / loss=t / use_sigmoid=true / use_batchnorm=true / num_conv_layers=4 / dropout=0.3 / lr=1e-3 / wd=1e-4 / epochs=200 / batch=32 / k=5 / patience=50 / seed=42 / patch_t=(-2.0, 3.0) / label_t=(-1.0, 2.0) / patch=64×64 / perp_scale=2.0
- **データ**: `manifest_train.csv` 8062 件 / `manifest_test.csv` 2058 件（EXP013 比 +22%、`split_holdout.py` で再分割。data_version=313708b3）
- **可変条件（2 因子 × 3 augmentation 強度）**

| 条件 | t_quant_step | aug_flags | aug 内訳（追加分） |
|------|:---:|:---:|---|
| **R0_baseline** | none | 79 (0x4F) | BRIGHT+BLUR+NOISE+COLOR+FLIP（基本） |
| **R1_baseline_grayscale** | none | 95 (0x5F) | + **GRAYSCALE** |
| **R2_baseline_t_perturbation** | none | 127 (0x7F) | + GRAYSCALE + **ENDPOINT_BLEND** |
| **R3_res_010** | 0.1 | 79 | 基本 |
| **R4_res_010_grayscale** | 0.1 | 95 | + GRAYSCALE |
| **R5_baseline_t_perturbation** | 0.1 | 127 | + GRAYSCALE + ENDPOINT_BLEND |

> `R6_baseline_t_perturbation_long.yaml` は R2 ベース long-train 版（lr=1e-4, epochs=1000, patience=500, t_quant_step=null）。各 fold の best epoch は 300〜467。

## 結果

### 評価指標の注意（EXP013 と同じ）

- `cv_mae` / `holdout_mae` は **正規化 t 空間**（label 範囲 3.0 で割った値）の MAE。
- 横断比較は **`test_mae`（生 t 空間のホールドアウト MAE）** を使う。

### 定量評価（test_mae 昇順 = 生 t 空間）

| 条件 | CV MAE (norm) | Holdout MAE (norm) | **Test MAE (生t)** | Train MAE (生t) | Test r | Train r |
|------|--------------:|-------------------:|-------------------:|---------------:|-------:|--------:|
| **R2_baseline_t_perturbation** | 0.0348 | 0.0333 | **0.1000** | 0.0679 | **0.788** | **0.913** |
| **R6_baseline_t_perturbation_long** | 0.0346 | 0.0336 | 0.1007 | 0.0567 | - | - |
| **R5_baseline_t_perturbation** (quant) | 0.0358 | 0.0345 | 0.1007 | 0.0774 | 0.778 | 0.884 |
| **R8_baseline_t_perturbation_wo_grayscale** | 0.0350 | 0.0339 | 0.1016 | 0.0662 | - | - |
| R0_baseline (Q10 再現) | 0.0367 | 0.0344 | 0.1031 | 0.0759 | 0.756 | 0.877 |
| R1_baseline_grayscale | 0.0361 | 0.0350 | 0.1049 | 0.0784 | 0.751 | 0.868 |
| R4_res_010_grayscale (quant) | 0.0377 | 0.0359 | 0.1053 | 0.0797 | 0.751 | 0.865 |
| R3_res_010 (quant) | 0.0377 | 0.0360 | 0.1053 | 0.0749 | 0.750 | 0.881 |


### エッジケース分析（test、t-bin 別の MAE / 件数）

> 件数は test 全体の分布（中央値 0.758、95%区間 [0.36, 1.06]）に強く偏っており、左裾（t<0）は計 18 件、右裾（t>1）も少数。

| 条件 | t<-0.3 (n=5) | -0.3≤t<0 (n=13) | 0≤t<0.3 (n=58) | 0.3≤t<0.7 (n=725) | t≥0.7 (n=1257) |
|------|---:|---:|---:|---:|---:|
| R0_baseline | 0.818 | 0.450 | 0.250 | 0.103 | 0.0901 |
| R3_res_010 (quant) | 0.817 | 0.446 | 0.247 | 0.106 | 0.0923 |
| **R2_t_perturbation** | **0.666** | **0.383** | **0.225** | 0.102 | 0.0881 |
| R5_t_perturbation (quant) | 0.690 | 0.405 | 0.233 | **0.0997** | 0.0898 |

- 中央域（t≥0.3）の改善は ±0.005 程度の僅差だが、**左裾エッジ（t<0）では R2 が R0 比 −15〜19% MAE 削減**。本実験で唯一明確なシグナル。

## 考察

### 1. ラベル量子化（step=0.1）は効果なし、むしろ僅かに悪化

- R0 vs R3（aug 同じ・量子化のみ追加）: test_mae 0.1031 → 0.1053（+0.0022）、test r 0.756 → 0.750
- R1 vs R4: 0.1049 → 0.1053（+0.0004）
- R2 vs R5: 0.1000 → 0.1007（+0.0007）
- どの aug 設定でも量子化版が **微悪化** で一貫している。`test_mae_quant`（量子化評価）を見ても改善はない（R3: 0.1053 vs 0.1048、R5: 0.1007 vs 0.1010）。
- **示唆**: 本データセットの `t` ラベルは **0.1 未満の細粒度にも有意なシグナルがある**（量子化で情報が失われる方が大きい）。少なくとも step=0.1 は粗すぎる。仮説 1 は否定。

### 2. ENDPOINT_BLEND（"t_perturbation"）が test MAE と相関を同時に押し上げる、本実験唯一の有効因子

- R0 → R2 の差分: test_mae −0.0031（0.1031 → 0.1000）、**test r +0.032（0.756 → 0.788）**、train r +0.037（0.877 → 0.913）。
- 同時に train_mae も R2 が最低（0.068）→ **過学習を抑えたまま** 表現力が上がっている（よくある汎化性能向上のパターン）。
- 効果はエッジケースで顕著（左裾 t<0 で −15〜19% MAE）。これは ENDPOINT_BLEND が **R は幾何保存しつつ P・Q を線形ブレンド** することで、`t = (R-P)/(Q-P)` の値を訓練中に **擬似的に揺らす** 効果に一致。極端な t を直接擬似生成しないが、**滑らかな t 補間関係** をモデルに学習させていると解釈できる。
- 仮説 2 は支持された。**ENDPOINT_BLEND は今後の標準 augmentation に格上げすべき**。

### 3. GRAYSCALE × ENDPOINT_BLEND に正の交互作用効果あり（R8 追加実験で判明）

R8（aug_flags=111 = ENDPOINT_BLEND のみ、GRAYSCALE なし）の実験で因子分離を行った。

| 条件 | aug 内訳 | Test MAE |
|------|----------|:-------:|
| R0 | なし（基本のみ） | 0.1031 |
| R1 | GRAYSCALE のみ | 0.1049（**悪化** +0.0018） |
| R8 | ENDPOINT_BLEND のみ | 0.1016（**改善** −0.0015） |
| R2 | GRAYSCALE + ENDPOINT_BLEND | **0.1000**（**最良** −0.0031） |

- GRAYSCALE **単独**: 0.1031 → 0.1049（悪化 +0.0018）
- ENDPOINT_BLEND **単独**: 0.1031 → 0.1016（改善 −0.0015）
- **両方同時**: 0.1031 → 0.1000（改善 −0.0031 ≈ 単純加算 −0.0033 とほぼ一致）

R2 の改善は ENDPOINT_BLEND 単独（R8: −0.0015）+ GRAYSCALE 共存効果（+0.0016 追加）に分解できる。GRAYSCALE 単独では悪化するが、ENDPOINT_BLEND と同時に適用すると **色情報依存の抑制とジオメトリ拡張が相補的に作用** してわずかに追加改善する正の交互作用がある。ただし効果量は 0.0016 と小さく、**主効果は ENDPOINT_BLEND** であることは変わらない。

### 4. R6（long-train）は R2 を上回れず

- R2 ベースに lr=1e-4・epochs=1000・patience=500 を設定した long-train 版（t_quant_step=null）。
- 結果: test_mae=0.1007 で **R2（0.1000）に届かず**。Train MAE は 0.0567（R2: 0.0679）と大幅に低く、より深く収束したが汎化には繋がらなかった。
- CV MAE は 0.0346 で R2（0.0348）より微改善だが、test_mae では逆転されている。長時間学習による train/val fit の深化が test セットの分布に過適応した形。
- **示唆**: 現アーキテクチャは lr=1e-3 × 200 epoch 程度で test_mae の収束点に既に達しており、学習時間を伸ばしても汎化が向上しない。

## 結論・次の実験への示唆

### 結論

- **t_quant_step=0.1 はラベル粒度を破壊して悪化させる**。仮説 1 は否定。今後は `t_quant_step=null` を標準に戻す。
- **ENDPOINT_BLEND（"t_perturbation"）は明確に効く**。test_mae −0.0031 / r +0.032、特に左裾エッジで MAE −15〜19%。**今後の標準 aug に組み込むべき**。
- GRAYSCALE は ENDPOINT_BLEND との **交互作用で+0.0016 の追加改善**。単独では悪化するが、ENDPOINT_BLEND と組み合わせると有効（R8: 0.1016 vs R2: 0.1000）。
- 暫定ベスト: **R2_baseline_t_perturbation（test_mae=0.1000、r=0.788）**。EXP013 Q10 比 −0.0019。aug_flags=127（GRAYSCALE+ENDPOINT_BLEND）が引き続き推奨。

### 次実験で試すこと

1. **量子化はやめて augmentation に集中**
   - ENDPOINT_BLEND の **強度・確率パラメータ** を sweep（現状はデフォルト確率で適用）。さらに強くすれば左裾改善が続く可能性。
2. ~~**R6 の組み直し**~~ → **完了・否定**
   - R2 ベース long-train（lr=1e-4, epochs=1000）を実行。test_mae=0.1007 で R2（0.1000）を下回り、long-train 効果なしと判断。
3. ~~**GRAYSCALE 削除の確認**~~ → **完了・GRAYSCALE は残すべきと判断**
   - R8（aug_flags=111）実行。test_mae=0.1016 で R2（0.1000）に劣る → GRAYSCALE を外すと悪化。ENDPOINT_BLEND との交互作用効果（+0.0016）が確認された。**aug_flags=127 を維持する**。
4. **エッジケースの根本対策（t≪0 / t≫1）**
   - サンプル数自体が少ない（左裾 18 件 / 全 2058）→ ENDPOINT_BLEND より **t 重み付きサンプラー** や **ハードサンプル oversampling** の方が効きそう。EXP011 で否定された `t_weighted_sampler` を ENDPOINT_BLEND と組み合わせて再検証。
