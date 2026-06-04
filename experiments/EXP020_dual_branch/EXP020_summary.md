# EXP020: デュアルブランチ fusion mode sweep

## 目的・仮説

EXP019 で「座標多様性アンサンブル（ENS4）が単一モデル比 −6.2%」であることが判明した。
アンサンブルを **単一モデル内のデュアルブランチ** で代替できれば、推論コスト増なしに同等の改善を得られると仮説した。

- 枝 A: EXP019 X0 と同一（生 PQ 座標系、raw_t ラベル）
- 枝 B: EXP019 X1 (eye_height) or X2 (eye_scale)（再スケール PQ 座標系、t' ラベル）
- 融合方式: independent（出力平均）/ crossfuse（Cross-Stitch 中間融合）/ attention（α 重み）

## 実験設計

| 条件 | アーキ | 枝A | 枝B | 融合 |
|------|--------|-----|-----|------|
| X0_none_baseline | 単一 | 生PQ | — | — |
| X0_eye_scale_baseline | 単一 | eye_scale P'Q' | — | — |
| X0_eye_height_baseline | 単一 | eye_height P'Q' | — | — |
| X1_independent_none_eye_scale | デュアル | 生PQ | eye_scale | 出力平均 |
| X1_independent_none_eye_height | デュアル | 生PQ | eye_height | 出力平均 |
| X2_crossfuse_none_eye_scale | デュアル | 生PQ | eye_scale | Cross-Stitch |
| X2_crossfuse_none_eye_height | デュアル | 生PQ | eye_height | Cross-Stitch |
| X3_attention_none_eye_scale | デュアル | 生PQ | eye_scale | α 重み出力 |
| X3_attention_none_eye_height | デュアル | 生PQ | eye_height | α 重み出力 |

- k=3 fold, epochs=200, patience=50, seed=42, custom backbone
- N_train=10710, N_holdout=7308, data_version=59ae3828

## 結果

### 全体 MAE

生 t 座標での infer MAE（N=7308）。X0_eye_scale/eye_height は t' 空間のため ※ 注意。

| 条件 | 生 t MAE | cv_mae (正規化) | holdout_mae (正規化) |
|------|----------|-----------------|----------------------|
| X0_none_baseline | 0.0884 | 0.03290 | 0.02933 |
| X0_eye_scale_baseline ※ | 0.0882 | 0.03424 | 0.03058 |
| X0_eye_height_baseline ※ | 0.0937 | 0.02792 | 0.02477 |
| **X1_independent_none_eye_scale** | **0.0854** | 0.03084 | 0.02848 |
| X1_independent_none_eye_height | 0.0865 | 0.03176 | 0.02884 |
| X2_crossfuse_none_eye_scale | 0.0919 | 0.03340 | 0.03062 |
| X2_crossfuse_none_eye_height | 0.0924 | 0.03413 | 0.03079 |
| X3_attention_none_eye_scale | 0.0869 | 0.03174 | 0.02896 |
| X3_attention_none_eye_height | 0.0896 | 0.03268 | 0.02988 |

※ X0_eye_scale/eye_height の生 t MAE は t' 空間（座標変換未適用）のため他条件と直接比較不可。
  正規化 MAE が小さく見えるのは label 範囲が 1.5（none の 1/2）であるため。

### 生 t バンド別 MAE

サンプル数: t<0=99, 0-0.3=291, 0.3-0.7=2533, 0.7-1.0=3794, 1.0-1.2=508, t≥1.2=83（N=7308）

| 条件 | t<0 | 0-0.3 | 0.3-0.7 | 0.7-1.0 | 1.0-1.2 | t≥1.2 | ALL |
|------|-----|-------|---------|---------|---------|-------|-----|
| X0_none_baseline | 0.4117 | 0.1668 | 0.0792 | 0.0729 | 0.1130 | 0.2635 | 0.0884 |
| X0_eye_scale_baseline ※ | 0.3080 | 0.1463 | 0.0872 | 0.0744 | 0.1015 | 0.2035 | 0.0882 |
| X0_eye_height_baseline ※ | 0.2751 | 0.1437 | 0.0976 | 0.0779 | 0.1123 | 0.1856 | 0.0937 |
| **X1_independent_none_eye_scale** | 0.3873 | 0.1642 | **0.0776** | **0.0700** | 0.1099 | 0.2444 | **0.0854** |
| X1_independent_none_eye_height | 0.3856 | 0.1631 | 0.0816 | 0.0699 | 0.1076 | 0.2421 | 0.0865 |
| X2_crossfuse_none_eye_scale | 0.3528 | 0.1644 | 0.0839 | 0.0766 | 0.1243 | 0.2687 | 0.0919 |
| X2_crossfuse_none_eye_height | 0.3971 | 0.1904 | 0.0848 | 0.0728 | 0.1247 | **0.3142** | 0.0924 |
| X3_attention_none_eye_scale | 0.3952 | 0.1718 | 0.0813 | **0.0687** | 0.1124 | 0.2691 | 0.0869 |
| X3_attention_none_eye_height | 0.4449 | 0.1726 | 0.0847 | 0.0698 | 0.1181 | 0.2595 | 0.0896 |

### 主要条件対応あり検定（paired t-test)

paired t-test（N_holdout=7308）。`***` p<.001 / `**` p<.01 / `*` p<.05 / `n.s.` 有意差なし。矢印は優位な側（有意差ありのみ）。

| 条件 | t<0 | 0-0.3 | 0.3-0.7 | 0.7-1.0 | 1.0-1.2 | t≥1.2 | ALL |
|------|-----|-------|---------|---------|---------|-------|-----|
| X0_none vs X0_eye_scale | *** (none↑) | *** (none↑) | *** (scale↑) | n.s. | *** (none↑) | *** (none↑) | n.s. |
| X0_none vs X1_independent_eye_scale | n.s. | n.s. | n.s. | *** (X1↑) | n.s. | n.s. | *** (X1↑) |
| X0_eye_scale vs X1_independent_eye_scale | *** (X1↑) | *** (X1↑) | *** (X1↑) | *** (X1↑) | *** (scale↑) | *** (scale↑) | *** (X1↑) |

## 考察

### 1. デュアルブランチ independent は有効

X1_independent_none_eye_scale が全体 MAE 0.0854 で X0_none_baseline（0.0884）を **−3.3%** 改善（paired t-test: p=5.1e-8）。
EXP019 の post-hoc アンサンブル ENS4（0.0874）を単一モデルで上回った。
統計的に確認された改善源は **0.7-1.0 帯**（p=6.7e-6、データの 52%）に限定される。0.3-0.7 帯は MAE 数値では改善しているが有意差なし（p=0.054）。

### 2. 複雑な融合（X2_crossfuse, X3_attention）は逆効果

crossfuse と attention は X0_none を下回った（0.0919〜0.0924）。
とくに X2_crossfuse_none_eye_height の t≥1.2 帯が 0.314 と突出して悪く、中間特徴量の強制共有が悪影響を与えた可能性がある。
融合のシンプルさ（独立→平均）が勝る。

### 3. X0_eye_height/eye_scale の正規化 MAE は見かけ上のベスト

正規化 MAE では X0_eye_height_baseline が最小（0.02477）に見えるが、
label 範囲が 1.5（none の 1/2）なため正規化 MAE が半分に圧縮されるだけ。
生 t 換算では最悪（0.0937）。eye_height 座標系単体での学習は効率が悪い。

### 4. eye_scale vs eye_height: eye_scale がやや優位

デュアル条件では eye_scale ブランチが eye_height をすべての fusionmode で上回った。
EXP019 では eye_height 単体 > eye_scale 単体だったが、
デュアルブランチの文脈では eye_scale との組み合わせが有効に働く可能性がある。

## まとめ

- **X1_independent_none_eye_scale が現時点のベスト（生 t MAE = 0.0854）**
- デュアルブランチ独立平均は EXP019 ENS4 を単一モデルで超えた
- 中間融合（crossfuse/attention）は効果なし → シンプルな出力平均が勝る
- eye_height 座標系は単体では非効率だがデュアルの相方としても eye_scale に劣る

## 次 EXP への示唆

1. **X1_independent_none_eye_scale をベースとした追加改善を検討**
   - より多様な座標系（mean_pq, 複数係数）をデュアル・トリプル枝へ拡張
   - 極端値帯（t<0, t≥1.2）の MAE はまだ大きい（0.39, 0.24）。裾野強化の余地あり

2. **eye_height 枝の代替を検討**
   - eye_height 単体・デュアル両方で eye_scale に劣る。mean_pq や pq_ratio 加重などを試す価値あり

3. **アンサンブル（X0_none + X1_independent_none_eye_scale）**
   - EXP019 では post-hoc アンサンブルが単体比 −6.2% だった
   - 現時点ベスト（0.0854）をさらに下げる可能性
