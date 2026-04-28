# 二重幅誤検出: EXP014 R2_baseline_t_perturbation vs EXP003 aligned_wide_large

## 背景

`analyses/double_fold_candidates/filter_candidates.py` の閾値
（upper の `pred_t_raw < 0.30`、face あたり 2 点以上）で抽出される
「二重幅誤検出候補」が、**EXP014 R2_baseline_t_perturbation** よりも
**EXP003 aligned_wide_large** で少なく出るケースがある。

EXP014 では test_mae ベスト（0.1000）であるにもかかわらず、二重まぶた線への
ロックが起きやすいのはなぜか — を、config 差分と `extract_patch` の実装から考察した。

## 設定差分

| パラメータ | R2 (EXP014) | aligned_wide_large (EXP003) |
|---|---|---|
| `patch_perp_scale` | **2.0** | **1.0** |
| `patch_h × patch_w` | **64 × 64** | **96 × 192** |
| `patch_t_min/max` | (-2, 3)（5 unit） | (-1, 2)（3 unit） |
| `label_t_min/max` | (-1, 2) | null → (-1, 2) |
| `num_conv_layers` | 4 | 3 |
| `dropout_rate` | 0.3 | 設定なし |
| `lr` | 1e-3 | 1e-4 |
| `epochs` | 200 | 100 |
| `aug_flags` | 127（基本 + GRAYSCALE + ENDPOINT_BLEND） | 79（基本のみ） |
| その他共通 | seed=42 / k=5 / patience=50 / loss=t / use_sigmoid=true / use_batchnorm=true |

## 実効解像度の比較（`utils/patch_utils.py:114,119` から導出）

|PQ| あたりのピクセル数:

| 軸 | aligned_wide_large | R2 | 差 |
|---|---|---|---|
| t 軸（アイライン方向）| (192-1) / 3 ≈ **64 px / t-unit** | (64-1) / 5 ≈ **12.6 px / t-unit** | **約 5× 低解像** |
| perp 軸（眉↔頬方向）| (96-1) / 1.0 ≈ **95 px / |PQ|** | (64-1) / 2.0 ≈ **31.5 px / |PQ|** | **約 3× 低解像** |
| perp 視野範囲 | ±0.5 |PQ| | ±1.0 |PQ| | R2 は倍の範囲を写す |

## 主因（誤検出への寄与が大きい順）

### 主因①: `perp_scale=2.0` で二重線をパッチに取り込んでいる

- 二重線はアイラインから眉側に **0.3〜0.7|PQ|** 程度上方に走る。
- aligned_wide_large（perp_scale=1.0、視野 ±0.5|PQ|）では二重線は
  **パッチ上端付近 or 外側** に追い出されやすい。
- R2（perp_scale=2.0、視野 ±1.0|PQ|）では二重線が **パッチ中段に明確に含まれる** →
  「濃い水平線が 2 本」両方が回帰候補になり、より強コントラスト側に
  t 予測がロックされる事故が起きやすい。

### 主因②: perp 軸の解像度 32 px では 2 本線を分離できない

- 32 px で perp 1.0|PQ| を表現すると、まつ毛ライン（〜数 px）と
  二重線（〜数 px）の **間隔（〜十数 px 相当）を分解する余裕がない**。
- aligned_wide_large は perp 96 px、視野も狭いので 1 px の物理スケールが
  細かく、CNN が二重線とまつ毛線を **別個の構造** として学習できる。

### 主因③: スクエア 64×64 ＋ 広い patch_t 範囲で本体特徴が潰れる

- patch_t=(-2, 3) は学習で意味のある label_t=(-1, 2) より
  **2/5（40%）が外周コンテキスト**（無関係領域）。
- 結果、アイライン本体の写る面積が相対的に小さくなり、
  二重線との位置差を判別する手がかりが薄まる。
- aligned_wide_large は **patch_t = label_t = (-1, 2)** で完全一致、
  横長 1:2 アスペクトもアイラインの細長い形状に整合。

## 副次因子（test MAE には効くが二重幅誤検出には逆向きに作用しうる）

### ENDPOINT_BLEND（aug=127）

- P, Q を ±a, ±b で線形ブレンドして t を擬似的に揺らす（`eyeliner/data/augmentation.py:124-135`）。
- **左裾エッジ（t<0）の MAE −15〜19% 改善** に効く一方、
  **「真の t と濃いラインの対応」をぼかす** ため、
  通常域で「最強コントラストの水平線（=二重線）」にロックする方向のバイアスを増やす可能性。
- aligned_wide_large は ENDPOINT_BLEND なし。

### 強表現力 + 強正則化（num_conv=4 / dropout=0.3 / lr=1e-3 / epochs=200）

- 「強い特徴」を選び抜く方向に学習が進む。
- 二重線がパッチ内で「最も濃い水平構造」になっている画像では、
  その強い特徴が二重線でも **ロックの引力源になりうる**。

## 切り分け検証案

R2 をベースに 1 因子ずつ切り替えて、二重幅誤検出件数の変化を計測する:

| 試行 | 仮説 | 期待 |
|---|---|---|
| R2 + `patch_perp_scale=1.0` のみ | 主因①が本命 | 大きく減る |
| R2 + `patch_h,w=96,192` のみ | 主因② | 中程度減る |
| R2 + `patch_t=(-1, 2)` のみ | 主因③ | 軽度減る |
| R2 + `aug_flags=63`（ENDPOINT_BLEND OFF） | 副次因子 | わずかに減る |

各条件で `scripts/infer.py` を unlabeled manifest に対して走らせ、
`analyses/double_fold_candidates/filter_candidates.py`
（`POINT_T_THRESHOLD=0.30`, `MIN_POINTS_PER_FACE=2`）で
**face 候補数 / point 候補数** を直接比較するのが最速。

## 結論

- aligned_wide_large の優位の本体は
  **「二重線をパッチに入れない（perp_scale 縮小）」** ＋
  **「2 本線を分離できる解像度の確保（patch_h 拡大、横長アスペクト）」** の 2 点。
- R2 は test_mae で勝っている代わりに、**エッジケース改善（左裾）と引き換えに
  二重線との混同を誘発** している構図と解釈できる。
- 二重幅誤検出を減らすには、現 R2 ベースに対して
  **perp_scale=1.0 / patch=96×192 / patch_t=label_t** の方向に input 設計を
  寄せた条件を再評価するのが第一候補。

## 参照

- `experiments/EXP014_t_resolution_sweep/config/R2_baseline_t_perturbation.yaml`
- `experiments/EXP003_input_sweep/config/aligned_wide_large.yaml`
- `utils/patch_utils.py` `extract_patch`（実効解像度の式）
- `eyeliner/data/augmentation.py` `augment_endpoints`（ENDPOINT_BLEND 実装）
- `analyses/double_fold_candidates/filter_candidates.py`（候補抽出ロジック）
- `experiments/EXP014_t_resolution_sweep/summary.md`
- `experiments/EXP003_input_sweep/summary.md`
