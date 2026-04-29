# xgboost_predskip: XGBoost skipフラグ予想

## 目的・仮説

- CNN とは別経路で、`samples.csv` の **非画像特徴量のみ**（eye_scale / eye_bbox / eye_aspect / face_pose / pq 幾何量 / is_upper / is_left_eye / point_index）から `skip_flag` を予想する。
- 副次目的: 既存のskipフラグの統計的特徴を分析する。

## データ特性

| 項目 | 値 |
|------|-----|
| 総行数 | 211,232 |
| skip=0（有効） | 88,368 (42%) |
| skip=1（除外） | 122,864 (58%) |
| ユニーク画像数 | 6,138 |
| 画像内skip混在 | 0件（各画像は均一にskip=0 or skip=1） |

重要: skip_flagは画像単位で一貫している。つまり行レベルの予測は実質画像レベルの予測と等価。

## 実験設計・条件

### 条件一覧

| 条件 | 特徴量 | cyclical | interaction | k-fold | Optuna |
|------|--------|----------|-------------|--------|--------|
| baseline | 全15種 | - | - | - | - |
| improved | 全17種 | o | o | 5-fold | 50trials |

### 分割方法
- image_path単位で train/val/test = 70/15/15 に分割（リーク防止）
- train: 148,456行 / 4,297画像
- val: 30,828行 / 921画像
- test: 31,948行 / 920画像

## 結果

### baseline

| 指標 | val | test |
|------|-----|------|
| Accuracy | 0.608 | 0.632 |
| Precision | 0.693 | 0.743 |
| Recall | 0.545 | 0.567 |
| F1 | 0.610 | 0.643 |
| **ROC-AUC** | **0.661** | **0.683** |

混同行列 (test):
```
             pred=0   pred=1
actual=0     9,610    3,662
actual=1     8,094   10,582
```

### 特徴量重要度 (baseline)

| 順位 | 特徴量 | 重要度 |
|------|--------|--------|
| 1 | **norm_eye_scale** | **0.283** |
| 2 | dist_from_center | 0.088 |
| 3 | pq_ratio | 0.067 |
| 4 | eye_scale | 0.067 |
| 5 | face_yaw | 0.065 |
| 6 | face_pitch | 0.062 |
| 7 | face_roll | 0.057 |
| 8 | eye_bbox_w | 0.056 |
| 9 | eye_bbox_h | 0.052 |
| 10 | eye_aspect | 0.051 |

### サブグループ別分析

| グループ | Accuracy | AUC |
|----------|----------|-----|
| eyeliner_left | 0.666 | 0.737 |
| eyeliner_right | 0.664 | 0.738 |
| upper | 0.664 | 0.735 |
| lower | 0.666 | 0.740 |
| point_index=3（中間） | 0.667 | 0.740 |
| point_index=0（端点） | 0.663 | 0.734 |

## 考察

### 1. norm_eye_scale が圧倒的トップ (重要度 0.283)

正規化された目のスケールが skip 判定に最も強く寄与している。これは**画像内で目が小さい（遠景の顔）サンプルが skip されやすい**ことを示唆する。アノテーション作業において、目が小さい画像は正確な t 値の付与が困難なため、アノテーターが skip しやすいという解釈が妥当。

### 2. AUC 0.66〜0.68 は中程度

ランダム (0.5) は明確に上回るが、実用レベル (0.9+) には遠い。これは非画像特徴量だけでは skip 判定を完全には捉えきれないことを示す。画像自体の品質（ボケ、ノイズ、照明）等の要因が大きいと推測される。

### 3. face_pose が上位に食い込む (yaw/pitch/roll 合計 0.184)

顔の角度が skip 判定に影響している。横顔や極端な傾きの顔は skip されやすい傾向。

### 4. point_index ごとの差は小さい (AUC 0.734〜0.740)

ポイント位置による skip 判定のしやすさに大きな差はない。画像単位で skip が決まるため、point_index の違いは間接的な影響にとどまる。

### 5. left/right, upper/lower の差も小さい

左右・上下による skip 判定の偏りはほぼ無い。

## 結論

- **norm_eye_scale が skip 判定の主因**（目の小さい画像は skip されやすい）
- 非画像特徴量だけでは AUC ~0.68 が上限。画像品質情報が追加で必要。
- このモデルは skip 傾向の**分析ツール**としては有用だが、自動 skip 判定器としては精度不足。
- improved 条件（cyclical + interaction + k-fold + Optuna）でどの程度改善するかは今後の課題。

## 再現手順

```bash
# manifest 生成
python -m analyses.xgboost_predskip.scr.split --config analyses/xgboost_predskip/config/baseline.yaml

# 学習
python -m analyses.xgboost_predskip.scr.train --config analyses/xgboost_predskip/config/baseline.yaml

# 推論 + 分析
python -m analyses.xgboost_predskip.scr.infer --config analyses/xgboost_predskip/config/baseline.yaml
```
