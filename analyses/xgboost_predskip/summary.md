# xgboost_predskip: XGBoost skipフラグ予想

## 目的・仮説

- CNN とは別経路で、`samples.csv` の **非画像特徴量のみ**（eye_scale / eye_bbox / eye_aspect / face_pose / pq 幾何量 / is_upper / is_left_eye / point_index / niqe_score）から `skip_flag` を予想する。
- 副次目的: 既存の skip フラグの統計的特徴を分析する。

## データ特性

| split | 全行数 (manifest) | 学習評価対象 (skip=1 ∪ true_t≠null) | 除外行 (skip=0 ∧ true_t=null) |
|-------|---:|---:|---:|
| train     | 96,096 | 93,506 | 2,590 |
| val       | 20,888 | 20,090 |   798 |
| test      | 20,412 | 19,642 |   770 |
| unlabeled | 73,836 |  ―     |  ―   |

- ラベル付き集合での **skip_flag=1 比率は約 91.6–91.9%**。多数派は positive。
- `scale_pos_weight=0.72` （= n_neg / n_pos ≈ 88,368 / 122,864）で多少の補正をかけている。

## 実験設計・条件

### 条件一覧

baseline 1 条件のみ（XGBClassifier, binary:logistic, 16 特徴量）。

| 項目 | 値 |
|---|---|
| 特徴量 | point_index, is_upper, is_left_eye, pq_dist, pq_angle, pq_ratio, dist_from_center, eye_scale, norm_eye_scale, eye_bbox_w, eye_bbox_h, eye_aspect, face_yaw, face_pitch, face_roll, niqe_score |
| n_estimators / max_depth / lr | 2000 / 6 / 0.05 |
| subsample / colsample_bytree | 0.8 / 0.8 |
| reg_alpha / reg_lambda / min_child_weight | 0.0 / 1.0 / 1 |
| early_stopping_rounds | 50 |
| scale_pos_weight | 0.72 |
| kfold / optuna | いずれも無効 |

### 分割方法

- 70% train / 15% val / 15% test, seed=42（`split.py`）。
- 学習・評価対象は `skip_flag=1` または `(skip_flag=0 ∧ true_t_raw≠null)` の行のみ（`train.py:_load_manifests`）。

## 結果

### baseline メトリクス

| split | accuracy | precision | recall | f1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| val  | 0.9092 | 0.9193 | 0.9876 | 0.9522 | 0.8931 |
| test | 0.9221 | 0.9271 | 0.9934 | 0.9591 | 0.8946 |

混同行列 (test, ラベル付き 19,642 行):

| | pred=0 | pred=1 |
|---|---:|---:|
| true=0 | 171  | 1,411 |
| true=1 | 120  | 17,940 |

- **specificity（skip=0 を当てる率）= 171 / 1,582 ≈ 10.8%**。
- recall（skip=1 を当てる率）≈ 99.3%。

参考: infer.py 側の混同行列（`(skip=0 ∧ true_t=null)` を含む全行で再評価）

| split | tn | fp | fn | tp | "fp 増加分" |
|---|---:|---:|---:|---:|---:|
| val  | 117 | 2,375 | 229 | 18,167 | +780 |
| test | 206 | 2,146 | 120 | 17,940 | +735 |

→ 学習時に除外した行（skip=0 だが true_t は欠損）の **97%以上 を XGBoost は skip=1 と予測** している。後述の通り重要なシグナル。

### 特徴量重要度 (baseline)

| rank | feature | importance |
|---:|---|---:|
| 1 | niqe_score       | 0.265 |
| 2 | norm_eye_scale   | 0.107 |
| 3 | eye_bbox_w       | 0.079 |
| 4 | dist_from_center | 0.077 |
| 5 | face_yaw         | 0.073 |
| 6 | face_pitch       | 0.069 |
| 7 | face_roll        | 0.061 |
| 8 | eye_bbox_h       | 0.059 |
| 9 | pq_ratio         | 0.054 |
| 10 | eye_aspect      | 0.052 |
| 11 | eye_scale       | 0.044 |
| 12 | is_left_eye     | 0.030 |
| 13 | point_index     | 0.014 |
| 14 | pq_angle        | 0.013 |
| 15 | pq_dist         | 0.004 |
| 16 | is_upper        | 0.000 |

- 先頭 1 特徴 (`niqe_score`) で 26.5%、上位 5 特徴 (`niqe_score, norm_eye_scale, eye_bbox_w, dist_from_center, face_yaw`) で 60.1% を占める。
- `is_upper` の重要度は **完全にゼロ**。`pq_dist`, `pq_angle`, `point_index` も極小。

### サブグループ別分析（train+val+test 結合, n=137,396）

| subgroup | n | n_pos | accuracy | precision | recall | f1 | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| part_name=eyeliner_left  | 68,698 | 61,390 | 0.9148 | 0.9170 | 0.9947 | 0.9543 | 0.928 |
| part_name=eyeliner_right | 68,698 | 61,390 | 0.9144 | 0.9156 | 0.9961 | 0.9541 | 0.938 |
| sub_key=upper            | 68,698 | 61,390 | 0.9146 | 0.9166 | 0.9950 | 0.9542 | 0.932 |
| sub_key=lower            | 68,698 | 61,390 | 0.9147 | 0.9160 | 0.9958 | 0.9542 | 0.934 |
| point_index=0..6         | 各 19,628 | 各 17,540 | ≈0.914 | ≈0.916 | ≈0.995 | ≈0.954 | ≈0.93 |

- **全サブグループで挙動がほぼ均一**（recall 99.5%, precision 91.6%, AUC 0.93 前後）。
- どのサブグループも同じ陽性率 (n_pos/n ≈ 0.894) で、データ不均衡そのものもサブグループ非依存。

## 考察

### 1. モデルは「ほぼ全部 skip=1」と返している
- skip=1 が 91.9% の極端な不均衡集合で、test の specificity が **10.8%** しかない。recall=99.3% / precision=92.7% は数値上は綺麗に見えるが、実態は **「skip=0（＝ラベル付け可能）」を 1,411 / 1,582 = 89% 取りこぼしている**。
- ROC-AUC 0.89 は分離能としては悪くないが、threshold=0.5 では完全に多数派（skip=1）に張り付いている。`scale_pos_weight=0.72` の効きが弱く、不均衡是正が不足している。

### 2. `niqe_score`（画質指標）が単独でドミナント
- 重要度の 26.5% を占め、norm_eye_scale を 2.5 倍引き離す。
- 画質起因のスキップ（ぼけ・低解像度）が skip=1 のかなりの割合を占める、という業務的直感とも整合。
- ただし `train.py` で `train_medians["niqe_score"]` を **train の最大値**で埋める特殊処理を行っている（欠損行は「最低画質」扱い）。これは欠損 = 画質計算不能（≒崩れ画像）という前提の置換だが、実装の意図はコード上で明示されておらず、後追い解析時に注意が必要。

### 3. 幾何特徴は単独では弱いが、顔向きは寄与する
- `face_yaw + face_pitch + face_roll` の合計 ≈ 20.3% で、顔向きの極端なケースで skip しがちな傾向を捉えている。
- 一方、`pq_dist`, `pq_angle`, `point_index`, `is_upper` はほぼ寄与しない。**「画像のどの点であるか」自体は skip 判定にほぼ無関係**で、判定は「フレーム全体の品質と顔の状態」で決まっている。

### 4. サブグループ依存性は無い
- left/right, upper/lower, point_index 0–6 のどの切り口でも、recall・precision・AUC が小数第 2 位までほぼ一致。
- これは「skip フラグが点固有の事情ではなく、画像 / 顔単位で決まっている」ことを強く示唆する。実際 skip フラグは顔・画像レベルの欠陥（被写体ぶれ、角度、目の大きさ）と相関している模様。

### 5. 「skip=0 ∧ true_t=null」行は実質 skip=1 と振る舞う
- 学習で除外した未ラベル行（val=798, test=770）に対しても、モデルは **97%以上を skip=1 と予測**（fp が val=+780, test=+735 増える）。
- これらは `auto_label` の取りこぼし、もしくは「アノテータが skip としていないが事実上ラベル不能だった」行に該当する可能性が高く、**ラベル整合性の問題**を示している。
  - 対策: skip=0 ∧ true_t=null の行を skip=1 に再ラベルするか、削除するか、再アノテーション対象とする。

### 6. このモデルを CNN パイプラインのゲートとして使うか
- 高確信 skip=1（pred_prob が 0.99 以上など）は誤って正常サンプルを落とすリスクが小さく、**データセット枝刈り用**には使える。
- 一方、高確信 skip=0（pred_prob が 0.05 以下）は**ほぼ存在しない**（specificity 10.8%）。ラベル可能サンプルを確実に拾うフィルタとしては不適。
- 単独ゲートで使うのは推奨しない。CNN 側のスコアと組み合わせる、もしくは閾値を最適化する必要あり。

## 結論

1. **非画像特徴量だけでも skip=1 の検出は容易（recall ≈ 99%, AUC ≈ 0.89）**。ただしデフォルト閾値ではほぼ全件 skip=1 と返す挙動で、specificity は 10.8% と低い。
2. **`niqe_score`（画質）が支配的**な特徴量で、次いで `norm_eye_scale` と顔向き 3 軸。`is_upper` や `point_index` は寄与しない。
3. skip フラグは **点固有ではなく、画像 / 顔単位の品質・姿勢に駆動されている**ことが示された（サブグループ非依存）。
4. **「skip=0 ∧ true_t=null」のラベル付け不整合**が val/test それぞれ ~800 行存在し、モデルはこれらをほぼ全て skip=1 と判定する。データ整備の見直しが必要。
5. baseline をデータセット枝刈り用途に使う場合は、ROC-AUC を最大化する閾値（または高確信側のみ採用する非対称閾値）に切り替えるべき。

### 次の一手（推奨）

- **threshold sweep**: ROC 上で youden_j（=tpr - fpr）または F0.5 最大点を選び、specificity と recall の妥協点を出す。
- **不均衡是正の強化**: `scale_pos_weight` を 0.4–0.5 まで下げる、または undersampling、focal-loss 等を試す。
- **Optuna HPO**: kfold + optuna は実装済みなので、`scale_pos_weight` を含めた CV AUC 最大化を回す。
- **データクレンジング**: `(skip=0 ∧ true_t=null)` 行の再ラベル付け（auto_label の再走 or 手動レビュー）。
- **特徴量整理**: `is_upper`, `pq_dist`, `pq_angle`, `point_index` は除去候補（重要度合計 < 3.1%）。

## 再現手順

```bash
# manifest 生成
python -m analyses.xgboost_predskip.scr.split --config analyses/xgboost_predskip/config/baseline.yaml

# 学習
python -m analyses.xgboost_predskip.scr.train --config analyses/xgboost_predskip/config/baseline.yaml

# 推論 + 分析
python -m analyses.xgboost_predskip.scr.infer --config analyses/xgboost_predskip/config/baseline.yaml
```
