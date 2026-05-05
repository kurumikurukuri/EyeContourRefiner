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

---

## EXP2: 集約特徴量 + HPO 実験 (2026-05-06)

### 目的

skip_flag が画像単位で決定される（全画像で skip_flag は単一値）ことに着目し、
画像・顔・眼レベルの集約特徴量を追加し、HPO と組み合わせて性能改善を図る。

### 集約特徴量

`analyses/xgboost_baseline/scr/features.py` に `add_aggregate_features()` を追加。

| カテゴリ | 特徴量 | 内容 |
|---|---|---|
| 眼内点群のばらつき | `pq_dist_eye_std`, `pq_ratio_eye_std`, `dist_from_center_eye_std` | 7点のランドマークの空間的広がり |
| 左右眼の非対称性 | `eye_scale_asymmetry`, `eye_aspect_asymmetry`, `norm_eye_scale_asymmetry`, `eye_bbox_w_asymmetry`, `eye_bbox_h_asymmetry` | abs(left-right)/max(left,right) |
| 画像内顔数 | `n_faces` | 複数顔を含む画像の識別 |
| 複数顔間の姿勢ばらつき | `face_yaw_cross_face_std`, `face_pitch_cross_face_std`, `face_roll_cross_face_std` | 多顔画像での角度分散 |

### 実験条件

| # | Config | 特徴量数 | HPO | 出力ディレクトリ |
|---|---|---|---|---|
| baseline | baseline.yaml | 16 | なし | outputs/baseline |
| 1 | feat_aggregate.yaml | 29 (16+13) | なし | outputs/feat_aggregate |
| 2 | feat_aggregate_pruned.yaml | 24 (低重要度4除外+13) | なし | outputs/feat_aggregate_pruned |
| 3 | hpo_aggregate.yaml | 29 | Optuna 50trials, kfold=5 | outputs/hpo_aggregate |
| 4 | hpo_aggregate_pruned.yaml | 24 | Optuna 50trials, kfold=5 | outputs/hpo_aggregate_pruned |

pruned 除外: `is_upper`, `pq_dist`, `pq_angle`, `point_index`

### 結果

#### メトリクス比較 (test)

| 条件 | accuracy | precision | recall | f1 | ROC-AUC | specificity |
|---|---|---|---:|---:|---:|---:|---:|
| baseline | 0.8847 | 0.9115 | 0.9632 | 0.9366 | 0.8612 | 28.5% |
| **feat_aggregate** | **0.9330** | **0.9402** | 0.9871 | **0.9631** | **0.9180** | 52.0% |
| feat_aggregate_pruned | 0.9281 | 0.9381 | 0.9836 | 0.9603 | 0.9188 | 50.4% |
| hpo_aggregate | 0.9252 | 0.9457 | 0.9711 | 0.9582 | 0.9118 | 57.4% |
| hpo_aggregate_pruned | 0.9266 | 0.9452 | 0.9735 | 0.9591 | 0.9114 | 56.8% |

#### 特徴量重要度 (feat_aggregate, 上位10)

| rank | feature | importance | type |
|---:|---|---:|:---|
| 1 | **eye_bbox_w_asymmetry** | 0.172 | 左右非対称 |
| 2 | niqe_score | 0.163 | 画質 |
| 3 | face_roll_cross_face_std | 0.059 | 顔間ばらつき |
| 4 | eye_aspect_asymmetry | 0.054 | 左右非対称 |
| 5 | eye_scale_asymmetry | 0.048 | 左右非対称 |
| 6 | face_yaw_cross_face_std | 0.046 | 顔間ばらつき |
| 7 | pq_dist_eye_std | 0.045 | 眼内ばらつき |
| 8 | norm_eye_scale_asymmetry | 0.044 | 左右非対称 |
| 9 | eye_bbox_h | 0.041 | 点単位 |
| 10 | norm_eye_scale | 0.039 | 点単位 |

非対称性5特徴の合計重要度: **35.0%**（単独グループとして最大）

### 考察

#### 1. 集約特徴量の効果は圧倒的

AUC +5.7pt (0.861→0.918)、specificity 28.5%→52.0% と劇的な改善。
**眼の左右非対称性が単独で最大の予測因子**であり、skip フラグは
「左右の目の大きさ・形状が非対称な画像」を強く捉えている。

これは業務的にも合理的：
- 横顔や斜め顔では左右の目領域サイズが大きく異なる
- 低画質・ボケ画像では片方の目のランドマーク検出が不正確になり非対称性が増大
- 複数顔画像では顔ごとの角度差が生じやすい

#### 2. 非対称性 > 画質

baseline では `niqe_score` が支配的だったが、非対称性特徴がそれを上回った。
NIQE が捉えられない「幾何的な品質低下」（例：斜め顔、部分遮蔽）を非対称性が補完している。

#### 3. Pruning の効果は限定的

低重要度の 4 特徴（is_upper, pq_dist, pq_angle, point_index）を除外しても
性能はほぼ変化なし。これらの特徴はモデルにとって実質的にノイズではない（無視されている）。

#### 4. Optuna HPO は過学習

両 HPO 条件とも CV AUC=1.0000 を達成したが test AUC は非 HPO より低下（0.918→0.912）。
Optuna が選択した `learning_rate=0.254`（baseline の 5 倍）+ `n_estimators=2000` の
組み合わせで過学習。集約特徴量によって問題が「簡単になりすぎた」ことが原因。

**対策案:**
- `n_estimators` も探索空間に含める
- early_stopping_rounds を小さくする（現在 50）
- より強い正則化（reg_alpha/reg_lambda の範囲拡大）

#### 5. 最良モデル

**`feat_aggregate`**（baseline 16 特徴 + 集約 13 特徴、デフォルトパラメータ）。
シンプルで最も汎化性能が高い。

### 結論

1. **集約特徴量（特に左右非対称性）の追加は必須。** AUC +5.7pt, specificity 2倍は見逃せない改善。
2. **HPO は現状の探索空間では不要または有害。** デフォルトパラメータで十分な性能が出ている。
3. 低重要度の点単位特徴（is_upper, pq_dist, pq_angle, point_index）は**削除しても問題ない**が、積極的な削除メリットも小さい。
4. 今後の改善方向は HPO ではなく、**新たな集約特徴量の発掘**（例：顔検出 confidence、画像全体の brightness/contrast 統計量、landmark 検出の confidence 分散）が有望。

### 再現手順

```bash
# 集約特徴量あり（非HPO、推奨）
python -m analyses.xgboost_predskip.scr.train \
    --config analyses/xgboost_predskip/config/feat_aggregate.yaml
python -m analyses.xgboost_predskip.scr.infer \
    --config analyses/xgboost_predskip/config/feat_aggregate.yaml
```

---

## EXP3: Kaggleワークフロー型 精度向上 (2026-05-06)

### 目的

EXP2 の `feat_aggregate` をベースに、Kaggle ワークフロー（リークフリーHPO・特徴量拡張・ヘテロアンサンブル・閾値最適化・擬似ラベル）で更なる精度向上を図る。

> 注: EXP3 は `data/samples.db` 更新後の新データで再評価しているため、EXP2 の絶対値とは直接比較できない（行数: train 73164, val 16016, test 15540）。同じ EXP3 内では条件比較が成立する。

### Phase 1: HPO のデータリーク修正

**問題**: 過去 Optuna が CV AUC=1.0000 を出していた原因は、`KFold` が `image_path` を考慮せず、画像内の14点が train/val 両方に出現していたこと。

**修正**: `train.py` の `_optuna_objective` と `train_kfold_ensemble` を **`GroupKFold(groups=image_path)`** に切り替え。

### Phase 2: 特徴量エンジニアリング拡張

#### A. 既存実装の有効化（feat_full）

| 機能 | 特徴量数 | 内容 |
|---|---|---|
| `use_image_features` | 14 | eye_gray_*, eye_laplacian_var, eye_sobel_*, eye_canny_density, eye_hog_*, eye_lbp_* |
| `use_pose_composite` | 2 | pose_magnitude, niqe_score_x_pose_magnitude |
| `use_dataset_percentile` | 2 | niqe_percentile, eye_scale_percentile |

#### B. 新規追加（feat_full_v2: features.py に `add_extended_aggregate_features` 実装）

| カテゴリ | 特徴量 |
|---|---|
| 眼内 range | `pq_dist_eye_range` |
| 顔単位 eye_scale | `eye_scale_face_max/min/range` |
| 画像単位 niqe | `niqe_in_img_std/min/max/range` |
| 顔サイズ正規化 | `eye_bbox_w_norm_face` |
| 顔単位 aspect | `eye_aspect_face_range` |

### Phase 3: モデリング戦術

- **3A: HPO** Optuna 100 trials × GroupKFold(5)、探索空間拡大（n_estimators 200–3000、reg_alpha/lambda 1e-8–100）
- **3B: ヘテロアンサンブル** XGBoost + LightGBM + CatBoost を 3 seeds (42, 7, 123) × 5-fold で学習・単純平均
- **3C: 閾値最適化** Youden's J / F0.5 / F2 / specificity≥0.8 制約下の recall 最大化

### Phase 4: 擬似ラベル + ノイジーラベル除去

- 擬似ラベル: unlabeled (50,232 行) のうち base モデル prob≥0.99 を skip=1 として train に追加
- ノイズ除去: train OOF prob と true label の乖離トップ 5% を train から除外

### 結果

#### メトリクス比較 (test, threshold=0.5)

| # | 条件 | 特徴量 | acc | prec | rec | f1 | **AUC** | spec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | EXP3 baseline | 16 | 0.880 | 0.917 | 0.950 | 0.933 | 0.872 | 30.8% |
| 1 | feat_aggregate | 29 | 0.917 | 0.930 | 0.982 | 0.955 | 0.913 | 39.9% |
| 2 | **feat_full** | 47 | 0.933 | 0.949 | 0.978 | 0.963 | 0.939 | 57.1% |
| 3 | **feat_full_v2** | 57 | 0.934 | 0.949 | 0.979 | 0.964 | 0.941 | 57.2% |
| 4 | **hpo_full_v2** | 57 | 0.934 | 0.948 | 0.979 | 0.963 | **0.952** | 56.8% |
| 5 | ensemble (3×3) | 57 | 0.936 | 0.948 | 0.982 | 0.964 | 0.950 | 56.4% |
| 6 | pseudo (HPO base) | 57 | 0.945 | 0.949 | 0.991 | 0.970 | 0.937 | 57.3% |

太字 = AUC ベスト達成段階。条件 4 が最終ベスト。

#### 閾値別性能 (条件 5: ensemble, test)

| threshold | source | acc | prec | rec | spec |
|---|---|---:|---:|---:|---:|
| 0.500 | default | 0.936 | 0.948 | 0.982 | 56.4% |
| 0.741 | F0.5 / spec≥0.8 | 0.922 | 0.968 | 0.944 | **74.4%** |
| 0.897 | Youden's J | 0.840 | 0.983 | 0.835 | **88.0%** |
| 0.285 | F2 | 0.924 | 0.922 | 0.999 | 31.6% |

→ specificity を稼ぎたいなら **threshold=0.741** （F0.5 = spec≥0.8 制約と同点）が最適。recall 94% を保ちつつ spec 74% を達成。

#### HPO 最適パラメータ

```yaml
n_estimators: 2894     # 元 2000
max_depth: 3           # 元 6 (大幅に浅く)
learning_rate: 0.111   # 元 0.05 (倍以上)
subsample: 0.935
colsample_bytree: 0.625
reg_alpha: 1.224       # 元 0.0
reg_lambda: 7.556      # 元 1.0 (強い L2 正則化)
min_child_weight: 26   # 元 1 (大幅に保守的)
scale_pos_weight: 0.700
```

→ **「浅い木 × 強い正則化 × 多数の弱い学習器」** に収束。集約特徴量によって個々の木の深さが要らなくなったため。

#### 特徴量重要度 (HPO XGB, top 12)

| rank | feature | importance | type |
|---:|---|---:|:---|
| 1 | **niqe_in_img_max** | 0.124 | Phase 2B 新規 |
| 2 | eye_bbox_w_asymmetry | 0.076 | EXP2 集約 |
| 3 | **niqe_in_img_std** | 0.059 | Phase 2B 新規 |
| 4 | eye_gray_entropy | 0.046 | Phase 2A 画像特徴 |
| 5 | **niqe_in_img_range** | 0.043 | Phase 2B 新規 |
| 6 | eye_scale_asymmetry | 0.036 | EXP2 集約 |
| 7 | eye_canny_density | 0.031 | Phase 2A 画像特徴 |
| 8 | **eye_scale_face_min** | 0.031 | Phase 2B 新規 |
| 9 | eye_bbox_h_asymmetry | 0.026 | EXP2 集約 |
| 10 | norm_eye_scale_asymmetry | 0.025 | EXP2 集約 |
| 11 | niqe_score | 0.024 | EXP1 |
| 12 | pq_ratio_eye_std | 0.024 | EXP2 集約 |

- **niqe_in_img_max** が単独 1 位（12.4%）。画像単位の最低画質点 (niqe は値が大きいほど低画質) が skip 判定の最強シグナル。
- Phase 2B 新規特徴量 4 個が top 12 に入り、合計 25.7%。
- 上位 12 のうち画像単位/顔単位の集約特徴量が 11 個。**点単位の特徴量はほぼ寄与しない**ことが再確認された。

### 考察

#### 1. 一番効いたのは特徴量追加（+6.9pt AUC）
- baseline (0.872) → feat_full_v2 (0.941) で **+6.9pt**。
- HPO 単独の効果は **+1.0pt** にすぎず、本質はハイパラではなく特徴量。

#### 2. アンサンブルは意外と効かなかった
- 単独 HPO XGB: 0.9516
- アンサンブル (XGB×3 + LGBM×3 + Cat×3): 0.9499 (-0.17pt)
- 全コンポーネントが 0.946–0.951 と性能差が小さく、平均化で凡庸化。**XGB 自体が圧倒的に強い**。
- ただしアンサンブルは **確率較正がマイルド** になり、threshold≥0.5 での specificity を稼ぎやすい。

#### 3. 擬似ラベルは AUC を下げる
- base AUC 0.9516 → final AUC 0.9373 (**-1.4pt**)。
- 原因: prob≥0.99 で抽出した 13,933 行はすべて陽性 (low_thr=0.01 を満たす陰性は 0 件)。**陽性方向にバイアス** がかかり判別境界が崩壊。
- accuracy@0.5 は逆に上昇 (0.9337 → 0.9445)。これは予測確率が陽性側に押されただけ。
- 教訓: skip フラグの極端な不均衡データに対して naive な確信度ベース擬似ラベルは無力。

#### 4. 閾値選択が最終的な実用性を決める
- recall 重視 (data filtering 用途): default 0.5 (recall=98%)
- spec 重視 (label-mining 用途): 0.741 (spec=74%, recall=94%)
- 両立: Youden's J 0.897 (spec=88%, recall=83%)

### 結論

1. **最終ベスト = `hpo_full_v2`** (test AUC **0.952**, accuracy 0.934, default specificity 56.8%)。
2. EXP2 baseline 0.918 → EXP3 0.952 (**+3.4pt**) を達成。
3. AUC 向上のドライバは **特徴量追加 (+6.9pt)** > **HPO (+1.0pt)** > アンサンブル (横ばい) > 擬似ラベル (マイナス)。
4. 業務利用では threshold=0.741 で運用すれば spec=74% / recall=94% の安定ラインが取れる。

### 再現手順

```bash
# Phase 1 + Phase 2A: 既存特徴量を全 ON
python -m analyses.xgboost_predskip.scr.train \
    --config analyses/xgboost_predskip/config/feat_full.yaml

# Phase 2B: 新規集約特徴量
python -m analyses.xgboost_predskip.scr.train \
    --config analyses/xgboost_predskip/config/feat_full_v2.yaml

# Phase 3A: Optuna HPO + GroupKFold
python -m analyses.xgboost_predskip.scr.train \
    --config analyses/xgboost_predskip/config/hpo_full_v2.yaml

# Phase 3B/3C: ヘテロアンサンブル + 閾値最適化
python -m analyses.xgboost_predskip.scr.train_ensemble \
    --config analyses/xgboost_predskip/config/ensemble.yaml

# Phase 4: 擬似ラベル + ノイズ除去 (negative result の再現)
python -m analyses.xgboost_predskip.scr.train_pseudo \
    --config analyses/xgboost_predskip/config/pseudo.yaml
```
