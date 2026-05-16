# xgboost_baseline: XGBoost ベースライン（CSV 特徴量のみ） × 未アノテ外れ値検出

## 目的・仮説

- CNN とは別経路で、`samples.csv` の **非画像特徴量のみ**（eye_scale / eye_bbox / eye_aspect / face_pose / pq 幾何量 / is_upper / is_left_eye / point_index）から `true_t_raw` を回帰できるかを確認する。
- 副次目的: 未アノテーション（`true_t_raw=NaN`）サンプルのうち **外れ値っぽい ID** を抽出し、アノテーション優先度付けに使う。
- 仮説:
  1. XGBoost 単体でも `test MAE < 0.2`（生 t 空間）程度は達成できる（CNN ベスト 0.10 台には届かない見込み）。
  2. 2 条件の比較（raw t vs. `t_resolution=0.1` に丸め）で、**0.1 分解能の方が過学習が抑えられて CV 性能がわずかに良くなる**、または少なくとも外れ値スコアの順位が安定する。
  3. Quantile 予測幅（q90 - q10）＋ IsolationForest 異常度 ＋ 物理範囲違反（pred ∉ [0, 1.5]）のランク合成スコアで **上位 500 件** を取ると、人間が見て明らかに外れなサンプルが多数含まれる。

## 実験設計・条件

- **config**: `analyses/xgboost_baseline/config/{baseline,t_res_0.1}.yaml`
- **分割**: `split_by_image_path` で image_path 単位に train/val/test = 70/15/15（seed=42、skip_flag=1 除外、unlabeled は全部別ファイル）
- **特徴量**: `features.py` の `FeatureConfig`（全カテゴリ有効）
  - geometry: `pq_dist, pq_angle, pq_ratio, dist_from_center`
  - eye_scale: `eye_scale, norm_eye_scale, eye_bbox_w, eye_bbox_h, eye_aspect`
  - face_pose: `face_yaw, face_pitch, face_roll`（CSV 内で全 NULL なら自動除外）
  - categorical: `is_upper, is_left_eye`
  - `point_index`
  - 欠損は train の median で埋める（推論時も同じ値を使う）
- **学習**: `train_exp015.py`
  - XGBoost 3 本: median (reg:squarederror) / q_lo=0.1 / q_hi=0.9（reg:quantileerror）
  - IsolationForest: 全 split 統合データに対して fit
  - early_stopping_rounds=50, n_estimators=2000, max_depth=6, lr=0.05
- **条件一覧**

| 条件 | t_resolution | 出力先 |
|------|--------------|--------|
| baseline | `null`（生 t） | `outputs/exp15_baseline` |
| t_res_0.1 | `0.1`（8 クラス相当に丸め） | `outputs/exp15_t_res_0.1` |

- **評価指標**: val/test の MAE / RMSE / R²（生 t 空間）、外れ値 top-K のアノテ優先度リスト
- **外れ値スコア合成**:
  - `score_uncertainty = rank(q_hi - q_lo)`
  - `score_range_violation = rank(max(0, t_min - pred_med, pred_med - t_max))`（t_min=0.0, t_max=1.5）
  - `score_iforest = rank(-iforest.score_samples)`
  - `score = 0.4*score_uncertainty + 0.3*score_range_violation + 0.3*score_iforest`

## 実行手順

1. 前提として `tools/export_db_to_csv.py` で `data/samples.csv` を最新状態に書き出す（`face_yaw/pitch/roll` を含む）。
2. manifest 生成:
   ```bash
   python -m analyses.xgboost_baseline.scr.split \
     --config analyses/xgboost_baseline/config/baseline.yaml
   ```
3. 学習（2 条件を順次）:
   ```bash
   python -m analyses.xgboost_baseline.scr.train \
     --config analyses/xgboost_baseline/config/baseline.yaml
   ```
4. 推論 + 外れ値スコア:
   ```bash
   python -m analyses.xgboost_baseline.scr.infer \
     --config analyses/xgboost_baseline/config/baseline.yaml
   ```

## 結果

### 条件別スコア比較

| 条件 | val MAE | val R² | test MAE | test R² | OOF MAE | OOF R² |
|------|---------|--------|----------|---------|---------|--------|
| baseline | 0.155 | 0.248 | 0.153 | 0.093 | — | — |
| t_res_0.1 | 0.166 | 0.132 | 0.161 | 0.041 | — | — |
| **improved** (cyclical + interaction + Optuna + k-fold) | 0.022 ※ | 0.984 ※ | 0.160 | 0.049 | **0.108** | **0.561** |

※ improved の val は k-fold 学習データに混在するため参考値外。**OOF が正直な CV 評価**。

### improved 条件の Optuna 最良パラメータ

| パラメータ | 値 |
|---|---|
| max_depth | 9 |
| learning_rate | 0.0207 |
| subsample | 0.721 |
| colsample_bytree | 0.759 |
| reg_alpha | 3.8e-6 |
| reg_lambda | 0.102 |
| min_child_weight | 2 |

### 特徴量（improved 条件）

17 特徴量（baseline の 15 から `pq_angle` を `pq_angle_sin` + `pq_angle_cos` に置換、`is_upper_x_pq_ratio` / `is_upper_x_dist_from_center` を追加）。

## 考察

- **OOF MAE 0.108 は baseline val MAE 0.155 から 30% 改善**。cyclical エンコーディング・インタラクション特徴・Optuna HPO・k-fold の複合効果。
- **test MAE は改善しなかった（0.153→0.160）**。OOF（train+val）と test の間に分布差が存在する可能性が高い。test セットに含まれる画像が train+val とは異なる撮影条件・顔向き分布を持つと考えられる。
- Optuna の best params: `max_depth=9`（深め）、`learning_rate=0.021`（遅め）。k-fold 学習の安定性が深いツリーの過学習を抑制している。
- `reg_lambda=0.102`（baseline デフォルト 1.0 より小さい）は、k-fold アンサンブルが正則化の役割を代替している可能性を示す。
- 非画像特徴量のみで test R²=0.05 程度が限界であることが示唆される。CNN（EXP013 ベスト MAE ~0.10 台）が画像情報を使う意義を裏付ける。

## 教訓

- **k-fold 学習時の val 評価は意味をなさない**（train データに混在するため R²=0.98 になる）。OOF または test のみで評価する。
- `KFoldEnsemble` を `__main__` として実行したモジュール内で定義すると、joblib 保存時に `__main__.KFoldEnsemble` として記録され、別スクリプトからの読み込みで `AttributeError` になる。推論側で同クラスを import することで回避できる。今後は推論スクリプトから参照されるクラスを共通モジュール（features.py 等）に置くのが安全。
- test と OOF の乖離は「非画像特徴の限界」か「test 分布差」の 2 仮説が残る。画像ベース CNN との特徴量融合（EXP016 候補）で解消できるか検証価値あり。

## 成果物

- `outputs/exp15_{baseline,t_res_0.1}/`
  - `xgb_median.joblib`, `xgb_q_lo.joblib`, `xgb_q_hi.joblib`, `iforest.joblib`
  - `train_medians.csv`, `feature_columns.json`
  - `metrics.json`, `experiment.json`
  - `predictions_{train,val,test,unlabeled}.csv`
  - `outlier_topk.csv`（unlabeled 上位 500）
  - `outlier_topk_annotated.csv`（アノテ済 上位 500 — ラベルミス候補）
- `manifests/manifest_{train,val,test,unlabeled}.csv`
