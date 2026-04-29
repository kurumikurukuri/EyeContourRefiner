# EXP015: interior / corner 分離モデル

## 目的・仮説

- **目的**: EXP014 ベスト条件（R2_baseline_t_perturbation、test_mae=0.1000）をベースラインに、
  keypoint を **corner**（point_index 0, 6：眼角・端点）と **interior**（1–5：中間点）に分割し、
  グループ専用モデルを学習することで全点一括モデルより精度が向上するか検証する。
- **背景（per-keypoint 分析より）**:
  - `left_lower_6`, `right_lower_6`, `right_lower_0`, `left_lower_0` は
    R2 train_mae が 0.10〜0.12（ノイズ床）と他点（≈ 0.04〜0.07）より著しく高い。
  - corner 点（0, 6）は `true_t_std` と `tail_share`（t∈[0,1] 外の比率）が共に高く、
    アノテーション座標系の物理的曖昧さに起因する固有困難が存在する。
  - per-keypoint **完全独立**モデル（EXP014 系の分析）はほぼ全点で均等劣化 → delta と難度は無相関。
  - グループ分離（corner / interior）は「特化学習」と「最低限の共有」を両立する中間策として有望。
- **仮説**:
  1. corner 専用モデルは、t 範囲を広く取った `corner_wide` 設定でエッジケース（t<0, t>1）の精度が向上する。
  2. interior 専用モデルは、垂直方向を絞った `interior_narrow`（h=96, w=192, perp_scale=1.0）で
     中間点の局所特徴をより精密に捉えられる。
  3. 専用モデルを組み合わせた評価（corner 点 → corner モデル、interior 点 → interior モデル）で
     全点一括の EXP014 R2（test_mae=0.1000）を下回る。

## 実験設計・条件

- **config**: `experiments/EXP015_interior_corner_model/config/*.yaml`
- **ベース設定（EXP014 R2 引き継ぎ）**: backbone=custom / loss=t / use_sigmoid=true / use_batchnorm=true /
  num_conv_layers=4 / dropout=0.3 / lr=1e-3 / wd=1e-4 / epochs=200 / batch=32 / k=5 / patience=50 / seed=42 /
  aug_flags=127（GRAYSCALE + ENDPOINT_BLEND）

### グループ定義

| グループ | point_index | manifest |
|---------|-------------|----------|
| corner | 0, 6（眼角・端点） | manifest_{train,test}_corner.csv |
| interior | 1–5（中間点） | manifest_{train,test}_interior.csv |

### 条件一覧

| 条件 | グループ | 主な変更点 | 目的 |
|------|---------|-----------|------|
| **corner** | corner | デフォルト（patch_t -2.0〜3.0, label_t -1〜2, perp 2.0, batch=16） | ベースライン |
| **corner_wide** | corner | patch_t -2.5〜3.5 / label_t -2〜3 / perp 2.5 | t 範囲外エッジケース対策 |
| **corner_long** | corner | epochs=1000 / patience=100 / | 収束深化（corner は難しいため） |
| **interior** | interior | デフォルト（patch_t -2.0〜3.0, label_t -1〜2, perp 2.0） | ベースライン |
| **interior_narrow** | interior | h=96 w=192 / perp 1.0 / label_t -0.5〜1.5 | 中間点の局所特徴に集中 |

### データ

- manifest は `split_holdout.py` で `data/samples.db` から派生（EXP014 の再分割と同 DB バージョン）
- corner  train: manifest_train_corner.csv / test: manifest_test_corner.csv
- interior train: manifest_train_interior.csv / test: manifest_test_interior.csv

## 結果

> 実行待ち。以下は結果記入欄。

### 定量評価（test_mae 昇順・グループ別）

**corner（point_index 0, 6）**

| 条件 | CV MAE (norm) | Holdout MAE (norm) | Test MAE (生t) | Train MAE (生t) |
|------|--------------:|-------------------:|---------------:|----------------:|
| corner | - | - | - | - |
| corner_wide | - | - | - | - |
| corner_long | - | - | - | - |

**interior（point_index 1–5）**

| 条件 | CV MAE (norm) | Holdout MAE (norm) | Test MAE (生t) | Train MAE (生t) |
|------|--------------:|-------------------:|---------------:|----------------:|
| interior | - | - | - | - |
| interior_narrow | - | - | - | - |

### 全点合算評価

> 各テストセットの予測を結合して、EXP014 R2 と同一スケールで比較する。

| 評価 | EXP014 R2（全点一括） | EXP015 組み合わせ |
|------|:---:|:---:|
| Test MAE (生t・全点) | 0.1000 | - |
| Test r (全点) | 0.788 | - |

## 考察

> 結果取得後に記入。

## 結論・次の実験への示唆

> 結果取得後に記入。
