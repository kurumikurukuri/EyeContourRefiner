# EXP015: interior / corner 分離モデル

## 目的・仮説

- **目的**: EXP014 ベスト条件をベースラインに、
  keypoint を **corner**（point_index 0, 6：眼角・端点）と **interior**（1–5：中間点）に分割し、
  グループ専用モデルを学習することで全点一括モデルより精度が向上するか検証する。
- **背景（per-keypoint 分析より）**:
  - `left_lower_6`, `right_lower_6`, `right_lower_0`, `left_lower_0` は
    R2 train_mae が 0.10〜0.12（ノイズ床）と他点（≈ 0.04〜0.07）より著しく高い。
  - corner 点（0, 6）は `true_t_std` と `tail_share`（t∈[0,1] 外の比率）が共に高く、
    アノテーション座標系の物理的曖昧さに起因する固有困難が存在する。
  - グループ分離（corner / interior）は「特化学習」と「最低限の共有」を両立する中間策として有望。
- **仮説**:
  1. corner 専用モデルは、t 範囲を広く取った `corner_wide` 設定でエッジケース（t<0, t>1）の精度が向上する。
  2. interior 専用モデルは、垂直方向を絞った `interior_narrow`（h=96, w=192, perp_scale=1.0）で
     中間点の局所特徴をより精密に捉えられる。
  3. 専用モデルを組み合わせた評価（corner 点 → corner モデル、interior 点 → interior モデル）で
     全点一括の EXP015 S1_baselineを下回る。

## 実験設計・条件

- **config**: `experiments/EXP015_interior_corner_model/config/*.yaml`
- **ベース設定（EXP014 R2 引き継ぎ）**: backbone=custom / loss=t / use_sigmoid=true / use_batchnorm=true /
  num_conv_layers=4 / dropout=0.3 / lr=1e-3 / wd=1e-4 / epochs=200 / batch=32(cornerは16) / k=5 / patience=50 / seed=42 /
  aug_flags=127

### グループ定義

| グループ | point_index | manifest |
|---------|-------------|----------|
| baseline | 全点 | manifest_{train,test}_baseline.csv |
| corner | 0, 6（眼角・端点） | manifest_{train,test}_corner.csv |
| interior | 1–5（中間点） | manifest_{train,test}_interior.csv |

### 条件一覧

| 条件 | グループ | 主な変更点 | 目的 |
|------|---------|-----------|------|
| **S1_baseline** | baseline | デフォルト（patch_t -2.0〜3.0, label_t -1〜2, perp 2.0） | 全点一括ベースライン |
| **S2_corner** | corner | デフォルト（patch_t -2.0〜3.0, label_t -1〜2, perp 2.0, batch=16） | データセット減のためbatch=16に半減 |
| **S3_corner_wide** | corner | patch_t -2.5〜3.5 / label_t -2〜3 / perp 2.5 / batch=16 | t 範囲外エッジケース対策 |
| **S4_corner_long** | corner | epochs=500 / patience=100 / batch=16 | 収束深化（corner は難しいため） |
| **S5_interior** | interior | デフォルト（patch_t -2.0〜3.0, label_t -1〜2, perp 2.0） | ベースライン |
| **S6_interior_narrow** | interior | h=96 w=192 / Lr=1.0e-4 / perp 1.0 /patch_t -1.0〜2.0/ label_t -0.5〜1.5 | aligned_wide_largeに近い条件 |
| **S7_interior_narrow_v2** | interior |  perp 1.0 / label_t -0.5〜1.5 | S6 の lr と入力を S5 相当に戻し、perp 縮小単独効果を切り分け |
| **S8_interior_narrow_v3** | interior |  perp 2.0 / patch_t -1.0〜2.0 / label_t -0.5〜1.5 / epochs 300 | perp は維持し t 範囲のみ縮小して効果を切り分け |

### データ

- manifest は `split_exp015.py` で `data/samples.db` から派生（EXP014 の再分割と同 DB バージョン）
- corner  train: manifest_train_corner.csv / test: manifest_test_corner.csv
- interior train: manifest_train_interior.csv / test: manifest_test_interior.csv

## 結果

### 評価指標の注意

- `cv_mae` / `holdout_mae` は **正規化 t 空間**（label 範囲で割った値）の MAE。
  - **S4_corner_wide のみ label_t=[-2,3]（範囲 5.0）** で他条件（範囲 3.0）と分母が異なるため、norm 値の直接比較は不可。
- 横断比較は **`test_mae`（生 t 空間のホールドアウト MAE）** で行う。

### 定量評価（グループ別・生 t 空間）

**baseline（全点・point 0–6）**

| 条件 | CV MAE (norm) | Holdout MAE (norm) | Test MAE (生t) | Train MAE (生t) |
|------|--------------:|-------------------:|---------------:|----------------:|
| S1_baseline | 0.0340 | 0.0329 | **0.0991** | 0.0764 |

**corner（point_index 0, 6 専用 / n_train=2428, n_holdout=576）**

| 条件 | CV MAE (norm) | Holdout MAE (norm) | Test MAE (生t) | Train MAE (生t) | 備考 |
|------|--------------:|-------------------:|---------------:|----------------:|------|
| S2_corner | 0.0444 | 0.0458 | 0.1389 | 0.1072 | デフォルト, batch=32 |
| S3_corner_long | 0.0437 | 0.0433 | **0.1313** | 0.0805 | epochs=500, patience=100, batch=16 |
| S4_corner_wide | 0.0262※ | 0.0267※ | 0.1336 | 0.0957 | label_t=[-2,3]（norm の分母 5.0） |

> ※ S4 の norm MAE は分母 5.0 によるスケール効果。生 t に直すと他条件と同等域。

**interior（point_index 1–5 専用 / n_train=6070, n_holdout=1440）**

| 条件 | CV MAE (norm) | Holdout MAE (norm) | Test MAE (生t) | Train MAE (生t) | 備考 |
|------|--------------:|-------------------:|---------------:|----------------:|------|
| S5_interior | 0.0303 | 0.0290 | **0.0870** | 0.0672 | デフォルト |
| S6_interior_narrow | 0.0487 | 0.0486 | 0.0972 | 0.0782 | h=96 w=192, lr=1e-4, perp=1.0, label_t=[-0.5,1.5] |
| S7_interior_narrow_v2 | 0.0470 | 0.0457 | 0.0913 | 0.0714 | perp=1.0, label_t=[-0.5,1.5] |
| S8_interior_narrow_v3 | 0.0450 | 0.0436 | 0.0871 | 0.0598 | perp=2.0, label_t=[-0.5,1.5], epochs 300 |

### 専用モデル組み合わせ評価（全 7 点 / n=2016）

> corner 点（0,6）→ corner モデル、interior 点（1–5）→ interior モデルで予測を結合。

| corner / interior | S5_interior | S6_interior_narrow | S7_interior_narrow_v2 | S8_interior_narrow_v3 |
|---|---:|---:|---:|---:|
| S2_corner          | 0.1019 | 0.1091 | 0.1058 | 0.1019 |
| S3_corner_long     | **0.0997** | 0.1070 | 0.1037 | 0.0998 |
| S4_corner_wide     | 0.1004 | 0.1076 | 0.1043 | 0.1004 |

**baseline 比較**

| 評価 | S1_baseline（全点一括） | EXP015 ベスト組み合わせ（S3_corner_long + S5_interior） |
|------|:---:|:---:|
| Test MAE (生 t・全 7 点) | **0.0991** | 0.0997 |
| Test r (全 7 点) | 0.789 | 0.786 |

### 点別 MAE（生 t）— baseline vs 専用モデル

| point_index | S1_baseline | corner 専用（最良） | interior S5 | interior S8 |
|---:|---:|---:|---:|---:|
| 0 (corner) | **0.1362** | 0.1389 (S3) | — | — |
| 1 (interior) | 0.0986 | — | **0.0961** | 0.1012 |
| 2 (interior) | **0.0838** | — | 0.0850 | 0.0843 |
| 3 (interior) | 0.0840 | — | 0.0853 | **0.0830** |
| 4 (interior) | 0.0838 | — | 0.0837 | **0.0809** |
| 5 (interior) | 0.0882 | — | **0.0850** | 0.0863 |
| 6 (corner) | **0.1192** | 0.1237 (S3) | — | — |
| corner 平均 | **0.1277** | 0.1313 (S3) | — | — |
| interior 平均 | **0.0877** | — | 0.0870 | 0.0871 |

## 考察

### 1. 仮説 3（専用モデル < 全点一括）— **棄却**

ベスト組み合わせ S3_corner_long + S5_interior でも test MAE 0.0997 となり、
全点一括 baseline 0.0991 を **下回らなかった**（差は 0.0006、ほぼ同等）。

理由は次の通り：

- **corner 専用化は明確に劣化**：どの設定でも baseline の corner 部分（0.128）に届かない（S2 0.139 / S3 0.131 / S4 0.134）。データ削減（8498→2428）による under-fitting が支配的。
- **interior 専用化は無利得**：S5（素直なデフォルト）が baseline interior（0.088）とほぼ同等の 0.087。
- 全点一括モデルが提供している **クロスポイント学習効果**（中間点の特徴が眼角点の予測にも有用、あるいは逆方向の汎化）が想定より大きいことが示唆される。完全分離はこの効果を切り捨ててしまう。

### 2. 仮説 1（corner_wide で外側エッジケース改善）— **確認できず**

S4_corner_wide は CV norm MAE が 0.0262 と一見ベストに見えるが、
これは **label_t 範囲を 3.0 → 5.0 に広げたことによる分母増加の見かけ効果**。
生 t に戻すと test MAE 0.1336 で S2/S3 と同等域。

「t 範囲を広げて学習」する戦略は、test 評価上は意味のない改善にしかならなかった。
むしろラベル範囲拡大によりタスクが難化し、サンプル数の少ない corner では train MAE 0.0957 → 学習が深まりきっていない。

### 3. 仮説 2（interior_narrow で局所特徴精密化）— **棄却**

入力解像度を上げて perp_scale を 1.0 に絞った S6 / S7 は、
ベース S5 より test MAE が悪化（S5 0.087 → S6 0.097 / S7 0.091）。

- S6 は lr=1e-4 と大幅に低くしたため、200 epoch では収束不足の可能性が高い。
- S7 は lr を戻しても改善せず → **perp 絞りそのものが interior 中間点の精度を損ねている** ことが示唆される。中間点でも横方向（perp 方向）の文脈情報は学習に貢献している。
- **S8（perp=2.0 を維持し t 範囲のみ [-0.5,1.5] に縮小）** は test MAE 0.0871 と S5（0.0870）にほぼ一致した。一方、train MAE は S8 0.060 < S5 0.067 と顕著に下がっている → **t 範囲縮小はタスクを易化させて train fit を上げるが、test には還元されない**。
- S6/S7/S8 の対比から、interior_narrow 系の劣化要因は **perp_scale=1.0** に局在することが切り分けできた。t 範囲縮小は単独では中立、**perp 縮小だけが劣化を生む**。
- 点別に見ると S8 は中央寄りの点（pt3, pt4）で S5 をわずかに上回る一方、両端寄りの pt1 で 0.096 → 0.101 と悪化しており、t 範囲縮小は端点側の汎化を損ねる傾向がある（端点は label_t=[-0.5,1.5] の窓に収まりにくい）。

### 4. corner_long の若干の改善

S3_corner_long（epochs=500, patience=100）は S2（200 epoch）から
test MAE 0.139 → 0.131 と改善。corner サンプルは少ないため、
**長く回す価値はある**が、それでも baseline の corner（0.128）には届かない。

### 5. CV-Test 乖離

- 全条件で `cv_mae` と `test_mae` の差が大きい（例：S1 で 0.034 vs 0.099、S5 で 0.030 vs 0.087）。
- これは正規化空間 vs 生 t 空間のスケール差が大半（label_t 範囲 3.0 で割った正規化）。
- 残差として、**ホールドアウトが train より系統的に難しい**（test_mae > train_mae × 1.3 程度）傾向は EXP014 から継続。

## 結論・次の実験への示唆

### 結論

- **「point グループ単位で独立モデルを学習する」アプローチは EXP015 のデータ規模・モデル構造では効果なし。**
- 全点一括モデルが暗黙に行っているクロスポイントの転移学習効果が大きく、完全分離はそれを上回るゲインを提供できない。
- 特に corner はデータ希少（n=2428）のため、独立モデル化はむしろ性能を落とす方向に働く。

### 次の実験への示唆

1. **ポイント分割モデルの方向は撤回**。次以降は全点一括モデル（EXP014 R2 / S1_baseline 系）をベースラインに戻す。
2. **corner 強化が必要なら別アプローチ**：
   - **マルチタスク構造**：shared backbone + per-group head（共有特徴を保ちつつ出力分岐）
   - **加重損失**：corner サンプルへの loss weight を上げる、または hard example mining
   - **corner 専用 augmentation**：端点ブレンド（ENDPOINT_BLEND）の強化版や、corner 周辺のクロップ多様化
3. **interior の perp_scale 縮小は逆効果**（S6/S7 で確認 → S8 の対照実験で perp 単独要因と切り分け済み）。今後の interior 改善実験では perp_scale を 2.0 以上で固定して別因子（patch_h、loss、augmentation）を回す。
   - **t 範囲（label_t）縮小は中立**（S5↔S8 で test 同等、train だけ良化）。窓を狭めるほど端点側の汎化が落ちる兆候があるため、デフォルト [-1, 2] を基準とし、t 範囲は今後の調整因子から外してよい。
4. **S1_baseline (test MAE 0.0991)** は EXP014 R2（0.1000）とほぼ同等で、データバージョン `e76f3dca` でのベースライン値として確定。今後の改善はこの数値を基準に評価する。
