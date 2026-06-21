# EXP023: OOF予測による誤ラベル候補検出パイプライン 仕様書

## 目的・背景

DB (`data/samples.db`) のアノテーション（t値）に含まれる誤ラベル（誤クリック・取り違え等）を、
CNNのOOF（Out-of-Fold）予測との残差を使って半自動的に検出し、annotatorでの目視レビュー対象を
絞り込む。

性能の良いモデルと悪いモデルの両方でOOF予測を行い、両者が共通して大きな残差を示す
サンプルを優先する形で二重チェックする（モデル個別の癖による誤検出を減らす狙い）。

- **良いモデル**: `bc32_pw8`（[EXP022](../EXP022_channel_pool_sweep/EXP022_summary.md) holdout raw-t MAE 最良 0.087129。少数派の極端t帯でも最良）
- **悪いモデル**: `bc16_pw1`（同 holdout raw-t MAE 最悪 0.092032。`pool_width=1`はFCヘッドに渡す特徴マップを1点まで圧縮し、空間情報が失われる）

## パイプライン構成

```
DB (samples.db, t IS NOT NULL 全件)
  │  export_full_manifest.py
  ▼
full_manifest.csv (20,986件)
  │  generate_folds.py → split_fixed_folds()
  ▼
fold1.csv 〜 fold5.csv（image_path単位 GroupKFold, seed固定で再現可能）
  │  run_oof.py --mode nested → run_nested_oof_kfold() / run_nested_dual_oof_kfold()
  ▼
oof_predictions.csv（モデルごとに1ファイル。各サンプルを学習に使っていないモデルでの予測）
  │  rank_candidates.py（bc16_pw1 / bc32_pw8 それぞれに対して実行）
  ▼
candidates_*.csv（多数派×高残差でランキング、annotator直リンク付き）
  │  （両モデルの候補リストを目視で突き合わせ → 人手レビュー、annotator WebUI）
  ▼
誤ラベル修正 → DB反映 → 再学習
```

### 0. 実装状況

単一ブランチ・デュアルブランチ両対応で実装済み。

- `eyeliner/training/oof.py`（`OofPrediction`, backbone非依存）
- `eyeliner/training/oof_kfold.py`
  - 単一ブランチ: `split_fixed_folds()` / `run_oof_kfold()` / `run_nested_oof_kfold()`
    （`eyeliner/training/kfold.py` の `_train_single_fold` / `_predict_records` を使用）
  - デュアルブランチ: `run_dual_oof_kfold()` / `run_nested_dual_oof_kfold()`
    （`eyeliner/training/dual_kfold.py` に追加した `_train_single_dual_fold` /
    `_predict_dual_records` を使用。`run_dual_kfold()` 自体もこの2関数を呼ぶ形に
    リファクタリングし、単一ブランチ版と対称な構造にした）
- `experiments/EXP023_oof_mislabel/scr/{export_full_manifest,generate_folds,run_oof,rank_candidates}.py`
  - `run_oof.py` は config の `backbone` フィールドが `custom_dual` かどうかで
    単一/デュアルブランチの2経路に自動分岐する。他実験の config を直接読みたい場合用に
    `--config-path` も用意しているが、EXP023用には `config/bc16_pw1.yaml` /
    `config/bc32_pw8.yaml` を自実験のconfigディレクトリにコピー済み（実体は
    `experiments/EXP022_channel_pool_sweep/config/` と同一。EXP023ディレクトリを
    自己完結させるための複製）。
- `tests/test_oof.py`, `tests/test_oof_kfold.py`（デュアル版のテストケース込み）, `tests/test_kfold.py`
- `experiments/EXP023_oof_mislabel/manifests/*.csv`
- `config/bc16_pw1.yaml` / `bc32_pw8.yaml`（デュアル良否比較用）

単一ブランチ版の試行（`woflip_none`）の config と outputs はリポジトリに含まれていない。

実行例:
```bash
python experiments/EXP023_oof_mislabel/scr/run_oof.py --config bc16_pw1
python experiments/EXP023_oof_mislabel/scr/run_oof.py --config bc32_pw8
```
`--mode nested`（デフォルト）かつ `--exp-name` 省略時は `outputs/bc16_pw1_nested/`,
`outputs/bc32_pw8_nested/` にそれぞれ出力される。fold{1..5}.csv は既に `manifests/`
にあるため2回とも同じ分割が使われる。各実行は5ラウンド分の訓練（nested CV）を要する
ため、フルデータ（20,986件）では長時間かかる（軽量動作確認は `--epochs 2 --patience 1`
等で代用）。

### 1. `export_full_manifest.py`

- 入力: `data/samples.db`
- 出力: `manifests/full_manifest.csv`
- `annotator.db.get_annotated_samples()` を使い `t IS NOT NULL` の全件を抽出。
- フィールド: `idx, image_path, face_index, part_name, sub_key, point_index, px, py, qx, qy, t`

### 2. `generate_folds.py` / `eyeliner.training.oof_kfold.split_fixed_folds()`

- 入力: `full_manifest.csv`
- 出力: `manifests/fold{1..5}.csv`
- `image_path` をグループキーとした `GroupKFold(n_splits=k, shuffle=True, random_state=seed)`。
- 既存ファイルがあればデフォルトでスキップ（`--force` で再生成）。シード固定のため決定的に
  同じ分割を再現できる。
- k=5, seed=42（デフォルト）。fold分割自体は良いモデル・悪いモデルで共通（同じ`fold{1..5}.csv`を使う）。

### 3. `run_oof.py` / `eyeliner.training.oof_kfold`

- 入力: `fold{1..5}.csv`,
  `experiments/EXP023_oof_mislabel/config/bc16_pw1.yaml`（悪いモデル）,
  `experiments/EXP023_oof_mislabel/config/bc32_pw8.yaml`（良いモデル）
- 出力: `outputs/<exp_name>/eyeliner_detection_fold{1..5}.pt`, `outputs/<exp_name>/oof_predictions.csv`
  （bc16_pw1, bc32_pw8 それぞれ別の `<exp_name>` で1回ずつ実行し、2系統のOOF予測を作る）
- **モード**:
  - `nested`（デフォルト）: `run_nested_oof_kfold()` / `run_nested_dual_oof_kfold()`。
    5ラウンドで `Test/Val/Train` をローテーション
    (`round i: Test=fold[i], Val=fold[(i+1)%k], Train=残りk-2個`)。
    各ラウンドの Test は学習・早期終了判定に一切使わず、保存されたベストモデルで推論する。
    k ラウンド終了後、全foldがTestを1回ずつ経験し、全件が重複なくOOF予測を持つ。
  - `simple`（旧方式）: `run_oof_kfold()` / `run_dual_oof_kfold()`。leave-one-fold-out、
    検証セット=OOF対象を兼用。早期終了がそのfoldに最適化されるわずかな楽観性がある。
- config の `backbone` が `custom_dual` ならデュアルブランチ経路（bc16_pw1/bc32_pw8 はこちら）、
  それ以外（`custom` 等）なら単一ブランチ経路。

#### OOF CSV スキーマ（`eyeliner/training/oof.py: OofPrediction`）

| カラム | 内容 |
|---|---|
| `idx` | DB上のサンプルID |
| `image_path` | 画像パス |
| `part_name` | `eyeliner_right` / `eyeliner_left` |
| `sub_key` | `upper` / `lower` |
| `point_index` | 0〜6 |
| `t_true` | manifestの生t（アノテーション値） |
| `t_pred` | OOFモデルによる生t予測値 |
| `fold_idx` | 予測に使ったラウンド番号 |

### 4. `rank_candidates.py`

- 入力: `oof_predictions.csv`（`--oof-csv` で必須指定。bc16_pw1/bc32_pw8 それぞれの OOF を指定する）
- 出力: `outputs/candidates_*.csv`（`rank, idx, ..., residual, error_rate, url` 列、
  `url` は `http://127.0.0.1:8000/?idx={idx}` 形式でannotator直リンク）
- **「多数派」の定義**: t_true分布を `bin_width`（既定0.05）でビニングし、ピークbinの
  半値（半値全幅・FWHM）以上のカウントを持つ連続bin群を多数派とする（`--majority-method fwhm`）。
  固定範囲指定も可能（`--majority-method range --t-min --t-max`）。
- **ランキング指標**: `residual = |t_pred - t_true|`（生t単位の絶対残差）、または
  `error_rate = residual / |t_true|`（相対誤差率）。`--sort-by` で選択。
- `--point-index` で対象point_indexを絞り込み可能（例: 目の中心点のみ）。
- **未実装**: bc16_pw1版・bc32_pw8版それぞれの`candidates_*.csv`を自動で突き合わせる
  （両モデルで共通して上位に出るサンプルだけを抽出する）スクリプトは現状ない。
  当面は2つのCSVを目視で突き合わせる運用とする。
