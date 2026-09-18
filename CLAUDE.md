# EyelinerDetector — アイライン輪郭検出 CNN の R&D パイプライン

CNN で目のパッチ画像から目のふち（eyeliner edge）を回帰し、[face_makeup](https://github.com/kurumikurukuri/face_makeup) 向けモデルを生成するプロジェクト。


---

## 環境

- **Python**: 3.11
- **仮想環境**: `.venv/`
- **パッケージ管理**: `pip` + `requirements.txt`
- **主要依存**: PyTorch, ONNX, FastAPI, OpenCV, scikit-learn, XGBoost, MediaPipe

```bash
pip install -r requirements.txt
pip install xgboost optuna joblib tqdm  # 分析用追加依存
```

---

## ディレクトリ構成

```
eyeliner/        コアライブラリ — モデル定義、データ読み込み、学習、評価、augmentation
  data/           dataset, augmentation, patch_utils, weights
  models/         CNN アーキテクチャ (eyeliner_cnn, dual_predictor, ensemble, soft_argmax)
  training/       kfold, trainer, scheduler, checkpoint, oof
  evaluation/     loss, metrics
annotator/       Web アノテーション UI (FastAPI server + dashboard + static/)
scripts/         実験パイプライン共通スクリプト — train/infer/sweep、評価(compute_passrate/paired_ttest/generate_summary)、
                 manifest分割(split_holdout)、ONNX変換等。ステップ2〜6で使用（pyproject.toml で正式パッケージ登録）
experiments/     CNN 学習実験（EXP{NNN}）— 各実験は独立
  EXP{NNN}/       config/, manifests/, outputs/, scr/, EXP{NNN}_summary.md
analyses/        分析・統計分析(XGBoost等)
tools/           DB メンテナンス、CSV エクスポート、アノテーション支援、画像管理
tests/           テスト
data/            samples.db（アノテーション DB）+ 画像ファイル（.gitignore 対象）
```

---

## コアコンセプト

### t 値（目のふちの幅）

- P, Q は目のランドマーク点。P が外側、Q が内側
- `t` は眼瞼縁（目のふち）の位置 R を表す正規化値: `R = t * P + (1 - t) * Q`
- `t = 0` → R = Q（内側寄り）、`t = 1` → R = P（外側寄り）
- 正常範囲: `[0, 1.2]`、極端値: `t < 0` or `t > 1.2`
- 学習時の正規化: `t_norm ∈ [0, 1]`

### データベース

- `data/samples.db` (SQLite) — **唯一の真実 (source of truth)**
- 主要カラム: `idx`, `image_path`, `part_name` (eyeliner_right/left), `sub_key` (upper/lower), `point_index` (0-6), `px/py/qx/qy`, `t`, `skip_flag`, `face_yaw/pitch/roll`, `eye_scale`, `norm_eye_scale`, `lap_var`
- 眼単位: 1画像あたり最大 2 眼 (left, right)。1 眼 = upper 7 点 + lower 7 点 の計 14 点（`part_name` で左右、`sub_key` で upper/lower を区別）

---

## 実験設計ルール

### 実験間の独立性（CRITICAL）

各実験は **完全に独立** していること。

- 実験間でコードやデータを直接 import / 参照しない
- 共通処理は `eyeliner/` コアライブラリに実装し、各実験スクリプトはそれを利用する
- ある EXP の outputs を別の EXP の scr/ が読み込む構成は禁止（パイプラインを組む場合は tools/ 経由）
- `sys.path` 操作で他実験フォルダをパスに追加するのも禁止

### 命名規則

| 対象 | 規則 | 例 |
|------|------|----|
| 実験フォルダ | `EXP{NNN}_{snake_case}` | `EXP027_perspective_aug` |
| サマリー | `EXP{NNN}_summary.md` | `EXP027_summary.md` |
| 訓練スクリプト | `train_exp{NNN}.py` | `train_exp027.py` |
| 推論スクリプト | `infer_exp{NNN}.py` | `infer_exp027.py` |
| 設定ファイル | `{condition_name}.yaml` | `fcp_p8.yaml` |
| manifest | `manifest_{split}.csv` | `manifest_train.csv` |

---

## 実験パイプライン

実験は以下の 6 ステップで進行する。各ステップに対応するスクリプトと実行コマンドを示す。

### ステップ 1: 実験企画

- 仮説・目的・過去実験との差分を明確にする
- 必要な場合 `architect` で設計判断、`planner` で実装計画を立てる
- 実験番号 `NNN` を確定し、ディレクトリ名 `EXP{NNN}_{snake_case}` を決める

### ステップ 2: 実験追加（ディレクトリ・設定・manifest 準備）

```bash
# 2a. 実験ディレクトリを作成
mkdir -p experiments/EXP{NNN}_{name}/{config,manifests,outputs,scr}

# 2b. train/infer スクリプトを作成（過去実験からコピーして改変）
# → experiments/EXP{NNN}_{name}/scr/train_exp{NNN}.py
# → experiments/EXP{NNN}_{name}/scr/infer_exp{NNN}.py

# 2c. 設定 YAML を作成
# → experiments/EXP{NNN}_{name}/config/{condition_name}.yaml

# 2d. manifest を生成（DB から train/test 分割）
python scripts/split_holdout.py --exp {NNN} --ratio 0.40 --seed 42

# 2e. summary テンプレートを初期化
python scripts/init_exp_summary.py --exp {NNN} --title "実験タイトル"
```

### ステップ 3: 学習開始

```bash
# 単一 config の場合
python experiments/EXP{NNN}_{name}/scr/train_exp{NNN}.py

# 複数 config を一括実行（sweep）
python scripts/sweep.py --exp {NNN}
python scripts/sweep.py --exp {NNN} --configs cond_A cond_B

# 実験固有スクリプトを直接指定
python experiments/EXP{NNN}_{name}/scr/train_exp{NNN}.py --configs cond_A
```

学習が完了すると `outputs/{config}/` 以下に fold モデル (`.pt`) と `experiment.json` が生成される。

### ステップ 4: 推論

```bash
# 単一 config
python experiments/EXP{NNN}_{name}/scr/infer_exp{NNN}.py

# 実験固有スクリプト
python experiments/EXP{NNN}_{name}/scr/infer_exp{NNN}.py --configs cond_A

# 汎用 infer スクリプト
python scripts/infer.py \
  --models-dir experiments/EXP{NNN}_{name}/outputs/cond_A \
  --manifest experiments/EXP{NNN}_{name}/manifests/manifest_test.csv \
  --output experiments/EXP{NNN}_{name}/outputs/cond_A/test_predictions.csv
```

推論が完了すると `outputs/{config}/` 以下に `predictions.csv`（または `test_predictions.csv`）が生成される。

### ステップ 5: paired t-test + passrate（定量評価）

```bash
# 5a. passrate 計算（全 config 一括）
python scripts/compute_passrate.py --exp EXP{NNN}
python scripts/compute_passrate.py --exp EXP{NNN} --configs cond_A cond_B

# 5b. paired t-test（baseline 対 全 config）
python scripts/paired_ttest.py --exp EXP{NNN} --baseline cond_A

# baseline と特定 config のみ比較
python scripts/paired_ttest.py --exp EXP{NNN} --baseline cond_A --targets cond_B

# error_ratio metric で比較（デフォルトは raw_t）
python scripts/paired_ttest.py --exp EXP{NNN} --baseline cond_A --targets cond_B --metric error_ratio
```

**評価指標**:
| 指標 | 説明 | スクリプト |
|------|------|-----------|
| MAE | Mean Absolute Error（t 値単位） | experiment.json に記録 |
| Passrate (Standard) | error_ratio ≤ 0.01788 の割合 | `compute_passrate.py` |
| Passrate (Strict) | error_ratio ≤ 0.01604 の割合 | `compute_passrate.py` |
| Paired t-test | per-sample 誤差の統計的検定（全体 + t帯別） | `paired_ttest.py` |


### ステップ 6: summary 生成

```bash
# 6a. 定量部分を自動生成（MAE, passrate, t帯別, t-test 結果）
python scripts/generate_summary.py --exp EXP{NNN}
python scripts/generate_summary.py --exp EXP{NNN} --baseline cond_A

# 既存 summary を上書き再生成
python scripts/generate_summary.py --exp EXP{NNN} --force
```

`generate_summary.py` は定量データを自動入力し、考察・結論セクションは `<!-- TODO -->` プレースホルダとして残す。

**6b. 手動追記**: 生成された `EXP{NNN}_summary.md` に以下を追記する:
- 実験の目的・仮説
- 結果の考察（なぜその結果になったか）
- 結論・次のステップ
- 必要に応じて `paired_ttest.py` の出力 JSON から詳細な t帯別比較を転記

---

## 規約

### 結果表示

分析結果のサンプル表示は annotator で直接開ける URL 形式:

```
http://127.0.0.1:8000/?idx=xxx
```

- annotator が起動中であればクリックで該当サンプルを開ける
- 対称の目 (eyeliner_right ↔ eyeliner_left) が必要な場合は明示的に「対称 upper[0]」等と併記

### データ分析コードの配置

考察に用いるデータ分析コードは各 scr ディレクトリ（experiments または analyses）に格納すること。


### コードスタイル

グローバルルール `~/.claude/rules/common/coding-style.md` に従う:
- **不変性（immutability）**: 常に新しいオブジェクトを返し、既存オブジェクトを変更しない
- **小さなファイル**: 200-400 行が標準、最大 800 行
- **エラー処理**: すべてのエラーを明示的に処理し、飲み込まない
- **入力検証**: システム境界で常に検証する

### Git コミット

- Conventional Commits に従う。ただしbodyは書かない
- フォーマット: `<type>: <description>`
- タイプ: feat, fix, refactor, docs, test, chore, perf, ci

---
