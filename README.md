# EyelinerDetector

CNN で目のパッチ画像から目のふち（eyeliner edge）位置を回帰するモデルの研究開発（R&D）
パイプライン。[face_makeup](https://github.com/kurumikurukuri/face_makeup) 向けの
ONNX モデルを生成する。

**このブランチ（`exp_3`）は研究・実験・学習の全履歴を持つ本体です。** データ準備、
アノテーションツール、k-fold 学習、augmentation、分析・統計検証などすべての実験
（EXP001〜EXP032）がここにある。本番配信用に絞った軽量パッケージは
[`main`](https://github.com/kurumikurukuri/EyelinerDetector/tree/main) ブランチを
参照。

## セットアップ

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows。Unix系は .venv/bin/activate
pip install -r requirements.txt
pip install xgboost optuna joblib tqdm  # 分析用追加依存
```

Python 3.11 以上が必要（`pyproject.toml` 参照）。

## ディレクトリ構成

```
eyeliner/        コアライブラリ — モデル定義、データ読み込み、学習、評価、augmentation
annotator/       Web アノテーション UI（FastAPI server + dashboard）
experiments/     CNN 学習実験（EXP{NNN}）— 各実験は独立
analyses/        分析・統計分析（XGBoost 等）
tools/           DB メンテナンス、CSV エクスポート、アノテーション支援、画像管理
scripts/         実験パイプライン共通スクリプト（split_holdout, sweep, infer, passrate, t-test 等）
tests/           テスト（pytest）
data/            samples.db（アノテーション DB）+ 画像ファイル（.gitignore 対象、非配布）
```

実験の企画から評価・summary生成までの一連のワークフロー、DB スキーマ、
`t` 値の定義など運用上の詳細は [`CLAUDE.md`](CLAUDE.md) を参照。

## テスト

```bash
pytest tests/ -m "not requires_data"
```

`requires_data` マーカー付きのテストは `data/samples.db` 等の大容量データ
（.gitignore 対象、非配布）を必要とするため、手元に無い環境ではスキップされる。

## ライセンス

MIT License。詳細は [`LICENSE`](LICENSE) を参照。

## 学習データについて

学習に使用した顔画像は [Open Images V7](https://storage.googleapis.com/openimages/web/index.html)
由来で、CC0 または CC BY 2.0 が付与された画像のみを対象に収集している。
著者名・出典URL・ライセンスURLは [`data/open_images_attribution.csv`](data/open_images_attribution.csv)
に収録している（CC BY 2.0 の表示義務に対応）。画像そのものは本リポジトリでは
再配布していない。

CC BY は撮影者の著作権についての許諾であり、写っている被写体本人の肖像権・
プライバシーまで許諾するものではない。本モデルを商用利用・再配布する場合、
必要な法務確認は利用者側の責任で行うこと。
