# EyeContourRefiner

顔ランドマーク検出器が出した目の輪郭点を、目のふち（眼瞼縁 / eyelid margin）の位置へ
補正する CNN モデルの配布パッケージ。目のパッチ画像から、アイラインを引くべき縁の位置を回帰する。
[face_makeup](https://github.com/kurumikurukuri/face_makeup) 向けの ONNX モデル
（`models/eye_contour_refiner.onnx`）と、その参考推論実装・コアライブラリを含む。

**このブランチ（`main`）はリリース用に絞った軽量パッケージです。** データ準備、
アノテーションツール、k-fold 学習、分析・統計検証などの研究開発の全履歴（EXP001〜EXP035）は
[`exp_3`](https://github.com/kurumikurukuri/EyeContourRefiner/tree/exp_3) ブランチを参照。

## セットアップ

```bash
py -3.11 -m venv .venv   # Windows。Unix系は python3.11 -m venv .venv
.venv/Scripts/activate   # Windows。Unix系は .venv/bin/activate
pip install -e .             # 推論・ONNX エクスポートに必要な依存
pip install -e ".[demo]"     # demo_visualize.py を使う場合（mediapipe）
pip install -e ".[dev]"      # テスト実行用
```

Python 3.11 以上が必要。`pip install -r requirements.txt` でも base の依存を入れられる。

## 使い方

```bash
python models/infer_example.py --dry-run                                   # 1点推論の参考実装
python models/demo_visualize.py --image path/to/face.jpg --output out.jpg  # 画像から両目の輪郭を検出・図示
```

モデルの入出力仕様（`t` 値の意味、各入力テンソルの作り方）・精度・再エクスポート方法は
[`models/README.md`](models/README.md) を参照。

## ディレクトリ構成

```
models/          配布モデル（ONNX）、参考推論実装、デモ、学習データの帰属表示
eye_contour/     コアライブラリ — モデル定義、パッチ切り出し、データ読み込み、学習、評価
scripts/         export_dual_onnx.py — fold チェックポイントから配布 ONNX を生成
tests/           テスト（pytest）
```

## テスト

```bash
pytest
```

学習済みチェックポイント（`models/eyeliner_detection_fold*.pt`）と元画像（`data/images/`）は
配布していないため、それらを使う ONNX エクスポートの golden テストは手元に無い環境では skip される。

## ライセンス

MIT License。詳細は [`LICENSE`](LICENSE) を参照。

## 学習データについて

学習に使用した顔画像は [Open Images V7](https://storage.googleapis.com/openimages/web/index.html)
由来で、CC0 または CC BY 2.0 が付与された画像のみを対象に収集している。
著者名・出典URL・ライセンスURLは [`models/DATA_ATTRIBUTION.csv`](models/DATA_ATTRIBUTION.csv)
に収録している（CC BY 2.0 の表示義務に対応）。画像そのものは本リポジトリでは
再配布していない。

CC BY は撮影者の著作権についての許諾であり、写っている被写体本人の肖像権・
プライバシーまで許諾するものではない。本モデルを商用利用・再配布する場合、
必要な法務確認は利用者側の責任で行うこと。
