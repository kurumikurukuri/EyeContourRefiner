# eyeliner_detection_dual.onnx

アイライン輪郭検出モデル（dual-branch標準構成）。目のパッチ画像から目のふち
（eyeliner edge）位置を回帰する。

- 由来: `eyeliner_detection_fold{1,2,3}.pt`
  （=[`exp_3`](https://github.com/kurumikurukuri/EyelinerDetector/tree/exp_3) ブランチの
  `experiments/EXP035_pq_jitter/outputs/J4_rot20`、PQ回転摂動±20°augmentation付き
  dual標準構成・k=3 fold）。精度: db_replica固定テストセットで
  pass_std=90.02% / holdout MAE(生t)=0.081145
  （詳細は `exp_3` ブランチの `experiments/EXP035_pq_jitter/EXP035_summary.md` 参照）。
- エクスポート: `scripts/export_dual_onnx.py`（`eyeliner.models.dual_ensemble.DualRawTOnnxModel`）。
  ImageNet正規化・fold平均・branch A/B融合・生t変換をすべてグラフに焼き込み済み。
  エクスポート時に実データ50サンプルで PyTorch参照（`fold_ensemble_raw_t`）との
  数値一致（max diff 1.8e-7）を確認済み。重み単体（.pt 3ファイル）は本リポジトリでは配布しない。手元で `models/` に置くと
  `tests/test_export_dual_onnx.py` のgoldenテストが同じ重みから再エクスポートして検証する。
- opset: 18（torch 2.11 の dynamo エクスポータ）。batch軸は動的。
  重みは単一 .onnx にインライン済み（外部 .data ファイル不要。エクスポートスクリプトが
  自動でインライン化する）。

## t値の意味

P, Q は目のランドマーク点（P=外側, Q=内側）。モデルが予測する `t` は
眼瞼縁（目のふち）の位置 R を表す正規化値:

```
R = t * P + (1 - t) * Q
```

`t = 0` → R = Q（内側寄り）、`t = 1` → R = P（外側寄り）。正常範囲は`[0, 1.2]`
（極端値: `t < 0` or `t > 1.2`）。

## 入力

パッチ画像は生画像から「アイライン推定対象の1点」ごとに、P-Q方向に沿って回転・
切り出ししたもの（`eyeliner.data.patch_utils.extract_patch`参照、詳細は
`infer_example.py`）。branch A/Bで異なる座標系のパッチを使う点に注意。

| 名前 | shape | dtype | 説明 |
|---|---|---|---|
| `patch_a` | (B, 3, 64, 64) | float32 | branch A用パッチ。**生のP,Q座標**でP-Q方向に沿って切り出し、`extract_patch(px,py,qx,qy, patch_size=(64,64), t_min=-2.0, t_max=3.0, perp_scale=2.0)`。値域[0,1]（0-255を255で割ったもの、ImageNet正規化などは不要） |
| `patch_b` | (B, 3, 64, 64) | float32 | branch B用パッチ。**eye_scaleでリスケールしたP',Q'座標**で切り出し。`compute_pq_rescale(px,py,qx,qy, t=0, rec={"eye_scale":eye_scale}, pq_rescale_mode="eye_scale", pq_rescale_factor=0.2)`でP',Q'を求めてから`extract_patch(..., patch_size=(64,64), t_min=-0.5, t_max=1.5, perp_scale=2.0)` |
| `is_upper` | (B, 1) | float32 | 上瞼のアイラインなら1.0、下瞼なら0.0 |
| `px`, `py` | (B, 1) | float32 | ランドマーク点Pの**元の**画像ピクセル座標（リスケール前） |
| `qx`, `qy` | (B, 1) | float32 | ランドマーク点Qの**元の**画像ピクセル座標（リスケール前） |
| `eye_scale` | (B, 1) | float32 | 対象の目の全14点（upper7+lower7のP,Q）のbounding box対角長（最小1.0）。`eyeliner.data.patch_utils.compute_eye_scale`参照 |

**注意**: `px,py,qx,qy`はリスケール**前**の元座標を渡すこと（`patch_b`切り出しに使った
リスケール後の座標P',Q'ではない）。グラフ内部で`pq_dist = sqrt((qx-px)^2+(qy-py)^2)`と
`pq_prime_dist = clamp(eye_scale * 0.2, min=1.0)`を計算し、branch Bの出力をこれらを使って
生t空間に変換してからbranch Aの出力と平均する。

`eye_scale`は1つの目（14点）で共通の値。7点それぞれを推論する際は、まず対象の目の
14点の生ランドマーク（推論前・アイライン検出前の初期ランドマーク、通常は顔ランドマーク
検出器の出力）から`eye_scale`を1回計算し、各点の推論で使い回す。

## 出力

| 名前 | shape | dtype | 説明 |
|---|---|---|---|
| `t` | (B, 1) | float32 | 生t値。`R = t*P + (1-t)*Q`で最終座標に変換する（元のP,Q、リスケール前） |

## 使い方

`infer_example.py`（1点だけの自己完結な参考実装、numpy/opencv/onnxruntimeのみ依存。
px,py,qx,qy,eye_scaleが既に分かっている前提）を参照。

```bash
python models/infer_example.py --dry-run   # サンプル値で動作確認
```

**画像を渡すだけで両目の輪郭を検出・図示したい場合**は`demo_visualize.py`を使う
（MediaPipe Face Meshで顔ランドマークを検出し、P,Qペアを自動構成してモデルに通す
エンドツーエンドのデモ）:

```bash
pip install -e ".[demo]"   # mediapipe>=0.10.21,<0.10.30（0.10.30以降は mp.solutions が無く動かない）
python models/demo_visualize.py --image path/to/face.jpg --output result.jpg
```

**注意**: `demo_visualize.py`が使うP,Qペアの組み方（MediaPipe公式の目輪郭16点ループから
隣接点を組にした自前実装）は、学習データ生成時に実際使われた
`face_makeup.constants.EYELINER_CORRECTION_MAP`（本リポジトリ外、face_makeup側で管理）
とは一致しない可能性がある。パイプライン全体の動作確認・可視化デモとしては正しいが、
本番のP,Qペアの厳密な再現ではない（`point_index`はモデルの入力ではないため、
モデル自体の予測精度には影響しない）。

## 再エクスポート方法（モデルを再学習した場合）

```bash
# 現行配布重みの .pt を models/ に置いて作り直す場合
python scripts/export_dual_onnx.py \
    --models-dir models \
    --output models/eyeliner_detection_dual.onnx \
    --verify-manifest <exp_3 の experiments/EXP035_pq_jitter/manifests/manifest_test.csv> \
    --verify-samples 50

# 任意のfold*.ptディレクトリから作り直す場合
python scripts/export_dual_onnx.py \
    --models-dir <fold*.ptが入ったディレクトリ> \
    --output models/eyeliner_detection_dual.onnx \
    --verify-manifest <manifest.csvのパス> --verify-samples 50
```

学習・実験コード（`experiments/`・`annotator/`・データ復元ツール等）は
[`exp_3`](https://github.com/kurumikurukuri/EyelinerDetector/tree/exp_3) ブランチを参照。
