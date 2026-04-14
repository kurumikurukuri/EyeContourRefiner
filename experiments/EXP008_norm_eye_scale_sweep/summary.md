# EXP008: norm_eye_scale 特徴量 sweep

## 目的

画像解像度で正規化した eye_scale（`eye_scale / sqrt(img_H * img_W)`）を
モデルのスカラー入力として追加したときの精度向上効果を確認する。

**仮説**: 画像間で解像度が異なる場合、生の eye_scale は絶対ピクセル数に依存するため
直接比較が困難。解像度で正規化することで、モデルが「目が画像全体の何割を占めるか」
という相対的スケール情報を学習でき、汎化性能が向上する可能性がある。

## 特徴量の定義

```
norm_eye_scale = eye_scale / sqrt(img_H * img_W)
```

- `eye_scale`: 同一画像・同一 part_name の全 P/Q 点群のバウンディングボックス対角長
- `sqrt(img_H * img_W)`: 画像の幾何平均解像度（対角長の近似）

## 条件

| 条件 | 対応ベースライン | 変更点 |
|---|---|---|
| J1_baseline_nes | EXP006 H1_baseline | use_norm_eye_scale: true を追加 |

manifests: EXP006 の manifests を共有（同一 train/test 分割で比較を保証）

## 実行手順

```bash
# EXP006 の manifests を使用するため split_holdout は不要

# 全条件を実行
python scripts/sweep.py --exp 008
```

## 結果

| 条件 | CV MAE | Holdout 生t MAE | 備考 |
|------|--------|-----------------|------|
| J1_baseline_nes | - | - | 実行待ち |
| H1_baseline（参考） | - | - | EXP006 参照 |

## 考察

（実行後に記入）
