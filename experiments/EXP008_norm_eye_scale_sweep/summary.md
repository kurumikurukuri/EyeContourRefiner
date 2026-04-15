# EXP008: norm_eye_scale sweep

## 目的

`norm_eye_scale`（画像解像度正規化済み eye_scale = eye_scale / sqrt(img_H × img_W)）を
モデルのスカラー補助入力として追加したときの効果を検証する。

**仮説**: 目のサイズ情報を与えることで、小さい目での t 予測精度が改善する。
wide patch range 構成（H2）との組み合わせが最も恩恵を受けやすい。

## 条件

| 条件 | ベースライン | norm_eye_scale | patch 範囲 |
|------|-------------|---------------|-----------|
| J1_baseline_nes | EXP006 H1_baseline | ✓ | -0.5〜1.5 |
| J2_wide_patch_nes | EXP006 H2_wide_patch | ✓ | -1.0〜2.0 |

manifests: EXP006 の manifests を共有（同一 train/test 分割で比較を保証）

## 比較ベースライン（EXP006）

| 条件 | CV MAE | Holdout MAE | 生t MAE |
|------|--------|-------------|---------|
| H1_baseline | 0.05835 | 0.06622 | 0.1324 |
| H2_wide_patch | 0.05641 | 0.06332 | 0.1266 |

## 実行手順

```bash
# 2グループ並行実行
python scripts/sweep.py --exp 008 --configs J1_baseline_nes &
python scripts/sweep.py --exp 008 --configs J2_wide_patch_nes &
```

## 結果

| 条件 | CV MAE | Holdout MAE | 生t MAE | EXP006比 |
|------|--------|-------------|---------|---------|
| J1_baseline_nes | - | - | - | vs H1 |
| J2_wide_patch_nes | - | - | - | vs H2 |

## 考察

（実行後に記入）
