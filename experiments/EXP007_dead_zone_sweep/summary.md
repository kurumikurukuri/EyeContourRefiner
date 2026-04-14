# EXP007: dead_zone_huber_loss sweep（EXP006との損失関数比較）

## 目的

EXP006（H1-H8）と完全に同じ入力条件・モデル構成のまま、損失関数を
`t`（Huber）→ `dead_zone`（デッドゾーン付き Huber）に変更して精度を比較する。

**仮説**: 残差が小さい（≈1px以内）サンプルへの無駄な勾配を排除することで、
mid帯の学習安定性が上がり、端点予測精度が改善する。

## デッドゾーン設定

| パラメータ | 値 | 意味 |
|---|---|---|
| `loss_epsilon` | 0.01 | デッドゾーン幅（正規化 t スケール ≈ 0.64px） |
| `loss_delta` | 0.05 | 二次→線形の遷移点 |

損失の挙動:
```
|残差| < 0.01           → loss = 0（勾配なし）
0.01 ≤ |残差| < 0.06   → 二次項（Huber 滑らか部）
|残差| ≥ 0.06          → 線形項（外れ値ロバスト）
```

## ベースライン（EXP006）

| 条件 | Holdout 生t MAE | 備考 |
|------|-----------------|------|
| H1_baseline | - | 未記入（EXP006結果を参照） |
| H2_wide_patch | - | 未記入（EXP006結果を参照） |

## 条件対応表

| EXP007 条件 | 対応 EXP006 条件 | 変更点 |
|---|---|---|
| I1_baseline | H1_baseline | loss_fn: t → dead_zone |
| I2_wide_patch | H2_wide_patch | loss_fn: t → dead_zone |


manifests: EXP006 の manifests を共有（同一 train/test 分割で比較を保証）

## 実行手順

```bash
# EXP006 の manifests を使用するため split_holdout は不要

# 全条件を実行
python scripts/sweep.py --exp 007

# 2グループ並行実行
python scripts/sweep.py --exp 007 --configs I1_baseline I2_wide_patch I3_deep &
python scripts/sweep.py --exp 007 --configs I4_sampler I5_wide_patch_short I6_wide_patch_w I7_wide_patch_scale I8_wide_patch_w_scale &
```

