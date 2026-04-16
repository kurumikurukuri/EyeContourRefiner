# EXP012: パッチサイズ格子 sweep

## 目的

`patch_t_min × patch_t_max × patch_perp_scale × patch_h × patch_w` の全組み合わせ (2×2×3×3×3 = **108条件**) を探索し、パッチ抽出パラメータのベスト構成を特定する。

## 条件

| パラメータ | 値 |
|-----------|-----|
| `patch_t_min` | -1.0, -1.25 |
| `patch_t_max` | 2.0, 2.25 |
| `patch_perp_scale` | 0.5, 1.0, 2.0 |
| `patch_h` | 32, 64, 128 |
| `patch_w` | 64, 128, 256 |

### グループ

| グループ | patch_t_min | patch_t_max |
|---------|-------------|-------------|
| A | -1.0 | 2.0（ベースライン） |
| B | -1.0 | 2.25 |
| C | -1.25 | 2.0 |
| D | -1.25 | 2.25 |

### ベースライン固定設定（EXP011 M0 相当）

```
backbone: custom / loss_fn: t / use_sigmoid: true / use_batchnorm: true
num_conv_layers: 3 / lr: 1e-4 / weight_decay: 1e-4
epochs: 100 / batch: 32 / k: 1 / patience: 25 / seed: 42
aug_flags: 79 / label_t_min: -0.5 / label_t_max: 1.5
```

## 実行方法

```bash
# manifest 準備（EXP011 から流用）
cp experiments/EXP011_hparam_sweep/manifests/*.csv \
   experiments/EXP012_patch_grid_sweep/manifests/

# 並行実行（GPU 2枚推奨）
python experiments/EXP012_patch_grid_sweep/scr/train_exp012.py --group A B  # GPU1
python experiments/EXP012_patch_grid_sweep/scr/train_exp012.py --group C D  # GPU2

# または sweep.py 経由
python scripts/sweep.py --exp 012
```

## 結果

（実験後に記入）

## 考察

（実験後に記入）
