# EXP{n}: 実験名

## 目的・仮説

## 実験設計・条件

- **config**: `experiments/EXP{n}_xxx/config/xxx.yaml`
- **主要ハイパーパラメータ**: epochs, batch_size, lr, patience など
- **データ**: manifest_train.csv / manifest_test.csv（件数）

## 結果

### 定量評価

| 条件 | MAE (正規化) | MAE (生t換算) | 相関係数 (r) |
|------|-------------|--------------|-------------|
|      |             |              |             |

### エッジケース分析

- tが極小・極大のサンプルに対して正しく予測できているか
- 平均回帰が起きていないか（t_pred の分布を確認）

## 考察

## 結論・次の実験への示唆

- **結論**:
- **次実験で試すこと**:

## 再現情報

- **ブランチ**: `feat/...`
- **実行環境**: Google Colab (GPU: )
- **実行日**: YYYY-MM-DD
