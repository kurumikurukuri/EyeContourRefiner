# EXP{n}: 実験名

## 目的・仮説

## 実験設計・条件

- **config**: `experiments/EXP{n}_xxx/config/xxx.yaml`
- **主要ハイパーパラメータ**: epochs, batch_size, lr, patience など
- **固定する条件 / 変える条件**: 何を固定し、どの軸をスイープするか
- **データ**: manifest_train.csv / manifest_test.csv（件数、train/holdout分割方法）
- **評価指標**: cv_mae, holdout_mae(norm/raw t), passrate(std/strict)。閾値は`experiments/Completion_Criteria.md`準拠
- **検定方法**: scipy paired t-test（比較対象との abs error 差を帯ごとに比較）

## 結果（hold-out, n=）

### 定量評価

| 条件 | (スイープ軸の値...) | cv_mae | holdout_mae(norm) | holdout_mae(raw t) | passrate(std) | passrate(strict) |
|---|---|---|---|---|---|---|
|   |   |   |   |   |   |   |

## 考察

- 全体MAE・passrateで見た優劣
- どの条件がどう効いているか（容量/分解度/損失/augなど軸ごとの傾向）

## t帯別結果

- tが極小・極大のサンプルに対して正しく予測できているか
- 平均回帰が起きていないか（t_pred の分布を確認）

| band | n | 条件1 | 条件2 | ... | baseline |
|---|---|---|---|---|---|
| t<0 |  |  |  |  |  |
| 0-0.3 |  |  |  |  |  |
| 0.3-0.7 |  |  |  |  |  |
| 0.7-1.0 |  |  |  |  |  |
| 1.0-1.2 |  |  |  |  |  |
| t>=1.2 |  |  |  |  |  |
| ALL |  |  |  |  |  |

## t帯別結果考察

- 少数派の極端t帯と多数派帯でのトレードオフの有無

## ペア比較(paired t-test)
ベースラインと採用候補条件を比較
\* = paired t-test で p<0.05。

### 条件A vs baseline

| band | n | MAE(条件A) | MAE(base) | diff | t_p |
|---|---|---|---|---|---|
| t<0     |   |   |   |   |   |
| 0-0.3   |   |   |   |   |   |
| 0.3-0.7 |   |   |   |   |   |
| 0.7-1.0 |   |   |   |   |   |
| 1.0-1.2 |   |   |   |   |   |
| t>=1.2  |   |   |   |   |   |
| **ALL** |   |   |   |   |   |

## ペア比較考察

- 帯ごとの有意差／非有意の解釈。ALLでn.s.でも帯別に有意差がある場合はその意味を明記

## 結論

- 採用/見送りの判断とその根拠（数値上の優劣だけでなく、用途・価値判断も明記）

## 今後の課題

- 今回検証できなかった軸・交互作用
- 次実験で試すこと