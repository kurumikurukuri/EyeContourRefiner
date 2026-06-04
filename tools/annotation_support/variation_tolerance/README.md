# variation_tolerance — アノテーター許容幅キャリブレーション & pass_rate 評価

[`experiments/Completion_Criteria.md`](../../../experiments/Completion_Criteria.md) の完了判定を支える 2 つのツールを提供するディレクトリ。

| スクリプト | 役割 |
|------------|------|
| `server.py` | **キャリブレーション Web ツール** — 目視で t の知覚的許容幅をマークし、合格閾値を実測導出する |
| `passrate.py` | **pass_rate 評価ツール** — 導出した閾値で全実験の予測を採点し、実測表を生成する |

`error_ratio` / `pass_rate` の定義と採用閾値の使い道は
[Completion_Criteria.md §2 判定指標](../../../experiments/Completion_Criteria.md#2-判定指標) が正本。
本 README は **閾値の導出過程**と**評価ツールの使い方**を扱う。

---

## ツール1: キャリブレーション (`server.py`)

### 目的

CNN の評価指標

```text
error_ratio = |t_pred_raw - t_true_raw| × pq_dist / eye_scale
```

の合格閾値を、アノテーター自身の知覚的許容幅から実測で導出する。
実際の目パッチを目視確認しながら t の許容幅（下限 L・上限 U）をマークし、
`ratio = pixel_range / eye_scale` の分布統計を得る。

### 起動

プロジェクトルートから実行:

```bash
uvicorn tools.annotation_support.variation_tolerance.server:app --port 8001
# → http://127.0.0.1:8001/
```

### 操作手順

1. **サンプル読み込み** — 起動時に 100 件（`upper`, `point_index=3`, `eye_scale > 60`）をランダム選択。
   前回セッションが `calibration_state.json` に保存されていれば自動復元される。

2. **L/U マーク** — 各サンプルで t スライダーを動かし、
   「これ以上ずれると目のふちを外れると感じる」下限 (L) と上限 (U) をマーク。
   - **パッチビュー**（左）: t 軸上の 1D 位置を可視化。緑=現在 t、黄=L、マゼンタ=U、青=P、赤=Q
   - **眼領域ビュー**（右）: 実際の目画像上で R 点の動きを目視確認

3. **中間保存** — "Save" ボタンで `calibration_state.json` に保存。ブラウザを閉じても復元可能。

4. **集計** — 全件マーク後に "Submit" → 各サンプルの `ratio` と統計サマリーが表示される。
   ratio は次式で算出:

   ```text
   ratio = |t_upper_norm - t_lower_norm| × T_RANGE × pq_dist / eye_scale
   ```

5. **CSV エクスポート** — "Export CSV" で `calibration_results.csv` をダウンロード。

### 出力ファイル

| ファイル | 内容 |
|----------|------|
| `calibration_state.json` | セッション状態（idx リスト＋中間マーク値）。中断再開用 |
| `calibration_results.csv` | 各サンプルの `idx`, `pixel_range`, `eye_scale`, `ratio` など |

Submit レスポンスの `summary` フィールドに count / mean / median / std / p75 / p90 / p95 / min / max が返る。

### 設定値（`server.py`）

| 定数 | 値 | 意味 |
|------|----|------|
| `NUM_SAMPLES` | 100 | 計測サンプル数 |
| `FILTER_EYE_SCALE_MIN` | 60 | 小さい目を除外する閾値 [px] |
| `DISPLAY_SCALE` | 6 | パッチ表示倍率 |

### キャリブレーション品質検証

キャリブレーションに使用した全サンプルについて、

> 元のアノテーション t が、アノテーター自身の設定した許容帯 [t_lower, t_upper] の内部に存在するか

を検証した。これは、許容帯中心を真値とみなし、その半幅（÷2）を誤差許容値として利用できるかを確認するための検証である。

| 指標 | 値 | 解釈 |
|--------|--------|--------|
| 帯内率 | 71 / 100 | 29% は帯外 |
| 帯内位置平均 | 0.315 | 下端寄り |
| 帯内位置中央値 | 0.371 | 下端寄り |
| 帯より下にはみ出し | 22件 | 下方向への偏り |
| 帯より上にはみ出し | 7件 | 上方向への偏り |
| 帯内位置標準偏差 | 0.51 | 大きなばらつき |

**結論:**

1. アノテーションは許容帯中央に位置していない。
2. 許容帯を対称な誤差モデルとして扱う根拠は弱い。
3. 29% のサンプルで元アノテーション自体が帯外であり、アノテーションノイズが無視できない。
4. このため、本プロジェクトでは許容帯半幅（÷2）ではなく、**許容帯幅そのもの**を閾値として採用する。

### 導出された閾値（ratio 分布）

上記キャリブレーションで得た ratio 分布。
[Completion_Criteria.md §2.3](../../../experiments/Completion_Criteria.md#23-採用閾値) はこの下位パーセンタイル
（**p10 = 標準 0.01788 / p5 = 厳格 0.01604**）をそのまま採用閾値とする。

条件: N = 100, `upper`, `point_index = 3`, `eye_scale > 60`

| 統計量 | ratio |
|--------|--------|
| min | 0.0104 |
| p5 | 0.0160 |
| p10 | 0.0179 |
| p25 | 0.0214 |
| median | 0.0251 |
| mean | 0.0281 |
| p75 | 0.0313 |
| p90 | 0.0393 |
| p95 | 0.0529 |
| max | 0.0924 |

詳細データ: `calibration_results.csv`

---

## ツール2: pass_rate 評価 (`passrate.py`)

導出した閾値で全実験条件の予測 CSV を採点し、
[Completion_Criteria.md §3 実測値](../../../experiments/Completion_Criteria.md#3-実測値) の表を生成する。

### 起動

プロジェクトルートから実行:

```bash
python tools/annotation_support/variation_tolerance/passrate.py

# 特定実験の t 帯別 pass rate を表示
python tools/annotation_support/variation_tolerance/passrate.py --band EXP021
```

`experiments/*/outputs/**/test_predictions.csv` を走査し、各点の error_ratio を
DB（`data/samples.db`）の PQ・eye_scale を用いて算出する。
idx が DB に存在しない行は除外し、警告として一覧表示する。

### 出力ファイル（`outputs/`）

| ファイル | 内容 |
|----------|------|
| `pass_rate_summary.csv` | 全 (exp, condition) の pass_std / pass_strict |
| `best_per_exp.md` | EXP ごとの pass_std 最良条件（→ Completion_Criteria.md §3 へ貼付） |
| `band_pass_rate_{EXP}.csv` | `--band` 指定時の t 帯別 pass rate |

### 閾値（`passrate.py` 冒頭定数）

| 定数 | 値 | 由来 |
|------|----|------|
| `THRESHOLD_STD` | 0.01788 | p10（標準） |
| `THRESHOLD_STRICT` | 0.01604 | p5（厳格） |

閾値を変更する場合は、本ファイルの定数と Completion_Criteria.md §2.3 の両方を更新すること。
