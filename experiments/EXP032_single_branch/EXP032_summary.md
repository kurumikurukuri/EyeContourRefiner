# EXP032: 単独ブランチ（dual fusionなし）現行ベストプラクティス再実験

## 目的・仮説

EyelinerDetector は EXP020 で dual-branch アーキテクチャ（生PQ座標 branch A + eye_scale
正規化 branch B の出力平均）を採用して以降、EXP021〜EXP031 の改善（t_l1損失、sigmoid
出力ヘッド、aug_flags=63のphotometricブロック、k=3/batch=32、OOF誤ラベル除去(EXP023)・
label noise対応(EXP028)によるDBクリーニング、db_replica固定テストセット(EXP029)）は
**すべてdual-branchモデルの上でのみ**検証・採用されてきた。単独ブランチ（dual fusion
なしの単一CNN）が最後に実際のEXPで使われたのはEXP018で、EXP019のX2_eye_scale
（pq_rescale_mode=eye_scale）が単独座標系として最良だった（raw PQ比 -1.5%）。

**仮説**: 単独ブランチ構成（branch A相当=生PQ、branch B相当=eye_scale）を現行の
ベストプラクティス・現行データセット・db_replica評価プロトコルで再学習すれば、
EXP020当時判明していた「dual-branchの優位性」がどの程度縮小/温存されるかを
定量化できる。dual-branchの推論コスト（単独ブランチのほぼ2倍）が今も正当化されるかの
判断材料とする。

**追記（推論性能ベンチマークの過程で発見・修正した設計変更）**: 精度比較とは別に
`tools/model_benchmark/`でPyTorch/ONNX推論性能を比較したところ、単独ブランチ
（`eyeliner/models/eyeliner_cnn.py`の`EyelinerCNN`）はヘッドの`AdaptiveAvgPool2d`の
出力幅がハードコード(32)されており、64×64パッチの実際の特徴マップ幅(8)と噛み合わず
**ONNXエクスポートが不可能**（`Unsupported: ONNX export of operator adaptive_avg_pool2d,
output size that are not factor of input size`）というバグが判明した。dual標準モデルは
同種の`AdaptiveAvgPool2d`ヘッドだが`pool_width`がパラメータ化されており、現行標準config
（`pool_width=8`）がたまたま特徴マップ幅と一致してエクスポートが通っていた——単独ブランチ
だけが取り残されていた形。ユーザー判断により、`eyeliner_cnn.py`に`pool_width`パラメータを
追加（dual側と同じ設計、デフォルト32で既存チェックポイントとの後方互換を維持）し、
`pool_width=8`で**SB_none・SB_eyescaleを再学習**した。本サマリーの数値は**すべて
`pool_width=8`版（再学習後）**のもの。

## 実験設計・条件

- **実験ディレクトリ**: `experiments\EXP032_single_branch`
- **config 数**: 2
- **config 一覧**: SB_eyescale, SB_none
- **評価指標**: cv_mae, holdout_mae(raw t), passrate(std/strict)
- **passrate 閾値**: std=0.01788 (p10), strict=0.01604 (p5)
- **検定方法**: scipy paired t-test（per-sample abs error）

**固定条件**（現行dual標準ベースライン=EXP028/029/030/031で一致確認済み構成をそのまま流用）:
`loss_fn=t_l1, use_sigmoid=true, use_batchnorm=true, num_conv_layers=4, dropout_rate=0.3,
lr=1e-3, weight_decay=1e-4, epochs=200(patience=50), batch=32, k=3, seed=42, aug_flags=63,
endpoint_blend_range=0.25, use_is_upper=true, pool_width=8`。`pool_width=8`はdual標準の
現行設定と同一値（ONNX互換性のために揃えた。詳細は「推論性能」章参照）。manifest は
EXP029 の `db_replica` 固定train/test（train=16,548点 / test=4,900点）をそのままコピーして
使用（新規split不要 — 現在のDB総アノテーション点数21,448はdb_replica生成時からほぼ増えて
いないことを確認済み）。

**変動軸**: `pq_rescale_mode`（dual fusionを外した単独ブランチの座標系）のみ。
- `SB_none`: `pq_rescale_mode=none`（生PQ、dual branch A相当。EXP019 X0_baseline相当）
- `SB_eyescale`: `pq_rescale_mode=eye_scale, factor=0.2`（dual branch B相当。EXP019
  X2_eye_scale相当、単独座標系としての旧ベスト）

patch_t_min/max, label_t_min/max, patch_perp_scale, patch_size（64×64）は
`eyeliner/training/dual_branch.py` の `branch_a_x0()` / `branch_b_eye_scale()` プリセットを
そのまま使用（dual側とズレないようにするため）。`SB_eyescale` の予測（P'Q'座標系のt'）は
`branch_pred_to_raw_t()` で生tへ逆変換してから評価（dual推論・EXP019のtest_mae評価と同じ変換）。

**比較対象（別実験・読み取り専用参照）**: 現行dual標準ベースライン
`experiments/EXP031_source_space_aug/outputs/SD0_baseline`（EXP028/029/030/031で一致確認済み
構成の最新再現確認済み結果、test MAE(生t)=0.084326、pass_std=89.04%）。`scripts/paired_ttest.py
--exp`は同一実験ディレクトリ内のconfigしか解決できないため、EXP024の先例
（`compare_exp024_vs_exp022.py`）に倣い `scr/compare_exp032_vs_dual_baseline.py` で
test_predictions.csvを直接読み取り専用参照して比較した（EXP032の学習/推論パイプライン自体は
他EXPのコード・manifestを一切importしない）。

## 結果（hold-out）

### 定量評価

| config | n_valid | cv_mae | holdout_mae(raw t) | pass_std% | pass_strict% |
|---|---|---|---|---|---|
| SB_eyescale | 4900 | 0.032254 | 0.090065 | 87.98 | 84.35 |
| SB_none | 4900 | 0.030990 | 0.087731 | 87.84 | 84.96 |

### t帯別 pass_std (%)

| config | t<0 | 0-0.3 | 0.3-0.7 | 0.7-1.0 | 1.0-1.2 | t>=1.2 | ALL |
|---|---|---|---|---|---|---|---|
| SB_eyescale | 65.9 | 73.5 | 90.3 | 89.7 | 86.0 | 73.6 | 87.98 |
| SB_none | 60.5 | 72.6 | 91.9 | 89.5 | 83.5 | 59.8 | 87.84 |

  n: t<0=129 0-0.3=219 0.3-0.7=1684 0.7-1.0=2430 1.0-1.2=351 t>=1.2=87

### t帯別 mae_raw_t

| config | t<0 | 0-0.3 | 0.3-0.7 | 0.7-1.0 | 1.0-1.2 | t>=1.2 | ALL |
|---|---|---|---|---|---|---|---|
| SB_eyescale | 0.290551 | 0.166439 | 0.088284 | 0.071165 | 0.089730 | 0.164239 | 0.090065 |
| SB_none | 0.322038 | 0.173202 | 0.075929 | 0.069939 | 0.096053 | 0.216997 | 0.087731 |

### 参考: 現行dual標準ベースライン（EXP031 SD0_baseline、読み取り専用参照）

| config | t<0 | 0-0.3 | 0.3-0.7 | 0.7-1.0 | 1.0-1.2 | t>=1.2 | ALL |
|---|---|---|---|---|---|---|---|
| dual標準(SD0_baseline) mae_raw_t | 0.273446 | 0.164433 | 0.076088 | 0.067414 | 0.091970 | 0.203235 | 0.084326 |
| dual標準(SD0_baseline) pass_std% | — | — | — | — | — | — | 89.04 |

## 考察

**pool_width=8での再学習は精度に実質影響しなかった**: 参考として、最初にpool_width=32
（デフォルト、ONNX非互換）で学習した際の結果はSB_none cv_mae=0.031060/pass_std=87.76%、
SB_eyescale cv_mae=0.032326/pass_std=87.31%だった。pool_width=8再学習後はSB_none
cv_mae=0.030990/pass_std=87.84%、SB_eyescale cv_mae=0.032254/pass_std=87.98%と、
いずれも誤差数十分の1ポイントの範囲で一致している。regressorの入力次元が
256×32=8192から256×8=2048へ縮小した（パラメータ数はfold単体で913,857→520,641、
約43%減）にもかかわらず精度が実質変わらないのは、この回帰ヘッドが8192次元も要さない
タスクだったことを示唆する（conv特徴の情報量に対してpool_width=32は過剰だった可能性）。

**単独ブランチ2条件の比較（SB_none vs SB_eyescale）**: 全体MAEではSB_none（生PQ）が
SB_eyescale（eye_scale）を有意に上回った（0.087731 vs 0.090065、+2.66%改善、p=0.045）。
これはEXP019時点の結果（X2_eye_scaleが単独座標系として最良、raw PQ比-1.5%）と**逆転**
している。EXP020以降に積み上がった改善（t_l1損失・photometric aug・DBクリーニング等）を
両条件に等しく適用した結果、生PQ座標系の方が現行のベストプラクティスとの相性が良くなった
と考えられる（要因の特定は本実験のスコープ外。t_l1損失がpq_rescale後の圧縮されたlabel
範囲[-0.25,1.25]よりraw範囲[-1,2]で相対的に効きやすい可能性等が候補）。ただし有意水準は
pool_width=32時点（p=0.0014）より弱まっており（p=0.045）、この逆転自体は頑健だが差の
大きさはやや不安定（1 seedのみの学習のため run-to-run 分散の影響を排除できない）。

**単独ブランチ vs 現行dual標準**: 両条件ともdual標準（pass_std=89.04%）に対して
有意に劣る（SB_none: 87.84% (-1.20pt)、SB_eyescale: 87.98% (-1.06pt)）。paired t-testでも
全体MAEで両条件ともp<0.001の強い有意差。**dual fusionは現行データ・現行ベストプラクティスの
下でも依然として明確な価値を持つ**ことが確認された。ただし差の絶対値（pass_std -1.1〜-1.2pt）
はEXP020当時に観測された差（X0_none 0.0884 → X1_independent 0.0854、-3.3%）と比べて
縮小して見える点は注目に値する。

## t帯別結果考察

**多数派帯（0.3-0.7, 0.7-1.0、データの84%）**: SB_none vs dual標準では0.3-0.7がn.s.
（+0.21%, p=0.881、実質同等）だが0.7-1.0は有意に悪化（-3.75%, p=0.0007）——
pool_width=32時点では両帯ともn.s.だったが、今回は0.7-1.0で有意差が出た。一方
SB_eyescale vs dual標準はこの2帯で明確に有意に悪化（-16.03%, -5.56%、共にp<0.001）。
SB_none vs SB_eyescaleの比較でも0.3-0.7でSB_noneが強く優位（-16.27%, p<0.001）、
0.7-1.0はn.s.（-1.75%, p=0.287）。多数派帯における弱さは主にeye_scale座標系
（SB_eyescale）に集中する傾向は維持されている。

**極端帯（t<0, t>=1.2、データの4.4%）**: SB_none vs dual標準はt<0で-17.77%(p=0.0019)、
0-0.3で-5.33%(p=0.049)と有意に悪化するが、t>=1.2は今回n.s.（-6.77%, p=0.078、
pool_width=32時点は-10.53%, p=0.004で有意だった）。対してSB_eyescale vs dual標準は
t<0で-6.25%(n.s.)とやや悪いのみで、**t>=1.2ではdual標準を強く有意に上回った**（+19.19%,
p<0.001——pool_width=32時点の+8.78%, p=0.044より効果量・有意性ともに強まった）。
SB_eyescale vs SB_noneの比較でもt>=1.2はSB_eyescaleが+24.31%(p<0.001)で明確に優位。
EXP020で観測された「eye_scale正規化は極端t帯を助ける」という傾向は、単独ブランチとして
見た場合も（t>=1.2で）現行データで一貫して再現されている。

**総合すると**: 多数派帯はSB_none、極端高帯(t>=1.2)はSB_eyescaleが強く、単独モデル1本では
両方を同時に取れない——これはEXP020がdual-branch fusionを導入した動機（座標系の多様性を
単一モデルに組み込む）と整合的な結果であり、dual fusionが依然として合理的な設計である
ことを補強する。

## ペア比較 (paired t-test)

ベースライン: **SB_none**  \* = paired t-test p<0.05

### SB_eyescale vs SB_none

| band | n | MAE(SB_eyescale) | MAE(SB_none) | diff% | t_p | sig |
|---|---|---|---|---|---|---|
| **ALL** | 4900 | 0.090065 | 0.087731 | -2.66% | 0.044611 | * |
| t<0 | 129 | 0.290551 | 0.322038 | +9.78% | 0.218159 | n.s. |
| 0-0.3 | 219 | 0.166439 | 0.173202 | +3.90% | 0.285581 | n.s. |
| 0.3-0.7 | 1684 | 0.088284 | 0.075929 | -16.27% | 0.000000 | *** |
| 0.7-1.0 | 2430 | 0.071165 | 0.069939 | -1.75% | 0.287175 | n.s. |
| 1.0-1.2 | 351 | 0.089730 | 0.096053 | +6.58% | 0.086088 | n.s. |
| t>=1.2 | 87 | 0.164239 | 0.216997 | +24.31% | 0.000058 | *** |

### SB_none vs 現行dual標準（SD0_baseline、読み取り専用参照。`scr/compare_exp032_vs_dual_baseline.py`）

| band | n | MAE(SB_none) | MAE(dual標準) | diff% | t_p | sig |
|---|---|---|---|---|---|---|
| **ALL** | 4900 | 0.087731 | 0.084326 | -4.04% | 0.000003 | *** |
| t<0 | 129 | 0.322038 | 0.273446 | -17.77% | 0.001862 | ** |
| 0-0.3 | 219 | 0.173202 | 0.164433 | -5.33% | 0.048596 | * |
| 0.3-0.7 | 1684 | 0.075929 | 0.076088 | +0.21% | 0.880543 | n.s. |
| 0.7-1.0 | 2430 | 0.069939 | 0.067414 | -3.75% | 0.000657 | *** |
| 1.0-1.2 | 351 | 0.096053 | 0.091970 | -4.44% | 0.115274 | n.s. |
| t>=1.2 | 87 | 0.216997 | 0.203235 | -6.77% | 0.078095 | n.s. |

### SB_eyescale vs 現行dual標準（SD0_baseline、読み取り専用参照）

| band | n | MAE(SB_eyescale) | MAE(dual標準) | diff% | t_p | sig |
|---|---|---|---|---|---|---|
| **ALL** | 4900 | 0.090065 | 0.084326 | -6.80% | 0.000000 | *** |
| t<0 | 129 | 0.290551 | 0.273446 | -6.25% | 0.308097 | n.s. |
| 0-0.3 | 219 | 0.166439 | 0.164433 | -1.22% | 0.689912 | n.s. |
| 0.3-0.7 | 1684 | 0.088284 | 0.076088 | -16.03% | 0.000000 | *** |
| 0.7-1.0 | 2430 | 0.071165 | 0.067414 | -5.56% | 0.000052 | *** |
| 1.0-1.2 | 351 | 0.089730 | 0.091970 | +2.44% | 0.385980 | n.s. |
| t>=1.2 | 87 | 0.164239 | 0.203235 | +19.19% | 0.000166 | *** |

## ペア比較考察

- **SB_eyescale vs SB_none**: ALLで有意にSB_noneが優位(p=0.045)。内訳は0.3-0.7帯で
  SB_none優位（p<0.001）、t>=1.2でSB_eyescale優位（p<0.001）、他4帯はn.s.。EXP019当時
  「eye_scaleが単独最良」だった構図が、現行ベストプラクティスの下では崩れ、生PQ座標系が
  総合的に優位に転じている。
- **SB_none vs dual標準**: ALLで有意にdual標準が優位(p<0.001)。0.3-0.7帯・1.0-1.2帯・
  t>=1.2帯はn.s.——**単独ブランチ(生PQ)は多数派帯の一角(0.3-0.7)ではdual標準と
  統計的に見分けがつかない**が、0.7-1.0帯・t<0帯・0-0.3帯では有意に劣る。
- **SB_eyescale vs dual標準**: SB_noneと対照的に、多数派2帯（0.3-0.7, 0.7-1.0）で強い
  有意差（p<0.001）がありdual標準に劣る。一方t>=1.2ではSB_eyescaleがdual標準を強く
  有意に上回った（+19.19%, p<0.001）。
- 全体として、単独ブランチのどちらの座標系も「dual標準の完全な代替」にはならないが、
  **弱点の所在が座標系ごとに異なり、かつdual標準自身がその両方を(branch A+Bの平均で)
  カバーしている**という構図が、現行データでも一貫して観測された。

## 結論

**単独ブランチは精度面では不採用。dual-branch fusionを引き続き精度重視の標準として
維持する。ただし、下記の推論性能ベンチマークにより、単独ブランチ側のONNX export bug
を修正した結果、単独ブランチは「dual標準よりも高速・小型」という明確なメリットを持つ
ことが判明し、レイテンシ/サイズ制約が強いユースケースでは現実的な選択肢になった。**

- 現行のベストプラクティス（t_l1損失・photometric aug・cleaned DB・db_replica評価）を
  単独ブランチに移植しても、dual標準（pass_std 89.04%）を上回る単独構成は得られなかった
  （SB_none: 87.84%、SB_eyescale: 87.98%）。EXP020の結論「dual fusionはコスト増なしに
  座標多様性の恩恵を得る」は現行データでも支持される。
- **単独ブランチの中では生PQ座標系(SB_none)がeye_scale座標系(SB_eyescale)より総合的に
  優れる**ことが判明した（EXP019時点の逆）。ただしSB_eyescaleは極端高帯(t>=1.2)で
  dual標準さえ上回るため、極端帯の頑健性を優先するならSB_eyescaleも選択肢になり得る。
- **ONNX export bugの修正（`pool_width`パラメータ化+再学習）により、単独ブランチは
  dual標準に対して速度・サイズの両方で明確に優位になった**（詳細は次節）。ONNX/CPU実測で
  単独ブランチのp50は0.6ms台、dual標準は1.8ms台——**約3倍高速**。ファイルサイズも
  単独ブランチ.onnx=6.0MB、dual標準.onnx=14.9MBと**半分以下**。精度差（pass_std -1.1〜
  -1.2pt）とのトレードオフとして、リアルタイム性・エッジデバイス制約が厳しい用途では
  単独ブランチ（特にSB_none）が実用上の選択肢になり得る。

## 推論性能（`tools/model_benchmark/`）

精度差だけでは「速度・サイズを優先して単独ブランチを選ぶ価値があるか」を判断できないため、
`tools/model_benchmark/cli.py`（PyTorch/ONNX Runtime比較ツール）でSB_none・SB_eyescale・
現行dual標準（EXP031 SD0_baseline、読み取り専用参照）のfold ensemble（実運用相当、3fold平均）を
PyTorch(cuda/cpu)・ONNX Runtime(cpu)の全組み合わせで測定した。

**発見したバグと対応（ツール側）**: 測定前、ツールがONNXダミー入力形状に
`eyeliner.data.patch_utils.PATCH_SIZE=(64,128)`（EXP019以前の旧世代値）を使っており、
現行モデルの実際の学習パッチサイズ(64,64)（`eyeliner/training/dual_branch.py`のbranch preset）
と食い違っていたことが判明した。`tools/model_benchmark/inputs.py`・`onnx_utils.py`・
`cli.py`のデフォルトを(64,64)に修正（`--patch-size`オプションも追加、
`tests/test_model_benchmark_inputs.py`更新・pytest 12件パス確認済み。
`tools/model_benchmark/`配下は共通ツールのためEXP032側からはコミットせず、別コミットで反映済み）。

**発見したバグと対応（モデル側）**: 正しい(64,64)入力でのONNX exportは、当初
dual標準では成功するが単独ブランチ2条件では失敗することが判明した。原因は
`eyeliner/models/eyeliner_cnn.py`の`EyelinerCNN`ヘッドが`nn.AdaptiveAvgPool2d((1, 32))`で
**出力幅32が固定**（dual側の`EyelinerCNNDual`と違い`pool_width`パラメータ化されていな
かった）だったため。64×64入力は4層conv+pool（3回のMaxPool2d）を経て幅8の特徴マップに
なり、8→32のpoolingはPyTorchのONNX exporter（opset17）が非対応のupsample方向の変換
になっていた。dual標準は`pool_width=8`（EXP022のchannel/pool sweepでパラメータ化済み）が
たまたま特徴マップ幅と一致し恒等写像になっていたため影響を受けなかった——設計として
単独ブランチより優れていたわけではなく偶然の一致。ユーザー判断で`eyeliner_cnn.py`に
`pool_width`パラメータを追加（デフォルト32、既存チェックポイントとの後方互換維持。
`eyeliner/models/ensemble.py`のbuild_model/load_model/save_model/save_ensemble、
`eyeliner/training/trainer.py`のTrainConfig、`eyeliner/training/kfold.py`にも配線。
既存テスト646件全パス確認済み）し、EXP032の2configを`pool_width=8`で**再学習**した。
本サマリーの全数値はこの再学習後のもの。

### レイテンシ・スループット（fold ensemble・3fold平均相当、batch=1/8/32、iterations=100）

| model | params | ファイルサイズ | runtime/device | batch=1 mean/p50 [ms] | batch=1 thr [/s] | batch=32 mean/p50 [ms] | batch=32 thr [/s] |
|---|---|---|---|---|---|---|---|
| dual標準(SD0_baseline) | 3,911,430 | .pt 15.0MB | PyTorch/CPU | 15.65 / 15.62 | 63.9 | 130.83 / 130.65 | 244.6 |
| dual標準(SD0_baseline) | 3,911,430 | .pt 15.0MB | PyTorch/CUDA | 5.66 / 5.71 | 176.7 | 12.60 / 13.46 | 2538.7 |
| dual標準(SD0_baseline) | 3,911,430 | .onnx 14.9MB | ONNX/CPU | 1.90 〜 2.07 / 1.83 〜 1.97 | 483.9〜525.0 | 38.39〜43.70 / 37.98〜43.40 | 732.2〜888.6 |
| SB_none | 1,561,923 | .pt 6.0MB | PyTorch/CPU | 5.27 / 5.24 | 189.8 | 49.23 / 49.11 | 650.0 |
| SB_none | 1,561,923 | .pt 6.0MB | PyTorch/CUDA | 2.01 / 1.86 | 498.6 | 4.63 / 4.57 | 6906.5 |
| SB_none | 1,561,923 | **.onnx 6.0MB** | **ONNX/CPU** | **0.67 〜 0.72 / 0.63 〜 0.66** | **1395.8〜1500.3** | **15.04〜15.62 / 14.76〜15.67** | **2048.5〜2127.4** |
| SB_eyescale | 1,561,923 | .pt 6.0MB | PyTorch/CPU | 5.18 / 5.11 | 193.0 | 48.93 / 48.72 | 654.0 |
| SB_eyescale | 1,561,923 | .pt 6.0MB | PyTorch/CUDA | 1.92 / 1.83 | 519.7 | 4.60 / 4.55 | 6956.3 |
| SB_eyescale | 1,561,923 | **.onnx 6.0MB** | **ONNX/CPU** | **0.63 〜 0.64 / 0.61 〜 0.64** | **1564.4〜1598.6** | **15.27〜18.52 / 14.84〜18.45** | **1727.9〜2096.0** |

（各モデルのONNX/CPU行はcuda測定回・cpu測定回それぞれでONNX exportをやり直したため2回分の
実測レンジを併記。いずれも誤差範囲内で一致しており測定の再現性は良好）

GPU側はレイテンシの変動係数(cv)が0.1〜0.7と大きい（モデルが小さくカーネル起動オーバーヘッドが
支配的なため）——meanよりp50を実効値として参照する方が妥当。CPU側はcv<0.25で相対的に安定。
この環境の`onnxruntime`はCPU版のみのため、`--device cuda`指定時もONNX側は自動的にCPUに
フォールバックする（ツール仕様通り、`onnxruntime-gpu`があればGPU実行も測定可能）。
`thop`未インストールのためMACs推定は全モデルで未測定（`pip install thop`で有効化可能、今回未実施）。

### 解釈: 精度・速度・サイズの総合トレードオフ

| 観点 | dual標準(SD0) | SB_none | SB_eyescale |
|---|---|---|---|
| pass_std (db_replica) | **89.04%** | 87.84% | 87.98% |
| params | 3,911,430 | 1,561,923 (**-60%**) | 1,561,923 (**-60%**) |
| .onnxサイズ | 14.9MB | **6.0MB (-60%)** | **6.0MB (-60%)** |
| ONNX/CPU p50 (bs=1) | 1.83〜1.97ms | **0.63〜0.66ms（約3倍高速）** | **0.61〜0.64ms（約3倍高速）** |
| ONNXデプロイ | 可能 | **可能（修正済み）** | **可能（修正済み）** |
| 動的量子化INT8(.onnx, MatMulのみ) | 14.9→10.4MB, 精度未検証, p50 -5% | 6.0→4.85MB(-19%), 精度実質無影響, p50 -17% | 6.0→4.85MB(-19%), 精度実質無影響, p50 -6% |
| 静的量子化INT8(QDQ, Conv含む) | 未検証 | 6.25→1.62MB(-74%), MAE+2.6%, **p50 +216%(悪化)** | 6.25→1.62MB(-74%), MAE+1.7%, **p50 +225%(悪化)** |

**最大の発見（訂正版）**: `pool_width`のハードコーディングを解消して再学習した結果、
単独ブランチは**ONNX/CPUでdual標準の約3倍高速**（p50 0.61〜0.66ms vs 1.83〜1.97ms）に
なり、かつファイルサイズ・パラメータ数も**60%削減**された。精度は依然dual標準が
上回る（pass_std -1.1〜-1.2pt）ため「精度」対「速度・サイズ」の**正当なトレードオフ**が
成立している。修正前（`pool_width=32`固定でONNX export不可）の時点では単独ブランチに
実利がほぼ無い（PyTorchでも2倍速止まりで、dual標準のONNX/CPUに負けていた）という結論
だったが、この修正によって単独ブランチの立ち位置は大きく改善した——用途がPyTorch/GPU
常駐サーバーでなく、エッジ・モバイル・CPU推論やレイテンシSLAが厳しい場面では、
SB_none（精度で優位）またはSB_eyescale（極端t帯の頑健性で優位）が現実的な選択肢になる。

**単独ブランチ vs dual標準の速度差の内訳**: params比は1,561,923/3,911,430=0.40倍
（60%減）。PyTorch実測でのレイテンシ比（SB_none/dual標準、p50）はCPU batch=1で
5.24/15.62=0.34倍（≈3.0倍高速）、GPU batch=1で1.86/5.71=0.33倍（≈3.1倍高速）——
paramsの削減幅(60%)よりもレイテンシの削減幅(約67-70%)の方が大きい。dual-branchは
2本のCNN forwardを毎回実行するため、paramsの増分以上に計算コストが乗っている。
ONNX/CPUではさらに差が開き（約3倍）、これはregressorのLinear層の入力次元縮小
（256×32=8192→256×8=2048）がONNX Runtimeの最適化パスと特によく噛み合った可能性がある
（未検証の推測）。SB_none・SB_eyescale間の速度差はほぼ無い（同一アーキテクチャ・
同一paramsのため当然）。

### 量子化 (1) ONNX INT8動的量子化

上記のONNX修正を受け、さらに量子化版も試した。`tools/model_benchmark/onnx_utils.py`に
`quantize_dynamic_onnx()`を追加し（`onnxruntime.quantization.quantize_dynamic`、
キャリブレーションデータ不要）、`cli.py`の`--runtimes`に`onnx_int8`を追加（`report.py`は
runtime文字列を汎用的に扱うため変更不要）。

**判明した制約**: 当初Conv層も含めて量子化すると、この環境の`onnxruntime==1.22.1`
CPUExecutionProviderに`ConvInteger`opの実装が無く実行時エラーになった
（`NOT_IMPLEMENTED: Could not find an implementation for ConvInteger(10)`）。
そのため`op_types_to_quantize=["MatMul"]`に限定——**regressorのLinear層のみ**INT8化し、
conv層（計算量の大半を占める）はfp32のまま。

**速度**: ensemble・batch=1・CPU・p50で、SB_none: 0.685ms(fp32)→0.566ms(int8、-17%)、
SB_eyescale: 0.721ms→0.678ms(-6%)、dual標準: 1.395ms→1.326ms(-5%)。batch=8/32では
ほぼ差が無いか僅かに悪化する条件もあった（量子化の動的レンジ計算オーバーヘッドが
conv側の計算量削減の恩恵を上回るため）。**MatMul限定量子化では速度面のメリットは
小さい（0〜17%程度）**——conv層を量子化できないこの環境の制約上、量子化による
高速化効果は限定的。

**ファイルサイズ**: SB_none/SB_eyescale .onnx 6.0MB→4.85MB(-19%)、dual標準
.onnx 14.9MB→10.4MB(-30%)。速度以上にサイズ削減効果はある。

**精度**: `scr/eval_onnx_quant_accuracy.py`でdb_replicaテストセット全件(n=4900)を
実際に推論し検証した（単独ブランチ2条件、ONNX(fp32)はPyTorch(fp32)と完全一致することも
確認——変換の正しさの裏付け）。

| config | MAE ONNX(fp32) | MAE ONNX(int8) | 差分% | pred差分 mean\|Δt\| | pred差分 max\|Δt\| |
|---|---|---|---|---|---|
| SB_none | 0.087731 | 0.087719 | -0.014% | 0.001358 | 0.006901 |
| SB_eyescale | 0.090065 | 0.090015 | -0.056% | 0.000744 | 0.008940 |

**精度への影響は実質ゼロ**（誤差の符号すら安定しない範囲、-0.01〜-0.06%）。per-sample
最大差分も0.007〜0.009（t値スケール）とpass判定閾値(error_ratio 0.01788相当)を動かす
ほどではない規模。dual標準側は、エクスポートされるONNXアンサンブルグラフが実運用の
`fold_ensemble_raw_t`（枝ごとのlabel_space変換を経た生t平均）ではなく正規化空間の単純
平均を出力する仕様のため、そのままでは真の生t MAEと比較できず、今回は精度検証をスキップ
した（速度・サイズのみ計測）。

**この時点の結論**: 動的量子化はConv層を量子化できないため速度面の恩恵が小さい
（0〜17%）。「本当にconv層は量子化できないのか」を公式ドキュメントで確認したところ
——次項参照。

### 量子化 (2) ONNX INT8静的量子化（QDQ, Conv含む） — 公式推奨手法での再検証

ONNX Runtime公式ドキュメント（onnxruntime.ai/docs/performance/model-optimizations/
quantization.html）を確認したところ、明確な記載があった:

> "it is recommended to use dynamic quantization for RNNs and transformer-based models,
> and static quantization for CNN models"（RNN/transformer系には動的量子化、CNN系には
> 静的量子化を推奨）

本モデルはCNNなので、動的量子化はそもそも公式に非推奨の組み合わせだった。ConvIntegerの
CPU未実装（GitHub issue #15888, #12558, #3130等で同種の報告多数、既知の制約）は、
この非推奨経路を無理に使おうとした結果とも言える。**公式推奨の静的量子化（QDQ形式、
QuantizeLinear/DequantizeLinearの標準opを使うためConvIntegerのような専用整数opに依存
せず、Convも問題なく量子化できる）で再検証した。**

`tools/model_benchmark/onnx_utils.py`に`quantize_static_onnx()`を追加
（`onnxruntime.quantization.quantize_static`, `quant_format=QuantFormat.QDQ`,
`op_types_to_quantize`未指定=Conv/MatMul両方が対象）。キャリブレーションデータは
実際のtrain manifestからランダムサンプリングした200件（`experiments/EXP032_single_branch/
scr/quantize_static_and_eval.py`の`PatchCalibrationDataReader`、`EyelinerPatchDataset`を
そのまま利用し実データでキャリブレーション）。

**結果（ensemble、CPU、db_replicaテストセット全件でMAE検証）**:

| config | ファイルサイズ | bs=1 p50 | bs=8 p50 | bs=32 p50 | MAE |
|---|---|---|---|---|---|
| SB_none fp32(onnx) | 6.25MB | 0.636ms | 3.352ms | 15.394ms | 0.087731 |
| SB_none static_int8(QDQ) | **1.62MB (-74%)** | **2.010ms (+216%)** | 4.909ms (+46%) | 16.506ms (+7%) | 0.090002 (+2.6%) |
| SB_eyescale fp32(onnx) | 6.25MB | 0.584ms | 3.378ms | 15.235ms | 0.090065 |
| SB_eyescale static_int8(QDQ) | **1.62MB (-74%)** | **1.898ms (+225%)** | 4.985ms (+48%) | 16.238ms (+7%) | 0.091607 (+1.7%) |

（校正サンプル数20→200で精度劣化幅が+9.5%→+2.6%へ縮小したことを確認済み——200件は
最低限の妥当性は確保できているが、より多くの校正データでさらに縮小する可能性は残る）

**公式推奨手法で検証しても、このモデルでは量子化は net negative（速度もアクセ精度も悪化）
だった**: ファイルサイズは劇的に縮小（-74%、Convの重みも含めて量子化されるため動的量子化
の-19%より大幅）するが、**レイテンシはbs=1で+216〜225%と大幅に悪化**（bs=32では+7%まで
縮小するが依然悪化）。MAEも+1.7〜2.6%と動的量子化（-0.01〜-0.06%、実質無影響）より
明確に悪い。

**原因の考察**: QDQ形式は量子化対象の各op（Conv/MatMul）の前後にQuantizeLinear/
DequantizeLinearノードを挿入する。本モデルは64×64パッチ・4層convという非常に小さい
ネットワークで、conv自体の計算量がそもそも小さいため、**INT8演算で浮く計算コストより
QDQノード自体のオーバーヘッド（fp32↔int8変換、ノード数増加によるディスパッチコスト）の
方が大きい**——量子化の恩恵はモデルサイズ・演算強度に比例するため、小さいモデルほど
逆効果になりやすいという一般的傾向と整合する（本調査の実測はこの傾向を裏付ける一次データ
だが、この特定の主張を直接支持する公式ドキュメントの記述は見つからなかった——推測として
明記する）。

**結論（量子化まとめ）**: 「本当にconv層を量子化できないか」という問いへの答えは
**「できる（静的量子化で可能、ConvIntegerは動的量子化特有の制約）」だが、
「速度・精度の両方で悪化するため、このモデル・この環境では推奨しない」**。
動的量子化（MatMulのみ、精度無影響・速度+0〜17%・サイズ-19%）の方が、この規模の
モデルには相対的にまだ実用的。ファイルサイズを最優先するユースケース（帯域制約が
厳しいOTA配信等）でなければ、量子化せずfp32のONNXをそのまま使うのが最も無難という
結論になる。

**`scripts/export_onnx.py`（本番用ONNXエクスポートスクリプト）について**: コード確認の結果、
このスクリプトは`eyeliner.models.eyeliner_cnn.EyelinerCNN`（単独ブランチ）のみを対象としており
（`eyeliner_cnn_dual`/`dual_ensemble`は一切import していない）、**現行dual標準モデルの
エクスポートには使われていない/使えない**古いスクリプトである可能性が高い。ダミー入力に
`eyeliner.data.patch_utils.PATCH_SIZE=(64,128)`（現行モデルの実際の学習形状(64,64)と
不一致）を使っている点、および今回の`pool_width`修正が未反映な点（`build_model`呼び出しに
`pool_width`を渡していない）から、現状のままでは単独ブランチであっても正しく動作しない。
`git log --all -- scripts/export_onnx.py`で確認したところ、最終の実質変更は2026-04-29
（`patch_utils.py`のディレクトリ移動に伴うimportパス修正のみ）で、EXP020のdual-branch
導入以降ロジックの更新が一切ない。実際に運用で使われているかはリポジトリ内の情報だけでは
判断できず、要ユーザー確認——**単独ブランチをONNXで本番配信する場合はこのスクリプトの
更新（`(64,64)`形状・`pool_width`引数対応）が必要**。

## 今後の課題

- **量子化は動的・静的の両方を検証済み、いずれも本モデルには推奨しない**という結論に
  達した（動的=Conv非対応で速度改善小、静的=公式推奨手法でConv含め量子化できるが
  このモデルには net negative）。さらなる高速化の検証候補としては`onnxruntime-gpu`
  （GPU上でのINT8）やQNN/OpenVINO等の別バックエンドが残っているが、モデルが既に
  非常に小さい（fp32 ONNX/CPUでp50 0.6ms台）ため追加の投資対効果は疑わしい。
  dual標準側の量子化精度検証も未実施（ONNXアンサンブルグラフが正規化空間平均を返す
  仕様のため生t MAEと直接比較できない——枝ごとに生tへ変換してから量子化影響を見る
  追加実装が必要）。
- 本実験は1 seedのみの学習。SB_eyescaleのt>=1.2での優位（+24.31%, n=87）やSB_noneの
  多数派帯の一部n.s.が真の効果かrun-to-run分散かは、複数seed再学習をしない限り完全には
  切り分けられない（EXP029/031で繰り返し指摘されている既知の限界と同じ）。
- SB_noneがSB_eyescaleを逆転した要因（t_l1損失とlabel範囲スケールの交互作用、DBクリーニングの
  影響等）は未特定。dual-branchのbranch A/Bそれぞれの重みや寄与度を分析すれば、
  fusionが具体的に何を補完しているかがより明確になる可能性がある。
- pool_width=8への縮小で精度が実質変わらなかったことから、regressorヘッドがさらに縮小
  可能な余地（例: pool_width=4）や、conv側（num_conv_layers/channel数）の縮小余地も
  示唆される——本実験のスコープ外だが、EXP031時点で計画倒れになった「モデルサイズ比較」を
  単独ブランチの文脈で改めて検討する価値があるかもしれない。
- EXP031summary が示唆していた BLUR/NOISE の元画像px空間再定義・対側眼aug/
  is_left_eye aux は本実験では扱っていない。次番号での着手を検討。
- **`scripts/export_onnx.py`の更新**: 単独ブランチを実際にONNXで配信する場合、
  `PATCH_SIZE`を(64,64)に、`build_model`呼び出しに`pool_width=8`を渡すよう更新が必要。
  face_makeup側が現行dual標準モデルをどうONNX化して消費しているのか（別の変換経路が
  あるはず）も未特定——先に確認すべき。
- `thop`未インストールのためMACs（乗算加算回数）は今回未測定。`pip install thop`で
  計算量ベースの比較も追加できる。
- `eyeliner_cnn.py`への`pool_width`パラメータ追加・`ensemble.py`/`trainer.py`/`kfold.py`
  の配線変更、および`tools/model_benchmark/`のpatch_size修正は、いずれもこのEXP032
  ディレクトリとは別に、コアライブラリ側の変更としてコミット済み。
