# data_management — データセットの復元と Open Images 画像の取得

`data/`（`samples.db` と画像）は `.gitignore` 対象で、クローン直後には存在しない。
このディレクトリのツールは、リポジトリに同梱の **実験 manifest** から学習・評価用データを復元し、
必要なら Open Images から画像を追加取得する。

```
experiments/EXP{NNN}_*/manifests/manifest_{train,test}.csv   ← 復元の基準（リポジトリ同梱）
        │
        ├─ fetch     画像を Open Images の公開 S3 から data/images/ へ取得
        ├─ db        samples.db を再構築（idx・アノテーション・eye_scale・norm_eye_scale）
        └─ features  FaceMesh を再実行して face_yaw/pitch/roll・lap_var を復元
```

## クイックスタート

```bash
pip install -r requirements.txt

python tools/data_management/restore_dataset.py --dry-run   # まず件数・容量の目安だけ確認
python tools/data_management/restore_dataset.py             # fetch → db → features を実行

# manifest に無い画像も追加して、アノテーション対象を増やす（詳細は「`--source all`」の節）
python tools/data_management/restore_dataset.py --source all --max-images 500
```

| オプション | 説明 |
|-----------|------|
| `--source manifest\|all` | 対象の切り替え。`manifest`（既定）は manifest の画像と行だけ。`all` は加えて Open Images から画像を追加取得し、手元の画像の未アノテーション行を DB に追記する |
| `--split validation test` | `all` 用: 追加で検索する split（既定: validation） |
| `--max-images 500` | `all` 用: 1 回の実行で追加取得する上限（既定 500。manifest の画像と手元にある画像は数えない） |
| `--cache-dir` | `all` 用: Open Images のメタデータのキャッシュ先（既定: `data/.open_images_cache`） |
| `--steps fetch db features` | 一部だけ実行（順序は常に fetch → db → features） |
| `--exp 32` | 使う manifest の実験番号（既定: 最新の実験） |
| `--dry-run` | 何も書き込まず、実行内容と件数だけ表示 |
| `--overwrite-db` | manifest と一致しない既存 DB を **バックアップ後に** 置換 |
| `--tol 0.25` | features の画像同一性の許容値 |
| `--db` / `--images-dir` / `--experiments-dir` | パスの変更（既定: `data/samples.db` / `data/images` / `experiments`） |

終了コードは、全ステップ成功で 0、いずれか失敗で 1。`db` / `features` が失敗すると以降のステップは中止する
（`fetch` で取得できなかった画像があっても、DB は画像なしで作れ、features は画像の無いものを飛ばすので続行する）。

## 復元の設計

### manifest の検出と検証（`manifest_source.py`）

- `experiments/EXP{NNN}_*/manifests/` に `manifest_train.csv` / `manifest_test.csv` を持つ実験のうち、
  **最新の実験番号** のものを使う（`--exp` で指定可）。
- 必須列: `idx, image_path, face_index, part_name, sub_key, point_index, px, py, qx, qy, t`
  （`nose_x, nose_y` は任意。鼻先未検出の `nan` は欠損として扱う）。
- **EXP001〜007 の manifest は `face_index` を持たないため復元に使えない**（列不足として明確なエラーにする）。
- `image_path` は外部入力として扱い、`data/images/<ID>.jpg`（ID は英数字・`_`・`-`）以外は拒否する。
  任意パスの読み書きに使われない。

### fetch — 画像の取得（`fetch_images.py` / `s3_download.py`）

`fetch_images.py` が取得計画（対象 ID の決定・シャード・除外）と CLI、`s3_download.py` が 1 枚ずつの取得・検証・
リトライ・並列化を担う。Open Images の画像を **公開 S3 から直接** 取得する（fiftyone 不要）。

```
https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg
```

fiftyone 経由で取得した画像とバイト単位で一致することを SHA-256 で確認済み。Flickr の `original_url` は
解像度が異なる場合があるため使わない。

- 既存ファイルはスキップ。`<id>.jpg.<PID>.part` に保存し、**Content-Length・サイズ上限（1 枚 64MiB）・JPEG マジックバイトを検証** してから rename する
  （一時ファイルは PID 入りなので、並列実行の別プロセスと衝突しない）
- 5xx・429・通信断は指数バックオフでリトライ、**400 / 401 / 403 は即失敗**、404 は次の split を探索
- リダイレクトは追従せず（失敗扱い）、圧縮転送も要求しない（`Accept-Encoding: identity`）
- Windows でウイルス対策ソフトが書き込み直後のファイルを一時的に掴んでも、rename を短く再試行する。1 枚の失敗で残りの取得は止まらない
- split が不明な manifest の画像は `train → validation → test` の順に探索する
- 途中で止まっても再実行で再開できる（残った `*.part` は削除してよい）

取得元は `--source` の 1 つだけで切り替える。

| `--source` | 取得するもの |
|-----------|-------------|
| `manifest`（既定） | 最新実験（`--exp` で指定可）の manifest の画像。アノテーション済みの画像で、復元用（`restore_dataset.py` が内部で使う） |
| `all` | **manifest の画像** + Open Images 全体（`--split`、条件は後述）から追加する画像 |

```bash
# manifest（既定）: manifest の画像を取得
python tools/data_management/fetch_images.py
python tools/data_management/fetch_images.py --source manifest --exp 32

# all: manifest の画像に加えて、Open Images 全体から条件（後述）に合う画像を追加で取得
# （split は既知なので 404 が出ない）
python tools/data_management/fetch_images.py --source all --split validation --max-images 500

python tools/data_management/fetch_images.py ... --dry-run           # 対象数と容量の目安（1 枚 約 320KB で見積り）
```

**シャード**: 大量取得は `--shard k --shard-count N` で並列化できる。処理順は
`条件に合う ID を抽出 → ID をソート → シャード分割 → 取得済みを除外 → --max-images 適用` で、
取得済み画像の有無でシャード境界が動かない。

**取得済みの扱い**: 手元（`data/images/`）にある画像は再取得しない。手元に無い画像は、attribution CSV に
登録済みでも取得する（リポジトリ同梱の attribution CSV は作者の取得履歴で、クローンした環境の手元の画像とは
一致しないため）。出力先に画像があるのに attribution へ未登録のもの（強制終了で記録が漏れた場合など）は、
再取得せず attribution だけ補完する（CC BY 2.0 の表示義務を取りこぼさないため）。

**上限**: `--source all` で `--max-images` を省略すると、Open Images から追加する分は 500 枚が上限
（誤って全件を取らないため）。これは 1 回の実行で追加する上限で、手元にある画像は数えないので、
繰り返すと次の N 枚を追加する。**manifest の画像**（アノテーションの付け先なので必ず取得する）と、手元にあって
attribution の補完だけが必要な画像は、この枚数に数えない。Open Images から追加せず manifest の画像だけを
取得するなら `--source manifest` を使う（`all` は `--max-images 0` でも Open Images のメタデータを読み込み、
attribution の補完を行う）。`--source manifest` の `--max-images` は既定で無制限。

**`all` は manifest を必要とする**: manifest の画像を必ず含めるため、manifest が見つからない
（`--experiments-dir` の既定は相対パスの `experiments`。リポジトリのルートで実行すること）と exit 2 で終了する。
`--shard` で並列実行するときは、全シャードで同じ `--exp`・同じ manifest の状態にすること
（manifest の内容がシャード境界に影響するため）。

### Open Images メタデータ（`oi_metadata.py`）

画像メタデータ（ImageID / License / OriginalURL / Author / Title）と bbox 注釈 CSV を取得してキャッシュする
（`data/.open_images_cache/`）。既存の fiftyone キャッシュ `~/fiftyone/open-images-v7` の注釈 CSV も
**読み取り専用で再利用** し、数 GB の再取得を避ける。再取得は `--refresh-cache`。

fiftyone キャッシュが無い環境（クローンした第三者など）では、`--source all` の初回に注釈 CSV とメタデータ CSV を
`data/.open_images_cache/` へ取得する。ダウンロード量の目安（split ごと）は validation 約 40MB、test 約 123MB、
**train 約 2.9GB**。既定の `--split validation` なら約 40MB で済む。

### 取得フィルタ（`oi_filter.py`）

`FilterConfig` の既定値（画像を落とす前に注釈 CSV だけで絞り込む）:

| 条件 | 既定値 |
|------|-------|
| 必須クラス | `Human face` |
| 顔 bbox 面積（正規化） | `>= 0.01` |
| 顔 bbox アスペクト比 | `0.2 〜 2.0` |
| 有効な顔数 | `1 〜 4` |
| 顔の属性 | `IsDepiction` / `IsGroupOf` / `IsInside` が 0 |
| ライセンス | CC0 / CC BY 2.0（管轄移転版・BY-SA・BY-NC・BY-ND は除外） |
| 除外クラス | Doll, Toy, Teddy bear, Poster, Sculpture, Infant bed |
| 子どもラベル（Boy / Girl） | 既定では除外しない（`--exclude-children` で除外） |

条件は `--min-face-area` `--min-face-aspect` `--max-face-aspect` `--max-faces` で変更できる。
フィルタ結果は条件のハッシュ付きで `{split}_filtered_<条件ハッシュ>.txt` にキャッシュされ、同じ条件なら再計算しない。

### attribution（`attribution.py`）

画像の取得に成功すると `data/open_images_attribution.csv` に著者・出典・ライセンスを追記する
（CC BY 2.0 の表示義務）。同一 Image ID は二重登録せず、`--shard` の並列実行でも壊れないよう
ロックファイル（`<csv>.lock`）で排他制御する。Open Images のメタデータを持たない画像（manifest だけから取る画像。
`--source manifest` の復元など）は、attribution には触れない。

> 注意: attribution CSV の著者・タイトルは Flickr 利用者が入力した文字列で、`=` `+` `-` `@` で始まるセルを含む
> （既存データにもある）。Excel 等で開くと数式として解釈されうるため、表計算ソフトで開くときは「テキスト」として
> 取り込むこと。ファイルの形式は他ツールとの互換のため変更していない。

### db — samples.db の再構築（`../sampledb_maintenance/rebuild_samples_db.py`）

manifest を基準に、`idx` とアノテーションをそのまま復元し、`eye_scale`（眼ごとの P/Q バウンディングボックス対角長）と
`norm_eye_scale`（`eye_scale / 画像対角長`。画像があれば）を計算する。fetch の後に実行するのは、画像があれば
`norm_eye_scale` まで復元できるため。

既存の DB を壊さないためのガード:

| 既存 DB の状態 | 動作 |
|---------------|------|
| 無い / 空 | 新規作成 |
| manifest と一致 | 何もしない（復元済みの特徴量も消さない） |
| 一致しない | **拒否**（DB は変更しない） |
| 一致しない + `--overwrite-db` | `samples.db.bak-<時刻>` にバックアップしてから置換 |

DB の比較は読み取り専用。置換は一時ファイルで新 DB を作ってから rename する。旧 DB の `-wal` / `-shm` /
`-journal` は、新規作成・置換のどちらでも（本体ファイルが無くても）新 DB に再生されないよう先に削除する。
ロック中など「壊れているわけではない理由で読めない」DB は、生コピーや置換をせずエラーで止まる。
置換の前に annotator など DB を開いているプロセスを止めること（Windows では置換に失敗して安全に中止する）。

`unchanged` のときも、画像が手元にあって `norm_eye_scale` が NULL の行は補完する（`fetch` が一部失敗したあと、
画像を取り直して再実行すれば埋まる）。

### features — 顔姿勢・画像特徴量の復元（`../sampledb_maintenance/restore_face_pose.py`）

画像を MediaPipe FaceMesh に通し直し、`face_yaw` / `face_pitch` / `face_roll`（顔ごと）と
`lap_var`（画像単位: 全ての目のラプラシアン分散の最大値。`images.db` の `max_lap_var` と同定義）を復元する。

**画像同一性の検証**: MediaPipe のバージョン差で P/Q 座標そのものは再現できないことがあるため、位置関係で
「今の画像がアノテーション時と同じか」を照合する。眼ごとに

```
score = |manifest の P/Q 点群の重心 − FaceMesh の目の輪郭ランドマークの重心| / 眼の対角長
```

を求め、**両目の最悪値** が `--tol`（既定 0.25）以下の画像だけ書き込む。照合できなかった画像には
`face_pose` も `lap_var` も書かない。EXP032 の 6 画像での実測は、同一画像 0.012〜0.032、別画像 1.1 以上
（2 桁近い差）で、既定値は余裕を持って両者を分離できる。

**誤上書きガード**: **DB の全行が manifest に含まれる場合のみ** 実行する。本来の大規模な `samples.db`
（未アノテーション行を含む）に、復元用の小さな manifest 由来の値を適用しないため。対象の画像は毎回再計算して
上書きする（冪等）。デコード前に Pillow でヘッダを読み、画素数が 2 億を超える画像は処理しない。

### `--source all` — 画像を追加して、未アノテーションの行を DB に追記する

manifest の復元に加えて、アノテーション対象の画像を増やす。取得と DB の再構築が 1 つのスイッチで連動する。

| ステップ | `--source manifest`（既定） | `--source all` |
|---------|---------------------------|----------------|
| fetch | manifest の画像 | manifest の画像 + Open Images 全体（`--split`）から追加（1 回の実行で最大 `--max-images` 枚） |
| db | manifest から再構築。DB が manifest と一致しなければ拒否 | DB が manifest の行をすべて含めばそのまま。加えて、手元にあって行が無い画像に未アノテーション行（`t` は NULL）を追記 |
| features | DB の全行が manifest に含まれる場合のみ。毎回再計算して上書き | 顔姿勢が NULL の manifest の行を持つ画像だけを処理し、NULL の列だけを埋める。既存の値は上書きせず、manifest 外の行（同じ顔の行を含む）には書かない |

- **追記のみ**: 既存の行の値は上書き・削除しない（`norm_eye_scale` と特徴量が NULL の行を埋めることだけは行う）。
  画像ごとに 1 つのトランザクションで書くので、途中で止まっても中途半端な画像は残らず、再実行すれば残りの画像だけが
  追記される。`idx` は書き込みロックを取ってから決めるので、別のプロセスが同時に行を足しても衝突しない。
- **追記される行**: P/Q 座標・顔番号・顔姿勢・鼻先・`eye_scale`・`norm_eye_scale`・`lap_var` まで埋まる（追加の補完ツールは不要）。
  P/Q・顔番号・姿勢は `annotator/preprocessor.py`、`lap_var` は `restore_face_pose.py` と同じ定義。
- **`idx`**: DB の現在の最大値の次から、画像 ID の昇順に画像ごとの連続ブロックで採番する。**`idx` はその環境の中だけで
  意味を持つ番号**で、作者の DB の番号と重なりうる。環境をまたぐ突き合わせ（アノテーションの受け渡しなど）は、
  自然キー `(image_path, face_index, part_name, sub_key, point_index)` で行うこと。
- **アノテーション**: 追記した行は annotator（`python annotator/server.py`）で開いて付ける。自分で付けたアノテーションや
  manifest の行の修正は、再実行しても消えない（既存の行は変更しないため。値の違いは「不一致」として見ない）。
- **`--source manifest` は厳密なまま**: 追記した行がある DB に `--source manifest` の `db` を実行すると
  「manifest と一致しない」と拒否する（DB は変更しない）。`--overwrite-db` は追記分を捨てるので付けないこと。
  追記後の再実行は `--source all` を使う。
- **`lap_var` は NULL でもよい**: 分析用の列で学習には使わない（annotator のギャラリーの「ブレ順」の並べ替えと表示、分析スクリプトが使う。NULL のときは「--」と表示し、並べ替えでは最後に回る）。そのため `--source all` は、`lap_var` だけが NULL の画像を処理しない（顔姿勢が NULL で処理する画像では、ついでに埋まる）。`tools/imagesdb_maintenance/backfill_db_features.py` が `images.db` の値で `lap_var` を置き換えて NULL になっても、そのままでよい。
- 顔が検出できなかった画像には行が入らず、実行のたびに解析し直される（結果は変わらない）。同様に、顔姿勢を算出
  できない manifest の画像は顔姿勢が NULL のまま残るので、実行のたびに解析し直される。
- FaceMesh（mediapipe の旧 API）が必要。行を追記する画像も、顔姿勢を埋める画像も無いときは使わない
  （何もすることが無い再実行は、mediapipe が使えなくても成功する）。
- `--max-images` と `--split` は `--source all` のときだけ指定できる（`--source manifest` は manifest の画像を
  すべて取得する。一部だけ取得すると、DB だけ全行を作る中途半端な復元になるため）。
- `--dry-run` の「行を追記する画像の候補」は、手元にある画像だけを数える（fetch がこれから取得する画像は含まない）。

## 復元できないもの・環境上の注意

- 復元できない: 未アノテーション行、`skip_flag=1` の行、`image_labels` テーブル
  （manifest はアノテーション済みサンプルだけを持つ）。
- MediaPipe のバージョン差により `lap_var`・顔姿勢の値が元の DB と微小に異なる場合がある。
- 新規に取得した画像に既存のアノテーションは付かない。`--source all` なら、未アノテーションの行が DB に追記される。
- `features` には MediaPipe の **旧 API（`mp.solutions`）** が必要で、`mediapipe>=0.10.21,<0.10.30` が前提
  （0.10.30 以降には旧 API が無い。`requirements.txt` で固定済み）。
- `requirements.txt` から入る PyTorch は **CPU 版**。GPU で学習する場合は別途 GPU 対応版を導入すること。

## 検証

```bash
pytest tests/test_restore_dataset.py tests/test_restore_manifest_source.py tests/test_restore_rebuild_db.py \
       tests/test_restore_face_pose.py tests/test_add_new_images.py \
       tests/test_restore_fetch_images.py tests/test_fetch_images_sources.py \
       tests/test_oi_metadata.py tests/test_oi_filter.py tests/test_attribution.py tests/test_data_management_helpers.py
```

ネットワークと FaceMesh はフェイクに差し替えており、`data/` や mediapipe が無くても実行できる
（同梱 manifest を読むテストのみ `experiments/` を使う）。

## その他のツール

`analyze_filter_exclusion.py` / `cleanup_orphans.py` / `delete_unreferenced_images.py` /
`find_unused_images.py` / `remove_no_face_images.py` は、手元の画像・DB の整理用。使い方は各ファイル先頭の
docstring を参照。旧 `download_open_images.py`（`scripts/` と本ディレクトリの 2 つ）は `fetch_images.py` に統合して削除した。
