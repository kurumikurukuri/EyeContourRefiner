"""EXP023: 5-Fold OOF 予測の永続化。

固定 fold{1..5}.csv（未生成なら自動生成）を使い、leave-one-fold-out で CNN を学習し、
各サンプルの OOF 予測（生 t）を CSV に永続化する。config の `backbone` が `custom_dual`
ならデュアルブランチ訓練（eyeliner.training.dual_kfold）経路を、それ以外は単一ブランチ
訓練（eyeliner.training.kfold）経路を使う。

デフォルトのデータソースは data/samples.db から再エクスポートした全件
full_manifest.csv（t IS NOT NULL の全件）。

mode:
    nested (デフォルト): Test/Val/Train を5ラウンドでローテーションする入れ子CV。
        各ラウンドの Test は学習・早期終了判定に一切使わず、保存されたベストモデルで
        推論する。全 fold が Test を1回ずつ経験するため、fold 内の全件が OOF 対象。
    simple: 旧来の方式（検証セット=OOF対象を兼用、早期終了にわずかな楽観性あり）。

スコアリング（残差ランキング・誤ラベル候補抽出）は本スクリプトの責務外
（rank_candidates.py / intersect_candidates.py が行う）で、OOF 予測 CSV の
生成までに限定する。

使い方:
    python experiments/EXP023_oof_mislabel/scr/run_oof.py
    python .../run_oof.py --epochs 2 --patience 1  # 軽量動作確認
    python .../run_oof.py --mode simple
    # デュアルブランチ（EXP022 bc16_pw1/bc32_pw8 の良否比較用、config/ にコピー済み）
    python .../run_oof.py --config bc16_pw1
    python .../run_oof.py --config bc32_pw8

進行ログ（fold/epoch毎のloss・val_mae）は標準出力と `outputs/<exp_name>/train.log` の両方に出る。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
CONFIG_DIR = EXP_DIR / "config"
MANIFEST_DIR = EXP_DIR / "manifests"
OUTPUT_DIR = EXP_DIR / "outputs"

DEFAULT_MANIFEST = MANIFEST_DIR / "full_manifest.csv"

sys.path.insert(0, str(ROOT))
from eyeliner.training.dual_kfold import DualTrainConfig  # noqa: E402
from eyeliner.training.oof import write_oof_csv  # noqa: E402
from eyeliner.training.oof_kfold import (  # noqa: E402
    run_dual_oof_kfold,
    run_nested_dual_oof_kfold,
    run_nested_oof_kfold,
    run_oof_kfold,
    split_fixed_folds,
)
from eyeliner.training.trainer import TrainConfig  # noqa: E402

DEFAULT_CONFIG_NAME = "woflip_none"
DUAL_BACKBONE = "custom_dual"


def build_train_config(cfg: dict, **overrides) -> TrainConfig:
    kwargs = dict(
        backbone=cfg.get("backbone", "custom"),
        loss_fn=cfg.get("loss_fn", "t"),
        use_sigmoid=cfg.get("use_sigmoid", True),
        use_batchnorm=cfg.get("use_batchnorm", True),
        num_conv_layers=cfg.get("num_conv_layers", 4),
        dropout_rate=cfg.get("dropout_rate", 0.3),
        use_is_upper=cfg.get("use_is_upper", True),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch"],
        patience=cfg["patience"],
        seed=cfg["seed"],
        aug_flags=cfg.get("aug_flags", 0x4F),
        patch_t_min=cfg["patch_t_min"],
        patch_t_max=cfg["patch_t_max"],
        patch_perp_scale=cfg["patch_perp_scale"],
        patch_h=cfg["patch_h"],
        patch_w=cfg["patch_w"],
        label_t_min=cfg["label_t_min"],
        label_t_max=cfg["label_t_max"],
        pq_rescale_mode=cfg.get("pq_rescale_mode", "none"),
        pq_rescale_factor=cfg.get("pq_rescale_factor", 1.0),
    )
    kwargs.update(overrides)
    return TrainConfig(**kwargs)


def build_dual_train_config(cfg: dict, **overrides) -> DualTrainConfig:
    kwargs = dict(
        fusion_mode=cfg.get("fusion_mode", "independent"),
        pq_rescale_mode_b=cfg.get("pq_rescale_mode_b", "eye_scale"),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch"],
        k=cfg.get("k", 5),
        patience=cfg["patience"],
        seed=cfg["seed"],
        num_conv_layers=cfg.get("num_conv_layers", 4),
        use_sigmoid=cfg.get("use_sigmoid", True),
        use_batchnorm=cfg.get("use_batchnorm", True),
        dropout_rate=cfg.get("dropout_rate", 0.3),
        aug_flags=cfg.get("aug_flags", 127),
        endpoint_blend_range=cfg.get("endpoint_blend_range", 0.25),
        loss_fn=cfg.get("loss_fn", "t"),
        base_channel=cfg.get("base_channel", 32),
        pool_width=cfg.get("pool_width", 32),
    )
    kwargs.update(overrides)
    return DualTrainConfig(**kwargs)


def main() -> int:
    p = argparse.ArgumentParser(description="EXP023 OOF 予測永続化")
    p.add_argument("--config", default=DEFAULT_CONFIG_NAME, help="EXP023 config/ 内の config 名(拡張子なし)")
    p.add_argument(
        "--config-path", type=Path, default=None,
        help="config yaml への直接パス(他実験の config を使う場合。例: EXP022のbc16_pw1.yaml)。"
             "指定時は --config を無視する",
    )
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--exp-name", default=None, help="出力サブディレクトリ名(省略時は config 名と同名)")
    p.add_argument("--k", type=int, default=None, help="fold数(省略時は config の k)")
    p.add_argument("--seed", type=int, default=None, help="fold分割シード(省略時は config の seed)")
    p.add_argument("--epochs", type=int, default=None, help="config 上書き(軽量検証用)")
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--force-folds", action="store_true", help="既存 fold ファイルを再生成する")
    p.add_argument("--mode", choices=["nested", "simple"], default="nested")
    args = p.parse_args()

    config_path = args.config_path or (CONFIG_DIR / f"{args.config}.yaml")
    config_name = config_path.stem
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    is_dual = cfg.get("backbone", "custom") == DUAL_BACKBONE

    k = args.k if args.k is not None else cfg.get("k", 5)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)

    fold_dir = MANIFEST_DIR
    existing_folds = sorted(fold_dir.glob("fold*.csv"))
    if not existing_folds or args.force_folds:
        if not args.manifest.exists():
            print(f"[ERROR] manifest 不在: {args.manifest}")
            return 1
        fold_paths = split_fixed_folds(args.manifest, fold_dir, k=k, seed=seed)
    else:
        fold_paths = existing_folds
        print(f"[skip] 既存 fold ファイルを使用: {[p.name for p in fold_paths]}")

    overrides = {"num_workers": args.num_workers}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.patience is not None:
        overrides["patience"] = args.patience

    if is_dual:
        config = build_dual_train_config(cfg, **overrides)
        run_fn = run_nested_dual_oof_kfold if args.mode == "nested" else run_dual_oof_kfold
    else:
        config = build_train_config(cfg, **overrides)
        run_fn = run_nested_oof_kfold if args.mode == "nested" else run_oof_kfold

    default_exp_name = f"{config_name}_nested" if args.mode == "nested" else config_name
    exp_name = args.exp_name or default_exp_name
    exp_dir = OUTPUT_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(exp_dir / "train.log"), encoding="utf-8"),
        ],
        force=True,
    )

    print(f"\n{'='*60}")
    print(f"  EXP023 OOF [{args.mode}{'/dual' if is_dual else ''}]: {exp_name}  "
          f"(k={len(fold_paths)}, loss={config.loss_fn}, aug_flags={config.aug_flags})")
    print(f"  output → {exp_dir}")
    print(f"{'='*60}")

    result = run_fn(fold_paths, exp_dir, config)

    oof_csv = exp_dir / "oof_predictions.csv"
    write_oof_csv(result.oof_predictions, oof_csv)

    mean_mae = sum(r.best_val_mae for r in result.fold_results) / len(result.fold_results)
    print(f"\n  CV MAE (正規化, 早期終了基準のVal): {mean_mae:.6f}")
    print(f"  OOF predictions: {len(result.oof_predictions)} 件 / 全 {result.n_total} 件 → {oof_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
