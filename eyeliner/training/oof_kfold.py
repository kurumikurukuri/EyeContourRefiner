"""固定 Fold ファイルによる OOF (Out-Of-Fold) 訓練オーケストレーション（単一/デュアルブランチ共通）。

run_kfold() / run_dual_kfold() は実行ごとに GroupKFold を動的計算するが、OOF 予測の
永続化・再現には「同じ分割を何度でも使い回せる」ことが重要なため、事前に固定の
fold{1..k}.csv を生成して保存し、それを読み込んで訓練する経路を提供する。

単一ブランチ（backbone: custom 等）には run_oof_kfold() / run_nested_oof_kfold() を、
デュアルブランチ（backbone: custom_dual）には run_dual_oof_kfold() /
run_nested_dual_oof_kfold() を使う。fold分割（split_fixed_folds）・OOF CSV スキーマ
（OofPrediction）はbackboneに依存しないため共通。

スコアリング（残差ランキング等）はこのモジュールの責務外。OOF 予測の収集・
永続化（CSV化）までを扱う。
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.model_selection import GroupKFold

from eyeliner.data.dataset import load_manifest
from eyeliner.data.weights import compute_eye_meta
from eyeliner.training.dual_branch import build_branch_configs
from eyeliner.training.dual_kfold import DualTrainConfig, _predict_dual_records, _train_single_dual_fold
from eyeliner.training.kfold import (
    FoldResult,
    _label_range_and_min,
    _predict_records,
    _train_single_fold,
)
from eyeliner.training.oof import OofPrediction
from eyeliner.training.trainer import TrainConfig, set_seed

import torch

logger = logging.getLogger(__name__)


def split_fixed_folds(
    manifest_path: str | Path,
    out_dir: str | Path,
    k: int = 5,
    seed: int = 42,
) -> list[Path]:
    """manifest を image_path 単位の GroupKFold で k 分割し、fold{1..k}.csv として保存する。

    保存後は同じファイルを読み込むことで、何度訓練を実行しても同一の分割を再現できる
    （sklearn GroupKFold(shuffle=True, random_state=seed) はシード固定で決定的）。

    Returns:
        書き出した fold ファイルのパスのリスト（fold1.csv, fold2.csv, ...）
    """
    records = load_manifest(manifest_path)
    if not records:
        raise ValueError(f"manifest にレコードがありません: {manifest_path}")

    groups = [r["image_path"] for r in records]
    kf = GroupKFold(n_splits=k, shuffle=True, random_state=seed)
    splits = list(kf.split(records, groups=groups))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())

    fold_paths: list[Path] = []
    for i, (_, test_idx) in enumerate(splits):
        fold_records = [records[j] for j in test_idx]
        fold_path = out_dir / f"fold{i + 1}.csv"
        with open(fold_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(fold_records)
        fold_paths.append(fold_path)
        logger.info("fold%d.csv: %d 件", i + 1, len(fold_records))

    return fold_paths


@dataclass
class OofKfoldResult:
    """run_oof_kfold() の戻り値。"""

    fold_results: list[FoldResult]
    oof_predictions: list[OofPrediction] = field(default_factory=list)
    n_total: int = 0


def run_oof_kfold(
    fold_manifest_paths: list[str | Path],
    exp_dir: str | Path,
    config: TrainConfig,
) -> OofKfoldResult:
    """固定 fold ファイル群を使い leave-one-fold-out で訓練し、OOF 予測を集約する。

    各 fold は他の全 fold を訓練データ、自身を検証データとして 1 回ずつ学習される
    （GroupKFold で生成された fold は image_path 単位で排他的なため、全サンプルが
    重複なく 1 回ずつ OOF 予測を持つ）。

    注意: 検証セット（早期終了の判定基準）と OOF 予測対象が同一であるため、
    早期終了が暗黙にその fold に最適化されるわずかな楽観性がある。これを避けたい
    場合は run_nested_oof_kfold() を使うこと。

    Args:
        fold_manifest_paths: split_fixed_folds() が生成した fold{1..k}.csv のパス群
        exp_dir:             モデル・OOF 予測を保存するディレクトリ
        config:               訓練設定
    """
    set_seed(config.seed)

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    fold_records_list = [load_manifest(p) for p in fold_manifest_paths]
    all_records = [r for fold in fold_records_list for r in fold]
    compute_eye_meta(all_records)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("デバイス: %s", device)

    use_transfer = config.backbone not in {"custom", "soft_argmax"}
    effective_freeze = config.freeze_epochs if use_transfer else 0
    label_range, label_t_min = _label_range_and_min(config)
    dl_kwargs = {"num_workers": config.num_workers, "persistent_workers": config.num_workers > 0}

    fold_results: list[FoldResult] = []
    oof_predictions: list[OofPrediction] = []

    k = len(fold_records_list)
    for i in range(k):
        logger.info("\n===== OOF Fold %d/%d =====", i + 1, k)
        val_records = fold_records_list[i]
        train_records = [r for j, fold in enumerate(fold_records_list) if j != i for r in fold]

        fold_result = _train_single_fold(
            train_records, val_records, i + 1, exp_dir, config,
            label_range, device, dl_kwargs,
            use_transfer, effective_freeze,
        )
        fold_results.append(fold_result)
        oof = _predict_records(
            fold_result.model_path, val_records, config,
            label_range, label_t_min, device, dl_kwargs, i + 1,
        )
        oof_predictions.extend(oof)

    return OofKfoldResult(
        fold_results=fold_results,
        oof_predictions=oof_predictions,
        n_total=len(all_records),
    )


def run_nested_oof_kfold(
    fold_manifest_paths: list[str | Path],
    exp_dir: str | Path,
    config: TrainConfig,
) -> OofKfoldResult:
    """入れ子 (nested) CV で、各 round の Test を学習・早期終了判定に一切使わずに OOF 予測する。

    fold{1..k}.csv（k>=3）を k ラウンドでローテーションする:

        round i: Test = fold[i], Val = fold[(i+1) % k], Train = 残り k-2 個

    Val は早期終了の判定にのみ使われ、保存されるベストモデルは Test を一度も学習に
    使わない。round 終了後、保存されたベストモデルで Test を推論し OOF 予測とする
    （run_oof_kfold の「Val=Test 兼用」より厳密だが、学習時間は変わらず k ラウンド）。

    k ラウンドを終えると全 fold が Test を 1 回ずつ経験するため、fold{1..k}.csv の
    全件（= 元 manifest の全件）が重複なく OOF 予測を持つ。

    Args:
        fold_manifest_paths: split_fixed_folds() が生成した fold{1..k}.csv のパス群（k>=3）
        exp_dir:             モデル・OOF 予測を保存するディレクトリ
        config:               訓練設定
    """
    set_seed(config.seed)

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    fold_records_list = [load_manifest(p) for p in fold_manifest_paths]
    k = len(fold_records_list)
    if k < 3:
        raise ValueError(f"nested CV には k>=3 が必要です（Test/Val/Train の3役）: k={k}")

    all_records = [r for fold in fold_records_list for r in fold]
    compute_eye_meta(all_records)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("デバイス: %s", device)

    use_transfer = config.backbone not in {"custom", "soft_argmax"}
    effective_freeze = config.freeze_epochs if use_transfer else 0
    label_range, label_t_min = _label_range_and_min(config)
    dl_kwargs = {"num_workers": config.num_workers, "persistent_workers": config.num_workers > 0}

    fold_results: list[FoldResult] = []
    oof_predictions: list[OofPrediction] = []

    for i in range(k):
        val_idx = (i + 1) % k
        test_records = fold_records_list[i]
        val_records = fold_records_list[val_idx]
        train_records = [
            r for j, fold in enumerate(fold_records_list)
            if j != i and j != val_idx for r in fold
        ]
        logger.info(
            "\n===== Nested OOF Round %d/%d ===== (test=fold%d, val=fold%d, train=残り%d fold)",
            i + 1, k, i + 1, val_idx + 1, k - 2,
        )

        fold_result = _train_single_fold(
            train_records, val_records, i + 1, exp_dir, config,
            label_range, device, dl_kwargs,
            use_transfer, effective_freeze,
        )
        fold_results.append(fold_result)
        oof = _predict_records(
            fold_result.model_path, test_records, config,
            label_range, label_t_min, device, dl_kwargs, i + 1,
        )
        oof_predictions.extend(oof)

    return OofKfoldResult(
        fold_results=fold_results,
        oof_predictions=oof_predictions,
        n_total=len(all_records),
    )


def run_dual_oof_kfold(
    fold_manifest_paths: list[str | Path],
    exp_dir: str | Path,
    config: DualTrainConfig,
) -> OofKfoldResult:
    """固定 fold ファイル群を使い leave-one-fold-out でデュアルブランチ訓練し、OOF 予測を集約する。

    run_oof_kfold() のデュアルブランチ版。検証セット（早期終了の判定基準）と OOF 予測対象が
    同一であるため、早期終了が暗黙にその fold に最適化されるわずかな楽観性がある。
    これを避けたい場合は run_nested_dual_oof_kfold() を使うこと。

    Args:
        fold_manifest_paths: split_fixed_folds() が生成した fold{1..k}.csv のパス群
        exp_dir:             モデル・OOF 予測を保存するディレクトリ
        config:               デュアルブランチ訓練設定
    """
    set_seed(config.seed)

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    fold_records_list = [load_manifest(p) for p in fold_manifest_paths]
    all_records = [r for fold in fold_records_list for r in fold]
    compute_eye_meta(all_records)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("デバイス: %s", device)

    cfg_a, cfg_b = build_branch_configs(config.pq_rescale_mode_b)
    dl_kwargs = {"num_workers": config.num_workers, "persistent_workers": config.num_workers > 0}

    fold_results: list[FoldResult] = []
    oof_predictions: list[OofPrediction] = []

    k = len(fold_records_list)
    for i in range(k):
        logger.info("\n===== Dual OOF Fold %d/%d =====", i + 1, k)
        val_records = fold_records_list[i]
        train_records = [r for j, fold in enumerate(fold_records_list) if j != i for r in fold]

        fold_result = _train_single_dual_fold(
            train_records, val_records, i + 1, exp_dir, config,
            cfg_a, cfg_b, device, dl_kwargs,
        )
        fold_results.append(fold_result)
        oof = _predict_dual_records(
            fold_result.model_path, val_records, config,
            cfg_a, cfg_b, device, dl_kwargs, i + 1,
        )
        oof_predictions.extend(oof)

    return OofKfoldResult(
        fold_results=fold_results,
        oof_predictions=oof_predictions,
        n_total=len(all_records),
    )


def run_nested_dual_oof_kfold(
    fold_manifest_paths: list[str | Path],
    exp_dir: str | Path,
    config: DualTrainConfig,
) -> OofKfoldResult:
    """入れ子 (nested) CV で、各 round の Test を学習・早期終了判定に一切使わずにデュアルブランチ
    モデルで OOF 予測する。run_nested_oof_kfold() のデュアルブランチ版。

    fold{1..k}.csv（k>=3）を k ラウンドでローテーションする:

        round i: Test = fold[i], Val = fold[(i+1) % k], Train = 残り k-2 個

    Args:
        fold_manifest_paths: split_fixed_folds() が生成した fold{1..k}.csv のパス群（k>=3）
        exp_dir:             モデル・OOF 予測を保存するディレクトリ
        config:               デュアルブランチ訓練設定
    """
    set_seed(config.seed)

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    fold_records_list = [load_manifest(p) for p in fold_manifest_paths]
    k = len(fold_records_list)
    if k < 3:
        raise ValueError(f"nested CV には k>=3 が必要です（Test/Val/Train の3役）: k={k}")

    all_records = [r for fold in fold_records_list for r in fold]
    compute_eye_meta(all_records)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("デバイス: %s", device)

    cfg_a, cfg_b = build_branch_configs(config.pq_rescale_mode_b)
    dl_kwargs = {"num_workers": config.num_workers, "persistent_workers": config.num_workers > 0}

    fold_results: list[FoldResult] = []
    oof_predictions: list[OofPrediction] = []

    for i in range(k):
        val_idx = (i + 1) % k
        test_records = fold_records_list[i]
        val_records = fold_records_list[val_idx]
        train_records = [
            r for j, fold in enumerate(fold_records_list)
            if j != i and j != val_idx for r in fold
        ]
        logger.info(
            "\n===== Nested Dual OOF Round %d/%d ===== (test=fold%d, val=fold%d, train=残り%d fold)",
            i + 1, k, i + 1, val_idx + 1, k - 2,
        )

        fold_result = _train_single_dual_fold(
            train_records, val_records, i + 1, exp_dir, config,
            cfg_a, cfg_b, device, dl_kwargs,
        )
        fold_results.append(fold_result)
        oof = _predict_dual_records(
            fold_result.model_path, test_records, config,
            cfg_a, cfg_b, device, dl_kwargs, i + 1,
        )
        oof_predictions.extend(oof)

    return OofKfoldResult(
        fold_results=fold_results,
        oof_predictions=oof_predictions,
        n_total=len(all_records),
    )
