"""EXP019 学習型ゲート訓練スクリプト。

凍結した X0/X1/X2/X3 全 4 モデルの予測をブレンドする
GateCNN を学習する。ゲートは画像パッチから softmax 重み (w0,w1,w2,w3) を出力し、
t_final = w0*t_x0 + w1*t_x1 + w2*t_x2 + w3*t_x3 で最終予測を生成する。

前提: train_predictions.csv / test_predictions.csv が全条件の outputs に存在すること。
（存在しない場合は先に `python experiments/EXP019_pq_transform/scr/infer_exp019.py` を実行）

使い方:
    # デフォルト設定で学習（4 experts）
    python experiments/EXP019_pq_transform/scr/ensemble/train_gate.py

    # 2 experts (X0+X2) に絞る
    python experiments/EXP019_pq_transform/scr/ensemble/train_gate.py --experts X0,X2
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[4]
EXP_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = EXP_DIR / "outputs"
MANIFEST_DIR = EXP_DIR / "manifests"
GATE_OUTPUT_DIR = OUTPUT_DIR / "_ensemble" / "gate"
# 実行ごとにタイムスタンプ付きサブディレクトリを作成（上書き防止）
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = GATE_OUTPUT_DIR / RUN_TIMESTAMP
ALL_VARIANTS = ["X0_baseline", "X1_eye_height", "X2_eye_scale", "X3_mean_pq"]
VARIANT_SHORT = {"X0_baseline": "X0", "X1_eye_height": "X1",
                 "X2_eye_scale": "X2", "X3_mean_pq": "X3"}

MANIFEST_TRAIN = MANIFEST_DIR / "manifest_train.csv"
MANIFEST_TEST = MANIFEST_DIR / "manifest_test.csv"

sys.path.insert(0, str(ROOT))
from eyeliner.data.dataset import EyelinerPatchDataset, load_manifest  # noqa: E402
from eyeliner.data.patch_utils import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from eyeliner.data.weights import compute_eye_meta  # noqa: E402
from eyeliner.models.ensemble import create_ensemble_from_dir  # noqa: E402
from eyeliner.training.trainer import set_seed  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))  # ensemble/ を import 可能に
from gate_cnn import (  # noqa: E402
    GateCNN, count_parameters, count_trainable_parameters,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 帯区分（EXP019 と統一）
BANDS = [
    ("t<0", -float("inf"), 0.0),
    ("[0,0.5)", 0.0, 0.5),
    ("[0.5,1.0)", 0.5, 1.0),
    ("[1.0,1.2]", 1.0, 1.2),
    ("t>1.2", 1.2, float("inf")),
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GateTrainConfig:
    """ゲート学習のハイパーパラメータ。"""

    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    patience: int = 20
    seed: int = 42
    weight_decay: float = 1e-4
    num_conv_layers: int = 3
    use_batchnorm: bool = True
    dropout_rate: float = 0.1
    band_weight: float = 5.0  # 裾（t<0, t>1.2）の損失倍率。0=無効
    aux_dim: int = 3  # 補助特徴量の次元数 [log(eye_scale), log(pq_dist), pq_ratio]
    experts: list = field(default_factory=lambda: ALL_VARIANTS)
    num_workers: int = 0
    # X0 パッチパラメータ（ゲートは生PQ座標系のパッチを見る）
    patch_t_min: float = -2.0
    patch_t_max: float = 3.0
    patch_perp_scale: float = 2.0
    patch_h: int = 64
    patch_w: int = 64
    label_t_min: float | None = -1.0
    label_t_max: float | None = 2.0
    norm_mean: list = field(default_factory=lambda: list(IMAGENET_MEAN))
    norm_std: list = field(default_factory=lambda: list(IMAGENET_STD))


# ---------------------------------------------------------------------------
# Key-based Dataset Wrapper
# ---------------------------------------------------------------------------


SampleKey = tuple  # (image_path, part_name, sub_key, point_index)


class GatePatchDataset(Dataset):
    """EyelinerPatchDataset のラッパー。サンプルキーと補助特徴量を追加で返す。

    __getitem__ は (x, is_upper, pq_ratio, y, weight, val_weight,
                     dist_from_center, norm_eye_scale, key, aux_features)
    の 10-tuple を返す。
    aux_features = [eye_scale, pq_dist, pq_ratio]（3次元）
    """

    def __init__(self, base_dataset: EyelinerPatchDataset) -> None:
        self.base = base_dataset
        self.records = base_dataset.records  # 参照を公開（ルックアップ用）

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, is_upper, pq_ratio, y, weight, val_weight, dist, norm_eye = self.base[idx]
        rec = self.records[idx]
        key: SampleKey = (
            rec["image_path"], rec["part_name"],
            rec["sub_key"], int(rec["point_index"]),
        )
        # 補助特徴量: log(eye_scale), log(pq_dist), pq_ratio
        eye_scale = float(rec.get("eye_scale", 1.0))
        px, py = float(rec["px"]), float(rec["py"])
        qx, qy = float(rec["qx"]), float(rec["qy"])
        pq_dist = ((qx - px) ** 2 + (qy - py) ** 2) ** 0.5
        pq_ratio_val = float(rec.get("pq_ratio", pq_dist / max(eye_scale, 1e-6)))
        # log 正規化（値域が 1〜3 桁異なるため）
        import math
        log_eye = math.log(max(eye_scale, 1.0))
        log_pq = math.log(max(pq_dist, 0.01))
        aux = torch.tensor([log_eye, log_pq, pq_ratio_val], dtype=torch.float32)
        return x, is_upper, pq_ratio, y, weight, val_weight, dist, norm_eye, key, aux


def _collate_gate(batch):
    """GatePatchDataset 用のカスタム collate_fn。"""
    xs, iu, pq, ys, ws, vw, dc, ne, keys, auxs = zip(*batch)
    return (
        torch.stack(xs),
        torch.stack(iu),
        torch.stack(pq),
        torch.stack(ys),
        torch.stack(ws),
        torch.stack(vw),
        torch.stack(dc),
        torch.stack(ne),
        list(keys),
        torch.stack(auxs),
    )


# ---------------------------------------------------------------------------
# Precomputed prediction loading
# ---------------------------------------------------------------------------


PredLookups = dict[str, dict[SampleKey, float]]  # variant_name → key → pred_t_raw


def load_pred_lookups(
    variants: list[str], split: str = "train",
) -> tuple[PredLookups, dict[SampleKey, float]]:
    """全 variant の推論 CSV から (preds, true_t) を返す。"""
    preds: PredLookups = {}
    true_t: dict[SampleKey, float] = {}

    for variant in variants:
        preds[variant] = {}
        path = OUTPUT_DIR / variant / f"{split}_predictions.csv"
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key: SampleKey = (
                    row["image_path"], row["part_name"],
                    row["sub_key"], int(row["point_index"]),
                )
                preds[variant][key] = float(row["pred_t_raw"])
                if variant == variants[0]:
                    true_t[key] = float(row["true_t_raw"])

    return preds, true_t


# ---------------------------------------------------------------------------
# Training / Validation
# ---------------------------------------------------------------------------


def run_epoch_gate(
    model: GateCNN,
    dataloader: DataLoader,
    preds_all: PredLookups,
    true_t_lookup: dict[SampleKey, float],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    band_weight: float = 0.0,
) -> dict:
    """ゲートの 1 エポック（訓練または検証）。

    Args:
        preds_all: {variant: {key: pred_t_raw}} — 全専門家の予測値
        optimizer: None なら検証モード
        band_weight: 裾サンプルの損失倍率（検証時は無効）

    Returns:
        {"loss", "mae", "preds", "truths",
         "expert_preds": {variant: np.ndarray},
         "weights": np.ndarray (B, num_experts),
         "keys": list[SampleKey]}
    """
    variant_names = list(preds_all.keys())
    n_experts = len(variant_names)
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    all_preds: list[np.ndarray] = []
    all_truths: list[np.ndarray] = []
    expert_collect: dict[str, list[np.ndarray]] = {v: [] for v in variant_names}
    all_weights: list[np.ndarray] = []
    all_keys: list[SampleKey] = []
    total_loss = 0.0
    n_samples = 0

    for batch in dataloader:
        x, is_upper, _, __, ___, ____, _____, ______, keys, aux = batch
        x = x.to(device)
        is_upper = is_upper.to(device)
        aux = aux.to(device)

        # 全専門家の予測値をキーでルックアップ → (B, n_experts)
        true_vals = [true_t_lookup.get(k, 0.0) for k in keys]
        t_experts_list = []
        for v in variant_names:
            t_experts_list.append([preds_all[v].get(k, 0.0) for k in keys])
        t_experts = torch.tensor(t_experts_list, dtype=torch.float32, device=device).T
        true_t = torch.tensor(true_vals, dtype=torch.float32, device=device).unsqueeze(1)

        # ゲート推論
        with torch.set_grad_enabled(is_train):
            w = model(x, is_upper, aux_features=aux)
            if n_experts == 2:
                t_pred = w * t_experts[:, 0:1] + (1.0 - w) * t_experts[:, 1:2]
            else:
                t_pred = (w * t_experts).sum(dim=1, keepdim=True)

        # 帯別重み付き L1 損失
        per_sample_loss = nn.functional.l1_loss(t_pred, true_t, reduction="none")
        if is_train and band_weight > 0:
            tail_mask = (true_t < 0) | (true_t > 1.2)
            loss_weights = torch.where(tail_mask, band_weight, 1.0)
            loss = (per_sample_loss * loss_weights).mean()
        else:
            loss = per_sample_loss.mean()

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(x)
        n_samples += len(x)

        all_preds.append(t_pred.detach().cpu().numpy())
        all_truths.append(true_t.detach().cpu().numpy())
        for i, v in enumerate(variant_names):
            expert_collect[v].append(t_experts[:, i].detach().cpu().numpy())
        all_weights.append(w.detach().cpu().numpy())
        all_keys.extend(keys)

    preds = np.concatenate(all_preds).flatten()
    truths = np.concatenate(all_truths).flatten()
    expert_preds = {v: np.concatenate(expert_collect[v]).flatten() for v in variant_names}
    weights = np.concatenate(all_weights)

    return {
        "loss": total_loss / max(n_samples, 1),
        "mae": float(np.abs(preds - truths).mean()),
        "preds": preds,
        "truths": truths,
        "expert_preds": expert_preds,
        "weights": weights,
        "keys": all_keys,
    }


def format_band_table(
    preds: np.ndarray,
    expert_preds: dict[str, np.ndarray],
    truths: np.ndarray,
) -> List[str]:
    """帯別ブレークダウンをフォーマットする（N 専門家対応）。"""
    variant_names = list(expert_preds.keys())
    n_experts = len(variant_names)
    # 単純平均
    mean_preds = np.stack(list(expert_preds.values())).mean(axis=0)

    header_cols = [f"{'band':<14}", f"{'N':>6}"]
    for v in variant_names:
        header_cols.append(f"{VARIANT_SHORT[v]:>8}")
    header_cols.append(f"{'mean':>8}")
    header_cols.append(f"{'gate':>8}")
    lines = ["".join(header_cols)]
    lines.append("-" * (22 + 10 * (n_experts + 2)))

    for band_name, low, high in BANDS:
        mask = (truths >= low) & (truths < high)
        n = int(mask.sum())
        if n == 0:
            continue
        row = f"{band_name:<14} {n:>6}"
        for v in variant_names:
            row += f" {float(np.abs(expert_preds[v][mask] - truths[mask]).mean()):>8.4f}"
        row += f" {float(np.abs(mean_preds[mask] - truths[mask]).mean()):>8.4f}"
        row += f" {float(np.abs(preds[mask] - truths[mask]).mean()):>8.4f}"
        lines.append(row)
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP019 学習型ゲート訓練")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-conv-layers", type=int, default=3)
    parser.add_argument("--no-batchnorm", action="store_true")
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--band-weight", type=float, default=5.0,
                        help="裾サンプル(t<0, t>1.2)の損失倍率。0=無効")
    parser.add_argument("--experts", type=str, default="X0,X1,X2,X3",
                        help="使用する専門家（カンマ区切り）。例: X0,X2")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # 専門家名をパース（カンマ区切り → variant 名にマッピング）
    expert_short = [e.strip() for e in args.experts.split(",")]
    name_to_variant = {v.split("_")[0]: v for v in ALL_VARIANTS}
    experts = [name_to_variant[e] for e in expert_short]

    config = GateTrainConfig(
        lr=args.lr, batch_size=args.batch, epochs=args.epochs,
        patience=args.patience, seed=args.seed, weight_decay=args.weight_decay,
        num_conv_layers=args.num_conv_layers,
        use_batchnorm=not args.no_batchnorm,
        dropout_rate=args.dropout_rate, num_workers=args.num_workers,
        band_weight=args.band_weight, experts=experts,
    )

    set_seed(config.seed)
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("デバイス: %s", device)

    # ------------------------------------------------------------------
    # 1. 凍結した全専門家モデルをロード
    # ------------------------------------------------------------------
    frozen_models: dict[str, nn.Module] = {}
    for variant in config.experts:
        logger.info("%s モデルをロード中...", variant)
        model, backbone = create_ensemble_from_dir(OUTPUT_DIR / variant)
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        frozen_models[variant] = model
        logger.info("  backbone=%s", backbone)

    # ------------------------------------------------------------------
    # 2. 事前計算済み予測値をロード
    # ------------------------------------------------------------------
    logger.info("事前計算済み予測値をロード中...")
    preds_train, true_t_train = load_pred_lookups(config.experts, "train")
    preds_test, true_t_test = load_pred_lookups(config.experts, "test")
    logger.info("  train: %d, test: %d", len(true_t_train), len(true_t_test))

    # ------------------------------------------------------------------
    # 3. DataLoader を作成
    # ------------------------------------------------------------------
    train_records = load_manifest(MANIFEST_TRAIN)
    test_records = load_manifest(MANIFEST_TEST)
    compute_eye_meta(train_records)
    compute_eye_meta(test_records)

    def _make_ds(records, augment: bool) -> GatePatchDataset:
        base = EyelinerPatchDataset(
            records,
            augment_data=augment,
            aug_flags=127 if augment else 0,
            norm_mean=config.norm_mean, norm_std=config.norm_std,
            patch_t_min=config.patch_t_min, patch_t_max=config.patch_t_max,
            patch_perp_scale=config.patch_perp_scale,
            patch_size=(config.patch_h, config.patch_w),
            use_t_norm=True,
            label_t_min=config.label_t_min, label_t_max=config.label_t_max,
            pq_rescale_mode="none", pq_rescale_factor=1.0,
        )
        return GatePatchDataset(base)

    train_ds = _make_ds(train_records, augment=True)
    test_ds = _make_ds(test_records, augment=False)
    logger.info("train records: %d, test records: %d",
                len(train_records), len(test_records))

    dl_kwargs = {"num_workers": config.num_workers,
                 "persistent_workers": config.num_workers > 0}
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        collate_fn=_collate_gate, **dl_kwargs)
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False,
        collate_fn=_collate_gate, **dl_kwargs)

    # ------------------------------------------------------------------
    # 4. ゲートモデルを作成
    # ------------------------------------------------------------------
    n_experts = len(config.experts)
    gate = GateCNN(
        num_experts=n_experts,
        num_conv_layers=config.num_conv_layers,
        aux_dim=config.aux_dim,
        use_batchnorm=config.use_batchnorm,
        dropout_rate=config.dropout_rate,
    ).to(device)
    logger.info("GateCNN: total=%d, trainable=%d",
                count_parameters(gate), count_trainable_parameters(gate))

    optimizer = torch.optim.Adam(
        gate.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    # ------------------------------------------------------------------
    # 5. 学習ループ
    # ------------------------------------------------------------------
    best_val_mae = float("inf")
    best_epoch = 0
    no_improve = 0

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        train_result = run_epoch_gate(
            gate, train_loader, preds_train, true_t_train,
            optimizer, device, band_weight=config.band_weight,
        )
        val_result = run_epoch_gate(
            gate, test_loader, preds_test, true_t_test,
            None, device, band_weight=0.0,
        )

        scheduler.step(val_result["mae"])
        current_lr = optimizer.param_groups[0]["lr"]

        # 各専門家の MAE をログ表示
        expert_maes = {v: float(np.abs(
            val_result["expert_preds"][v] - val_result["truths"]
        ).mean()) for v in config.experts}
        mean_pred = np.stack(list(val_result["expert_preds"].values())).mean(axis=0)
        mae_mean = float(np.abs(mean_pred - val_result["truths"]).mean())

        mae_str = " ".join(f"{VARIANT_SHORT[v]}={expert_maes[v]:.4f}" for v in config.experts)
        logger.info(
            "Epoch %3d  loss=%.6f  gate=%.6f  %s  mean=%.6f  lr=%.2e",
            epoch + 1, train_result["loss"], val_result["mae"],
            mae_str, mae_mean, current_lr,
        )

        if val_result["mae"] < best_val_mae:
            best_val_mae = val_result["mae"]
            best_epoch = epoch + 1
            no_improve = 0
            torch.save({
                "type": "gate",
                "state_dict": gate.state_dict(),
                "num_experts": n_experts,
                "num_conv_layers": config.num_conv_layers,
                "aux_dim": config.aux_dim,
                "use_batchnorm": config.use_batchnorm,
                "dropout_rate": config.dropout_rate,
            }, RUN_DIR / "gate.pt")
        else:
            no_improve += 1
            if no_improve >= config.patience:
                logger.info("Early stopping (epoch %d)", epoch + 1)
                break

    logger.info("Best val MAE: %.6f (epoch %d)", best_val_mae, best_epoch)

    # ------------------------------------------------------------------
    # 6. 最終評価
    # ------------------------------------------------------------------
    best_ckpt = torch.load(RUN_DIR / "gate.pt", map_location="cpu",
                           weights_only=True)
    gate.load_state_dict(best_ckpt["state_dict"])
    gate.to(device).eval()

    final = run_epoch_gate(
        gate, test_loader, preds_test, true_t_test,
        None, device, band_weight=0.0,
    )

    band_lines = format_band_table(
        final["preds"], final["expert_preds"], final["truths"],
    )
    logger.info("\n===== 帯別ブレークダウン (test, 生t MAE) =====\n%s",
                "\n".join(band_lines))

    # ------------------------------------------------------------------
    # 7. 結果を保存
    # ------------------------------------------------------------------
    expert_maes_final = {v: float(np.abs(
        final["expert_preds"][v] - final["truths"]
    ).mean()) for v in config.experts}
    mean_pred = np.stack(list(final["expert_preds"].values())).mean(axis=0)
    mae_mean_final = float(np.abs(mean_pred - final["truths"]).mean())

    results = {
        "best_val_mae": best_val_mae, "best_epoch": best_epoch,
        "test_mae_gate": float(final["mae"]),
        "test_mae_mean": mae_mean_final,
        "n_test_samples": len(final["preds"]),
    }
    for v in config.experts:
        results[f"test_mae_{VARIANT_SHORT[v]}"] = expert_maes_final[v]

    # 重みの統計
    w = final["weights"]
    if w.ndim == 1:
        results["mean_weight"] = float(w.mean())
        results["weight_std"] = float(w.std())
    else:
        for i, v in enumerate(config.experts):
            results[f"mean_w_{VARIANT_SHORT[v]}"] = float(w[:, i].mean())
            results[f"std_w_{VARIANT_SHORT[v]}"] = float(w[:, i].std())

    experiment_json = {
        "exp_name": "EXP019_gate",
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": {
            "gate_experts": config.experts,
            "gate_lr": config.lr, "gate_batch_size": config.batch_size,
            "gate_epochs": config.epochs, "gate_patience": config.patience,
            "gate_seed": config.seed, "gate_weight_decay": config.weight_decay,
            "gate_num_conv_layers": config.num_conv_layers,
            "gate_use_batchnorm": config.use_batchnorm,
            "gate_dropout_rate": config.dropout_rate,
            "gate_band_weight": config.band_weight,
            "gate_aux_dim": config.aux_dim,
        },
        "results": results,
    }

    with open(RUN_DIR / "experiment.json", "w", encoding="utf-8") as f:
        json.dump(experiment_json, f, indent=2, ensure_ascii=False)

    logger.info("\n[saved] %s", RUN_DIR / "gate.pt")
    logger.info("[saved] %s", RUN_DIR / "experiment.json")

    best_single = min(expert_maes_final.values())
    best_name = [v for v, m in expert_maes_final.items() if m == best_single][0]
    imp_vs_best = (final["mae"] - best_single) / best_single * 100
    imp_vs_mean = (final["mae"] - mae_mean_final) / mae_mean_final * 100
    logger.info("\n===== サマリー =====")
    logger.info("gate MAE:       %.6f  (vs best %s %+.1f%%, vs mean %+.1f%%)",
                final["mae"], VARIANT_SHORT[best_name], imp_vs_best, imp_vs_mean)
    for v in config.experts:
        logger.info("%s single MAE:  %.6f", VARIANT_SHORT[v], expert_maes_final[v])
    logger.info("mean ensemble:  %.6f", mae_mean_final)
    if w.ndim > 1:
        w_str = " ".join(f"{VARIANT_SHORT[v]}={float(w[:, i].mean()):.3f}"
                         for i, v in enumerate(config.experts))
        logger.info("mean weights: %s", w_str)


if __name__ == "__main__":
    main()
