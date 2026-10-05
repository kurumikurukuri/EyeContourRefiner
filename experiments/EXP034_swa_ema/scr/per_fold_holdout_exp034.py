"""EXP034: fold 単体モデルの hold-out MAE（生t）を config ごとに比較する。

CV（fold 単体モデルの val MAE）では EMA が base 比 約 −2% だったが、hold-out（3 fold
アンサンブル）では差が消えた。重み平均の効果が fold アンサンブルの平均化と重複して
吸収されたのかを、fold 単体モデルの hold-out MAE で確認する。

使い方:
    python experiments/EXP034_swa_ema/scr/per_fold_holdout_exp034.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent.parent.parent
EXP_DIR = Path(__file__).parent.parent
MANIFEST_DIR = EXP_DIR / "manifests"
OUTPUT_DIR = EXP_DIR / "outputs"
sys.path.insert(0, str(ROOT))

from eyeliner.data.dataset import load_manifest  # noqa: E402
from eyeliner.data.dataset_dual import EyelinerPatchDualDataset  # noqa: E402
from eyeliner.data.weights import compute_eye_meta  # noqa: E402
from eyeliner.models.dual_ensemble import load_dual_model  # noqa: E402
from eyeliner.training.dual_branch import (  # noqa: E402
    build_branch_configs,
    model_pred_raw_t,
)

CONFIGS = ("base", "base_ema0999", "base_ema09998", "base_swa")
FOLDS = (1, 2, 3)


@torch.no_grad()
def fold_mae(model: torch.nn.Module, loader: DataLoader, t_true: torch.Tensor,
             cfg_a, cfg_b, device: torch.device) -> float:
    model.eval()
    preds: list[torch.Tensor] = []
    for batch in loader:
        t = model_pred_raw_t(
            model, batch[0].to(device), batch[1].to(device), batch[2].to(device),
            "independent", cfg_a, cfg_b, batch[12].to(device), batch[13].to(device), None,
        )
        preds.append(t.squeeze(-1).cpu())
    return float((torch.cat(preds) - t_true).abs().mean())


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_manifest(MANIFEST_DIR / "manifest_test.csv")
    compute_eye_meta(records)
    t_true = torch.tensor([float(r["t"]) for r in records])
    cfg_a, cfg_b = build_branch_configs("eye_scale")
    ds = EyelinerPatchDualDataset(records, cfg_a, cfg_b, augment_data=False, preload_images=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    results: dict[str, list[float]] = {}
    for config in CONFIGS:
        maes = []
        for fold in FOLDS:
            path = OUTPUT_DIR / config / f"eyeliner_detection_fold{fold}.pt"
            if not path.exists():
                raise FileNotFoundError(f"fold モデルが見つかりません: {path}")
            maes.append(fold_mae(load_dual_model(path).to(device), loader, t_true, cfg_a, cfg_b, device))
        results[config] = maes

    base_mean = sum(results["base"]) / len(FOLDS)
    print(f"\n  {'config':<15} {'fold1':>9} {'fold2':>9} {'fold3':>9} {'mean':>9} {'vs base':>8}")
    for config, maes in results.items():
        mean = sum(maes) / len(maes)
        print(f"  {config:<15} " + " ".join(f"{m:>9.6f}" for m in maes)
              + f" {mean:>9.6f} {(base_mean - mean) / base_mean * 100:>+7.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
