"""評価メトリクス。

指標の役割を明確に分離する:
  cv_mae      → ハイパーパラメータ選定（K-Fold CV の Best MAE 平均）
  holdout_mae → 汎化性能の不偏推定（固定 Hold-out セット、訓練後に一度だけ）
  pixel_mae   → 実用誤差の把握（ピクセル単位）
  relative_mae→ 目サイズ正規化誤差

重要: holdout_mae はハイパーパラメータ選定に使ってはならない。
     使うたびに「汎化性能の不偏推定」という性質が失われる。
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def cv_mae(fold_maes: list[float]) -> float:
    """K-Fold 各 Fold の Best MAE 平均。ハイパーパラメータ選定に使用する。"""
    return float(np.mean(fold_maes))


def holdout_mae(
    model: torch.nn.Module,
    holdout_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_pq_ratio: bool = False,
    use_distance_from_center: bool = False,
    use_norm_eye_scale: bool = False,
    use_aux_face_shape: bool = False,
) -> float:
    """固定 Hold-out セットでの MAE 評価。汎化性能の不偏推定に使用する。

    注意: この値を見てハイパーパラメータを調整することは禁止。
         訓練完了後に一度だけ呼び出すこと。
    """
    model.eval()
    all_preds: list[Tensor] = []
    all_labels: list[Tensor] = []

    with torch.no_grad():
        for batch in holdout_loader:
            x = batch[0].to(device)
            is_upper = batch[1].to(device)
            pq_ratio = batch[2]
            y = batch[3]
            dist_from_center = batch[6]
            norm_eye_scale = batch[7]
            pq_ratio_dev = pq_ratio.to(device) if use_pq_ratio else None
            dist_dev = dist_from_center.to(device) if use_distance_from_center else None
            nes_dev = norm_eye_scale.to(device) if use_norm_eye_scale else None
            aux_dev = batch[8].to(device) if use_aux_face_shape else None
            pred = model(x, is_upper, pq_ratio_dev, dist_dev, nes_dev, aux_dev)
            all_preds.append(pred.cpu())
            all_labels.append(y)

    preds_t = torch.cat(all_preds)
    labels_t = torch.cat(all_labels)
    return float((preds_t - labels_t).abs().mean().item())


def pixel_mae(
    t_pred: np.ndarray,
    t_true: np.ndarray,
    p_points: np.ndarray,
    q_points: np.ndarray,
) -> float:
    """ピクセル空間での平均絶対誤差。dt * pq_dist で計算する。

    Args:
        t_pred:   予測 t 値の配列 (N,)
        t_true:   正解 t 値の配列 (N,)
        p_points: P 点座標の配列 (N, 2)
        q_points: Q 点座標の配列 (N, 2)

    Returns:
        ピクセル単位の平均絶対誤差（float）
    """
    pq_dist = np.linalg.norm(q_points - p_points, axis=1)
    return float(np.mean(np.abs(t_pred - t_true) * pq_dist))


def relative_mae(
    t_pred: np.ndarray,
    t_true: np.ndarray,
    p_points: np.ndarray,
    q_points: np.ndarray,
    eye_scales: np.ndarray,
) -> float:
    """目サイズで正規化した相対誤差。目の大小に依らず公平に評価する。

    Args:
        t_pred:     予測 t 値の配列 (N,)
        t_true:     正解 t 値の配列 (N,)
        p_points:   P 点座標の配列 (N, 2)
        q_points:   Q 点座標の配列 (N, 2)
        eye_scales: 目スケール（目の横幅）の配列 (N,)

    Returns:
        目サイズ正規化誤差（float）
    """
    pq_dist = np.linalg.norm(q_points - p_points, axis=1)
    return float(np.mean(np.abs(t_pred - t_true) * pq_dist / np.maximum(eye_scales, 1.0)))
