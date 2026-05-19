"""Losses used in Phase 1 training.

Training signal: GT pointmap supervision (unproject TUM Kinect depth).
Auxiliary regularizer: multi-view consistency in world frame.

Both are scale-aware (TUM gives metric depth in metres).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vggt_mamba.eval.metrics import multi_view_consistency


def pointmap_l1_loss(
    pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Mean L1 over valid pixels.

    Args:
        pred:  (B, T, 3, H, W) predicted points in camera frame.
        gt:    (B, T, 3, H, W) gt points in camera frame.
        valid: (B, T, H, W) bool mask.
    """
    diff = (pred - gt).abs().sum(dim=2)  # (B, T, H, W)
    m = valid.float()
    denom = m.sum().clamp_min(1.0)
    return (diff * m).sum() / denom


def pointmap_log_loss(
    pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, eps: float = 1e-3
) -> torch.Tensor:
    """Scale-invariant log-space loss (Eigen-style) on the depth channel."""
    pred_d = pred[:, :, 2].clamp_min(eps)
    gt_d = gt[:, :, 2].clamp_min(eps)
    d = torch.log(pred_d) - torch.log(gt_d)
    m = valid.float()
    denom = m.sum().clamp_min(1.0)
    sq = (d ** 2 * m).sum() / denom
    mean = (d * m).sum() / denom
    return sq - 0.5 * mean ** 2


def phase1_loss(
    pred_pmap_cam: torch.Tensor,
    gt_pmap_cam: torch.Tensor,
    valid: torch.Tensor,
    poses_w_c: torch.Tensor,
    consistency_weight: float = 0.1,
    consistency_samples: int = 1024,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined supervision: L1 + scale-invariant log + multi-view consistency.

    Returns (total_loss, scalar_log_dict).
    """
    l1 = pointmap_l1_loss(pred_pmap_cam, gt_pmap_cam, valid)
    logl = pointmap_log_loss(pred_pmap_cam, gt_pmap_cam, valid)
    if consistency_weight > 0:
        cons = multi_view_consistency(
            pred_pmap_cam, valid, poses_w_c, n_samples=consistency_samples
        )
    else:
        cons = pred_pmap_cam.new_zeros(())
    total = l1 + logl + consistency_weight * cons
    return total, {
        "loss_total": float(total.detach()),
        "loss_l1": float(l1.detach()),
        "loss_log": float(logl.detach()),
        "loss_mvc": float(cons.detach()),
    }
