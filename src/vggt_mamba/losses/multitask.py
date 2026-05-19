"""Multi-task loss for GeoMamba.

Combines:
  - depth/pointmap L1 + scale-invariant log
  - multi-view consistency Chamfer (auxiliary)
  - camera translation (Huber) + camera rotation (geodesic)
  - optional track L1

All losses are normalised so weights of order 1 produce comparable
gradient magnitudes. Returns (total, scalar_log_dict).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vggt_mamba.eval.metrics import multi_view_consistency
from vggt_mamba.losses.pointmap import pointmap_l1_loss, pointmap_log_loss


def _quat_geodesic_loss(pred_q: torch.Tensor, gt_q: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between two unit quaternions (radians).

    Args:
        pred_q, gt_q: (..., 4) — already (or nearly) unit-norm.
    """
    pred_q = F.normalize(pred_q, dim=-1)
    gt_q = F.normalize(gt_q, dim=-1)
    dot = (pred_q * gt_q).sum(dim=-1).abs().clamp(0.0, 1.0)
    return (2.0 * torch.acos(dot)).mean()


def camera_loss(pred_cam: torch.Tensor, gt_cam: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """pred_cam, gt_cam: (B, T, 9)  [tx,ty,tz,qx,qy,qz,qw,fovx,fovy]."""
    pred_t = pred_cam[..., :3]
    gt_t = gt_cam[..., :3]
    pred_q = pred_cam[..., 3:7]
    gt_q = gt_cam[..., 3:7]
    pred_fov = pred_cam[..., 7:]
    gt_fov = gt_cam[..., 7:]

    trans_huber = F.smooth_l1_loss(pred_t, gt_t)
    rot_geo = _quat_geodesic_loss(pred_q, gt_q)
    fov_l1 = F.l1_loss(pred_fov, gt_fov)
    total = trans_huber + rot_geo + 0.1 * fov_l1
    return total, {
        "cam_trans": float(trans_huber.detach()),
        "cam_rot": float(rot_geo.detach()),
        "cam_fov": float(fov_l1.detach()),
    }


def track_loss(pred_tracks: torch.Tensor, gt_tracks: torch.Tensor,
               valid: torch.Tensor | None = None) -> torch.Tensor:
    """L1 in normalized image coordinates. (B, T, 2)."""
    if valid is None:
        return F.l1_loss(pred_tracks, gt_tracks)
    diff = (pred_tracks - gt_tracks).abs().sum(dim=-1)        # (B, T)
    m = valid.float()
    return (diff * m).sum() / m.sum().clamp_min(1.0)


def geomamba_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    w_l1: float = 1.0,
    w_log: float = 0.5,
    w_mvc: float = 0.1,
    w_cam: float = 1.0,
    w_track: float = 1.0,
    w_pred: float = 0.5,
    mvc_samples: int = 1024,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined multi-task loss.

    predictions:
        "pointmap": (B, T, 3, H, W) in each frame's camera frame
        "camera":   (B, T, 9)
        "tracks":   (B, T, 2)               optional
    targets:
        "gt_pointmap_cam": (B, T, 3, H, W)
        "valid":           (B, T, H, W) bool
        "poses_w_c":       (B, T, 4, 4)
        "camera_gt":       (B, T, 9)
        "gt_tracks":       (B, T, 2)        optional
        "track_valid":     (B, T) bool      optional
    """
    log: dict[str, float] = {}

    # Pointmap.
    pred_pmap = predictions["pointmap"]
    gt_pmap = targets["gt_pointmap_cam"]
    valid = targets["valid"]
    l1 = pointmap_l1_loss(pred_pmap, gt_pmap, valid)
    logl = pointmap_log_loss(pred_pmap, gt_pmap, valid)
    log["loss_pmap_l1"] = float(l1.detach())
    log["loss_pmap_log"] = float(logl.detach())

    # Multi-view consistency.
    mvc = multi_view_consistency(pred_pmap.float(), valid,
                                 targets["poses_w_c"], n_samples=mvc_samples)
    log["loss_mvc"] = float(mvc.detach())

    # Camera.
    cam_t, cam_log = camera_loss(predictions["camera"], targets["camera_gt"])
    log.update(cam_log)
    log["loss_cam"] = float(cam_t.detach())

    total = w_l1 * l1 + w_log * logl + w_mvc * mvc + w_cam * cam_t

    # Tracks (optional).
    if "tracks" in predictions and "gt_tracks" in targets:
        tl = track_loss(predictions["tracks"], targets["gt_tracks"],
                        targets.get("track_valid"))
        total = total + w_track * tl
        log["loss_track"] = float(tl.detach())

    # World-model regularizer (optional): predicted next-frame summary tokens
    # vs EMA-target. MSE in latent space. JEPA recipe — target already detached.
    if "predicted_next" in predictions and "target_next" in predictions:
        pl = F.mse_loss(predictions["predicted_next"].float(),
                        predictions["target_next"].float())
        total = total + w_pred * pl
        log["loss_pred"] = float(pl.detach())

    log["loss_total"] = float(total.detach())
    return total, log
