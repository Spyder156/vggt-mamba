"""Multi-task loss for TerraWM.

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

from vggt_mamba.eval.metrics import multi_view_consistency, project_points_to_pixels
from vggt_mamba.losses.pointmap import pointmap_l1_loss, pointmap_log_loss
from vggt_mamba.models.aggregators.anchor_pool import cam9_to_pose_w_c


def _quat_geodesic_loss(pred_q: torch.Tensor, gt_q: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between two unit quaternions (radians).

    Args:
        pred_q, gt_q: (..., 4) — already (or nearly) unit-norm.

    Note: dot is clamped to [0, 1 - 1e-6] to keep acos away from the +1
    singularity. acos'(1) = -1/sqrt(1-1²) = -inf — and the delta-pose
    formulation has the model produce EXACTLY identity at frame 0 (zero-init
    + coverage=0 → identity quat), matched against GT identity delta,
    yielding dot=1 exactly and infinite backward gradient. Clamping at
    1-1e-6 means a max forward error of ~0.0014 rad on the matched case
    and finite backward everywhere.
    """
    pred_q = F.normalize(pred_q, dim=-1)
    gt_q = F.normalize(gt_q, dim=-1)
    dot = (pred_q * gt_q).sum(dim=-1).abs().clamp(0.0, 1.0 - 1e-6)
    return (2.0 * torch.acos(dot)).mean()


def camera_loss(pred_cam: torch.Tensor, gt_cam: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """pred_cam, gt_cam: (B, T, 9)  [tx,ty,tz,qx,qy,qz,qw,fovx,fovy].

    For absolute-pose mode, gt_cam is per-frame world-from-camera pose components.
    For TerraWM delta mode, both pred_cam and gt_cam are per-frame relative
    motion from frame t-1 to frame t (with frame 0 = identity); FOV stays absolute.
    The math is the same — the *interpretation* of the targets is what changes,
    and that's the caller's responsibility.
    """
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


def camera_tracking_loss(
    pred_cam: torch.Tensor,
    gt_cam: torch.Tensor,
    w_l1: float = 0.1,
    w_rel: float = 1.0,
    w_cos: float = 1.0,
    cos_mag_floor_m: float = 0.002,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, dict]:
    """Tracking-focused pose loss. Designed to break the 'output a constant
    delta' lazy-solution failure mode that pure per-frame L1 falls into when
    GT deltas are small.

    Three terms on the delta translation:
      - smooth_l1 (background, small weight)
      - relative-L1: |pred - gt|.mean() / (mean GT magnitude in batch).clamp_min(eps).
        Forces tracking AT the right magnitude — a constant output can't satisfy this
        when GT magnitude is small but constant output magnitude is large.
      - masked cosine: (1 - cos(pred, gt)) averaged ONLY on frames where
        |gt_dt| > cos_mag_floor_m. Penalizes direction mismatch when GT actually
        has a direction; skipped on still frames (where GT direction is meaningless
        noise — would produce garbage gradient).

    Rotation: existing geodesic loss. FOV: L1 with small weight.
    """
    pred_t = pred_cam[..., :3]
    gt_t = gt_cam[..., :3]
    pred_q = pred_cam[..., 3:7]
    gt_q = gt_cam[..., 3:7]
    pred_fov = pred_cam[..., 7:]
    gt_fov = gt_cam[..., 7:]

    # Background term (kept small).
    trans_huber = F.smooth_l1_loss(pred_t, gt_t)

    # Relative-L1: rescale by per-batch mean GT magnitude.
    gt_mag = gt_t.norm(dim=-1)                                       # (B, T)
    scale = gt_mag.mean().clamp_min(eps)                             # scalar
    rel_l1 = (pred_t - gt_t).abs().mean() / scale

    # Masked cosine direction — only on frames where GT has real motion.
    pred_norm = pred_t.norm(dim=-1).clamp_min(eps)                   # (B, T)
    gt_norm = gt_t.norm(dim=-1).clamp_min(eps)                       # (B, T)
    cos_sim = (pred_t * gt_t).sum(dim=-1) / (pred_norm * gt_norm)    # (B, T) in [-1, 1]
    cos_mask = (gt_mag > cos_mag_floor_m).float()                    # (B, T)
    cos_mask_sum = cos_mask.sum().clamp_min(1.0)
    direction = ((1.0 - cos_sim) * cos_mask).sum() / cos_mask_sum

    rot_geo = _quat_geodesic_loss(pred_q, gt_q)
    fov_l1 = F.l1_loss(pred_fov, gt_fov)

    total = (w_l1 * trans_huber + w_rel * rel_l1 + w_cos * direction
             + rot_geo + 0.1 * fov_l1)
    # Diagnostic: scale ratio + cosine-on-tracked-frames (verdict metrics).
    with torch.no_grad():
        scale_ratio = (pred_t.norm(dim=-1).mean() / gt_mag.mean().clamp_min(eps))
        mean_cos_on_tracked = (cos_sim * cos_mask).sum() / cos_mask_sum
    return total, {
        "cam_trans": float(trans_huber.detach()),
        "cam_rel": float(rel_l1.detach()),
        "cam_direction": float(direction.detach()),
        "cam_rot": float(rot_geo.detach()),
        "cam_fov": float(fov_l1.detach()),
        "cam_scale_ratio": float(scale_ratio.detach()),
        "cam_cos_tracked": float(mean_cos_on_tracked.detach()),
        "cam_track_frac": float(cos_mask.mean().detach()),
    }


def camera_scale_invariant_loss(
    pred_cam: torch.Tensor,
    gt_cam: torch.Tensor,
    w_scale_inv: float = 1.0,
    w_cos: float = 1.0,
    w_rot: float = 1.0,
    w_fov: float = 0.1,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """Scale-invariant pose loss. Designed to break the 'output constant
    magnitude' failure mode that bounded-output + metric-supervision falls into
    when GT magnitudes vary across frames.

    Three terms on the delta translation:
      - **scale-invariant log-variance** (Eigen-style): penalize variance of
        log(|pred| / |gt|) across the batch. A model that outputs pred = k * gt
        for ANY constant k gets zero loss (loss is invariant to a global
        multiplicative scale on pred). A model that outputs constant pred
        regardless of GT magnitude has nonzero variance → nonzero loss. A
        collapsed pred → 0 model (the lazy escape under naive align-by-regression
        losses) also gets nonzero loss because log(eps) - log(gt) still varies
        with gt. The architecture's metric anchors (grid bounds + bootstrap
        depth supervision) implicitly pull the chosen scale toward GT.
      - **cosine direction** (already scale-free): no metric mask — cosine on
        near-zero gt vectors is noisy but bounded, doesn't bias the loss.
      - **rotation geodesic** (already scale-free): unchanged.

    The diagnostic `cam_mag_corr` is the PRIMARY verdict metric: Pearson
    correlation between per-frame |pred| and |gt|. Magnitude tracking will
    register here even if absolute scale doesn't match.

    The diagnostic `cam_pred_mag_mean` is the GUARD against collapse: if it
    shrinks toward zero, the loss is being gamed — fail it even if
    cam_scale_inv is small.
    """
    pred_t = pred_cam[..., :3]
    gt_t = gt_cam[..., :3]
    pred_q = pred_cam[..., 3:7]
    gt_q = gt_cam[..., 3:7]
    pred_fov = pred_cam[..., 7:]
    gt_fov = gt_cam[..., 7:]

    # Scale-invariant magnitude loss (Eigen log-variance).
    pred_mag = pred_t.norm(dim=-1).clamp_min(eps)                    # (B, T)
    gt_mag = gt_t.norm(dim=-1).clamp_min(eps)                        # (B, T)
    log_ratio = pred_mag.log() - gt_mag.log()                        # (B, T)
    # Center by per-batch mean to absorb any constant multiplicative scale.
    log_ratio_centered = log_ratio - log_ratio.mean()
    scale_inv = (log_ratio_centered ** 2).mean()

    # Cosine direction (no metric mask; eps in norm prevents divide-by-zero).
    cos_sim = (pred_t * gt_t).sum(dim=-1) / (pred_mag * gt_mag)      # (B, T) in [-1, 1]
    direction = (1.0 - cos_sim).mean()

    # Rotation: geodesic distance (already scale-free).
    rot_geo = _quat_geodesic_loss(pred_q, gt_q)

    # FOV (intrinsic — keep metric, light weight).
    fov_l1 = F.l1_loss(pred_fov, gt_fov)

    total = (w_scale_inv * scale_inv + w_cos * direction
             + w_rot * rot_geo + w_fov * fov_l1)

    # Diagnostics for live monitoring + post-hoc verdict.
    with torch.no_grad():
        pred_mag_mean = pred_mag.mean()
        gt_mag_mean = gt_mag.mean()
        scale_ratio = pred_mag_mean / gt_mag_mean.clamp_min(eps)
        # PRIMARY VERDICT: per-frame magnitude correlation. Tracks whether the
        # model's steps vary WITH gt's steps (the eyes-on-TB observation:
        # does step size match per-frame, or is it constant?).
        pm_flat = pred_mag.flatten()
        gm_flat = gt_mag.flatten()
        if pm_flat.std() > eps and gm_flat.std() > eps:
            stacked = torch.stack([pm_flat, gm_flat])
            mag_corr = torch.corrcoef(stacked)[0, 1]
        else:
            mag_corr = torch.tensor(float("nan"), device=pred_t.device)

    return total, {
        "cam_trans": float(scale_inv.detach()),       # for TB backward compat
        "cam_scale_inv": float(scale_inv.detach()),
        "cam_direction": float(direction.detach()),
        "cam_rot": float(rot_geo.detach()),
        "cam_fov": float(fov_l1.detach()),
        "cam_pred_mag_mean": float(pred_mag_mean.detach()),
        "cam_gt_mag_mean": float(gt_mag_mean.detach()),
        "cam_scale_ratio": float(scale_ratio.detach()),
        "cam_mag_corr": float(mag_corr.detach()),     # PRIMARY VERDICT METRIC
    }


def photometric_l1(
    rgb_pred: torch.Tensor,            # (B, T, 3, H, W) ∈ [0, 1]
    rgb_target: torch.Tensor,          # (B, T, 3, H, W) ∈ [0, 1]
    depth_mask: torch.Tensor,          # (B, T, H, W) bool — only score where grid covers
) -> torch.Tensor:
    """L1 photometric loss, masked on depth_mask.

    The drift-blind check (terrawm_d_drift_blind_check.py) confirmed the
    geometric render-vs-current channel INVERTS with drift (β_disp = -2.18).
    Photometric is the candidate fix: predict per-patch RGB from rendered
    voxel features and supervise against current frame's RGB.

    Mask is on depth_mask (where the grid has content). Unwritten regions
    render to ~0 features → trivially predict gray → would dominate the
    loss without masking and let the color head game it without engaging
    the grid.
    """
    diff = (rgb_pred - rgb_target).abs()                                     # (B, T, 3, H, W)
    mask = depth_mask.unsqueeze(2).float()                                   # (B, T, 1, H, W)
    masked = (diff * mask).sum()
    return masked / (mask.sum() * 3.0).clamp_min(1.0)


def terrawm_d_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    w_render_l1: float = 1.0,
    w_render_log: float = 0.5,
    w_bootstrap: float = 1.0,
    w_pose: float = 1.0,
    w_mvc: float = 0.1,
    mvc_samples: int = 1024,
    pose_tracking: bool = False,
    pose_w_l1: float = 0.1,
    pose_w_rel: float = 1.0,
    pose_w_cos: float = 1.0,
    pose_cos_mag_floor_m: float = 0.002,
    pose_scale_invariant: bool = False,
    pose_w_scale_inv: float = 1.0,
    pose_w_rot: float = 1.0,
    pose_w_fov: float = 0.1,
    w_photometric: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """D's loss: bootstrap-depth (write supervision) + rendered-depth (masked
    on unwritten voxels) + delta-pose. No predictor, no VICReg, no anchor.

    predictions:
      depth:                  (B, T, H, W) rendered depth
      depth_mask:             (B, T, H, W) bool — pixels with voxel coverage
      bootstrap_depth_patch:  (B, T, P) per-patch write-time depth hypothesis
      camera:                 (B, T, 9) corrected pose [t, q, fov]
      pointmap:               (B, T, 3, H, W) for MVC

    targets:
      gt_depth_full:          (B, T, H, W) GT depth
      gt_depth_patch:         (B, T, P)    GT per-patch depth
      gt_depth_patch_valid:   (B, T, P)    bool
      poses_w_c:              (B, T, 4, 4) GT world-from-camera
      camera_delta_gt:        (B, T, 9)    GT relative motion + fov
      valid:                  (B, T, H, W) GT depth validity mask
    """
    from vggt_mamba.models.heads.bootstrap_depth import bootstrap_depth_loss
    log: dict[str, float] = {}

    # 1. Rendered-depth loss. Masked: only pixels where BOTH (a) GT is valid
    #    AND (b) voxel grid has been written to along the ray.
    depth = predictions["depth"].float()                     # (B, T, H, W)
    depth_mask = predictions["depth_mask"]                   # (B, T, H, W) bool
    gt_depth_full = targets["gt_depth_full"].float()         # (B, T, H, W)
    gt_valid = targets["valid"]                              # (B, T, H, W) bool
    full_mask = depth_mask & gt_valid                        # (B, T, H, W)
    eps = 1e-3
    diff = (depth.clamp_min(eps) - gt_depth_full.clamp_min(eps)).abs()
    render_l1 = (diff * full_mask.float()).sum() / full_mask.float().sum().clamp_min(1.0)
    log["loss_render_l1"] = float(render_l1.detach())
    # Scale-invariant log loss on rendered.
    log_diff = (depth.clamp_min(eps).log() - gt_depth_full.clamp_min(eps).log())
    render_log = (log_diff.pow(2) * full_mask.float()).sum() / full_mask.float().sum().clamp_min(1.0)
    log["loss_render_log"] = float(render_log.detach())
    log["depth_mask_coverage"] = float(depth_mask.float().mean().detach())

    # 2. Bootstrap depth (write hypothesis) — per-patch L1 in log space, masked
    #    on GT-valid patches. This trains the write path only; firewalled from
    #    the rendered output by the .detach() inside the model's write.
    bs = bootstrap_depth_loss(
        predictions["bootstrap_depth_patch"].float(),
        targets["gt_depth_patch"].float(),
        targets["gt_depth_patch_valid"],
    )
    log["loss_bootstrap"] = float(bs.detach())

    # 3. Delta-pose: corrected pose vs GT delta pose. Three modes:
    #    - pose_scale_invariant: Eigen log-variance + cosine + geodesic (NEW).
    #      Removes the absolute-scale supervision; the grid + bootstrap depth
    #      losses are the implicit scale anchors. PRIMARY verdict = cam_mag_corr.
    #    - pose_tracking: relative-L1 + masked cosine + smooth_l1 (older).
    #    - default: smooth_l1 + geodesic + fov.
    if pose_scale_invariant:
        cam_t, cam_log = camera_scale_invariant_loss(
            predictions["camera"], targets["camera_delta_gt"],
            w_scale_inv=pose_w_scale_inv, w_cos=pose_w_cos,
            w_rot=pose_w_rot, w_fov=pose_w_fov,
        )
    elif pose_tracking:
        cam_t, cam_log = camera_tracking_loss(
            predictions["camera"], targets["camera_delta_gt"],
            w_l1=pose_w_l1, w_rel=pose_w_rel, w_cos=pose_w_cos,
            cos_mag_floor_m=pose_cos_mag_floor_m,
        )
    else:
        cam_t, cam_log = camera_loss(predictions["camera"], targets["camera_delta_gt"])
    log.update(cam_log)
    log["loss_pose"] = float(cam_t.detach())

    # 4. Multi-view consistency (optional, lightweight; uses Z from rendered pointmap).
    mvc = multi_view_consistency(
        predictions["pointmap"].float(), gt_valid, targets["poses_w_c"], n_samples=mvc_samples,
    )
    log["loss_mvc"] = float(mvc.detach())

    # 5. Photometric — fix for the geometric-channel inversion (β_disp_geom = -2.18).
    #    L1 between predicted RGB (from ColorHead on rendered voxel features) and the
    #    current frame's RGB, masked on depth_mask. Required: predictions["rgb_pred"]
    #    AND targets["rgb_target"]. w_photometric=0 disables; > 0 activates.
    photo = depth.new_zeros(())
    if w_photometric > 0 and "rgb_pred" in predictions and "rgb_target" in targets:
        photo = photometric_l1(
            predictions["rgb_pred"].float(), targets["rgb_target"].float(), depth_mask,
        )
        log["loss_photometric"] = float(photo.detach())

    total = (w_render_l1 * render_l1 + w_render_log * render_log +
             w_bootstrap * bs + w_pose * cam_t + w_mvc * mvc + w_photometric * photo)
    log["loss_total"] = float(total.detach())
    return total, log


def vicreg_variance_loss(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """VICReg variance regularizer: per-dim std should be ≥ gamma.

    z: any shape ending in (..., D). Flattens all leading dims as batch.
    Loss = mean over D of max(0, gamma - std(z_d)).
    """
    flat = z.reshape(-1, z.shape[-1]).float()
    # eps inside sqrt for numerical stability
    std = (flat.var(dim=0, unbiased=False) + 1e-4).sqrt()                 # (D,)
    return torch.relu(gamma - std).mean()


def vicreg_covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """VICReg covariance regularizer: off-diagonal covariance → 0.

    Computes sum_{i≠j} cov(z)[i,j]² / D.
    """
    flat = z.reshape(-1, z.shape[-1]).float()
    flat = flat - flat.mean(dim=0, keepdim=True)
    N = flat.shape[0]
    D = flat.shape[1]
    cov = (flat.T @ flat) / max(N - 1, 1)                                  # (D, D)
    off_diag_sq = (cov.pow(2).sum() - cov.diagonal().pow(2).sum())
    return off_diag_sq / D


def track_loss(pred_tracks: torch.Tensor, gt_tracks: torch.Tensor,
               valid: torch.Tensor | None = None) -> torch.Tensor:
    """L1 in normalized image coordinates. (B, T, 2)."""
    if valid is None:
        return F.l1_loss(pred_tracks, gt_tracks)
    diff = (pred_tracks - gt_tracks).abs().sum(dim=-1)        # (B, T)
    m = valid.float()
    return (diff * m).sum() / m.sum().clamp_min(1.0)


def anchor_consistency_loss(
    predictions: dict[str, torch.Tensor],
    threshold: float = 0.5,
    img_size_px: float = 512.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Re-projection error for matched (patch, anchor) pairs above threshold.

    predictions must contain:
      camera:           (B, T, 9) — the corrected pose
      anchor_scores:    (B, T, P, K_a) — match scores in (0, 1)
      anchor_positions: (B, T, K_a, 3) — anchor world positions BEFORE the
                        current frame's write (so what was actually visible
                        to the read step).
      patch_pixel:      (B, P, 2) — pixel center per patch
      K_intrinsics:     (B, 3, 3)

    Loss = mean over matched pairs of score * ||proj(anchor_pos, pose) - patch_pixel||²,
    in normalized image coordinates (divided by img_size). Mask out behind-camera
    projections and invalid anchors (which already have score=0).
    """
    cam = predictions["camera"]                                     # (B, T, 9)
    scores = predictions["anchor_scores"]                            # (B, T, P, K_a)
    anchor_pos = predictions["anchor_positions"].float()             # (B, T, K_a, 3)
    patch_pixel = predictions["patch_pixel"].float()                 # (B, P, 2)
    K = predictions["K_intrinsics"].float()                          # (B, 3, 3)
    B, T, P, K_a = scores.shape

    # Build (B, T, 4, 4) pose stack and (B, T, 3, 3) K stack.
    poses_w_c = cam9_to_pose_w_c(cam)                                # (B, T, 4, 4)
    K_bt = K.unsqueeze(1).expand(B, T, 3, 3).contiguous()

    # Project all K_a anchors through each (B, T) pose.
    anchor_pix, in_front = project_points_to_pixels(anchor_pos, poses_w_c, K_bt)
    # anchor_pix: (B, T, K_a, 2)
    # Broadcast diff: (B, T, 1, K_a, 2) - (B, 1, P, 1, 2) → ... wait, need both in same shape
    pp = patch_pixel.unsqueeze(1).unsqueeze(-2)                      # (B, 1, P, 1, 2)
    ap = anchor_pix.unsqueeze(2)                                     # (B, T, 1, K_a, 2)
    pix_diff_sq = ((pp - ap) ** 2).sum(dim=-1)                       # (B, T, P, K_a)
    # Normalize by image size so the loss is in roughly [0, 1] units.
    pix_diff_norm = pix_diff_sq / (img_size_px ** 2)
    # Mask: only count pairs with score>threshold AND anchor projects in front.
    score_mask = (scores > threshold).float()                        # (B, T, P, K_a)
    in_front_mask = in_front.unsqueeze(2).float()                    # (B, T, 1, K_a)
    mask = score_mask * in_front_mask
    weighted = scores * pix_diff_norm * mask
    n_valid = mask.sum().clamp_min(1.0)
    loss = weighted.sum() / n_valid
    n_matches = float(score_mask.sum().detach())
    return loss, {
        "loss_anchor_consistency": float(loss.detach()),
        "anchor_n_matches": n_matches,
        "anchor_mean_score": float(scores.mean().detach()),
        "anchor_max_score": float(scores.max().detach()),
    }


def terrawm_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    w_l1: float = 1.0,
    w_log: float = 0.5,
    w_mvc: float = 0.1,
    w_cam: float = 1.0,
    w_track: float = 1.0,
    w_pred: float = 0.5,
    w_anchor: float = 0.5,
    w_vic_var: float = 0.0,            # VICReg variance reg (TerraWM)
    w_vic_cov: float = 0.0,            # VICReg covariance reg (TerraWM)
    anchor_threshold: float = 0.5,
    mvc_samples: int = 1024,
    pred_motion_weights: torch.Tensor | None = None,
    cam_target_key: str = "camera_gt", # "camera_gt" for absolute, "camera_delta_gt" for TerraWM
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

    # Camera. In TerraWM mode the predicted camera output is interpreted as
    # per-frame relative motion (Δt, Δq, fov) and the GT key is "camera_delta_gt";
    # in absolute mode it's per-frame world-from-camera and the key is "camera_gt".
    cam_gt = targets[cam_target_key]
    cam_t, cam_log = camera_loss(predictions["camera"], cam_gt)
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
    # If pred_motion_weights is supplied (Experiment 1a), weight each (b, t)
    # pair by the ego-motion magnitude between gt frame t and t+1, normalized
    # by the batch median. Locked formula — do not retune mid-flight.
    if "predicted_next" in predictions and "target_next" in predictions:
        pn = predictions["predicted_next"].float()                          # (B, T-1, K, D)
        tn = predictions["target_next"].float()                             # (B, T-1, K, D)
        if pred_motion_weights is None:
            pl = F.mse_loss(pn, tn)
        else:
            # pred_motion_weights: (B, T-1) — per-pair weight
            w = pred_motion_weights.to(pn.device, pn.dtype)                 # (B, T-1)
            elem = (pn - tn).pow(2)                                         # (B, T-1, K, D)
            pl = (elem * w[:, :, None, None]).mean()
        total = total + w_pred * pl
        log["loss_pred"] = float(pl.detach())

        # VICReg variance + covariance regs (TerraWM). Applied to both predictor
        # output and EMA target to prevent collapse on either side. Default w=0
        # means no-op for non-TerraWM configs.
        if w_vic_var > 0.0 or w_vic_cov > 0.0:
            var_p = vicreg_variance_loss(pn, gamma=1.0)
            var_t = vicreg_variance_loss(tn, gamma=1.0)
            cov_p = vicreg_covariance_loss(pn)
            cov_t = vicreg_covariance_loss(tn)
            total = total + w_vic_var * (var_p + var_t) + w_vic_cov * (cov_p + cov_t)
            log["vic_var_pred"] = float(var_p.detach())
            log["vic_var_target"] = float(var_t.detach())
            log["vic_cov_pred"] = float(cov_p.detach())
            log["vic_cov_target"] = float(cov_t.detach())
            # Diagnostic: predictor output's per-dim std (mean over D).
            # Healthy values are around 1.0 (matching VICReg γ=1).
            with torch.no_grad():
                std_pred = pn.reshape(-1, pn.shape[-1]).float().std(dim=0).mean()
                log["predictor_dim_std"] = float(std_pred)

    # Anchor-pool consistency loss (Experiment 2). Only fires if the model
    # emitted anchor_scores in its predictions.
    if "anchor_scores" in predictions:
        anc_l, anc_log = anchor_consistency_loss(predictions, threshold=anchor_threshold)
        total = total + w_anchor * anc_l
        log.update(anc_log)

    log["loss_total"] = float(total.detach())
    return total, log
