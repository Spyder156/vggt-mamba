"""TerraWM-D — voxel-grid world model.

The architecture from-scratch principles:
  - Recurrent state = bounded 3D voxel feature grid (frame-0-anchored, fixed bounds).
  - Every per-frame output STRUCTURALLY routes through the voxel grid:
      depth = volumetric rendering of grid features along camera rays
      pose  = render-and-compare against the grid at an initial pose estimate
  - No parallel "summary tokens" path. No cross-frame Mamba scan. No latent
    predictor. No anchor pool. All of those are bypasses that the optimizer
    has historically routed around the recurrent state via.
  - Bootstrap depth head is WRITE-ONLY (firewall): it produces the per-patch
    depth hypothesis used to project patches into 3D for voxel writes. Its
    output is never read by the dense head.

Per-frame pipeline:
  1. encoder (frozen) → patches
  2. intra-frame self-attention → refined patches
  3. bootstrap depth head → per-patch depth hypothesis (write-only)
  4. RENDER at initial pose estimate → per-ray rendered features + total weight
  5. render-and-compare pose head → corrected pose (delta from initial)
  6. backproject (patch pixel + bootstrap depth + corrected pose) → world points
  7. trilinear voxel write with projected patch features
  8. RE-RENDER at corrected pose → dense depth (rendered) + unwritten mask
  9. upsample patch-resolution depth to full resolution (bilinear, MVP)
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregators.anchor_pool import cam9_to_pose_w_c, build_patch_pixel_grid
from .aggregators.intraframe_attn import IntraFrameTransformer
from .encoders import DINOv2Encoder, DINOv3Encoder, EncoderOutput, VJEPAEncoder
from .heads.bootstrap_depth import BootstrapDepthHead
from .heads.color import ColorHead
from .heads.render_compare import RenderCompareHead
from .heads.write_confidence import WriteConfidenceHead
from .pose_utils import gt_relative_motion_from_abs_poses, pose_w_c_to_T, T_to_pose_w_c
from .voxel_grid import (
    VoxelGridConfig,
    VoxelGridState,
    backproject_patches_to_world,
    build_rays_from_pose,
    init_voxel_state,
    render_rays_volumetric,
    reset_voxel_state,
    write_voxels_trilinear,
)


def _pose_T_to_cam9(T: torch.Tensor, fov: torch.Tensor) -> torch.Tensor:
    """(B, 4, 4) world-from-cam + (B, 2) fov → (B, 9) [t, q, fov]."""
    t, q = T_to_pose_w_c(T)
    return torch.cat([t, q, fov], dim=-1)


class TerraWM_D(nn.Module):
    """Voxel-grid TerraWM. No Mamba scan, no summary tokens, no predictor."""

    def __init__(
        self,
        encoder: VJEPAEncoder | DINOv2Encoder | DINOv3Encoder,
        n_intraframe_layers: int = 4,
        # Voxel grid:
        voxel_bounds: tuple[float, float, float, float, float, float] = (-4.0, -4.0, -4.0, 4.0, 4.0, 4.0),
        voxel_resolution: tuple[int, int, int] = (64, 64, 64),
        voxel_feature_dim: int = 32,
        # Rendering:
        n_render_samples: int = 64,
        render_near: float = 0.1,
        render_far: float = 8.0,
        # Heads:
        bootstrap_hidden: int = 128,
        bootstrap_max_depth: float = 10.0,
        pose_head_hidden: int = 256,
        pose_max_dt: float = 0.30,
        pose_max_dq: float = 0.15,
        # Unwritten-mask threshold (rays below this total_weight are excluded from dense loss).
        unwritten_mask_threshold: float = 1e-3,
        # Write-confidence head: makes mass DIFFERENTIABLE via a learned per-patch scalar.
        # When True, write weights = sigmoid(WriteConfidenceHead(patches)) instead of
        # constant 1.0. Gives render-loss gradient a path through mass → depth, which
        # is structurally missing otherwise. See write_confidence.py for the design.
        use_write_confidence: bool = False,
        write_confidence_hidden: int = 64,
        # Pose-head gate mode (no-bypass multiplier on the MLP output). See class
        # body for the cold-start-vs-drift-freeze distinction.
        pose_gate_mode: str = "coverage",
        # Differentiable write geometry: when True, bootstrap_d is NOT detached at the
        # write step, so render-loss gradient can flow back into bootstrap depth via
        # the trilinear-write's position dependence (sub-voxel position correction).
        # Closes the map-adjusts-to-image half of bundle adjustment that the original
        # detached-write design left open. Position gradient is local (bounded by voxel
        # size) due to discrete corner-index assignment in write_voxels_trilinear, so
        # this is the SOFT closed-loop test — sub-voxel correction works, cross-voxel
        # relocation needs continuous-position representations (e.g. Gaussians).
        differentiable_write_geometry: bool = False,
        # PHOTOMETRIC — fix for the geometric-channel inversion (β_disp_geom = -2.18
        # confirmed). Predict per-patch RGB from rendered voxel features (post-write,
        # corrected-pose render) and supervise against current frame's RGB.
        use_photometric: bool = False,
        photometric_hidden: int = 64,
        # When True, the 2nd render's pose is NOT detached → photometric gradient
        # flows back into the pose head. This is the WHOLE POINT of photometric in
        # the inversion regime: replace the bad geometric pose signal with a good
        # photometric one. Co-guarded by the scene-state ablation (post-reset Δt).
        photometric_pose_gradient: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.dim = encoder.dim
        self.img_size = encoder.img_size
        self.grid_h = encoder.grid
        self.grid_w = encoder.grid
        self.n_patches = self.grid_h * self.grid_w
        self.n_render_samples = n_render_samples
        self.render_near = render_near
        self.render_far = render_far
        self.unwritten_mask_threshold = unwritten_mask_threshold

        # Intra-frame attention.
        self.intraframe = IntraFrameTransformer(dim=self.dim, n_layers=n_intraframe_layers)

        # Bootstrap depth head — write-only firewall.
        self.bootstrap_depth = BootstrapDepthHead(
            dim=self.dim, hidden=bootstrap_hidden, max_depth=bootstrap_max_depth,
        )

        # Patch → voxel feature projection. Learnable (not detached at write).
        self.patch_to_voxel = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, voxel_feature_dim),
        )

        # Render-and-compare pose head.
        self.pose_head = RenderCompareHead(
            patch_dim=self.dim, voxel_dim=voxel_feature_dim,
            hidden=pose_head_hidden, max_dt=pose_max_dt, max_dq=pose_max_dq,
        )

        # Optional: write-confidence head (makes mass differentiable).
        self.use_write_confidence = use_write_confidence
        if use_write_confidence:
            self.write_confidence = WriteConfidenceHead(
                dim=self.dim, hidden=write_confidence_hidden,
            )

        # Optional: differentiable write geometry. Controls .detach() on bootstrap_d
        # and the no-grad wrap around the write step.
        self.differentiable_write_geometry = differentiable_write_geometry

        # Photometric — color head + pose-gradient flag. ColorHead reads only
        # rendered features (structural bypass guard). When enabled, _frame_step
        # additionally produces per-patch + dense RGB predictions.
        self.use_photometric = use_photometric
        if use_photometric:
            self.color_head = ColorHead(
                voxel_dim=voxel_feature_dim, hidden=photometric_hidden,
            )
        # Photometric pose-gradient flag: when True (and use_photometric), the
        # 2nd render's pose is NOT detached → photometric gradient flows back
        # into the pose head. Co-guarded by scene-state ablation.
        self.photometric_pose_gradient = photometric_pose_gradient

        # Pose-head no-bypass gate mode. The gate multiplies the MLP output to
        # force delta=identity when "the grid has nothing to say." Two definitions:
        #   "coverage" (default): use per-render-call ray coverage. Fires on cold
        #     start (good) BUT also fires when the camera drifts to a region the
        #     populated grid doesn't cover (bad → self-perpetuating freeze).
        #   "grid_mass": use a sigmoid on total voxel mass. Only fires when the
        #     grid is genuinely empty. Doesn't fire on the drift case.
        # The grid-mass mode is the inference-time fix for the long-horizon
        # collapse (frame ~900 freeze on fr1/room with pure1 ckpt).
        self.pose_gate_mode = pose_gate_mode

        # Voxel grid config (state is allocated externally).
        self.voxel_cfg = VoxelGridConfig(
            bounds=voxel_bounds, resolution=voxel_resolution,
            feature_dim=voxel_feature_dim,
        )

        # Pre-computed per-patch pixel centers (constant per resolution).
        self.register_buffer(
            "_patch_pixel_grid",
            build_patch_pixel_grid(self.grid_h, self.grid_w, self.img_size, device="cpu"),
            persistent=False,
        )
        # Default FOV vector (constant fovx, fovy in radians-ish — pass-through from input).
        self.register_buffer(
            "_default_fov",
            torch.tensor([1.0, 1.0]),
            persistent=False,
        )

    # ---------- helpers ----------

    def _encode_frame(self, rgb_frame: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, P, D) intra-attended patches."""
        with torch.no_grad():
            enc_out: EncoderOutput = self.encoder(rgb_frame)
        patches = enc_out.patches.unsqueeze(1)                              # (B, 1, P, D)
        refined = self.intraframe(patches).squeeze(1)                       # (B, P, D)
        return refined

    def _patch_pixel(self, B: int, device) -> torch.Tensor:
        return self._patch_pixel_grid.to(device).unsqueeze(0).expand(B, -1, -1)

    # ---------- per-frame step (single frame, used by both batched and streaming) ----------

    def _frame_step(
        self,
        patches: torch.Tensor,            # (B, P, D) intra-attended patches
        voxel_state: VoxelGridState,
        initial_pose_T: torch.Tensor,    # (B, 4, 4)
        K_intrinsics: torch.Tensor,      # (B, 3, 3)
        fov_passthrough: torch.Tensor,   # (B, 2)
        return_diagnostics: bool = False,
        rgb_frame: torch.Tensor | None = None,   # (B, 3, H, W) — current frame, for photo mismatch diagnostic
    ) -> dict[str, torch.Tensor]:
        B = patches.shape[0]
        device = patches.device
        patch_pixel = self._patch_pixel(B, device)                          # (B, P, 2)

        # Bootstrap depth (write-only).
        bootstrap_d = self.bootstrap_depth(patches)                         # (B, P) — DIFFERENTIABLE, supervised by Loss_bootstrap

        # 1st render: at initial pose, to drive the pose head.
        ray_o1, ray_d1 = build_rays_from_pose(initial_pose_T, K_intrinsics, patch_pixel)
        render1 = render_rays_volumetric(
            voxel_state, ray_o1, ray_d1,
            n_samples=self.n_render_samples, near=self.render_near, far=self.render_far,
        )
        rendered_feat = render1["feature"]                                  # (B, P, voxel_dim)
        ray_total_w1 = render1["total_weight"]                              # (B, P)

        # Pose head: render-vs-current → DELTA pose (camera-frame relative motion).
        initial_pose_9 = _pose_T_to_cam9(initial_pose_T, fov_passthrough)   # (B, 9)
        # Compute gate signal. "coverage" → None → render_compare uses its
        # default coverage-based gate. "grid_mass" → soft sigmoid on total mass:
        # ≈ 0 when grid is genuinely empty (cold start), ≈ 1 when there's any
        # meaningful population. Threshold 1e3 puts the sigmoid at half-saturation
        # around the mass typically seen after a few frames of writes.
        if self.pose_gate_mode == "grid_mass":
            mass_total = voxel_state.write_mass.sum().detach()
            mass_gate = torch.sigmoid((mass_total - 1e3) / 1e2)
            external_gate = mass_gate.expand(B).unsqueeze(-1).to(initial_pose_9.dtype)
        else:
            external_gate = None  # default behavior
        delta_pose_9 = self.pose_head(
            patches, rendered_feat, ray_total_w1, initial_pose_9,
            external_gate=external_gate,
        )                                                                    # (B, 9) DELTA — see render_compare.RenderCompareHead.forward
        # Compose with the previous absolute pose to get this frame's absolute
        # pose for write + 2nd render: T_world_t = T_world_{t-1} @ T_delta.
        delta_pose_T = cam9_to_pose_w_c(delta_pose_9)                       # (B, 4, 4)
        corrected_pose_T = initial_pose_T.float() @ delta_pose_T            # (B, 4, 4)

        # WRITE: project patches to 3D via (corrected abs pose, bootstrap depth), scatter.
        # Pose is ALWAYS detached at the write step (firewalls render-loss from pushing
        # the pose head — separate stability concern). Bootstrap depth is conditionally
        # detached: when differentiable_write_geometry=True, bootstrap_d keeps its
        # gradient so render-loss can correct write positions via the trilinear-weight
        # gradient (sub-voxel position correction). This is the soft closed-loop test
        # for the map-adjusts-to-image half of bundle adjustment.
        if self.differentiable_write_geometry:
            # Keep bootstrap_d's gradient; pose still detached.
            world_pts = backproject_patches_to_world(
                patch_pixel, bootstrap_d, K_intrinsics,
                corrected_pose_T.detach(),
            )                                                                # (B, P, 3) — has grad through bootstrap_d
        else:
            with torch.no_grad():
                world_pts = backproject_patches_to_world(
                    patch_pixel, bootstrap_d.detach(), K_intrinsics,
                    corrected_pose_T.detach(),
                )                                                            # (B, P, 3) — fully detached
        # Patch → voxel feature projection IS differentiable (so Loss_render
        # flows back through it into the projection weights and patches).
        voxel_feat = self.patch_to_voxel(patches)                           # (B, P, voxel_dim)
        # Write confidence: scalar [0,1] per patch. Makes mass differentiable
        # when enabled. None → defaults to constant 1.0 inside write_voxels_trilinear.
        write_weights = self.write_confidence(patches) if self.use_write_confidence else None
        write_voxels_trilinear(voxel_state, world_pts, voxel_feat, weights=write_weights)

        # 2nd render: at corrected abs pose, for dense depth output (and RGB
        # when photometric is enabled).
        # POSE FIREWALL DECISION:
        #   - Legacy / geometric-only: detach corrected_pose_T → render loss
        #     only updates voxel features, not pose head. Pose was supervised
        #     ONLY by Loss_pose. This was correct given geometric channel
        #     INVERTS with drift (β_disp = -2.18) — letting render-loss into
        #     pose would be actively harmful.
        #   - Photometric pose-gradient: when self.photometric_pose_gradient,
        #     do NOT detach. Photometric loss flows back into the pose head as
        #     a (hypothesized) correct-direction signal. Gated behind a flag
        #     because the gain depends on photometric β_disp coming back
        #     positive (the pre-registered gate post-train). Scene-state
        #     ablation co-guards against bypass induction.
        pose_for_render2 = (
            corrected_pose_T
            if (self.use_photometric and self.photometric_pose_gradient)
            else corrected_pose_T.detach()
        )
        ray_o2, ray_d2 = build_rays_from_pose(pose_for_render2, K_intrinsics, patch_pixel)
        render2 = render_rays_volumetric(
            voxel_state, ray_o2, ray_d2,
            n_samples=self.n_render_samples, near=self.render_near, far=self.render_far,
        )
        patch_depth = render2["depth"]                                       # (B, P)
        patch_mass = render2["total_weight"]                                 # (B, P)
        # Photometric output: per-patch + dense RGB prediction from rendered features.
        rendered_feat_dense = render2["feature"]                             # (B, P, voxel_dim)
        if self.use_photometric:
            patch_rgb_pred = self.color_head(rendered_feat_dense)            # (B, P, 3) ∈ [0, 1]
            # Upsample patch RGB → image resolution for dense photometric loss.
            dense_rgb_pred = F.interpolate(
                patch_rgb_pred.permute(0, 2, 1).contiguous().view(B, 3, self.grid_h, self.grid_w),
                size=(self.img_size, self.img_size), mode="bilinear", align_corners=True,
            )                                                                 # (B, 3, H, W) ∈ [0, 1]
        else:
            patch_rgb_pred = None
            dense_rgb_pred = None

        # Upsample patch-resolution depth + mass to full image resolution.
        H = W = self.img_size
        dense_depth = F.interpolate(
            patch_depth.view(B, 1, self.grid_h, self.grid_w),
            size=(H, W), mode="bilinear", align_corners=True,
        ).squeeze(1)                                                          # (B, H, W)
        dense_mass = F.interpolate(
            patch_mass.view(B, 1, self.grid_h, self.grid_w),
            size=(H, W), mode="bilinear", align_corners=True,
        ).squeeze(1)                                                          # (B, H, W)
        dense_mask = dense_mass > self.unwritten_mask_threshold              # (B, H, W) bool

        out = {
            "depth": dense_depth,                                            # (B, H, W) rendered, supervised by Loss_render
            "depth_mask": dense_mask,                                        # (B, H, W) bool: which pixels are write-covered
            "depth_mass": dense_mass,                                        # (B, H, W) per-pixel render weight (for diagnostics)
            "bootstrap_depth_patch": bootstrap_d,                            # (B, P) for Loss_bootstrap
            "camera": delta_pose_9,                                          # (B, 9) DELTA pose — directly comparable to camera_delta_gt
            "corrected_pose_T": corrected_pose_T,                            # (B, 4, 4) absolute pose at this frame (for streaming inference)
            "patch_depth_render": patch_depth,                               # (B, P) per-patch rendered depth (diagnostic)
            "patch_mass_render": patch_mass,                                 # (B, P)
        }
        if self.use_photometric:
            out["rgb_pred"] = dense_rgb_pred                                 # (B, 3, H, W) ∈ [0, 1] — photometric loss target
            out["patch_rgb_pred"] = patch_rgb_pred                           # (B, P, 3) — diagnostic + future re-grounding
        if return_diagnostics:
            # === TB Tier-4 diagnostics ===
            # Compute the EXACT signal the pose head reads internally
            # (current_proj, pooled_diff, pooled_cur) so we can log it without
            # threading it out of RenderCompareHead. Detached — diagnostics only.
            with torch.no_grad():
                current_proj_d = self.pose_head.current_proj(patches.detach()) # (B, P, voxel_dim)
                diff_d = current_proj_d - rendered_feat.detach()
                w_d = ray_total_w1.detach().unsqueeze(-1).clamp(min=0.0)
                w_sum_d = w_d.sum(dim=1).clamp_min(1e-6)
                pooled_diff_d = (diff_d * w_d).sum(dim=1) / w_sum_d            # (B, voxel_dim)
                pooled_cur_d = (current_proj_d * w_d).sum(dim=1) / w_sum_d
                mismatch_l2 = pooled_diff_d.norm(dim=-1)                        # (B,)
                cur_norm = pooled_cur_d.norm(dim=-1).clamp_min(1e-6)            # (B,)
                mismatch_rel = mismatch_l2 / cur_norm
                coverage_b = (ray_total_w1.detach() > 1e-3).float().mean(dim=1) # (B,)
                # Rendered feature norm per batch element (mean per-patch L2).
                rendered_feat_norm = rendered_feat.detach().norm(dim=-1).mean(dim=1)  # (B,)
                # Rendered depth std across patches (degenerate when ~constant).
                patch_depth_std = patch_depth.detach().float().std(dim=1)        # (B,)
                # Bootstrap depth std (input variance proxy).
                bootstrap_d_std = bootstrap_d.detach().float().std(dim=1)        # (B,)
                # Pose-head action magnitude per frame.
                dt_mag = delta_pose_9[:, :3].detach().float().norm(dim=-1)       # (B,)
                # Grid mass (single scalar broadcast to B for logging convenience).
                grid_mass_total = voxel_state.write_mass.sum().detach().expand(B).float()
                # Encoder feature norm — Tier-1 input panel.
                enc_norm = patches.detach().float().norm(dim=-1).mean(dim=1)     # (B,)
                # Gate value that multiplied the pose-head output.
                if external_gate is not None:
                    gate_value = external_gate.detach().squeeze(-1).float()     # (B,)
                else:
                    gate_value = coverage_b
                # Write confidence mean (Tier-2 health metric); 0 if disabled.
                if self.use_write_confidence and write_weights is not None:
                    ww = write_weights.detach().float()                          # (B, P) or (B,)
                    wc_mean = ww.mean(dim=1) if ww.dim() == 2 else ww
                else:
                    wc_mean = torch.zeros(B, device=patches.device)
                # === Photometric mismatch (the inversion-validation signal) ===
                # When photometric head exists AND rgb_frame is provided, compute
                # patch-pooled photometric mismatch in the SAME form as geometric:
                #   photo_diff = ‖rgb_pred_patch - rgb_current_patch‖₂
                #   photo_pooled = (photo_diff · w) / Σw     where w = ray_total_w
                # Then photo_mismatch_rel = photo_pooled / ‖rgb_current_patch‖.
                # Pre-registered: gate on β_disp_photo > +1.0 after retrain.
                if self.use_photometric and rgb_frame is not None and patch_rgb_pred is not None:
                    # Per-patch target RGB: average rgb_frame over each patch's pixel block.
                    rgb_target_patch = F.adaptive_avg_pool2d(
                        rgb_frame.detach().float(), (self.grid_h, self.grid_w)
                    ).flatten(2).transpose(1, 2)                                # (B, P, 3)
                    photo_diff_p = (patch_rgb_pred.detach().float() - rgb_target_patch).norm(dim=-1)  # (B, P)
                    w_pp = ray_total_w1.detach().float()                       # (B, P) — same weight as geometric mismatch
                    photo_l2 = (photo_diff_p * w_pp).sum(dim=1) / w_pp.sum(dim=1).clamp_min(1e-6)
                    target_norm = rgb_target_patch.norm(dim=-1).mean(dim=1).clamp_min(1e-6)
                    photo_rel = photo_l2 / target_norm
                else:
                    photo_l2 = torch.zeros(B, device=patches.device)
                    photo_rel = torch.zeros(B, device=patches.device)
            out["diagnostics"] = {
                "mismatch_l2": mismatch_l2,                                     # (B,) pose-head input signal magnitude (GEOMETRIC — INVERTED, β_disp=-2.18)
                "mismatch_rel": mismatch_rel,                                   # (B,) GEOMETRIC normalized — broken-channel diagnostic
                "photo_mismatch_l2": photo_l2,                                  # (B,) PHOTOMETRIC mismatch — gate signal
                "photo_mismatch_rel": photo_rel,                                # (B,) PHOTOMETRIC normalized — PRIMARY new signal
                "pooled_cur_norm": cur_norm,                                    # (B,) feature scale
                "render_coverage": coverage_b,                                  # (B,) ray-weight coverage
                "render_feat_norm": rendered_feat_norm,                         # (B,) mean per-patch ‖rendered_feat‖
                "render_depth_std": patch_depth_std,                            # (B,) degeneracy guard
                "bootstrap_d_std": bootstrap_d_std,                             # (B,)
                "dt_mag": dt_mag,                                                # (B,) pose-head action magnitude
                "grid_mass_total": grid_mass_total,                             # (B,) broadcast scalar
                "enc_norm": enc_norm,                                            # (B,) encoder input scale
                "gate_value": gate_value,                                        # (B,) what the pose-head output was multiplied by
                "write_conf_mean": wc_mean,                                      # (B,) 0 if write-conf disabled
            }
        return out

    # ---------- batched (training) forward ----------

    def forward(
        self,
        rgb: torch.Tensor,                # (B, T, 3, H, W)
        K_intrinsics: torch.Tensor,       # (B, 3, 3)
        gt_poses_w_c: torch.Tensor | None = None,   # (B, T, 4, 4) for teacher-forced initial pose (training)
        fov: torch.Tensor | None = None,            # (B, T, 2)  pass-through fov; defaults to constant
        return_voxel_state: bool = False,           # if True, add "voxel_write_mass" + "voxel_features" to output
        return_diagnostics: bool = False,           # if True, add "diagnostics" dict — Tier-4 TB panels
    ) -> dict[str, torch.Tensor]:
        B, T, _, H, W = rgb.shape
        device = rgb.device
        if fov is None:
            fov = self._default_fov.to(device).unsqueeze(0).unsqueeze(0).expand(B, T, 2)

        # Allocate voxel grid for this window.
        voxel_state = init_voxel_state(self.voxel_cfg, batch_size=B, device=device,
                                       dtype=torch.float32)

        out_depth = []
        out_mask = []
        out_mass = []
        out_bootstrap = []
        out_camera = []
        out_rgb_pred: list[torch.Tensor] = []
        diag_per_t: list[dict[str, torch.Tensor]] = []

        for ti in range(T):
            patches = self._encode_frame(rgb[:, ti])                         # (B, P, D)
            # Initial pose: teacher-forced from GT[t-1]; identity at t=0 unless GT[0] supplied.
            if gt_poses_w_c is not None:
                if ti == 0:
                    initial_T = gt_poses_w_c[:, 0]                           # (B, 4, 4)
                else:
                    initial_T = gt_poses_w_c[:, ti - 1]
            else:
                initial_T = torch.eye(4, device=device).expand(B, 4, 4).contiguous()
            step_out = self._frame_step(
                patches, voxel_state, initial_T, K_intrinsics, fov[:, ti],
                return_diagnostics=return_diagnostics,
                rgb_frame=rgb[:, ti] if (self.use_photometric or return_diagnostics) else None,
            )
            out_depth.append(step_out["depth"])
            out_mask.append(step_out["depth_mask"])
            out_mass.append(step_out["depth_mass"])
            out_bootstrap.append(step_out["bootstrap_depth_patch"])
            out_camera.append(step_out["camera"])
            if self.use_photometric and step_out.get("rgb_pred") is not None:
                out_rgb_pred.append(step_out["rgb_pred"])
            if return_diagnostics:
                diag_per_t.append(step_out["diagnostics"])

        # Stack to (B, T, ...).
        depth = torch.stack(out_depth, dim=1)                                # (B, T, H, W)
        mask = torch.stack(out_mask, dim=1)                                  # (B, T, H, W)
        mass = torch.stack(out_mass, dim=1)
        bootstrap = torch.stack(out_bootstrap, dim=1)                        # (B, T, P)
        cam = torch.stack(out_camera, dim=1)                                 # (B, T, 9)
        # Pointmap output is depth-only (z channel) for now; xy can be derived if needed.
        pmap = torch.zeros(B, T, 3, H, W, device=device, dtype=depth.dtype)
        pmap[:, :, 2] = depth

        out = {
            "pointmap": pmap,                                                # (B, T, 3, H, W) Z=rendered depth
            "depth": depth,
            "depth_mask": mask,                                              # (B, T, H, W) — Loss_render must mask on this
            "depth_mass": mass,
            "bootstrap_depth_patch": bootstrap,                              # (B, T, P)
            "camera": cam,                                                   # (B, T, 9) corrected poses
        }
        if out_rgb_pred:
            out["rgb_pred"] = torch.stack(out_rgb_pred, dim=1)               # (B, T, 3, H, W) ∈ [0, 1]
        if return_voxel_state:
            out["voxel_write_mass"] = voxel_state.write_mass.detach()        # (B, V_x, V_y, V_z, 1)
        if return_diagnostics:
            # Stack per-frame (B,) tensors → (B, T) for each key.
            keys = diag_per_t[0].keys()
            out["diagnostics"] = {
                k: torch.stack([d[k] for d in diag_per_t], dim=1) for k in keys
            }                                                                 # each (B, T)
        return out

    # ---------- streaming inference ----------

    @torch.no_grad()
    def init_voxel_state(self, batch_size: int = 1, device: str = "cuda",
                          dtype: torch.dtype = torch.float32) -> VoxelGridState:
        return init_voxel_state(self.voxel_cfg, batch_size, device, dtype)

    @torch.no_grad()
    def streaming_forward(
        self,
        rgb_frame: torch.Tensor,          # (1, 3, H, W) single frame
        voxel_state: VoxelGridState,
        prev_pose_9: torch.Tensor,        # (1, 9) ABSOLUTE world-from-camera pose of frame t-1
        K_intrinsics: torch.Tensor,       # (1, 3, 3)
        fov: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Streaming inference. Caller passes the previous ABSOLUTE pose; we
        return the new absolute pose composed from the predicted delta. So
        existing callers (`prev_pose_9 = corrected`) keep working unchanged —
        but `out["camera"]` now exposes the predicted delta directly for any
        diagnostic that wants delta semantics.
        """
        device = rgb_frame.device
        if fov is None:
            fov = self._default_fov.to(device).unsqueeze(0)
        patches = self._encode_frame(rgb_frame)
        initial_T = cam9_to_pose_w_c(prev_pose_9)
        step_out = self._frame_step(patches, voxel_state, initial_T, K_intrinsics, fov)
        # Convert the new absolute pose (4x4) back to 9-vec for the caller.
        new_abs_T = step_out["corrected_pose_T"]                             # (1, 4, 4)
        new_abs_9 = _pose_T_to_cam9(new_abs_T, fov)                          # (1, 9)
        # Repack to match batched-forward conventions.
        H = W = self.img_size
        pmap = torch.zeros(1, 1, 3, H, W, device=device, dtype=step_out["depth"].dtype)
        pmap[:, 0, 2] = step_out["depth"]
        out = {
            "pointmap": pmap,
            "depth": step_out["depth"].unsqueeze(1),
            "depth_mask": step_out["depth_mask"].unsqueeze(1),
            "camera": step_out["camera"].unsqueeze(1),                       # (1, 1, 9) DELTA
            "camera_abs": new_abs_9.unsqueeze(1),                            # (1, 1, 9) absolute (for trajectory diagnostics)
            "patch_depth_render": step_out["patch_depth_render"],
            "patch_mass_render": step_out["patch_mass_render"],
        }
        return out, new_abs_9                                                 # caller feeds this back as prev_pose_9


def build_terrawm_d(
    encoder_name: Literal["vjepa", "dinov2", "dinov3"],
    weights_root: str,
    n_intraframe_layers: int = 4,
    voxel_bounds: tuple[float, float, float, float, float, float] = (-4.0, -4.0, -4.0, 4.0, 4.0, 4.0),
    voxel_resolution: tuple[int, int, int] = (64, 64, 64),
    voxel_feature_dim: int = 32,
    n_render_samples: int = 64,
    render_near: float = 0.1,
    render_far: float = 8.0,
    bootstrap_hidden: int = 128,
    bootstrap_max_depth: float = 10.0,
    pose_head_hidden: int = 256,
    pose_max_dt: float = 0.30,
    pose_max_dq: float = 0.15,
    unwritten_mask_threshold: float = 1e-3,
    use_write_confidence: bool = False,
    write_confidence_hidden: int = 64,
    differentiable_write_geometry: bool = False,
    pose_gate_mode: str = "coverage",
    use_photometric: bool = False,
    photometric_hidden: int = 64,
    photometric_pose_gradient: bool = False,
) -> TerraWM_D:
    weights_root = Path(weights_root)
    if encoder_name == "dinov3":
        enc = DINOv3Encoder(
            repo_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
            img_size=512, freeze=True,
        )
    elif encoder_name == "dinov2":
        enc = DINOv2Encoder(
            weights_root / "dinov2-large/dinov2_vitl14_pretrain.pth",
            img_size=518, freeze=True,
        )
    elif encoder_name == "vjepa":
        enc = VJEPAEncoder(
            weights_root / "vjepa2-vitl/vjepa2_1_vitl_dist_vitG_384.pt",
            img_size=384, freeze=True,
        )
    else:
        raise ValueError(f"unknown encoder {encoder_name!r}")
    return TerraWM_D(
        enc,
        n_intraframe_layers=n_intraframe_layers,
        voxel_bounds=voxel_bounds,
        voxel_resolution=voxel_resolution,
        voxel_feature_dim=voxel_feature_dim,
        n_render_samples=n_render_samples,
        render_near=render_near,
        render_far=render_far,
        bootstrap_hidden=bootstrap_hidden,
        bootstrap_max_depth=bootstrap_max_depth,
        pose_head_hidden=pose_head_hidden,
        pose_max_dt=pose_max_dt,
        pose_max_dq=pose_max_dq,
        unwritten_mask_threshold=unwritten_mask_threshold,
        use_write_confidence=use_write_confidence,
        write_confidence_hidden=write_confidence_hidden,
        differentiable_write_geometry=differentiable_write_geometry,
        pose_gate_mode=pose_gate_mode,
        use_photometric=use_photometric,
        photometric_hidden=photometric_hidden,
        photometric_pose_gradient=photometric_pose_gradient,
    )


if __name__ == "__main__":
    import os
    root = os.environ.get("VGGT_MAMBA_DATA_ROOT", "/workspace/datasets") + "/weights"
    m = build_terrawm_d(
        "dinov3", root,
        n_intraframe_layers=2,
        voxel_resolution=(32, 32, 16),
    ).cuda()
    print(f"[terrawm-d] trainable params: {sum(p.numel() for p in m.parameters() if p.requires_grad)/1e6:.2f}M")

    B, T = 1, 4
    s = m.img_size
    rgb = torch.rand(B, T, 3, s, s, device="cuda")
    K = torch.tensor([[[420., 0., 256.], [0., 575., 264.], [0., 0., 1.]]], device="cuda")
    gt_poses = torch.eye(4, device="cuda").expand(B, T, 4, 4).contiguous()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = m(rgb, K_intrinsics=K, gt_poses_w_c=gt_poses)
    print(f"[terrawm-d] outputs:")
    for k, v in out.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {tuple(v.shape)}  dtype={v.dtype}")
    pct_covered = out["depth_mask"].float().mean().item() * 100
    print(f"[terrawm-d] dense-mask coverage: {pct_covered:.1f}% of pixels have voxel hits")
    print(f"[terrawm-d] depth range: [{out['depth'][out['depth_mask']].min():.3f}, "
          f"{out['depth'][out['depth_mask']].max():.3f}] m (masked)")
    print(f"[terrawm-d] camera[:, 0, :3] (translations): {out['camera'][0, :, :3].tolist()}")
    print(f"[terrawm-d] PASS")
