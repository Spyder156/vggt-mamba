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
from .heads.render_compare import RenderCompareHead
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

        # Pose head: render-vs-current → corrected pose.
        initial_pose_9 = _pose_T_to_cam9(initial_pose_T, fov_passthrough)   # (B, 9)
        corrected_pose_9 = self.pose_head(
            patches, rendered_feat, ray_total_w1, initial_pose_9,
        )                                                                    # (B, 9)
        corrected_pose_T = cam9_to_pose_w_c(corrected_pose_9)               # (B, 4, 4) differentiable

        # WRITE: project patches to 3D via (corrected pose, bootstrap depth), scatter.
        # Both pose and depth detached so the geometric write doesn't backprop
        # into them — write is a one-way deposit, firewalled.
        with torch.no_grad():
            world_pts = backproject_patches_to_world(
                patch_pixel, bootstrap_d.detach(), K_intrinsics,
                corrected_pose_T.detach(),
            )                                                                # (B, P, 3)
        # Patch → voxel feature projection IS differentiable (so Loss_render
        # flows back through it into the projection weights and patches).
        voxel_feat = self.patch_to_voxel(patches)                           # (B, P, voxel_dim)
        write_voxels_trilinear(voxel_state, world_pts, voxel_feat)

        # 2nd render: at corrected pose, for dense depth output.
        ray_o2, ray_d2 = build_rays_from_pose(corrected_pose_T, K_intrinsics, patch_pixel)
        render2 = render_rays_volumetric(
            voxel_state, ray_o2, ray_d2,
            n_samples=self.n_render_samples, near=self.render_near, far=self.render_far,
        )
        patch_depth = render2["depth"]                                       # (B, P)
        patch_mass = render2["total_weight"]                                 # (B, P)

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

        return {
            "depth": dense_depth,                                            # (B, H, W) rendered, supervised by Loss_render
            "depth_mask": dense_mask,                                        # (B, H, W) bool: which pixels are write-covered
            "depth_mass": dense_mass,                                        # (B, H, W) per-pixel render weight (for diagnostics)
            "bootstrap_depth_patch": bootstrap_d,                            # (B, P) for Loss_bootstrap
            "camera": corrected_pose_9,                                      # (B, 9) corrected pose (or delta if interpreted thus)
            "patch_depth_render": patch_depth,                               # (B, P) per-patch rendered depth (diagnostic)
            "patch_mass_render": patch_mass,                                 # (B, P)
        }

    # ---------- batched (training) forward ----------

    def forward(
        self,
        rgb: torch.Tensor,                # (B, T, 3, H, W)
        K_intrinsics: torch.Tensor,       # (B, 3, 3)
        gt_poses_w_c: torch.Tensor | None = None,   # (B, T, 4, 4) for teacher-forced initial pose (training)
        fov: torch.Tensor | None = None,            # (B, T, 2)  pass-through fov; defaults to constant
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
            )
            out_depth.append(step_out["depth"])
            out_mask.append(step_out["depth_mask"])
            out_mass.append(step_out["depth_mass"])
            out_bootstrap.append(step_out["bootstrap_depth_patch"])
            out_camera.append(step_out["camera"])

        # Stack to (B, T, ...).
        depth = torch.stack(out_depth, dim=1)                                # (B, T, H, W)
        mask = torch.stack(out_mask, dim=1)                                  # (B, T, H, W)
        mass = torch.stack(out_mass, dim=1)
        bootstrap = torch.stack(out_bootstrap, dim=1)                        # (B, T, P)
        cam = torch.stack(out_camera, dim=1)                                 # (B, T, 9)
        # Pointmap output is depth-only (z channel) for now; xy can be derived if needed.
        pmap = torch.zeros(B, T, 3, H, W, device=device, dtype=depth.dtype)
        pmap[:, :, 2] = depth

        return {
            "pointmap": pmap,                                                # (B, T, 3, H, W) Z=rendered depth
            "depth": depth,
            "depth_mask": mask,                                              # (B, T, H, W) — Loss_render must mask on this
            "depth_mass": mass,
            "bootstrap_depth_patch": bootstrap,                              # (B, T, P)
            "camera": cam,                                                   # (B, T, 9) corrected poses
        }

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
        prev_pose_9: torch.Tensor,        # (1, 9) previous predicted pose
        K_intrinsics: torch.Tensor,       # (1, 3, 3)
        fov: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        device = rgb_frame.device
        if fov is None:
            fov = self._default_fov.to(device).unsqueeze(0)
        patches = self._encode_frame(rgb_frame)
        initial_T = cam9_to_pose_w_c(prev_pose_9)
        step_out = self._frame_step(patches, voxel_state, initial_T, K_intrinsics, fov)
        # Repack to match batched-forward conventions.
        H = W = self.img_size
        pmap = torch.zeros(1, 1, 3, H, W, device=device, dtype=step_out["depth"].dtype)
        pmap[:, 0, 2] = step_out["depth"]
        out = {
            "pointmap": pmap,
            "depth": step_out["depth"].unsqueeze(1),
            "depth_mask": step_out["depth_mask"].unsqueeze(1),
            "camera": step_out["camera"].unsqueeze(1),                       # (1, 1, 9)
            "patch_depth_render": step_out["patch_depth_render"],
            "patch_mass_render": step_out["patch_mass_render"],
        }
        return out, step_out["camera"]                                       # return corrected pose to feed back next frame


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
