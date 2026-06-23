"""TerraWM-D: bounded, frame-0-anchored voxel feature grid.

The grid IS the recurrent state in D. Every per-frame output is structurally
routed through it (depth rendered from voxel features along rays; pose
predicted by render-and-compare against the grid). There is no parallel
"summary token" or "scanned patch" path — those bypasses are gone.

Shape:
  features:   (B, V_x, V_y, V_z, D)  per-voxel learned features
  write_mass: (B, V_x, V_y, V_z, 1)  accumulated write weight (per-voxel)
  bounds:     (x_min, y_min, z_min, x_max, y_max, z_max) — fixed at config

Coordinate convention: world frame is anchored at frame 0's camera pose.
The grid covers a fixed box around the origin (default 8×8×4 m). If the
camera leaves the box, voxel-grid coverage becomes degraded — documented
limitation, fine for TUM-Freiburg3 room-scale sequences.

Writes are GEOMETRIC: each input patch is projected to a 3D world position
via (depth + camera pose + intrinsics), and its features are trilinearly
scattered into the 8 surrounding voxels. No descriptor matching anywhere.
The write-mass tracks how many writes each voxel has received (weighted).

Reads are RENDERING: per output pixel, cast a ray from the camera through
the pixel, sample the voxel grid along the ray, accumulate features and
write-mass via volumetric integration. Returns per-pixel features, depth,
and the cumulative write-mass (used to mask pixels whose rays hit nothing).

Constant memory by construction: fixed (V_x, V_y, V_z, D+1) shape.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VoxelGridConfig:
    """Static configuration; the grid state is separate from this."""
    bounds: tuple[float, float, float, float, float, float] = (-4.0, -4.0, -2.0, 4.0, 4.0, 2.0)
    resolution: tuple[int, int, int] = (64, 64, 32)
    feature_dim: int = 32

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        x_min, y_min, z_min, x_max, y_max, z_max = self.bounds
        v_x, v_y, v_z = self.resolution
        return ((x_max - x_min) / v_x, (y_max - y_min) / v_y, (z_max - z_min) / v_z)

    @property
    def total_voxels(self) -> int:
        return self.resolution[0] * self.resolution[1] * self.resolution[2]


@dataclass
class VoxelGridState:
    """Per-batch state. Pre-allocated; reset_state zeros in place."""
    features: torch.Tensor      # (B, V_x, V_y, V_z, D)
    write_mass: torch.Tensor    # (B, V_x, V_y, V_z, 1)
    cfg: VoxelGridConfig


def init_voxel_state(cfg: VoxelGridConfig, batch_size: int = 1,
                     device: str = "cuda", dtype: torch.dtype = torch.float32) -> VoxelGridState:
    v_x, v_y, v_z = cfg.resolution
    return VoxelGridState(
        features=torch.zeros(batch_size, v_x, v_y, v_z, cfg.feature_dim,
                             device=device, dtype=dtype),
        write_mass=torch.zeros(batch_size, v_x, v_y, v_z, 1,
                               device=device, dtype=dtype),
        cfg=cfg,
    )


def reset_voxel_state(state: VoxelGridState) -> None:
    """Zero in place (preserves tensor addresses for graph compatibility)."""
    state.features.zero_()
    state.write_mass.zero_()


def world_to_grid_coords(
    points_world: torch.Tensor,    # (..., 3)
    cfg: VoxelGridConfig,
) -> torch.Tensor:
    """World coordinates → continuous voxel-grid indices (in [0, V) per axis).
    Returns (..., 3) with float voxel indices. Out-of-bounds points are NOT
    clipped here — callers can mask them via the in_bounds helper.
    """
    bounds_min = points_world.new_tensor([cfg.bounds[0], cfg.bounds[1], cfg.bounds[2]])
    vx, vy, vz = cfg.voxel_size
    voxel_size = points_world.new_tensor([vx, vy, vz])
    return (points_world - bounds_min) / voxel_size


def in_bounds_mask(grid_coords: torch.Tensor, cfg: VoxelGridConfig) -> torch.Tensor:
    """(..., 3) → (...,) bool, True iff coords are inside the grid (with a
    one-voxel margin for safe trilinear sampling).
    """
    v_x, v_y, v_z = cfg.resolution
    x, y, z = grid_coords.unbind(-1)
    return ((x >= 0.5) & (x <= v_x - 1.5) &
            (y >= 0.5) & (y <= v_y - 1.5) &
            (z >= 0.5) & (z <= v_z - 1.5))


def write_voxels_trilinear(
    state: VoxelGridState,
    points_world: torch.Tensor,    # (B, N, 3)  world positions of points to write
    features: torch.Tensor,        # (B, N, D)  features per point
    weights: torch.Tensor | None = None,   # (B, N)  optional per-point weight (default 1.0)
) -> None:
    """Scatter-write features into the voxel grid with trilinear weights.

    Per point: compute its (continuous) grid coords, find the 8 surrounding
    voxel corners, compute trilinear weights to those corners, accumulate
    feature * weight into features tensor and weight into write_mass tensor.

    Differentiable through `features` (and `weights`). Not differentiable
    through `points_world` (we never propagate gradient through 3D positions
    — they come from the bootstrap depth head + pose, both of which are
    detached at write time to keep the write strictly geometric).

    Mutates state in place via index_add_ on flattened indices.
    """
    cfg = state.cfg
    B, N, D = features.shape
    v_x, v_y, v_z = cfg.resolution
    assert points_world.shape == (B, N, 3)

    if weights is None:
        weights = torch.ones(B, N, device=features.device, dtype=features.dtype)

    # World → continuous grid coords.
    gc = world_to_grid_coords(points_world, cfg)            # (B, N, 3)
    mask = in_bounds_mask(gc, cfg)                          # (B, N) bool
    # Floor + frac for trilinear weights.
    gc_floor = gc.floor().long()                            # (B, N, 3)
    gc_frac = gc - gc_floor.float()                         # (B, N, 3)

    # 8 corner offsets.
    corners = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
         [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        device=features.device, dtype=torch.long,
    )                                                        # (8, 3)

    for c in range(8):
        ox, oy, oz = corners[c].tolist()
        # Corner voxel index per (b, n)
        ix = gc_floor[..., 0] + ox                          # (B, N)
        iy = gc_floor[..., 1] + oy
        iz = gc_floor[..., 2] + oz
        # Trilinear weight for this corner.
        wx = gc_frac[..., 0] if ox == 1 else (1.0 - gc_frac[..., 0])
        wy = gc_frac[..., 1] if oy == 1 else (1.0 - gc_frac[..., 1])
        wz = gc_frac[..., 2] if oz == 1 else (1.0 - gc_frac[..., 2])
        w_corner = wx * wy * wz * mask.float() * weights    # (B, N)

        # Flat voxel index per (b, n).
        in_corner_bounds = ((ix >= 0) & (ix < v_x) &
                            (iy >= 0) & (iy < v_y) &
                            (iz >= 0) & (iz < v_z))
        w_corner = w_corner * in_corner_bounds.float()

        flat_idx = ((ix.clamp(0, v_x - 1) * v_y + iy.clamp(0, v_y - 1)) * v_z
                    + iz.clamp(0, v_z - 1))                  # (B, N)
        for b in range(B):
            # Accumulate per-batch: features += sum(features[b] * w[:, None]) over points sharing voxel.
            f_b = features[b] * w_corner[b].unsqueeze(-1)    # (N, D)
            m_b = w_corner[b].unsqueeze(-1)                  # (N, 1)
            feat_flat = state.features[b].view(-1, D)        # (V_x*V_y*V_z, D)
            mass_flat = state.write_mass[b].view(-1, 1)
            feat_flat.index_add_(0, flat_idx[b], f_b)
            mass_flat.index_add_(0, flat_idx[b], m_b)


def trilinear_sample_grid(
    state: VoxelGridState,
    points_world: torch.Tensor,    # (B, N, 3)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample voxel features + write_mass at world points via trilinear interp.
    Differentiable through `state.features` and `state.write_mass`.
    Returns:
      sampled_feat: (B, N, D)
      sampled_mass: (B, N, 1)
    """
    cfg = state.cfg
    B, N, _ = points_world.shape
    v_x, v_y, v_z = cfg.resolution
    D = cfg.feature_dim

    # Use F.grid_sample with mode='bilinear' (trilinear for 5D inputs).
    # grid_sample expects features in (B, C, D_z, H_y, W_x) and grid in
    # (B, D_out, H_out, W_out, 3) with coords in [-1, 1] order (x, y, z).
    # Map our (B, V_x, V_y, V_z, D) to (B, D, V_z, V_y, V_x):
    feat_5d = state.features.permute(0, 4, 3, 2, 1).contiguous()       # (B, D, V_z, V_y, V_x)
    mass_5d = state.write_mass.permute(0, 4, 3, 2, 1).contiguous()     # (B, 1, V_z, V_y, V_x)

    # World → continuous voxel coords [0, V) per axis → grid_sample's [-1, 1].
    gc = world_to_grid_coords(points_world, cfg)            # (B, N, 3) [vx, vy, vz indices]
    # grid_sample expects normalized coords in (x, y, z) where x is the
    # *fastest* spatial dimension (last axis of the 5D tensor). We permuted
    # so the last axis is V_x → x = gc[..., 0] / (V_x-1) * 2 - 1.
    norm_x = gc[..., 0] / (v_x - 1) * 2.0 - 1.0
    norm_y = gc[..., 1] / (v_y - 1) * 2.0 - 1.0
    norm_z = gc[..., 2] / (v_z - 1) * 2.0 - 1.0
    grid = torch.stack([norm_x, norm_y, norm_z], dim=-1)    # (B, N, 3)
    grid = grid.view(B, N, 1, 1, 3)                          # (B, N, 1, 1, 3)
    sampled_feat = F.grid_sample(feat_5d, grid, mode="bilinear",
                                  padding_mode="zeros", align_corners=True)
    # sampled_feat shape: (B, D, N, 1, 1) → squeeze → (B, D, N) → transpose → (B, N, D)
    sampled_feat = sampled_feat.squeeze(-1).squeeze(-1).transpose(1, 2).contiguous()
    sampled_mass = F.grid_sample(mass_5d, grid, mode="bilinear",
                                  padding_mode="zeros", align_corners=True)
    sampled_mass = sampled_mass.squeeze(-1).squeeze(-1).transpose(1, 2).contiguous()
    return sampled_feat, sampled_mass


def build_rays_from_pose(
    pose_w_c: torch.Tensor,        # (B, 4, 4) world-from-camera
    K: torch.Tensor,               # (B, 3, 3) intrinsics
    pixel_xy: torch.Tensor,        # (B, R, 2) per-ray pixel centers (u, v)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build world-frame ray origins and UNNORMALIZED directions, using the
    along-z depth convention (matches backproject_patches_to_world + bootstrap_d).

    Origin = camera center = pose_w_c[:3, 3] (broadcast over R rays).
    Direction = pose_w_c.rotation @ K^-1 @ [u, v, 1], NOT normalized.

    K^-1 @ [u, v, 1] has z=1 in camera frame, so a sample point
        origin + t · dir
    has camera-frame z = t. That is: t = depth-along-z, the same convention
    bootstrap_d / TUM GT depth / backproject_patches_to_world use. The
    along-z norm of dir_world equals ||cam_dir||_2, which is >= 1 and grows
    radially outward from the principal point. The renderer must scale the
    density-vs-dt math by ||dir|| to integrate against the physical ray
    length, but `t` itself is z-distance everywhere.

    Previously dir_world was normalized, which made `origin + t · dir` give
    along-ray distance instead of along-z. That was inconsistent with the
    write path (backproject treats depth as z) and bootstrap_d (trained
    against TUM GT depth, which is z). Confirmed by terrawm_d_convention_check
    09 + 10: 25-30cm discrepancy at corners at depth=1.5m, growing radially.
    """
    B, R, _ = pixel_xy.shape
    K_inv = torch.linalg.inv(K.float())                                 # (B, 3, 3)
    ones = torch.ones(B, R, 1, device=pixel_xy.device, dtype=pixel_xy.dtype)
    pix_h = torch.cat([pixel_xy.float(), ones.float()], dim=-1)         # (B, R, 3)
    # Ray dir in camera frame: (K^-1 @ [u, v, 1]), unnormalized → z = 1.
    dir_cam = torch.einsum("bij,brj->bri", K_inv, pix_h)                 # (B, R, 3)
    # Rotate to world. NOT normalized — keep ||dir|| = ||cam_dir|| so the
    # renderer's z-convention math is consistent with backproject's.
    R_wc = pose_w_c[:, :3, :3].float()
    dir_world = torch.einsum("bij,brj->bri", R_wc, dir_cam)             # (B, R, 3)
    t_wc = pose_w_c[:, :3, 3].float()                                   # (B, 3)
    origin = t_wc.unsqueeze(1).expand(B, R, 3)                          # (B, R, 3)
    return origin, dir_world


def backproject_patches_to_world(
    pixel_xy: torch.Tensor,        # (B, P, 2) per-patch pixel centers
    patch_depth: torch.Tensor,     # (B, P) per-patch depth (m)
    K: torch.Tensor,               # (B, 3, 3)
    pose_w_c: torch.Tensor,        # (B, 4, 4)
) -> torch.Tensor:
    """(pixel + depth) + camera pose → world-frame 3D position. (B, P, 3)."""
    B, P, _ = pixel_xy.shape
    K_inv = torch.linalg.inv(K.float())
    ones = torch.ones(B, P, 1, device=pixel_xy.device, dtype=torch.float32)
    pix_h = torch.cat([pixel_xy.float(), ones], dim=-1)
    cam_dir = torch.einsum("bij,bpj->bpi", K_inv, pix_h)                # (B, P, 3) normalized? not yet
    P_cam = cam_dir * patch_depth.float().unsqueeze(-1)                 # (B, P, 3) point in camera frame
    R_wc = pose_w_c[:, :3, :3].float()
    t_wc = pose_w_c[:, :3, 3].float()
    P_world = torch.einsum("bij,bpj->bpi", R_wc, P_cam) + t_wc.unsqueeze(1)
    return P_world


def render_rays_volumetric(
    state: VoxelGridState,
    ray_origins: torch.Tensor,     # (B, R, 3)  per-ray origin (world frame)
    ray_dirs: torch.Tensor,        # (B, R, 3)  per-ray direction (UNNORMALIZED, z-convention)
    n_samples: int = 64,
    near: float = 0.1,
    far: float = 8.0,
) -> dict[str, torch.Tensor]:
    """Volumetric rendering using the along-z depth convention.

    ray_dirs is the world-frame ray direction with camera-frame z=1 (i.e.
    NOT unit-normalized). With this convention, sample positions
        pt_i = origin + t_i · ray_dir
    have camera-frame z = t_i, so t directly represents along-z depth — the
    same units bootstrap_d / TUM GT depth / backproject_patches_to_world use.

    The physical length traversed per t-step is dt · ||ray_dir||, which
    varies per pixel (||ray_dir|| ≥ 1, ≈ 1 at the principal point, larger
    radially). The density-vs-path-length integration must use that physical
    length so alpha is consistent across pixels.

    Returns:
      depth:        (B, R)     expected depth along CAMERA-Z axis (matches bootstrap_d / GT)
      feature:      (B, R, D)  accumulated feature
      total_weight: (B, R)     cumulative accumulated alpha-weight
    """
    cfg = state.cfg
    B, R, _ = ray_origins.shape
    device = ray_origins.device

    # Sample positions in along-z depth (t in [near, far] is z-distance).
    t_vals = torch.linspace(near, far, n_samples, device=device, dtype=ray_origins.dtype)
    t_vals = t_vals.view(1, 1, n_samples, 1)                # (1, 1, S, 1)
    # pt = origin + t · ray_dir. Since ray_dir has z=1 in cam frame,
    # the resulting pt has z = t (in cam frame), i.e. t IS along-z depth.
    pts = ray_origins.unsqueeze(2) + ray_dirs.unsqueeze(2) * t_vals  # (B, R, S, 3)
    pts_flat = pts.reshape(B, R * n_samples, 3)
    sampled_feat, sampled_mass = trilinear_sample_grid(state, pts_flat)
    sampled_feat = sampled_feat.view(B, R, n_samples, cfg.feature_dim)
    sampled_mass = sampled_mass.view(B, R, n_samples)

    # NeRF-style accumulation with PER-RAY physical path length.
    # Each t-step covers dt·||ray_dir|| of actual 3D length.
    # alpha = 1 - exp(-density · physical_step_length) so corners (where
    # ||ray_dir|| > 1) accumulate more per sample than center pixels — that's
    # the correct geometry, not a bug, because each sample on a corner ray
    # spans a longer physical path through the grid.
    dt = (far - near) / n_samples
    ray_norm = ray_dirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)        # (B, R, 1)
    step_length = dt * ray_norm                                             # (B, R, 1)
    density = torch.relu(sampled_mass)                                      # (B, R, S)
    alpha = 1.0 - torch.exp(-density * step_length)                         # (B, R, S)
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    T = torch.cat([torch.ones_like(T[..., :1]), T[..., :-1]], dim=-1)
    w = alpha * T
    total_w = w.sum(dim=-1)
    # Expected depth = ∑ w · t. With t = along-z, depth is along-z directly.
    t_per_sample = t_vals.view(1, 1, n_samples).expand(B, R, n_samples)
    depth = (w * t_per_sample).sum(dim=-1)
    feature = (w.unsqueeze(-1) * sampled_feat).sum(dim=-2)
    return {"depth": depth, "feature": feature, "total_weight": total_w}


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = VoxelGridConfig(bounds=(-4.0, -4.0, -2.0, 4.0, 4.0, 2.0),
                          resolution=(32, 32, 16), feature_dim=8)
    state = init_voxel_state(cfg, batch_size=1, device="cuda", dtype=torch.float32)
    print(f"[voxel] config: bounds={cfg.bounds}, res={cfg.resolution}, voxel_size={cfg.voxel_size}")
    print(f"[voxel] state: features {tuple(state.features.shape)}, "
          f"write_mass {tuple(state.write_mass.shape)}")
    print(f"[voxel] total voxels: {cfg.total_voxels}, bytes: {state.features.numel()*4 + state.write_mass.numel()*4}")

    # Write 256 random points with random features, check the grid populates.
    points = (torch.rand(1, 256, 3, device="cuda") * 8.0 - 4.0)   # in [-4, 4]^3 roughly
    points[..., 2] = points[..., 2] * 0.5                          # squeeze z to [-2, 2]
    feats = torch.randn(1, 256, 8, device="cuda")
    write_voxels_trilinear(state, points.detach(), feats.detach())
    n_nonzero = (state.write_mass.abs() > 0).float().sum().item()
    print(f"[voxel] after writing 256 points: nonzero voxels = {int(n_nonzero)} / {cfg.total_voxels}")
    print(f"[voxel] write_mass: total={state.write_mass.sum().item():.2f} (expect ~256 from trilinear partition)")

    # Sample at the same points: features should match (modulo trilinear interp).
    samp_feat, samp_mass = trilinear_sample_grid(state, points.detach())
    err = (samp_feat - feats).abs().mean().item()
    print(f"[voxel] sample-at-write-point mean feature error: {err:.4f} "
          f"(trilinear interp introduces small error)")

    # Render 100 random rays.
    origins = torch.zeros(1, 100, 3, device="cuda")
    dirs = torch.randn(1, 100, 3, device="cuda")
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    out = render_rays_volumetric(state, origins, dirs, n_samples=32, near=0.1, far=6.0)
    print(f"[voxel] render: depth mean {out['depth'].mean():.3f}, "
          f"total_weight mean {out['total_weight'].mean():.4f}, "
          f"feature norm {out['feature'].norm(dim=-1).mean():.3f}")

    # Reset.
    reset_voxel_state(state)
    print(f"[voxel] after reset: write_mass total = {state.write_mass.sum().item()}")
    print(f"[voxel] PASS")
