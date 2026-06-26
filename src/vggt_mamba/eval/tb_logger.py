"""TensorBoard logger for TerraWM-D training.

Comprehensive visualization for every full training run. Covers:
  - Scalars: all losses + diagnostic numbers (grid mass, pose bias, etc.)
  - Histograms: pose deltas, bootstrap depth, voxel mass, rendered depth
  - Images:
      RGB inputs
      GT depth vs predicted rendered depth (side-by-side colormap)
      Depth mask + render coverage
      Bootstrap depth (upsampled, colormap)
      Per-pixel L1 error map
      Voxel grid 3-view marginal projections (xy, xz, yz)
      Voxel grid 1D marginals along each axis (boundary-clip detection)
      Predicted vs GT camera trajectory (xz plane)

Cheap when not called (no overhead when intervals aren't hit). Matplotlib
figures are built only at image-logging steps.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch


def _depth_colormap(depth: np.ndarray, vmin: float, vmax: float, cmap: str = "turbo") -> np.ndarray:
    """(H, W) float depth -> (3, H, W) RGB uint8 via matplotlib colormap."""
    import matplotlib
    cm = matplotlib.colormaps[cmap]
    d = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgb = (cm(d)[..., :3] * 255.0).astype(np.uint8)                     # (H, W, 3)
    return np.transpose(rgb, (2, 0, 1))                                  # (3, H, W)


def _fig_to_image(fig) -> np.ndarray:
    """matplotlib Figure -> (3, H, W) uint8 RGB ndarray for SummaryWriter."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    from PIL import Image
    img = np.asarray(Image.open(buf).convert("RGB"))                    # (H, W, 3)
    return np.transpose(img, (2, 0, 1))


class TBLogger:
    """Thin wrapper around SummaryWriter with TerraWM-D-specific helpers."""

    def __init__(self, log_dir: Path | str):
        from torch.utils.tensorboard import SummaryWriter
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))
        print(f"[tb] logging to {log_dir}")

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()

    # ---------- scalars ----------

    def log_scalars(self, prefix: str, scalars: dict[str, float], step: int) -> None:
        for k, v in scalars.items():
            self.writer.add_scalar(f"{prefix}/{k}", float(v), step)

    # ---------- histograms ----------

    def log_histograms(self, prefix: str, tensors: dict[str, torch.Tensor], step: int) -> None:
        for k, t in tensors.items():
            if t.numel() == 0:
                continue
            flat = t.detach().float().flatten().cpu()
            if not torch.isfinite(flat).any():
                continue
            self.writer.add_histogram(f"{prefix}/{k}", flat, step)

    # ---------- training-batch image panel ----------

    def log_train_batch(
        self,
        rgb: torch.Tensor,                    # (B, T, 3, H, W) in [0, 1]
        gt_depth: torch.Tensor,               # (B, T, H, W)
        pred_depth: torch.Tensor,             # (B, T, H, W)
        depth_mask: torch.Tensor,             # (B, T, H, W) bool
        depth_mass: torch.Tensor,             # (B, T, H, W)
        bootstrap_depth_patch: torch.Tensor,  # (B, T, P)
        grid_h: int, grid_w: int,
        gt_valid: torch.Tensor,               # (B, T, H, W) bool
        step: int,
        depth_max_m: float = 8.0,
        frames_to_show: tuple[int, ...] = (0, -1),
    ) -> None:
        """Log per-window image panels for one batch (B=1 expected)."""
        import torch.nn.functional as F
        B, T, _, H, W = rgb.shape
        # Pick frames to show.
        frame_idx = [(i if i >= 0 else T + i) for i in frames_to_show]
        for fi in frame_idx:
            # RGB.
            rgb_img = rgb[0, fi].detach().float().clamp(0, 1).cpu().numpy()       # (3, H, W)
            self.writer.add_image(f"frame{fi:02d}/rgb", (rgb_img * 255).astype(np.uint8), step)
            # GT depth.
            gt = gt_depth[0, fi].detach().float().cpu().numpy()
            self.writer.add_image(f"frame{fi:02d}/gt_depth",
                                  _depth_colormap(gt, 0.0, depth_max_m), step)
            # Pred depth.
            pd = pred_depth[0, fi].detach().float().cpu().numpy()
            self.writer.add_image(f"frame{fi:02d}/pred_depth",
                                  _depth_colormap(pd, 0.0, depth_max_m), step)
            # Depth mask (coverage).
            dm = depth_mask[0, fi].detach().float().cpu().numpy() * 255
            self.writer.add_image(f"frame{fi:02d}/depth_mask",
                                  dm.astype(np.uint8)[None, :, :], step)
            # Render mass (heatmap).
            mass = depth_mass[0, fi].detach().float().cpu().numpy()
            mass_norm = mass / max(mass.max(), 1e-6)
            self.writer.add_image(f"frame{fi:02d}/render_mass",
                                  _depth_colormap(mass_norm, 0.0, 1.0, cmap="hot"), step)
            # Bootstrap depth (upsampled from patches).
            bd_patch = bootstrap_depth_patch[0, fi].detach().float()              # (P,)
            bd_grid = bd_patch.view(1, 1, grid_h, grid_w)
            bd_dense = F.interpolate(bd_grid, size=(H, W), mode="bilinear",
                                     align_corners=True).squeeze().cpu().numpy()
            self.writer.add_image(f"frame{fi:02d}/bootstrap_depth",
                                  _depth_colormap(bd_dense, 0.0, depth_max_m), step)
            # Per-pixel L1 error (masked on depth_mask AND gt_valid).
            valid_m = (depth_mask[0, fi] & gt_valid[0, fi]).float().cpu().numpy()
            err = np.abs(pd - gt) * valid_m
            self.writer.add_image(f"frame{fi:02d}/depth_error",
                                  _depth_colormap(err, 0.0, 1.0), step)
            # Combined panel: RGB | GT | pred | error
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(np.transpose(rgb_img, (1, 2, 0))); axes[0].set_title("RGB"); axes[0].axis("off")
            axes[1].imshow(gt, cmap="turbo", vmin=0, vmax=depth_max_m); axes[1].set_title("GT depth"); axes[1].axis("off")
            axes[2].imshow(pd, cmap="turbo", vmin=0, vmax=depth_max_m); axes[2].set_title("pred depth"); axes[2].axis("off")
            axes[3].imshow(err, cmap="turbo", vmin=0, vmax=1.0); axes[3].set_title("|pred-gt| (masked)"); axes[3].axis("off")
            fig.suptitle(f"step {step}  frame {fi}")
            plt.tight_layout()
            self.writer.add_image(f"panels/frame{fi:02d}", _fig_to_image(fig), step)
            plt.close(fig)

    # ---------- voxel grid figures ----------

    def log_voxel_grid(
        self,
        voxel_write_mass: torch.Tensor,       # (B, V_x, V_y, V_z, 1)
        voxel_bounds: tuple[float, float, float, float, float, float],
        step: int,
    ) -> None:
        """Log 3-view marginal projections + 1D marginals + nonzero-voxel cloud."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        wm = voxel_write_mass[0, ..., 0].detach().float().cpu().numpy()            # (V_x, V_y, V_z)
        v_x, v_y, v_z = wm.shape
        total = wm.sum()
        nonzero = (wm > 0).sum()

        # 3-view marginal projections.
        m_xy = wm.sum(axis=2)                                                       # view from +z
        m_xz = wm.sum(axis=1)                                                       # view from +y
        m_yz = wm.sum(axis=0)                                                       # view from +x
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(m_xy.T, origin="lower", cmap="hot", aspect="auto",
                       extent=[voxel_bounds[0], voxel_bounds[3], voxel_bounds[1], voxel_bounds[4]])
        axes[0].set_title("xy (sum-z)  view from +z"); axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
        axes[1].imshow(m_xz.T, origin="lower", cmap="hot", aspect="auto",
                       extent=[voxel_bounds[0], voxel_bounds[3], voxel_bounds[2], voxel_bounds[5]])
        axes[1].set_title("xz (sum-y)  view from +y"); axes[1].set_xlabel("x (m)"); axes[1].set_ylabel("z (m)")
        axes[2].imshow(m_yz.T, origin="lower", cmap="hot", aspect="auto",
                       extent=[voxel_bounds[1], voxel_bounds[4], voxel_bounds[2], voxel_bounds[5]])
        axes[2].set_title("yz (sum-x)  view from +x"); axes[2].set_xlabel("y (m)"); axes[2].set_ylabel("z (m)")
        fig.suptitle(f"voxel mass projections — step {step}  total={total:.0f}  nonzero={nonzero}/{v_x*v_y*v_z}")
        plt.tight_layout()
        self.writer.add_image("voxel/projections_xy_xz_yz", _fig_to_image(fig), step)
        plt.close(fig)

        # 1D marginals — the boundary-clip indicator.
        m_x = wm.sum(axis=(1, 2))
        m_y = wm.sum(axis=(0, 2))
        m_z = wm.sum(axis=(0, 1))
        x_pos = np.linspace(voxel_bounds[0], voxel_bounds[3], v_x)
        y_pos = np.linspace(voxel_bounds[1], voxel_bounds[4], v_y)
        z_pos = np.linspace(voxel_bounds[2], voxel_bounds[5], v_z)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, pos, m, label, bmin, bmax in zip(
            axes, [x_pos, y_pos, z_pos], [m_x, m_y, m_z],
            ["x", "y", "z"],
            [voxel_bounds[0], voxel_bounds[1], voxel_bounds[2]],
            [voxel_bounds[3], voxel_bounds[4], voxel_bounds[5]],
        ):
            width = (bmax - bmin) / len(pos)
            ax.bar(pos, m, width=width, color="tab:blue")
            ax.axvline(bmin, color="red", linestyle=":", label="bounds")
            ax.axvline(bmax, color="red", linestyle=":")
            ax.set_title(f"mass marginal along {label} (boundary spikes = OOB clipping)")
            ax.set_xlabel(f"{label} (m)")
        fig.suptitle(f"voxel 1D marginals — step {step}")
        plt.tight_layout()
        self.writer.add_image("voxel/marginals_1d", _fig_to_image(fig), step)
        plt.close(fig)

    # ---------- trajectory figure ----------

    def log_trajectory(
        self,
        pred_delta_9: torch.Tensor,           # (B, T, 9)  predicted relative motion
        gt_delta_9: torch.Tensor,             # (B, T, 9)  GT relative motion
        step: int,
    ) -> None:
        """Plot predicted vs GT integrated trajectory in xz plane.

        Integrates per-frame deltas to give absolute positions; if predicted
        deltas have systematic bias, the integrated curves separate visibly.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pred_t = pred_delta_9[0, :, :3].detach().float().cpu().numpy()              # (T, 3)
        gt_t = gt_delta_9[0, :, :3].detach().float().cpu().numpy()                  # (T, 3)
        # Cumulative integration (frame 0 = origin).
        pred_pos = np.cumsum(pred_t, axis=0)
        gt_pos = np.cumsum(gt_t, axis=0)
        pred_pos = np.concatenate([np.zeros((1, 3)), pred_pos], axis=0)
        gt_pos = np.concatenate([np.zeros((1, 3)), gt_pos], axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(gt_pos[:, 0], gt_pos[:, 2], "o-", color="green", label="GT")
        axes[0].plot(pred_pos[:, 0], pred_pos[:, 2], "o-", color="red", label="pred")
        axes[0].set_title("integrated trajectory (xz plane)"); axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("z (m)")
        axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_aspect("equal")
        # Per-frame delta magnitudes.
        pred_mag = np.linalg.norm(pred_t, axis=-1)
        gt_mag = np.linalg.norm(gt_t, axis=-1)
        axes[1].plot(np.arange(len(pred_mag)), pred_mag, "o-", color="red", label="pred |Δt|")
        axes[1].plot(np.arange(len(gt_mag)), gt_mag, "o-", color="green", label="GT |Δt|")
        axes[1].set_title("per-frame translation delta magnitude")
        axes[1].set_xlabel("frame"); axes[1].set_ylabel("|Δt| (m)")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.suptitle(f"trajectory diagnostics — step {step}")
        plt.tight_layout()
        self.writer.add_image("trajectory/integrated_and_delta", _fig_to_image(fig), step)
        plt.close(fig)
