"""TerraWM-D long-horizon collapse diagnostic.

The trajectory is shape-correct at ~600 frames but collapses to a scribble
by ~1300 frames. This is now blocking the loop-closure test (the trajectory
is broken before revisits can show their re-grounding effect).

This script streams a long sequence and records, per frame:
  - Predicted absolute position (and ATE vs Sim3-aligned GT).
  - Voxel grid mass + nonzero-voxel count.
  - 1st-render statistics: mean coverage, mean total-weight per ray, depth
    mean/std (the actual input the pose head reads — degraded render → bad
    pose signal).
  - Pose head's output delta magnitude (per-frame |Δt| and rotation magnitude).

Then plots all of these against frame index and reports:
  - Where does each metric start degrading?
  - Which goes first — grid mass change, render quality, pose magnitude, or
    integrated position?
  - That tells us the mechanism: grid-saturation, feedback-collapse, or
    horizon-OOD.

Pure inference; no training. ~1 minute on the 5070 Ti.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.data.tum_rgbd import sync_sequence, intrinsics_for                 # noqa: E402
from vggt_mamba.eval.metrics import umeyama_sim3                                    # noqa: E402
from vggt_mamba.models.aggregators.anchor_pool import cam9_to_pose_w_c              # noqa: E402
from vggt_mamba.models.terrawm_d import build_terrawm_d, _pose_T_to_cam9             # noqa: E402
from vggt_mamba.models.voxel_grid import (                                          # noqa: E402
    backproject_patches_to_world, build_rays_from_pose,
    render_rays_volumetric, write_voxels_trilinear,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--seq", default="rgbd_dataset_freiburg1_room")
    p.add_argument("--n-frames", type=int, default=1362)
    p.add_argument("--out-dir", type=Path,
                   default=Path("viz/output/terrawm_d_long_horizon_diag"))
    p.add_argument("--pose-gate-mode", choices=["coverage", "grid_mass"], default=None,
                   help="override the pose gate mode (default: use ckpt's saved value)")
    return p.parse_args()


def load_model(ckpt_path, weights_root, pose_gate_mode_override=None):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if pose_gate_mode_override is not None:
        cfg["model"]["pose_gate_mode"] = pose_gate_mode_override
        print(f"[d-long] OVERRIDING pose_gate_mode = {pose_gate_mode_override}")
    model = build_terrawm_d(
        cfg["encoder"], str(weights_root),
        n_intraframe_layers=cfg["model"]["n_intraframe_layers"],
        voxel_bounds=tuple(cfg["model"]["voxel_bounds"]),
        voxel_resolution=tuple(cfg["model"]["voxel_resolution"]),
        voxel_feature_dim=cfg["model"]["voxel_feature_dim"],
        n_render_samples=cfg["model"]["n_render_samples"],
        render_near=cfg["model"]["render_near"],
        render_far=cfg["model"]["render_far"],
        bootstrap_hidden=cfg["model"]["bootstrap_hidden"],
        bootstrap_max_depth=cfg["model"]["bootstrap_max_depth"],
        pose_head_hidden=cfg["model"]["pose_head_hidden"],
        pose_max_dt=cfg["model"]["pose_max_dt"],
        pose_max_dq=cfg["model"]["pose_max_dq"],
        unwritten_mask_threshold=cfg["model"]["unwritten_mask_threshold"],
        use_write_confidence=cfg["model"].get("use_write_confidence", False),
        write_confidence_hidden=cfg["model"].get("write_confidence_hidden", 64),
        differentiable_write_geometry=cfg["model"].get("differentiable_write_geometry", False),
        pose_gate_mode=cfg["model"].get("pose_gate_mode", "coverage"),
    )
    model.load_state_dict(ckpt["model"], strict=False)
    return model.cuda().eval(), cfg


def load_rgb(rec, img_size):
    from PIL import Image
    img = Image.open(rec.rgb_path).convert("RGB").resize((img_size, img_size))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0
                            ).permute(2, 0, 1).unsqueeze(0).contiguous()


@torch.no_grad()
def stream_with_diagnostics(model, recs, K, fov):
    """Stream and record per-frame internal stats. Replicates _frame_step's
    logic so we can inspect intermediate values.
    """
    img_size = model.img_size
    voxel_state = model.init_voxel_state(batch_size=1, device="cuda", dtype=torch.float32)
    prev_pose_9 = torch.tensor([[0., 0., 0., 0., 0., 0., 1., 1.0, 1.0]],
                                device="cuda", dtype=torch.float32)
    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0).float()           # (1, P, 2)

    pred_pos = []
    grid_mass = []
    grid_nonzero = []
    render_coverage = []
    render_mean_weight = []
    render_depth_mean = []
    render_depth_std = []
    render_feat_norm = []
    delta_t_mag = []
    delta_q_dist = []
    bootstrap_d_mean = []
    bootstrap_d_std = []

    for i, rec in enumerate(recs):
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        initial_T = cam9_to_pose_w_c(prev_pose_9)                                # (1, 4, 4)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            patches = model._encode_frame(rgb)
            bootstrap_d = model.bootstrap_depth(patches).float()                # (1, P)
            voxel_feat = model.patch_to_voxel(patches).float()
            write_w = model.write_confidence(patches).float() if model.use_write_confidence else None
            # 1st render at initial pose (pose head's input).
            ray_o1, ray_d1 = build_rays_from_pose(initial_T, K, patch_pixel)
            render1 = render_rays_volumetric(
                voxel_state, ray_o1, ray_d1,
                n_samples=model.n_render_samples, near=model.render_near, far=model.render_far,
            )
            rendered_feat = render1["feature"]
            ray_total_w1 = render1["total_weight"]
            initial_pose_9 = _pose_T_to_cam9(initial_T, fov)
            # Mirror _frame_step's gate logic for whichever pose_gate_mode the model is configured for.
            if model.pose_gate_mode == "grid_mass":
                mass_total = voxel_state.write_mass.sum().detach()
                mass_gate = torch.sigmoid((mass_total - 1e3) / 1e2)
                external_gate = mass_gate.expand(1).unsqueeze(-1).to(initial_pose_9.dtype)
            else:
                external_gate = None
            delta_pose_9 = model.pose_head(patches, rendered_feat, ray_total_w1, initial_pose_9,
                                            external_gate=external_gate)

        # Compose and write (mimicking _frame_step).
        delta_pose_T = cam9_to_pose_w_c(delta_pose_9)
        corrected_pose_T = initial_T.float() @ delta_pose_T
        # Write at corrected pose.
        world_pts = backproject_patches_to_world(patch_pixel, bootstrap_d, K, corrected_pose_T.detach())
        write_voxels_trilinear(voxel_state, world_pts, voxel_feat, weights=write_w)

        # Record stats.
        pred_pos.append(corrected_pose_T[0, :3, 3].cpu().numpy())
        grid_mass.append(float(voxel_state.write_mass.sum()))
        grid_nonzero.append(int((voxel_state.write_mass > 1e-3).sum()))
        render_coverage.append(float((ray_total_w1 > 1e-3).float().mean()))
        render_mean_weight.append(float(ray_total_w1.mean()))
        render_depth_mean.append(float(render1["depth"].mean()))
        render_depth_std.append(float(render1["depth"].std()))
        render_feat_norm.append(float(rendered_feat.norm(dim=-1).mean()))
        # Delta translation magnitude (camera-frame).
        delta_t_mag.append(float(delta_pose_9[0, :3].norm()))
        # Quaternion distance from identity (0,0,0,1).
        dq = delta_pose_9[0, 3:7].float().cpu().numpy()
        identity_q = np.array([0., 0., 0., 1.])
        delta_q_dist.append(float(np.linalg.norm(dq - identity_q)))
        bootstrap_d_mean.append(float(bootstrap_d.mean()))
        bootstrap_d_std.append(float(bootstrap_d.std()))

        # Prepare next iteration.
        new_abs_9 = _pose_T_to_cam9(corrected_pose_T, fov)
        prev_pose_9 = new_abs_9.float()

        if (i + 1) % 200 == 0:
            print(f"[d-long]   streamed {i + 1}/{len(recs)} frames  "
                  f"mass={grid_mass[-1]:.0f}  cov={render_coverage[-1]:.2f}  "
                  f"|dt|={delta_t_mag[-1]:.4f}m")

    return {
        "pred_pos": np.stack(pred_pos),
        "grid_mass": np.array(grid_mass),
        "grid_nonzero": np.array(grid_nonzero),
        "render_coverage": np.array(render_coverage),
        "render_mean_weight": np.array(render_mean_weight),
        "render_depth_mean": np.array(render_depth_mean),
        "render_depth_std": np.array(render_depth_std),
        "render_feat_norm": np.array(render_feat_norm),
        "delta_t_mag": np.array(delta_t_mag),
        "delta_q_dist": np.array(delta_q_dist),
        "bootstrap_d_mean": np.array(bootstrap_d_mean),
        "bootstrap_d_std": np.array(bootstrap_d_std),
    }


def find_breakpoint(series: np.ndarray, baseline_window: int = 200,
                     threshold_factor: float = 0.5) -> int | None:
    """Find earliest frame where series departs from baseline by > threshold_factor × baseline_std.
    Returns None if no breakpoint detected.
    """
    if len(series) < 2 * baseline_window:
        return None
    baseline = series[:baseline_window]
    base_mean = baseline.mean()
    base_std = baseline.std() + 1e-9
    threshold = threshold_factor * base_std
    for i in range(baseline_window, len(series)):
        if abs(series[i] - base_mean) > max(threshold, 0.05 * abs(base_mean)):
            return i
    return None


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model, cfg = load_model(args.ckpt, args.weights_root, args.pose_gate_mode)
    img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]
    fx, fy, cx, cy = intrinsics_for(args.seq)
    sx, sy = img_size / 640.0, img_size / 480.0
    K = torch.tensor([[[fx * sx, 0., cx * sx], [0., fy * sy, cy * sy], [0., 0., 1.]]],
                     device="cuda")
    fov = torch.tensor([[1.0, 1.0]], device="cuda")
    recs = sync_sequence(args.data_root / args.seq)[:args.n_frames]
    print(f"[d-long] {args.seq}: streaming {len(recs)} frames + collecting diagnostics")

    t0 = time.perf_counter()
    stats = stream_with_diagnostics(model, recs, K, fov)
    dt = time.perf_counter() - t0
    print(f"[d-long] streamed in {dt:.1f}s")

    # GT for ATE.
    gt_abs_raw = np.stack([r.pose_w_c for r in recs])
    gt_rel = np.einsum("ij,njk->nik", np.linalg.inv(gt_abs_raw[0]), gt_abs_raw)
    gt_t = gt_rel[:, :3, 3]
    pred_t = stats["pred_pos"]
    # Sim3 alignment.
    s, R, t = umeyama_sim3(pred_t, gt_t)
    pred_aligned = (s * (R @ pred_t.T)).T + t
    per_frame_ate = np.linalg.norm(pred_aligned - gt_t, axis=-1)
    stats["per_frame_ate"] = per_frame_ate
    stats["pred_aligned"] = pred_aligned
    print(f"[d-long] Sim3 scale: {s:.4f}  "
          f"ATE mean: {per_frame_ate.mean():.3f}m  final: {per_frame_ate[-1]:.3f}m")

    # === Find breakpoint per metric ===
    metrics_to_check = {
        "grid_mass": stats["grid_mass"],
        "grid_nonzero": stats["grid_nonzero"],
        "render_coverage": stats["render_coverage"],
        "render_mean_weight": stats["render_mean_weight"],
        "render_depth_std": stats["render_depth_std"],
        "render_feat_norm": stats["render_feat_norm"],
        "delta_t_mag": stats["delta_t_mag"],
        "delta_q_dist": stats["delta_q_dist"],
        "bootstrap_d_std": stats["bootstrap_d_std"],
        "per_frame_ate": per_frame_ate,
    }
    print(f"\n[d-long] === BREAKPOINT DETECTION (earliest frame departing 0.5σ from first-200 baseline) ===")
    breakpoints = {}
    for name, series in metrics_to_check.items():
        bp = find_breakpoint(series)
        breakpoints[name] = bp
        baseline_mean = series[:200].mean()
        final_val = series[-1]
        change_pct = (final_val - baseline_mean) / max(abs(baseline_mean), 1e-9) * 100
        bp_str = f"frame {bp}" if bp is not None else "no breakpoint"
        print(f"  {name:25s} baseline {baseline_mean:9.4f}  final {final_val:9.4f}  Δ{change_pct:+7.1f}%  {bp_str}")

    # === Visualizations ===

    # 1. Multi-panel time series.
    fig, axes = plt.subplots(5, 2, figsize=(15, 16), sharex=True)
    frames = np.arange(len(per_frame_ate))

    # Row 1: ATE + Sim3 alignment.
    axes[0, 0].plot(frames, per_frame_ate, color="tab:red", linewidth=0.8)
    axes[0, 0].set_ylabel("per-frame ATE (m)"); axes[0, 0].set_title(f"ATE (Sim3 scale = {s:.3f})")
    axes[0, 0].grid(alpha=0.3)
    pred_traj_len = np.linalg.norm(np.diff(pred_t, axis=0), axis=-1).cumsum()
    gt_traj_len = np.linalg.norm(np.diff(gt_t, axis=0), axis=-1).cumsum()
    axes[0, 1].plot(frames[1:], pred_traj_len, color="tab:red", label="pred")
    axes[0, 1].plot(frames[1:], gt_traj_len, color="tab:green", label="GT")
    axes[0, 1].set_ylabel("cumulative traj length (m)"); axes[0, 1].set_title("Cumulative path length")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # Row 2: Grid stats.
    axes[1, 0].plot(frames, stats["grid_mass"], color="tab:blue", linewidth=0.8)
    axes[1, 0].set_ylabel("grid mass total"); axes[1, 0].set_title("Grid mass total")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 1].plot(frames, stats["grid_nonzero"], color="tab:blue", linewidth=0.8)
    axes[1, 1].set_ylabel("grid nonzero voxels"); axes[1, 1].set_title("Grid nonzero count")
    axes[1, 1].grid(alpha=0.3)

    # Row 3: Render quality.
    axes[2, 0].plot(frames, stats["render_coverage"], color="tab:purple", linewidth=0.8)
    axes[2, 0].set_ylabel("render coverage"); axes[2, 0].set_title("Render coverage (pose-head input)")
    axes[2, 0].grid(alpha=0.3); axes[2, 0].set_ylim(-0.05, 1.05)
    axes[2, 1].plot(frames, stats["render_feat_norm"], color="tab:purple", linewidth=0.8)
    axes[2, 1].set_ylabel("rendered feature norm"); axes[2, 1].set_title("Rendered feature norm (pose-head input)")
    axes[2, 1].grid(alpha=0.3)

    # Row 4: Render depth distribution.
    axes[3, 0].plot(frames, stats["render_depth_mean"], color="tab:orange", linewidth=0.8)
    axes[3, 0].set_ylabel("rendered depth mean (m)"); axes[3, 0].set_title("Rendered depth — mean across pixels")
    axes[3, 0].grid(alpha=0.3)
    axes[3, 1].plot(frames, stats["render_depth_std"], color="tab:orange", linewidth=0.8)
    axes[3, 1].set_ylabel("rendered depth std (m)"); axes[3, 1].set_title("Rendered depth — std (blob=low, sharp=high)")
    axes[3, 1].grid(alpha=0.3)

    # Row 5: Pose-head outputs.
    axes[4, 0].plot(frames, stats["delta_t_mag"], color="tab:green", linewidth=0.8)
    axes[4, 0].set_ylabel("|Δt| per frame (m)"); axes[4, 0].set_title("Pose-head: per-frame translation magnitude")
    axes[4, 0].set_xlabel("frame"); axes[4, 0].grid(alpha=0.3)
    axes[4, 1].plot(frames, stats["delta_q_dist"], color="tab:green", linewidth=0.8)
    axes[4, 1].set_ylabel("|Δq - identity|"); axes[4, 1].set_title("Pose-head: per-frame rotation magnitude")
    axes[4, 1].set_xlabel("frame"); axes[4, 1].grid(alpha=0.3)

    # Mark earliest breakpoint frame across all metrics.
    bp_frames = [v for v in breakpoints.values() if v is not None]
    if bp_frames:
        earliest = min(bp_frames)
        which = min(breakpoints.items(), key=lambda kv: kv[1] if kv[1] is not None else 1e9)
        print(f"\n[d-long] EARLIEST BREAK: frame {earliest}  ({which[0]})")
        for ax_row in axes:
            for ax in ax_row:
                ax.axvline(earliest, color="black", linestyle="--", alpha=0.4, linewidth=1)

    fig.suptitle(f"Long-horizon collapse diagnostic — {args.seq}  "
                  f"({len(recs)} frames, ckpt={args.ckpt.name})\n"
                  f"Find where things break and which goes first (grid → render → pose, or pose → write → grid)",
                  fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out_dir / "long_horizon_diagnostic.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[d-long] saved long_horizon_diagnostic.png")

    # 2. XZ trajectory plot with frame-index coloring.
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.plot(gt_t[:, 0], gt_t[:, 2], "-", color="green", linewidth=1.5, alpha=0.7, label="GT")
    sc = ax.scatter(pred_aligned[:, 0], pred_aligned[:, 2], c=frames, cmap="plasma",
                     s=3, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="frame index")
    if bp_frames:
        earliest = min(bp_frames)
        ax.scatter(pred_aligned[earliest, 0], pred_aligned[earliest, 2], s=100, marker="*",
                   color="red", edgecolors="black", zorder=5, label=f"earliest break (f={earliest})")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.set_title(f"Pred trajectory colored by frame index — {args.seq}")
    ax.legend(); ax.grid(alpha=0.3); ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(args.out_dir / "trajectory_colored.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[d-long] saved trajectory_colored.png")

    # === Mechanism diagnosis ===
    # Order of breaks tells us the mechanism.
    print(f"\n[d-long] === MECHANISM DIAGNOSIS ===")
    ordered = sorted([(k, v) for k, v in breakpoints.items() if v is not None],
                      key=lambda kv: kv[1])
    if ordered:
        print("  Break order (earliest first):")
        for name, bp in ordered:
            print(f"    f={bp}: {name}")
        first = ordered[0][0]
        if first in ("delta_t_mag", "delta_q_dist"):
            mechanism = "POSE-FIRST: pose-head output changed before grid/render. " \
                        "Likely: pose distribution shift at long horizon (horizon-OOD) " \
                        "OR feedback collapse where small pose error compounds to bad writes."
        elif first in ("grid_mass", "grid_nonzero"):
            mechanism = "GRID-FIRST: voxel state changed before pose output. " \
                        "Likely: grid saturation / memory exhaustion as writes accumulate."
        elif first in ("render_coverage", "render_mean_weight", "render_depth_std", "render_feat_norm"):
            mechanism = "RENDER-FIRST: render statistics shifted before pose or grid mass. " \
                        "Likely: grid content quality degrading even though mass is stable."
        else:
            mechanism = f"UNCLEAR ({first} first)"
        print(f"\n  Likely mechanism: {mechanism}")
    else:
        print("  No breakpoints detected — the trajectory may be qualitatively similar throughout.")
        mechanism = "NO_BREAKPOINT"

    summary = {
        "ckpt": str(args.ckpt),
        "seq": args.seq,
        "n_frames": len(recs),
        "sim3_scale": float(s),
        "ate_mean": float(per_frame_ate.mean()),
        "ate_final": float(per_frame_ate[-1]),
        "pred_traj_length": float(pred_traj_len[-1]),
        "gt_traj_length": float(gt_traj_len[-1]),
        "pred_gt_traj_length_ratio": float(pred_traj_len[-1] / max(gt_traj_len[-1], 1e-9)),
        "breakpoints": {k: int(v) if v is not None else None for k, v in breakpoints.items()},
        "earliest_break_frame": int(min([v for v in breakpoints.values() if v is not None]))
            if any(v is not None for v in breakpoints.values()) else None,
        "earliest_break_metric": min(
            ((k, v) for k, v in breakpoints.items() if v is not None),
            key=lambda kv: kv[1], default=(None, None))[0],
        "mechanism": mechanism,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[d-long] saved {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
