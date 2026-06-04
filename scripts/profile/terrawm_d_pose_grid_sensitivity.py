"""TerraWM-D pose-head grid-sensitivity test.

Isolates "does the pose head READ the grid" from confounds in the
trajectory-divergence ablation (which conflates grid-dependence with
output-volatility under wipe).

Method:
  1. Stream a sequence with GT poses to populate the grid up to frame K
     (continuous), AND with resets at R ∈ {100, 200, 300, 400, 500}, snap
     the grid at the same eval frame for each.
  2. Pick a test frame's RGB → encode patches once (so patches are IDENTICAL
     across the comparison).
  3. For each grid snapshot, run pose_head with:
       - the SAME patches
       - the SAME initial_pose (GT pose at test frame)
       - the DIFFERENT grid (everything else equal)
  4. Compare pose_head's predicted DELTA across the snapshots.

Reading:
  - If pred_delta varies strongly with grid → pose head reads the grid.
  - If pred_delta is similar across all grids → pose head is grid-insensitive
    (the trajectory-divergence ablation was measuring output volatility, not
    grid dependence).

This is the cleanest possible test of pose-head's functional dependence on
grid state. No integration dynamics; single forward call per grid.
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
from vggt_mamba.models.terrawm_d import build_terrawm_d                            # noqa: E402
from vggt_mamba.models.aggregators.anchor_pool import cam9_to_pose_w_c              # noqa: E402
from vggt_mamba.models.voxel_grid import (                                          # noqa: E402
    backproject_patches_to_world, build_rays_from_pose,
    render_rays_volumetric, reset_voxel_state, write_voxels_trilinear,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--seq", default="rgbd_dataset_freiburg3_sitting_xyz")
    p.add_argument("--reset-points", type=int, nargs="+", default=[100, 200, 300, 400, 500])
    p.add_argument("--eval-frame", type=int, default=600)
    p.add_argument("--out-dir", type=Path,
                   default=Path("viz/output/terrawm_d_pose_grid_sensitivity"))
    return p.parse_args()


def load_model(ckpt_path, weights_root):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
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
    )
    model.load_state_dict(ckpt["model"], strict=False)
    return model.cuda().eval(), cfg


def load_rgb(rec, img_size):
    from PIL import Image
    img = Image.open(rec.rgb_path).convert("RGB").resize((img_size, img_size))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0
                            ).permute(2, 0, 1).unsqueeze(0).contiguous()


@torch.no_grad()
def stream_to_snapshot(model, recs, gt_rel_T, K, fov, reset_frame, eval_frame):
    """Stream with GT poses, snapshot voxel state at eval_frame."""
    img_size = model.img_size
    voxel_state = model.init_voxel_state(batch_size=1, device="cuda", dtype=torch.float32)
    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0).float()
    snap = None
    for i, rec in enumerate(recs):
        if reset_frame is not None and i == reset_frame:
            reset_voxel_state(voxel_state)
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        gt_pose_T = gt_rel_T[i:i + 1]
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            patches = model._encode_frame(rgb)
            bootstrap_d = model.bootstrap_depth(patches).float()
            voxel_feat = model.patch_to_voxel(patches).float()
            write_w = model.write_confidence(patches).float() if model.use_write_confidence else None
        world_pts = backproject_patches_to_world(patch_pixel, bootstrap_d, K, gt_pose_T)
        write_voxels_trilinear(voxel_state, world_pts, voxel_feat, weights=write_w)
        if i == eval_frame:
            snap = {
                "features": voxel_state.features.detach().clone().cpu(),
                "write_mass": voxel_state.write_mass.detach().clone().cpu(),
            }
    return snap


def restore_voxel_state(model, snapshot, device="cuda"):
    state = model.init_voxel_state(batch_size=1, device=device, dtype=torch.float32)
    state.features.copy_(snapshot["features"].to(device))
    state.write_mass.copy_(snapshot["write_mass"].to(device))
    return state


@torch.no_grad()
def pose_head_response(model, patches, voxel_state, initial_pose_T, K, fov):
    """Run pose_head end-to-end with the given grid + patches + initial pose.
    Returns the predicted DELTA (B, 9) and the render's coverage and feature norms.
    """
    B = patches.shape[0]
    patch_pixel = model._patch_pixel(B, patches.device)
    ray_o, ray_d = build_rays_from_pose(initial_pose_T, K, patch_pixel)
    render = render_rays_volumetric(
        voxel_state, ray_o, ray_d,
        n_samples=model.n_render_samples, near=model.render_near, far=model.render_far,
    )
    rendered_feat = render["feature"]
    ray_total_w = render["total_weight"]
    coverage_frac = float((ray_total_w > 1e-3).float().mean())
    feat_norm = float(rendered_feat.norm(dim=-1).mean())
    from vggt_mamba.models.terrawm_d import _pose_T_to_cam9
    initial_pose_9 = _pose_T_to_cam9(initial_pose_T, fov)
    delta_9 = model.pose_head(patches, rendered_feat, ray_total_w, initial_pose_9)
    return delta_9, coverage_frac, feat_norm


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model, cfg = load_model(args.ckpt, args.weights_root)
    img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]
    fx, fy, cx, cy = intrinsics_for(args.seq)
    sx, sy = img_size / 640.0, img_size / 480.0
    K = torch.tensor([[[fx * sx, 0., cx * sx], [0., fy * sy, cy * sy], [0., 0., 1.]]],
                     device="cuda")
    fov = torch.tensor([[1.0, 1.0]], device="cuda")
    recs = sync_sequence(args.data_root / args.seq)[:args.eval_frame + 1]

    # Relativize GT poses to frame 0.
    gt_poses_w_c = np.stack([r.pose_w_c for r in recs])
    P0_inv = np.linalg.inv(gt_poses_w_c[0])
    gt_rel = np.einsum("ij,njk->nik", P0_inv, gt_poses_w_c)
    gt_rel_T = torch.from_numpy(gt_rel).float().cuda()
    print(f"[d-sense] {args.seq}: ckpt={args.ckpt.name}  eval @ frame {args.eval_frame}")
    print(f"[d-sense] reset points: {args.reset_points}")

    # === Step 1: stream + snapshot grid for each reset point ===
    R_to_run = [("continuous", None)] + [(f"reset_{R}", R) for R in args.reset_points]
    snapshots = {}
    for label, R in R_to_run:
        t0 = time.perf_counter()
        snap = stream_to_snapshot(model, recs, gt_rel_T, K, fov, R, args.eval_frame)
        dt = time.perf_counter() - t0
        mass = snap["write_mass"].sum().item()
        nz = (snap["write_mass"] > 1e-3).sum().item()
        snapshots[label] = snap
        print(f"[d-sense]   snap {label:15s}: {dt:.1f}s  mass={mass:.0f}  nonzero={nz}")

    # === Step 2: encode TEST frame's patches ONCE (frame eval_frame) ===
    test_rec = recs[args.eval_frame]
    test_rgb = load_rgb(test_rec, img_size).cuda(non_blocking=True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        test_patches = model._encode_frame(test_rgb)
    test_pose_T = gt_rel_T[args.eval_frame:args.eval_frame + 1]
    print(f"\n[d-sense] === Pose-head response under fixed patches + pose, varying grid ===")
    print(f"[d-sense] patches: {tuple(test_patches.shape)}  test_pose translation: {test_pose_T[0, :3, 3].cpu().numpy()}")

    # === Step 3: for each snapshot, get pose-head response ===
    print(f"\n{'grid':>15s}  {'coverage':>9s}  {'feat_norm':>10s}  {'dt_x':>9s} {'dt_y':>9s} {'dt_z':>9s}  {'|dt|':>9s}  {'dq_norm':>9s}")
    results = []
    for label, _ in R_to_run:
        state = restore_voxel_state(model, snapshots[label])
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            delta_9, cov, feat_norm = pose_head_response(model, test_patches, state, test_pose_T, K, fov)
        d = delta_9[0].float().cpu().numpy()                                    # (9,)
        dt = d[:3]
        dq = d[3:7]
        dt_mag = float(np.linalg.norm(dt))
        dq_mag = float(np.linalg.norm(dq - np.array([0., 0., 0., 1.])))  # distance from identity quat
        print(f"{label:>15s}  {cov:9.3f}  {feat_norm:10.4f}  "
              f"{dt[0]:+9.5f} {dt[1]:+9.5f} {dt[2]:+9.5f}  "
              f"{dt_mag:9.5f}  {dq_mag:9.5f}")
        results.append({
            "grid": label,
            "coverage": cov,
            "feat_norm": feat_norm,
            "dt": dt.tolist(),
            "dt_magnitude": dt_mag,
            "dq_dist_from_identity": dq_mag,
            "delta_9": d.tolist(),
        })

    # === Step 4: sensitivity analysis ===
    # If pose head reads grid, deltas should DIFFER substantially across grids.
    # If pose head ignores grid, deltas should be similar (especially for non-empty grids).
    dts = np.array([r["dt"] for r in results])                                   # (n, 3)
    cont_dt = dts[0]
    rel_diffs = []
    print(f"\n[d-sense] === Sensitivity (delta vs continuous grid) ===")
    print(f"{'grid':>15s}  {'|dt - cont_dt|':>17s}  {'|dt - cont_dt|/|cont_dt|':>26s}")
    for i, label in enumerate([r['grid'] for r in results]):
        diff = float(np.linalg.norm(dts[i] - cont_dt))
        rel = diff / max(float(np.linalg.norm(cont_dt)), 1e-6) if i > 0 else 0.0
        rel_diffs.append(diff)
        print(f"{label:>15s}  {diff:17.5f}  {rel:26.3f}")

    spread = float(np.std([np.linalg.norm(d) for d in dts]))
    range_mag = float(max(np.linalg.norm(d) for d in dts) - min(np.linalg.norm(d) for d in dts))
    print(f"\n[d-sense] |dt| std across all grids:   {spread:.5f}")
    print(f"[d-sense] |dt| range across all grids: {range_mag:.5f}")
    print(f"[d-sense] |dt| of continuous grid:     {float(np.linalg.norm(cont_dt)):.5f}")

    # === VERDICT ===
    cont_mag = float(np.linalg.norm(cont_dt))
    if cont_mag < 1e-6:
        verdict = "CONTINUOUS_DELTA_NEAR_ZERO  (degenerate test condition)"
    else:
        max_rel_diff = max(d / cont_mag for d in rel_diffs)
        if max_rel_diff > 0.5:
            verdict = f"GRID-DEPENDENT  (max relative diff {max_rel_diff:.2f} >> 0.5)"
        elif max_rel_diff > 0.1:
            verdict = f"PARTIALLY-DEPENDENT  (max relative diff {max_rel_diff:.2f})"
        else:
            verdict = f"GRID-INSENSITIVE  (max relative diff {max_rel_diff:.2f} << 0.5 — pose head ignores grid)"
    print(f"\n[d-sense] === VERDICT ===")
    print(f"[d-sense] {verdict}")

    # === Plot: dt components and magnitude across grids ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = [r["grid"] for r in results]
    x = np.arange(len(labels))
    # Per-axis dt.
    for ax_idx, axis_name in enumerate(["x", "y", "z"]):
        axes[0].plot(x, dts[:, ax_idx], "o-", label=f"dt_{axis_name}")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("predicted dt component (m)")
    axes[0].set_title("Pose-head dt vs grid (fixed patches+pose)")
    axes[0].grid(alpha=0.3); axes[0].legend()
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
    # Magnitude.
    mags = [np.linalg.norm(d) for d in dts]
    axes[1].bar(x, mags, color="tab:blue")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("|dt| (m)")
    axes[1].set_title("Pose-head dt magnitude vs grid")
    axes[1].grid(alpha=0.3)
    # Coverage + feat_norm.
    covs = [r["coverage"] for r in results]
    feats = [r["feat_norm"] for r in results]
    ax2 = axes[2].twinx()
    bars1 = axes[2].bar(x - 0.2, covs, width=0.4, color="tab:green", label="coverage")
    bars2 = ax2.bar(x + 0.2, feats, width=0.4, color="tab:orange", label="feat_norm")
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("coverage", color="tab:green")
    ax2.set_ylabel("feat_norm", color="tab:orange")
    axes[2].set_title("Render statistics vs grid")
    axes[2].grid(alpha=0.3); axes[2].legend(loc="upper left"); ax2.legend(loc="upper right")
    fig.suptitle(f"Pose-head grid-sensitivity — {args.ckpt.name}  ({args.seq} @ f={args.eval_frame})", fontsize=10)
    plt.tight_layout()
    plt.savefig(args.out_dir / "pose_head_sensitivity.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[d-sense] saved plot {args.out_dir / 'pose_head_sensitivity.png'}")

    summary = {
        "ckpt": str(args.ckpt),
        "seq": args.seq,
        "eval_frame": args.eval_frame,
        "results": results,
        "dt_std_across_grids": spread,
        "dt_range_across_grids": range_mag,
        "continuous_dt_magnitude": cont_mag,
        "max_relative_diff_vs_continuous": max(d / max(cont_mag, 1e-6) for d in rel_diffs) if cont_mag > 1e-6 else None,
        "verdict": verdict,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[d-sense] saved summary {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
