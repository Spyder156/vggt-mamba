"""Depth pipeline visualization — GT poses substituted at every step.

WHY GT POSES: when streaming uses predicted poses, the writes happen at
drifted world locations and the rendered depth is contaminated by pose
drift. We can't see the depth half's actual quality through that. By
replacing the predicted pose with the GT pose at every frame, the WRITE
lands at the true world location and the RENDER reads from the true camera
position. What we see is then THE PURE DEPTH PIPELINE quality.

WHAT GETS SAVED per test frame (--out/frame_XXXX/):
  pipeline.png — six-panel comparison of the depth pipeline stages:
    RGB | GT depth (full-res) | bootstrap_d (patch-res, kron-upsampled)
    rendered_patch_depth (patch-res, kron-upsampled) | dense_depth (bilinear-
        upsampled, the model's actual output) | |dense - GT| / GT relative
        error map
  scatter.png — three scatter plots showing each pipeline stage vs GT:
    bootstrap_d vs GT_patch | rendered_patch vs GT_patch | dense vs GT_full
    with mean abs_rel and correlation in each title

PRINTED stage-by-stage abs_rel per test frame, so the cap is identifiable:
  - abs_rel(bootstrap_d, GT_patch)       ← encoder quality
  - abs_rel(rendered_patch_depth, GT)    ← voxel/render quality
  - abs_rel(dense_depth, GT_full)        ← final output (matches training)

If the three numbers are close to each other, the cap is in the encoder
(can't get better than bootstrap_d). If rendered_patch is worse than
bootstrap_d, the voxel/render is the cap. If dense_depth is much worse
than rendered_patch, the bilinear upsample is the cap.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.data.tum_rgbd import sync_sequence, intrinsics_for                 # noqa: E402
from vggt_mamba.models.voxel_grid import (                                          # noqa: E402
    backproject_patches_to_world, build_rays_from_pose,
    init_voxel_state, render_rays_volumetric, write_voxels_trilinear,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from terrawm_d_regrounding_stream import load_model, load_rgb                       # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--seq", default="rgbd_dataset_freiburg2_desk")
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--out", type=Path, default=Path("viz/output/depth_pipeline_gt"))
    p.add_argument("--test-frames", type=int, nargs="+",
                    default=[10, 100, 500, 1000, 1500, 2000])
    p.add_argument("--n-frames", type=int, default=2100)
    return p.parse_args()


def load_gt_depth_image(rec, img_size: int, depth_max_m: float = 8.0):
    from PIL import Image
    d = Image.open(rec.depth_path).resize((img_size, img_size), Image.NEAREST)
    d = np.asarray(d, dtype=np.float32) / 5000.0
    d = np.where((d > 0) & (d < depth_max_m), d, np.nan)
    return d


@torch.no_grad()
def stream_with_gt_poses(model, recs, K, gt_rel_T, test_frame_set):
    """Stream the sequence, but REPLACE the model's predicted pose with the GT
    pose at every step. WRITE + RENDER use GT poses. Snapshot the voxel state
    at each test frame so we can run the per-frame diagnostics on it."""
    img_size = model.img_size
    voxel_state = model.init_voxel_state(1, "cuda", torch.float32)
    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0).float()
    snaps = {}
    for i, rec in enumerate(recs):
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        # === THE CHANGE: pose is GT, not predicted. ===
        # (pose head is not even run — we don't need its output for write/render)
        pose_T = gt_rel_T[i:i+1].float()                                              # (1, 4, 4)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            patches = model._encode_frame(rgb)
            bootstrap_d = model.bootstrap_depth(patches).float()
            voxel_feat = model.patch_to_voxel(patches).float()
            wc = model.write_confidence(patches).float() if model.use_write_confidence else None
            # WRITE at GT pose.
            wp = backproject_patches_to_world(patch_pixel, bootstrap_d, K, pose_T)
            write_voxels_trilinear(voxel_state, wp, voxel_feat, weights=wc)
        if i in test_frame_set:
            snaps[i] = {
                "voxel_state": {
                    "features": voxel_state.features.detach().clone(),
                    "write_mass": voxel_state.write_mass.detach().clone(),
                    "cfg": voxel_state.cfg,
                },
                "pose_T": pose_T[0].detach().float().cpu().numpy().copy(),
                "bootstrap_d_patch": bootstrap_d[0].detach().float().cpu().numpy().copy(),
            }
        if (i + 1) % 200 == 0:
            print(f"[depth-pipe] streamed {i+1}/{len(recs)}  "
                   f"total_mass={float(voxel_state.write_mass.sum()):.0f}")
    return snaps


def restore_voxel_state(snap):
    s = init_voxel_state(snap["voxel_state"]["cfg"], 1, "cuda", torch.float32)
    s.features.copy_(snap["voxel_state"]["features"])
    s.write_mass.copy_(snap["voxel_state"]["write_mass"])
    return s


def viz_pipeline(out_dir: Path, frame_idx: int, rgb_np: np.ndarray,
                  gt_depth_full: np.ndarray, bootstrap_d_patch: np.ndarray,
                  rendered_patch: np.ndarray, dense_depth: np.ndarray,
                  img_size: int, grid_h: int, grid_w: int):
    """Six-panel depth pipeline comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Kron-upsample patch-resolution depths to image resolution for visual comparison.
    block_h = img_size // grid_h
    block_w = img_size // grid_w
    bootstrap_d_img = np.kron(bootstrap_d_patch.reshape(grid_h, grid_w),
                                np.ones((block_h, block_w)))
    rendered_patch_img = np.kron(rendered_patch.reshape(grid_h, grid_w),
                                   np.ones((block_h, block_w)))

    # Consistent colormap range across all depth views.
    valid_gt = gt_depth_full[~np.isnan(gt_depth_full)]
    all_depths = np.concatenate([
        valid_gt,
        bootstrap_d_img.flatten(),
        rendered_patch_img.flatten(),
        dense_depth.flatten(),
    ])
    vmin = float(np.percentile(all_depths, 1))
    vmax = float(np.percentile(all_depths, 99))

    # Per-pixel relative error |dense - GT| / GT.
    valid = ~np.isnan(gt_depth_full)
    relerr = np.zeros_like(gt_depth_full)
    relerr[valid] = np.abs(dense_depth[valid] - gt_depth_full[valid]) / np.clip(
        gt_depth_full[valid], 0.1, None)
    relerr[~valid] = np.nan

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(rgb_np)
    axes[0, 0].set_title(f"RGB (frame {frame_idx})")
    axes[0, 0].axis("off")

    im = axes[0, 1].imshow(gt_depth_full, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f"GT depth (full-res)\n"
                          f"range [{np.nanmin(gt_depth_full):.2f}, {np.nanmax(gt_depth_full):.2f}]m")
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    im = axes[0, 2].imshow(bootstrap_d_img, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f"bootstrap_d (encoder, 32×32 kron-upsampled)\n"
                          f"range [{bootstrap_d_patch.min():.2f}, {bootstrap_d_patch.max():.2f}]m")
    axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    im = axes[1, 0].imshow(rendered_patch_img, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(f"rendered_patch_depth (voxel render, 32×32 kron)\n"
                          f"range [{rendered_patch.min():.2f}, {rendered_patch.max():.2f}]m")
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    im = axes[1, 1].imshow(dense_depth, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[1, 1].set_title(f"dense_depth (bilinear-upsampled, model output)\n"
                          f"range [{dense_depth.min():.2f}, {dense_depth.max():.2f}]m")
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    im = axes[1, 2].imshow(relerr, cmap="hot", vmin=0, vmax=1.0)
    abs_rel_image = float(np.nanmean(relerr))
    axes[1, 2].set_title(f"|dense - GT|/GT per-pixel relative error\n"
                          f"mean abs_rel = {abs_rel_image:.3f}")
    axes[1, 2].axis("off")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    fig.suptitle(f"Frame {frame_idx} depth pipeline @ GT POSE (every step). "
                  f"Same colormap [{vmin:.2f}, {vmax:.2f}]m across depth panels.",
                  fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / f"frame_{frame_idx:04d}/pipeline.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def viz_scatter(out_dir: Path, frame_idx: int,
                 bootstrap_d_patch: np.ndarray, gt_depth_patch: np.ndarray,
                 rendered_patch: np.ndarray, dense_depth: np.ndarray,
                 gt_depth_full: np.ndarray):
    """Three stages vs GT: per-patch scatter plots with abs_rel + correlation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    valid_p = (gt_depth_patch > 0.1) & (gt_depth_patch < 8.0)
    gt_p = gt_depth_patch[valid_p]

    # 1. bootstrap_d vs GT (patch resolution)
    vals = bootstrap_d_patch[valid_p]
    abs_rel = float(np.mean(np.abs(vals - gt_p) / np.clip(gt_p, 0.1, None)))
    corr = float(np.corrcoef(vals, gt_p)[0, 1]) if len(vals) > 1 else float("nan")
    axes[0].scatter(gt_p, vals, s=4, alpha=0.4)
    lim = max(gt_p.max(), vals.max()) * 1.05
    axes[0].plot([0, lim], [0, lim], "g-", alpha=0.5, label="y=x")
    axes[0].set_xlim(0, lim); axes[0].set_ylim(0, lim)
    axes[0].set_xlabel("GT depth (m)"); axes[0].set_ylabel("bootstrap_d (m)")
    axes[0].set_title(f"bootstrap_d vs GT (patch res)\n"
                       f"abs_rel = {abs_rel:.3f}, corr = {corr:+.3f}")
    axes[0].grid(alpha=0.3); axes[0].legend()

    # 2. rendered_patch vs GT (patch resolution)
    vals = rendered_patch[valid_p]
    abs_rel = float(np.mean(np.abs(vals - gt_p) / np.clip(gt_p, 0.1, None)))
    corr = float(np.corrcoef(vals, gt_p)[0, 1]) if len(vals) > 1 else float("nan")
    axes[1].scatter(gt_p, vals, s=4, alpha=0.4)
    lim = max(gt_p.max(), vals.max()) * 1.05
    axes[1].plot([0, lim], [0, lim], "g-", alpha=0.5, label="y=x")
    axes[1].set_xlim(0, lim); axes[1].set_ylim(0, lim)
    axes[1].set_xlabel("GT depth (m)"); axes[1].set_ylabel("rendered_patch_depth (m)")
    axes[1].set_title(f"rendered_patch vs GT (patch res)\n"
                       f"abs_rel = {abs_rel:.3f}, corr = {corr:+.3f}")
    axes[1].grid(alpha=0.3); axes[1].legend()

    # 3. dense_depth vs GT (full resolution)
    valid_full = ~np.isnan(gt_depth_full)
    gt_d = gt_depth_full[valid_full]
    dd = dense_depth[valid_full]
    # Subsample to keep plot readable
    if len(gt_d) > 50000:
        idx = np.random.choice(len(gt_d), 50000, replace=False)
        gt_d, dd = gt_d[idx], dd[idx]
    abs_rel = float(np.mean(np.abs(dd - gt_d) / np.clip(gt_d, 0.1, None)))
    corr = float(np.corrcoef(dd, gt_d)[0, 1]) if len(dd) > 1 else float("nan")
    axes[2].scatter(gt_d, dd, s=2, alpha=0.2)
    lim = max(gt_d.max(), dd.max()) * 1.05
    axes[2].plot([0, lim], [0, lim], "g-", alpha=0.5, label="y=x")
    axes[2].set_xlim(0, lim); axes[2].set_ylim(0, lim)
    axes[2].set_xlabel("GT depth (m)"); axes[2].set_ylabel("dense_depth (m)")
    axes[2].set_title(f"dense_depth (model output) vs GT (full res)\n"
                       f"abs_rel = {abs_rel:.3f}, corr = {corr:+.3f}")
    axes[2].grid(alpha=0.3); axes[2].legend()

    fig.suptitle(f"Frame {frame_idx} depth pipeline scatter (GT poses throughout)",
                  fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / f"frame_{frame_idx:04d}/scatter.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.test_frames:
        (args.out / f"frame_{f:04d}").mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(args.ckpt, args.weights_root)
    img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]
    fx, fy, cx, cy = intrinsics_for(args.seq)
    sx, sy = img_size / 640.0, img_size / 480.0
    K = torch.tensor([[[fx*sx, 0, cx*sx], [0, fy*sy, cy*sy], [0, 0, 1]]], device="cuda")
    recs = sync_sequence(args.data_root / args.seq)[:args.n_frames]
    print(f"[depth-pipe] {args.seq}: {len(recs)} frames, GT poses substituted")

    gt_poses = np.stack([r.pose_w_c for r in recs])
    P0_inv = np.linalg.inv(gt_poses[0])
    gt_rel = np.einsum("ij,njk->nik", P0_inv, gt_poses)
    gt_rel_T = torch.from_numpy(gt_rel).float().cuda()

    test_frame_set = set(args.test_frames)
    snaps = stream_with_gt_poses(model, recs, K, gt_rel_T, test_frame_set)

    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0).float()
    print(f"\n[depth-pipe] === STAGE-BY-STAGE ABS_REL PER TEST FRAME ===")
    print(f"  Stage 1 (encoder):     abs_rel(bootstrap_d,     GT_patch)")
    print(f"  Stage 2 (voxel/render): abs_rel(rendered_patch,  GT_patch)")
    print(f"  Stage 3 (final output): abs_rel(dense_depth,     GT_full)")
    print(f"  {'frame':>6}  {'boot':>8}  {'render':>8}  {'dense':>8}  {'Δ render-boot':>14}  {'Δ dense-render':>14}")
    for fi in args.test_frames:
        if fi not in snaps:
            continue
        snap = snaps[fi]
        voxel_state = restore_voxel_state(snap)
        gt_T = torch.from_numpy(snap["pose_T"]).unsqueeze(0).cuda()
        bootstrap_d_patch = snap["bootstrap_d_patch"]                                 # (P,)

        rec = recs[fi]
        gt_depth_full = load_gt_depth_image(rec, img_size, cfg["data"]["depth_max_m"])
        rgb_pil = load_rgb(rec, img_size).squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Render at GT pose (the model's render2 path, replicated for visualization)
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                ray_o, ray_d = build_rays_from_pose(gt_T, K, patch_pixel)
                r = render_rays_volumetric(voxel_state, ray_o, ray_d,
                    n_samples=model.n_render_samples,
                    near=model.render_near, far=model.render_far)
        rendered_patch = r["depth"][0].float().cpu().numpy()                          # (P,)

        # Reproduce the model's bilinear upsample of rendered_patch to full image res.
        rendered_t = torch.from_numpy(rendered_patch).view(1, 1, model.grid_h, model.grid_w).float()
        dense_depth = F.interpolate(rendered_t, size=(img_size, img_size),
                                      mode="bilinear", align_corners=True
                                      ).squeeze().numpy()                              # (H, W)

        # GT depth at patch resolution (for stage 1 + 2 comparisons)
        gt_depth_t = torch.from_numpy(np.nan_to_num(gt_depth_full, nan=0)
                                        ).float().unsqueeze(0).unsqueeze(0)
        gt_depth_patch = F.adaptive_avg_pool2d(gt_depth_t, (model.grid_h, model.grid_w)
                                                ).squeeze().numpy().flatten()         # (P,)

        # Stage abs_rels.
        valid_p = (gt_depth_patch > 0.1) & (gt_depth_patch < 8.0)
        boot_abs = float(np.mean(np.abs(bootstrap_d_patch[valid_p] - gt_depth_patch[valid_p])
                                  / np.clip(gt_depth_patch[valid_p], 0.1, None)))
        rend_abs = float(np.mean(np.abs(rendered_patch[valid_p] - gt_depth_patch[valid_p])
                                  / np.clip(gt_depth_patch[valid_p], 0.1, None)))
        valid_full = ~np.isnan(gt_depth_full)
        dense_abs = float(np.mean(np.abs(dense_depth[valid_full] - gt_depth_full[valid_full])
                                    / np.clip(gt_depth_full[valid_full], 0.1, None)))
        print(f"  {fi:>6}  {boot_abs:>8.3f}  {rend_abs:>8.3f}  {dense_abs:>8.3f}  "
               f"{rend_abs - boot_abs:>+14.3f}  {dense_abs - rend_abs:>+14.3f}")

        viz_pipeline(args.out, fi, rgb_pil, gt_depth_full, bootstrap_d_patch,
                      rendered_patch, dense_depth, img_size, model.grid_h, model.grid_w)
        viz_scatter(args.out, fi, bootstrap_d_patch, gt_depth_patch, rendered_patch,
                     dense_depth, gt_depth_full)

    print(f"\n[depth-pipe] saved to {args.out}/")
    print(f"  per frame: frame_XXXX/pipeline.png  frame_XXXX/scatter.png")


if __name__ == "__main__":
    main()
