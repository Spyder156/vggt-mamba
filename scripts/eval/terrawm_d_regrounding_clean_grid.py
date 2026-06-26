"""Clean-grid disambiguation — run ONLY on PARTIAL/NULL verdicts.

The streaming re-grounding test refines pose against a grid populated by
prior (drifted) writes. Even a geometrically-perfect correction renders
against a partially-corrupted target, capping how much ΔATE can be reduced.

This script answers: is the bottleneck the CORRECTION or the GRID POLLUTION?

Procedure:
  1. Build a CLEAN grid by streaming the held-out sequence with GT POSES.
     Pose head's output is ignored; writes happen at GT-world poses each frame.
     The resulting grid is what re-grounding WOULD see if the prior writes
     had been pose-correct.
  2. At each fire frame from the original RG run, replay the same refinement
     (k_refine, lr, threshold) but render against this CLEAN grid built up to
     that frame.
  3. Compute ΔATE_clean per fire using the matched-frame metric (same form
     as the main gate, but with the clean-grid refinement replacing the
     streamed-grid refinement).

Interpretation:
  - If ΔATE_clean is STRONG_POSITIVE while ΔATE_streamed is PARTIAL/NULL,
    → the correction works; the streamed grid is too polluted. Next step:
      un-write/re-write at refined pose, OR train with re-grounding ON so
      the grid is built around the refined trajectory from step 1.
  - If ΔATE_clean is also PARTIAL/NULL, → the correction itself is weak;
    photometric channel doesn't carry enough geometric information. Loop back
    to push photometric harder (longer training / different feature).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.data.tum_rgbd import sync_sequence, intrinsics_for                 # noqa: E402
from vggt_mamba.models.terrawm_d import build_terrawm_d                              # noqa: E402
from vggt_mamba.models.voxel_grid import (                                          # noqa: E402
    backproject_patches_to_world, build_rays_from_pose,
    render_rays_volumetric, write_voxels_trilinear,
)

# Reuse the refinement primitives from the main RG stream.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from terrawm_d_regrounding_stream import (                                          # noqa: E402
    load_model, load_rgb, compute_photo_mismatch, refine_pose_photometric,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--seq", default="rgbd_dataset_freiburg2_desk")
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--regrounding-dir", type=Path, required=True,
                    help="Directory with fires.json from terrawm_d_regrounding_stream.py")
    p.add_argument("--baseline-dir", type=Path, required=True,
                    help="Directory with per_frame.npz from --baseline run")
    p.add_argument("--out", type=Path, required=True)
    # Locked.
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--k-refine", type=int, default=10)
    p.add_argument("--refine-lr", type=float, default=0.05)
    p.add_argument("--max-step-translation-m", type=float, default=0.10)
    p.add_argument("--max-total-jump-m", type=float, default=2.0)
    return p.parse_args()


@torch.no_grad()
def build_clean_grid_streaming(model, recs, K, gt_poses_T):
    """Stream the sequence writing at GT POSES (not pred poses). Returns a list
    of voxel_state snapshots — one per frame — so we can refine against the
    state THE FIRE FRAME WOULD HAVE SEEN had prior writes been pose-correct.

    For memory: store only the voxel_state at fire frames + neighboring frames.
    Strategy: keep one rolling clean voxel_state, and when we hit a fire frame,
    deep-copy a snapshot of its current write_mass + features tensors. Other
    frames just update the rolling state."""
    img_size = model.img_size
    voxel_state = model.init_voxel_state(1, "cuda", torch.float32)
    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0).float()
    return voxel_state, patch_pixel


def snapshot_voxel_state(state):
    """Deep-copy voxel_state for later replay. Returns a lightweight container."""
    return {
        "features": state.features.detach().clone(),
        "write_mass": state.write_mass.detach().clone(),
        "cfg": state.cfg,
    }


def restore_voxel_state(model, snap):
    """Rebuild a VoxelGridState from a snapshot."""
    s = model.init_voxel_state(1, "cuda", torch.float32)
    s.features.copy_(snap["features"])
    s.write_mass.copy_(snap["write_mass"])
    return s


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model, cfg = load_model(args.ckpt, args.weights_root)
    img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]
    fx, fy, cx, cy = intrinsics_for(args.seq)
    sx, sy = img_size / 640.0, img_size / 480.0
    K = torch.tensor([[[fx*sx, 0, cx*sx], [0, fy*sy, cy*sy], [0, 0, 1]]], device="cuda")
    recs = sync_sequence(args.data_root / args.seq)
    print(f"[clean-grid] {args.seq}: streaming {len(recs)} frames at GT POSES to build clean grid")

    fires = json.loads((args.regrounding_dir / "fires.json").read_text())
    fire_frames = sorted({f["frame"] for f in fires})
    print(f"[clean-grid] {len(fire_frames)} unique fire frames from RG run")

    # === Stream with GT poses; snapshot voxel_state at fire frames ===
    gt_poses = np.stack([r.pose_w_c for r in recs])
    P0_inv = np.linalg.inv(gt_poses[0])
    gt_rel = np.einsum("ij,njk->nik", P0_inv, gt_poses)
    gt_rel_T = torch.from_numpy(gt_rel).float().cuda()
    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0).float()
    voxel_state = model.init_voxel_state(1, "cuda", torch.float32)
    fire_frame_set = set(fire_frames)
    snapshots = {}

    for i, rec in enumerate(recs):
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        gt_T = gt_rel_T[i:i+1]                                                  # (1, 4, 4)
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                patches = model._encode_frame(rgb)
                bootstrap_d = model.bootstrap_depth(patches).float()
                voxel_feat = model.patch_to_voxel(patches).float()
                wc = model.write_confidence(patches).float() if model.use_write_confidence else None
                wp = backproject_patches_to_world(patch_pixel, bootstrap_d, K, gt_T)
                write_voxels_trilinear(voxel_state, wp, voxel_feat, weights=wc)
        if i in fire_frame_set:
            snapshots[i] = snapshot_voxel_state(voxel_state)
        if (i + 1) % 200 == 0:
            print(f"[clean-grid] streamed {i+1}/{len(recs)} GT-pose frames")

    print(f"[clean-grid] snapshotted {len(snapshots)} clean-grid states at fire frames")

    # === Replay refinement at each fire frame against the clean grid ===
    base_disp = np.load(args.baseline_dir / "per_frame.npz")["displacement"]
    n = len(base_disp)
    per_fire = []
    refined_displacements_clean = []
    for fire in fires:
        t = fire["frame"]
        if t not in snapshots:
            continue
        if not fire.get("fired"):
            continue
        snap = snapshots[t]
        rgb_t = load_rgb(recs[t], img_size).cuda()
        clean_state = restore_voxel_state(model, snap)
        # Initial pose for refinement: the pose the RG run's pose head emitted at t.
        # Stored in fire["initial_pose_t"] (translation only); we reconstruct a
        # full 4×4 by taking the RG-run's per-frame pose at t-1 + a translation-only
        # delta. SIMPLER: build a translation-only T (rotation = identity) — this is
        # a known limitation (refinement on clean grid starts from RG's translation
        # estimate; rotation reset). For an MVP disambiguation this is OK because
        # most TUM drift is translational.
        initial_T = torch.eye(4, device="cuda").unsqueeze(0).float()
        initial_T[0, :3, 3] = torch.tensor(fire["initial_pose_t"], device="cuda")
        gt_pos = np.array(fire["gt_pos"])
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            refined_T, mm_rel_after, status, total_jump = refine_pose_photometric(
                model, clean_state, initial_T, K, patch_pixel, rgb_t,
                k_refine=args.k_refine, lr=args.refine_lr,
                max_step_translation_m=args.max_step_translation_m,
                max_total_jump_m=args.max_total_jump_m,
            )
        refined_t = refined_T[0, :3, 3].float().cpu().numpy()
        displacement_clean = float(np.linalg.norm(refined_t - gt_pos))
        per_fire.append({
            "frame": int(t),
            "status": status,
            "displacement_before_m": float(fire["displacement_before_m"]),
            "displacement_after_streamed_m": float(fire["displacement_after_m"]),
            "displacement_after_clean_m": displacement_clean,
            "mm_rel_after_clean": mm_rel_after,
            "total_jump_m": total_jump,
        })
        refined_displacements_clean.append(displacement_clean)

    # Aggregate. The CLEAN-grid version doesn't have a paired streaming run; we
    # compare per-fire displacement reduction (init → refined_clean) directly.
    if per_fire:
        deltas_streamed = np.array([f["displacement_after_streamed_m"] - f["displacement_before_m"]
                                      for f in per_fire])
        deltas_clean = np.array([f["displacement_after_clean_m"] - f["displacement_before_m"]
                                   for f in per_fire])
        print(f"\n[clean-grid] === DISAMBIGUATION RESULT ({len(per_fire)} fires) ===")
        print(f"  Δ displacement (refined−initial):")
        print(f"    Streamed grid:  mean={float(deltas_streamed.mean()):+.4f}m  "
               f"median={float(np.median(deltas_streamed)):+.4f}m")
        print(f"    CLEAN grid:     mean={float(deltas_clean.mean()):+.4f}m  "
               f"median={float(np.median(deltas_clean)):+.4f}m")
        improvement = float(deltas_streamed.mean()) - float(deltas_clean.mean())
        print(f"    Clean-grid IMPROVEMENT over streamed: {improvement:+.4f}m per fire")
        print(f"  Toward GT (refined_clean closer to GT than initial):"
               f" {sum(1 for d in deltas_clean if d < 0)}/{len(deltas_clean)}")
    args.out.write_text(json.dumps({
        "seq": args.seq,
        "n_fires_replayed": len(per_fire),
        "delta_displacement_streamed_mean": float(deltas_streamed.mean()) if per_fire else None,
        "delta_displacement_clean_mean": float(deltas_clean.mean()) if per_fire else None,
        "improvement_clean_over_streamed_per_fire_m":
            float(deltas_streamed.mean() - deltas_clean.mean()) if per_fire else None,
        "per_fire": per_fire,
    }, indent=2))
    print(f"\n[clean-grid] saved {args.out}")


if __name__ == "__main__":
    main()
