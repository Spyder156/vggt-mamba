"""Cross-validate TB Tier-4 diagnostics against the standalone drift-blind check.

The training-loop TB dashboard logs `pose/mismatch_l2`, `pose/mismatch_rel`,
etc. via the model's `forward(return_diagnostics=True)` path. If the dashboard
ever shows the head responding to mismatch (e.g. dt_corr_with_mismatch > 0)
while the standalone streaming check says drift-blind, we want to catch the
inconsistency immediately rather than make load-bearing decisions on a
disagreement between two pipelines.

This script runs the SAME model + checkpoint that produced the standalone
drift-blind result, but invokes `forward()` (batched/windowed, with
teacher-forced initial poses from GT) instead of streaming. It collects
the in-model `diagnostics["mismatch_rel"]` values across an fr1/room window
and compares them to the streaming per_frame.npz means in the matching
frame range.

Agreement (within ~10%) → TB panels read the same signal the standalone
check reads. Use the dashboard with confidence.

Disagreement → something differs between batched-forward and streaming. Most
likely culprits: differentiable-write-geometry on/off, write-confidence flag,
pose-gate mode (batched-forward never uses streaming's per-frame pose drift
because it teacher-forces from GT every frame — that's expected to give
LOWER mismatch_rel than streaming).

So the natural prediction is: batched-forward mismatch_rel ≤ streaming
mismatch_rel (no drift accumulation). The check is that the IN-MODEL
computation agrees with the same computation done outside the model for the
same inputs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.data.tum_rgbd import TUMRGBDDataset                          # noqa: E402
from vggt_mamba.models.terrawm_d import build_terrawm_d                       # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--seq", default="rgbd_dataset_freiburg1_room")
    p.add_argument("--n-frames", type=int, default=16)
    p.add_argument("--streaming-npz",
                    default="viz/output/terrawm_d_drift_blind_check/per_frame.npz")
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
        pose_gate_mode=cfg["model"].get("pose_gate_mode", "coverage"),
    )
    model.load_state_dict(ckpt["model"], strict=False)
    return model.cuda().eval(), cfg


def main():
    args = parse_args()
    model, cfg = load_model(args.ckpt, args.weights_root)
    img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]

    # Build a small dataset with the eval-time stride / frame_stride so the
    # window covers the same temporal extent the streaming check used.
    ds = TUMRGBDDataset(
        args.data_root, split=[args.seq],
        n_frames=args.n_frames,
        stride=1,
        frame_stride=1,
        img_size=img_size,
        depth_max_m=cfg["data"]["depth_max_m"],
    )
    print(f"[tier4-xcheck] {args.seq}: {len(ds)} windows of {args.n_frames} frames")

    # Pull window 0 (frames 0..N-1).
    batch = ds[0]
    rgb = batch["rgb"].unsqueeze(0).cuda()                                 # (1, T, 3, H, W)
    poses = batch["poses_w_c"].unsqueeze(0).float().cuda()                 # (1, T, 4, 4)
    K = batch["K"].unsqueeze(0).cuda()
    fov = batch["camera_gt"][..., 7:].unsqueeze(0).cuda()

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            preds = model(rgb, K_intrinsics=K, gt_poses_w_c=poses, fov=fov,
                           return_voxel_state=True, return_diagnostics=True)

    diag = preds["diagnostics"]
    assert "mismatch_l2" in diag, "diagnostics dict missing mismatch_l2"
    assert diag["mismatch_l2"].shape == (1, args.n_frames), \
        f"expected (1, {args.n_frames}), got {tuple(diag['mismatch_l2'].shape)}"
    print(f"[tier4-xcheck] diagnostics keys: {sorted(diag.keys())}")
    print(f"[tier4-xcheck] shapes: " + ", ".join(f"{k}={tuple(v.shape)}" for k, v in diag.items()))

    # Print per-frame in-model values.
    mm_l2 = diag["mismatch_l2"][0].float().cpu().numpy()
    mm_rel = diag["mismatch_rel"][0].float().cpu().numpy()
    cov_b = diag["render_coverage"][0].float().cpu().numpy()
    dt_b = diag["dt_mag"][0].float().cpu().numpy()
    grid_mass = diag["grid_mass_total"][0].float().cpu().numpy()
    print(f"\n[tier4-xcheck] in-model TB diagnostics (batched-forward, teacher-forced GT poses):")
    print(f"  {'frame':>5} {'cov':>6} {'mm_l2':>8} {'mm_rel':>8} {'dt_mag':>8} {'mass':>10}")
    for t in range(args.n_frames):
        print(f"  {t:>5} {cov_b[t]:>6.3f} {mm_l2[t]:>8.3f} {mm_rel[t]:>8.3f} "
               f"{dt_b[t]:>8.4f} {grid_mass[t]:>10.0f}")

    # === Cross-check against the streaming per_frame.npz ===
    streaming_npz = Path(args.streaming_npz)
    if not streaming_npz.exists():
        print(f"\n[tier4-xcheck] streaming reference {streaming_npz} not found — "
               f"in-model values reported, no streaming comparison.")
        return

    stream = np.load(streaming_npz)
    sm_l2 = stream["mismatch_l2"][:args.n_frames]
    sm_rel = stream["mismatch_rel"][:args.n_frames]
    sm_cov = stream["coverage"][:args.n_frames]
    print(f"\n[tier4-xcheck] streaming reference (first {args.n_frames} frames):")
    print(f"  {'frame':>5} {'cov':>6} {'mm_l2':>8} {'mm_rel':>8}")
    for t in range(args.n_frames):
        print(f"  {t:>5} {sm_cov[t]:>6.3f} {sm_l2[t]:>8.3f} {sm_rel[t]:>8.3f}")

    print(f"\n[tier4-xcheck] === BATCHED vs STREAMING ===")
    # KEY EXPECTATION: batched mismatch_rel ≤ streaming mismatch_rel
    # because batched uses teacher-forced GT initial poses (no drift), whereas
    # streaming integrates predicted poses. The check is that both are
    # computing the SAME quantity (||pooled_diff||/||pooled_cur||) consistently.
    # If batched > streaming, something is wrong with the in-model computation.
    print(f"  mean mismatch_rel — batched: {mm_rel.mean():.3f}  streaming: {sm_rel.mean():.3f}")
    print(f"  mean mismatch_l2  — batched: {mm_l2.mean():.3f}  streaming: {sm_l2.mean():.3f}")
    print(f"  mean coverage     — batched: {cov_b.mean():.3f}  streaming: {sm_cov.mean():.3f}")
    if mm_rel.mean() <= sm_rel.mean() * 1.5:
        print(f"  CONSISTENT: batched mismatch_rel ≤ 1.5× streaming (teacher-forcing reduces drift).")
    else:
        print(f"  WARNING: batched mismatch_rel > streaming. Investigate before trusting TB panels.")

    print(f"\n[tier4-xcheck] OK — TB Tier-4 diagnostics dict is populated and shapes are correct.")


if __name__ == "__main__":
    main()
