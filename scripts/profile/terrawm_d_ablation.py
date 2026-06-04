"""TerraWM-D scene-state ablation — the locked MVP verdict.

Protocol (locked before training):
  - Stream sitting_xyz, two runs in lockstep
    Run A: continuous (voxel grid evolves normally over all frames)
    Run B: identical to A through frame K-1, then voxel grid is reset to
           zero (features + write_mass) at frame K. Continue streaming.
  - Measure: mean Δt-magnitude difference between A's corrected pose and
    B's corrected pose, over frames K..K+49.

Locked verdict thresholds:
  Strong positive: ≥ 0.10 m   (50× TerraWM baseline 0.00192 m)
  Partial:         0.020 – 0.10 m
  Negative:        < 0.020 m

Why pose not depth: zeroing the grid masks the dense depth output entirely
(loss is gated on `depth_mask`). Pose comes from render-and-compare, which
arithmetically cannot work with an empty grid — that's the test that
exercises the no-bypass property.
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
from vggt_mamba.models.voxel_grid import reset_voxel_state                          # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--seq", default="rgbd_dataset_freiburg3_sitting_xyz")
    p.add_argument("--reset-frame", type=int, default=200)
    p.add_argument("--n-frames", type=int, default=300)
    p.add_argument("--out", type=Path, default=Path("viz/output/terrawm_d_ablation/diag.json"))
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
def stream_one_run(model, recs, K, fov, reset_frame: int | None) -> np.ndarray:
    """Stream the sequence; if reset_frame is given, zero the voxel grid at
    that frame index BEFORE the frame's forward. Return per-frame (T, 9)
    corrected poses."""
    img_size = model.img_size
    voxel_state = model.init_voxel_state(batch_size=1, device="cuda", dtype=torch.float32)
    # Initial pose: identity (matches training convention for frame 0).
    prev_pose_9 = torch.tensor([[0., 0., 0., 0., 0., 0., 1., 1.0, 1.0]],
                                device="cuda", dtype=torch.float32)
    out_cams = []
    for i, rec in enumerate(recs):
        if reset_frame is not None and i == reset_frame:
            reset_voxel_state(voxel_state)
            # Note: we deliberately KEEP prev_pose_9 from frame i-1 (it's an
            # external state). Only the voxel grid is wiped — that's the
            # diagnostic we care about.
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out, corrected_pose_9 = model.streaming_forward(
                rgb, voxel_state, prev_pose_9, K, fov=fov,
            )
        out_cams.append(corrected_pose_9[0].float().cpu().numpy())
        prev_pose_9 = corrected_pose_9.float()
    return np.stack(out_cams)                            # (T, 9)


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model, cfg = load_model(args.ckpt, args.weights_root)
    img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]
    fx, fy, cx, cy = intrinsics_for(args.seq)
    sx, sy = img_size / 640.0, img_size / 480.0
    K = torch.tensor([[[fx * sx, 0., cx * sx], [0., fy * sy, cy * sy], [0., 0., 1.]]],
                     device="cuda")
    fov = torch.tensor([[1.0, 1.0]], device="cuda")
    recs = sync_sequence(args.data_root / args.seq)[:args.n_frames]
    print(f"[d-ablation] {args.seq}: streaming {len(recs)} frames, reset at frame {args.reset_frame}")

    # Run A: continuous
    t0 = time.perf_counter()
    cam_a = stream_one_run(model, recs, K, fov, reset_frame=None)
    dt_a = time.perf_counter() - t0
    # Run B: reset at reset_frame
    t1 = time.perf_counter()
    cam_b = stream_one_run(model, recs, K, fov, reset_frame=args.reset_frame)
    dt_b = time.perf_counter() - t1
    print(f"[d-ablation] run A (continuous): {dt_a:.1f}s   run B (reset): {dt_b:.1f}s")

    # Per-frame Δt-magnitude difference (translation only — fov+quat are bonus).
    t_diff = np.linalg.norm(cam_a[:, :3] - cam_b[:, :3], axis=-1)   # (T,)
    pre_reset = t_diff[:args.reset_frame]
    immediate = t_diff[args.reset_frame:args.reset_frame + 50]
    late = t_diff[args.reset_frame + 100:]

    diag = {
        "ckpt": str(args.ckpt),
        "seq": args.seq,
        "n_frames": len(recs),
        "reset_frame": args.reset_frame,
        "pre_reset_t_diff_max_m": float(pre_reset.max() if pre_reset.size else 0.0),
        "post_reset_immediate_mean_m": float(immediate.mean()),
        "post_reset_immediate_max_m": float(immediate.max()),
        "post_reset_late_mean_m": float(late.mean() if late.size else 0.0),
        "post_reset_late_max_m": float(late.max() if late.size else 0.0),
    }
    print(f"[d-ablation] pre-reset Δt max:        {diag['pre_reset_t_diff_max_m']:.5f} m   (should be ~0)")
    print(f"[d-ablation] post-reset immediate (50 frames):")
    print(f"   mean: {diag['post_reset_immediate_mean_m']:.5f} m   max: {diag['post_reset_immediate_max_m']:.5f} m")
    print(f"[d-ablation] post-reset late:")
    print(f"   mean: {diag['post_reset_late_mean_m']:.5f} m   max: {diag['post_reset_late_max_m']:.5f} m")

    # Apply locked verdict.
    val = diag["post_reset_immediate_mean_m"]
    print(f"\n[d-ablation] === LOCKED VERDICT ===")
    print(f"   primary metric: post-reset immediate mean Δt-diff = {val:.5f} m")
    print(f"   threshold (locked):")
    print(f"     Strong positive (≥ 0.10 m):    voxel grid is load-bearing.")
    print(f"     Partial         (0.020–0.10):  state-dependent but mild.")
    print(f"     Negative        (< 0.020 m):   voxel grid went inert.")
    if val >= 0.10:
        verdict = "STRONG_POSITIVE"
    elif val >= 0.020:
        verdict = "PARTIAL"
    else:
        verdict = "NEGATIVE"
    print(f"\n[d-ablation] VERDICT: {verdict}  (post-reset mean = {val:.5f} m)")
    diag["verdict"] = verdict
    args.out.write_text(json.dumps(diag, indent=2))
    print(f"[d-ablation] saved {args.out}")


if __name__ == "__main__":
    main()
