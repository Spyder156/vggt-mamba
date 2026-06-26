"""Held-out evaluation for TerraWM-Linear on a TUM sequence.

Loads a checkpoint (or an untrained model with --no-ckpt for smoke), picks
N evenly-sampled frames from a TUM sequence, runs forward, and reports:

    - ATE (Sim(3)-aligned)  ← primary metric for Path A
    - ATE (SE(3)-aligned)   ← would be primary if we had metric output
    - RPE@1 (translation + rotation)
    - depth abs_rel after single global scale fit to GT
    - depth abs_rel under Sim(3) scale (sanity)

This is THE script every checkpoint should run through. Numbers it prints are
what we'll graph in the paper.

Usage:
    python -m scripts.eval.terrawm_linear_held_out \
        --ckpt path/to/ckpt.pt --seq rgbd_dataset_freiburg3_walking_xyz --n-frames 16

    # Or as untrained-model smoke (random init, no ckpt):
    python -m scripts.eval.terrawm_linear_held_out --no-ckpt --n-frames 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.data.tum_rgbd import sync_sequence, intrinsics_for, TRAIN_SEQS, EVAL_SEQS
from vggt_mamba.eval.metrics import (
    absolute_translation_error,
    depth_abs_rel,
    relative_pose_error,
    umeyama_sim3,
)
from vggt_mamba.eval.pose_enc import pose_enc_to_intrinsics, pose_enc_to_world_from_cam
from vggt_mamba.models.terrawm_linear import TerraWMConfig, TerraWMLinear


DATA_ROOT = Path("/workspace/datasets/tum_rgbd")


def load_frames(seq_name: str, n_frames: int, img_size: int) -> dict:
    """Pick n_frames evenly-sampled frames from a TUM sequence, preprocess
    to (n, 3, img_size, img_size) [0,1] tensor with TUM GT depth+poses+K kept
    at full image_size (resize-matched)."""
    recs_all = sync_sequence(DATA_ROOT / seq_name)
    if len(recs_all) < n_frames:
        raise SystemExit(f"sequence {seq_name} has only {len(recs_all)} frames")
    idx = np.linspace(0, len(recs_all) - 1, n_frames).astype(int).tolist()
    recs = [recs_all[i] for i in idx]

    rgb_list, depth_list, pose_list = [], [], []
    for r in recs:
        rgb = Image.open(r.rgb_path).convert("RGB").resize((img_size, img_size), Image.BICUBIC)
        rgb_list.append(np.asarray(rgb, dtype=np.float32) / 255.0)
        d = np.asarray(
            Image.open(r.depth_path).resize((img_size, img_size), Image.NEAREST),
            dtype=np.float32,
        ) / 5000.0
        depth_list.append(d)
        pose_list.append(r.pose_w_c.astype(np.float32))
    rgb = np.stack(rgb_list)                                                       # (T, H, W, 3)
    depth = np.stack(depth_list)                                                    # (T, H, W)
    pose_w_c = np.stack(pose_list)                                                  # (T, 4, 4)

    fx, fy, cx, cy = intrinsics_for(seq_name)
    sx = img_size / 640.0
    sy = img_size / 480.0
    K = np.array(
        [[fx*sx, 0, cx*sx], [0, fy*sy, cy*sy], [0, 0, 1]],
        dtype=np.float32,
    )

    valid = (depth > 0.01) & (depth < 8.0)
    return {
        "rgb": torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous(),               # (T, 3, H, W)
        "depth": torch.from_numpy(depth),                                            # (T, H, W)
        "valid": torch.from_numpy(valid),                                            # (T, H, W) bool
        "pose_w_c": torch.from_numpy(pose_w_c),                                      # (T, 4, 4)
        "K": torch.from_numpy(K),                                                    # (3, 3)
        "frame_indices": idx,
        "seq_name": seq_name,
    }


def fit_global_scale(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """Single scalar s minimizing || s*pred - gt ||^2 over valid pixels."""
    p = pred[valid]
    g = gt[valid]
    return float((p @ g) / max((p @ p), 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--no-ckpt", action="store_true",
                    help="Run an untrained model (smoke test of the pipeline).")
    p.add_argument("--seq", type=str, default="rgbd_dataset_freiburg2_desk")
    p.add_argument("--n-frames", type=int, default=16)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-latents", type=int, default=512)
    p.add_argument("--n-write-blocks", type=int, default=4)
    p.add_argument("--n-decode-blocks", type=int, default=2)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"seq: {args.seq}, n_frames: {args.n_frames}, img_size: {args.img_size}")
    if args.seq in TRAIN_SEQS:
        print(f"  WARNING: {args.seq} is a TRAIN seq. Held-out should be one of: {EVAL_SEQS}")

    cfg = TerraWMConfig(
        img_size=args.img_size,
        d_model=args.d_model,
        n_latents=args.n_latents,
        n_write_blocks=args.n_write_blocks,
        n_decode_blocks=args.n_decode_blocks,
    )
    model = TerraWMLinear(cfg).to(device).eval()
    if args.ckpt is not None and not args.no_ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        print(f"loaded ckpt: {args.ckpt}")
    else:
        print(f"NO CKPT — untrained model (smoke mode). Errors will be large.")

    batch = load_frames(args.seq, args.n_frames, args.img_size)
    rgb = batch["rgb"].unsqueeze(0).to(device)                                        # (1, T, 3, H, W)

    with torch.no_grad():
        out = model(rgb)
    cameras = out["cameras"][0].float().cpu()                                         # (T, 9)
    depths = out["depths"][0].float().cpu()                                           # (T, H, W)

    # Decode predicted: pose_enc → pose_w_c (world-from-cam)
    pred_pose_w_c = pose_enc_to_world_from_cam(cameras).numpy()                       # (T, 4, 4)
    gt_pose_w_c = batch["pose_w_c"].numpy()                                           # (T, 4, 4)

    # ATE (Sim(3) — Path A primary)
    ate_sim3 = absolute_translation_error(pred_pose_w_c, gt_pose_w_c, align="sim3")
    ate_se3 = absolute_translation_error(pred_pose_w_c, gt_pose_w_c, align="se3")
    rpe = relative_pose_error(pred_pose_w_c, gt_pose_w_c, delta=1)

    # Depth: fit one global scale to GT, then abs_rel
    pred_d_np = depths.numpy()                                                        # (T, H, W)
    gt_d_np = batch["depth"].numpy()                                                  # (T, H, W)
    valid_np = batch["valid"].numpy()                                                 # (T, H, W) bool
    s_depth = fit_global_scale(pred_d_np, gt_d_np, valid_np)
    abs_rel_scaled = depth_abs_rel(pred_d_np * s_depth, gt_d_np, valid_np)
    abs_rel_raw = depth_abs_rel(pred_d_np, gt_d_np, valid_np)
    # Also Sim(3) scale for comparison
    s_sim3 = ate_sim3.get("align_scale", None)                                        # we'll compute outside if needed

    print(f"\n=== TerraWM-Linear held-out eval on {args.seq} ===")
    print(f"frames: {args.n_frames} from {batch['frame_indices'][0]} to {batch['frame_indices'][-1]}")
    print()
    print(f"  POSE (Path A primary = Sim(3)):")
    print(f"    ATE Sim(3) rmse: {ate_sim3['ate_rmse_m']:.4f} m   "
            f"mean: {ate_sim3['ate_mean_m']:.4f} m   median: {ate_sim3['ate_median_m']:.4f} m")
    print(f"    ATE SE(3) rmse:  {ate_se3['ate_rmse_m']:.4f} m   "
            f"(would be primary if metric)")
    print(f"    RPE@1 trans rmse: {rpe['rpe_trans_rmse_m']:.4f} m")
    print(f"    RPE@1 rot rmse:   {rpe['rpe_rot_rmse_deg']:.4f} deg")
    print()
    print(f"  DEPTH:")
    print(f"    abs_rel raw:           {abs_rel_raw:.4f}   (no alignment)")
    print(f"    abs_rel after scale s: {abs_rel_scaled:.4f}   (s = {s_depth:.4f})")
    print()
    print(f"  EXPECTED FOR UNTRAINED MODEL: huge numbers, this is just plumbing check.")
    print(f"  Once we start training, ATE Sim(3) is the headline metric.")


if __name__ == "__main__":
    main()
