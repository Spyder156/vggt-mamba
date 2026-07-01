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
    p.add_argument("--rrd-path", type=str, default=None,
                    help="If set, also write a Rerun .rrd with predicted+GT cameras and "
                         "point clouds (Sim(3)-aligned predicted on top of TUM-metric GT).")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"seq: {args.seq}, n_frames: {args.n_frames}, img_size: {args.img_size}")
    if args.seq in TRAIN_SEQS:
        print(f"  WARNING: {args.seq} is a TRAIN seq. Held-out should be one of: {EVAL_SEQS}")

    # If a ckpt is provided AND it stored the training cfg, use that — otherwise
    # fall back to CLI flags. This guarantees model dims match the ckpt's
    # shapes (frame_embed, latents, etc.).
    ckpt_state = None
    ckpt_cfg = None
    if args.ckpt is not None and not args.no_ckpt:
        ckpt_state = torch.load(args.ckpt, map_location="cpu")
        ckpt_cfg = ckpt_state.get("cfg") if isinstance(ckpt_state, dict) else None

    if ckpt_cfg is not None and "model" in ckpt_cfg:
        m = ckpt_cfg["model"]
        cfg = TerraWMConfig(
            img_size=m["img_size"],
            d_enc=m.get("d_enc", 1024),
            d_model=m["d_model"],
            n_heads=m.get("n_heads", 12),
            n_latents=m["n_latents"],
            n_write_blocks=m["n_write_blocks"],
            n_decode_blocks=m["n_decode_blocks"],
            encoder_repo=m.get("encoder_repo", "facebook/dinov3-vitl16-pretrain-lvd1689m"),
            freeze_encoder=m.get("freeze_encoder", True),
            max_frames=m.get("max_frames", 64),
            pose_supervision_mode=m.get("pose_supervision_mode", "predicted"),
        )
        print(f"using model cfg from ckpt (pose_mode={cfg.pose_supervision_mode}, "
                f"max_frames={cfg.max_frames}, d_model={cfg.d_model})")
    else:
        cfg = TerraWMConfig(
            img_size=args.img_size,
            d_model=args.d_model,
            n_latents=args.n_latents,
            n_write_blocks=args.n_write_blocks,
            n_decode_blocks=args.n_decode_blocks,
        )

    model = TerraWMLinear(cfg).to(device).eval()
    if ckpt_state is not None:
        state = ckpt_state["model"] if "model" in ckpt_state else ckpt_state
        model.load_state_dict(state, strict=False)
        print(f"loaded ckpt: {args.ckpt}")
    else:
        print(f"NO CKPT — untrained model (smoke mode). Errors will be large.")

    batch = load_frames(args.seq, args.n_frames, args.img_size)
    rgb = batch["rgb"].unsqueeze(0).to(device)                                        # (1, T, 3, H, W)
    pose_w_c_b = batch["pose_w_c"].unsqueeze(0).to(device)                            # (1, T, 4, 4)
    depth_b = batch["depth"].unsqueeze(0).to(device)                                  # (1, T, H, W)
    valid_b = batch["valid"].unsqueeze(0).to(device)
    K_b = batch["K"].unsqueeze(0).to(device)                                          # (1, 3, 3)

    # Cheat-pose mode: feed normalized GT pose enc into the model.
    pose_mode = getattr(cfg, "pose_supervision_mode", "predicted")
    gt_pose_enc_in = None
    if pose_mode == "gt_replace":
        from vggt_mamba.eval.normalize import normalize_scene_by_mean_distance
        from vggt_mamba.eval.pose_enc import (
            fov_from_intrinsics as _fov_fn,
            world_from_cam_to_pose_enc as _to_pose_enc,
        )
        pose_normed, _, _ = normalize_scene_by_mean_distance(
            pose_w_c_b.float(), depth_b.float(), valid_b, K_b.float()
        )
        T = pose_normed.shape[1]
        fh, fw = _fov_fn(K_b, (args.img_size, args.img_size))
        gt_pose_enc_in = _to_pose_enc(
            pose_normed, fh[:, None].expand(1, T), fw[:, None].expand(1, T)
        )
        print(f"feeding GT pose enc into model (gt_replace mode)")

    with torch.no_grad():
        if gt_pose_enc_in is not None:
            out = model(rgb, gt_pose_enc=gt_pose_enc_in)
        else:
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

    # === Optional Rerun visualization ===
    if args.rrd_path is not None:
        # In gt_replace mode the pose head got zero gradient — its predicted
        # cameras are meaningless. What we ACTUALLY want to look at is:
        # "given correct poses (which we fed the model), what depth did it
        # produce?" So we use GT poses for both cam viz AND for back-projecting
        # the predicted depth. In predicted mode we keep pred poses.
        if pose_mode == "gt_replace":
            pred_pose_for_viz = gt_pose_w_c
            pred_K_for_viz = batch["K"].numpy()[None].repeat(gt_pose_w_c.shape[0], axis=0)
        else:
            pred_pose_for_viz = pred_pose_w_c
            pred_K_for_viz = pose_enc_to_K_per_frame(cameras, args.img_size)
        write_rerun_rrd(
            args.rrd_path,
            seq=args.seq,
            recs_idx=batch["frame_indices"],
            images=batch["rgb"].numpy(),                                              # (T, 3, H, W) in [0,1]
            pred_pose_w_c=pred_pose_for_viz,                                          # (T, 4, 4)
            gt_pose_w_c=gt_pose_w_c,
            pred_depth=pred_d_np,                                                     # (T, H, W) normalized
            gt_depth=gt_d_np,                                                         # (T, H, W) metric
            valid=valid_np,
            pred_intrinsics=pred_K_for_viz,
            gt_K=batch["K"].numpy(),
            img_size=args.img_size,
            skip_sim3_align=(pose_mode == "gt_replace"),
        )
        print(f"  RRD: wrote {args.rrd_path}")


# ============================================================================
# Rerun visualization
# ============================================================================

def pose_enc_to_K_per_frame(cameras: torch.Tensor, img_size: int) -> np.ndarray:
    """Decode predicted (T, 9) cam-from-world pose enc into per-frame 3x3 K."""
    from vggt_mamba.eval.pose_enc import pose_enc_to_intrinsics
    K = pose_enc_to_intrinsics(cameras, (img_size, img_size)).numpy()                  # (T, 3, 3)
    return K


def umeyama_sim3_np(P: np.ndarray, Q: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Reuse the metrics.py implementation."""
    from vggt_mamba.eval.metrics import umeyama_sim3
    return umeyama_sim3(P, Q)


def backproject_depth_to_world(
    depth: np.ndarray, valid: np.ndarray, K: np.ndarray, pose_w_c: np.ndarray
) -> np.ndarray:
    """Single-frame back-projection: depth + K + cam-to-world -> (M, 3) world points."""
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    mask = valid & (depth > 1e-3)
    uu = us[mask].astype(np.float32)
    vv = vs[mask].astype(np.float32)
    dd = depth[mask]
    x_cam = (uu - cx) * dd / fx
    y_cam = (vv - cy) * dd / fy
    z_cam = dd
    P_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    R = pose_w_c[:3, :3]
    t = pose_w_c[:3, 3]
    return P_cam @ R.T + t


def write_rerun_rrd(
    rrd_path: str,
    *,
    seq: str,
    recs_idx: list[int],
    images: np.ndarray,                                                                # (T, 3, H, W) [0,1]
    pred_pose_w_c: np.ndarray,                                                         # (T, 4, 4) in MODEL (normalized) scale
    gt_pose_w_c: np.ndarray,                                                           # (T, 4, 4) in TUM metric
    pred_depth: np.ndarray,                                                            # (T, H, W) MODEL scale
    gt_depth: np.ndarray,                                                              # (T, H, W) TUM metric
    valid: np.ndarray,                                                                 # (T, H, W) bool
    pred_intrinsics: np.ndarray,                                                       # (T, 3, 3) — VGGT's predicted K (center PP)
    gt_K: np.ndarray,                                                                  # (3, 3) — TUM K
    img_size: int,
    skip_sim3_align: bool = False,                                                     # cheatpose mode: pred cams ARE GT already
) -> None:
    """Write a Rerun .rrd with:
        world/points_pred_aligned : predicted cloud, aligned to TUM metric (one cloud)
        world/points_gt           : GT cloud in TUM metric (one cloud)
        world/cam_pred_aligned[i] : predicted cameras (Sim(3) or GT-as-fed), animated
        world/cam_gt[i]           : GT cameras, animated

    skip_sim3_align: if True (cheatpose mode), the pred_pose_w_c IS the GT pose
        that was fed into the model, so no Umeyama needed. We instead fit a single
        scalar between pred_depth and gt_depth to bring pred depth into metric.
    """
    import rerun as rr
    from pathlib import Path

    Path(rrd_path).parent.mkdir(parents=True, exist_ok=True)
    rr.init("terrawm_linear_held_out", spawn=False)
    rr.save(rrd_path)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    T = pred_pose_w_c.shape[0]

    if skip_sim3_align:
        # cheatpose: cams stay as fed. scale predicted depth to metric via a
        # single scalar minimising ||s*pred - gt||^2 over valid pixels.
        p_flat = pred_depth[valid]
        g_flat = gt_depth[valid]
        s = float((p_flat @ g_flat) / max((p_flat @ p_flat), 1e-12))
        pred_pose_aligned = pred_pose_w_c.copy()
        print(f"  [rrd cheatpose] depth-fit scale s = {s:.4f}, cams = GT (no Umeyama)")
    else:
        # predicted mode: Sim(3)-align pred cams to GT cams, use same s for depth.
        pred_centers = pred_pose_w_c[:, :3, 3]
        gt_centers = gt_pose_w_c[:, :3, 3]
        s, R_align, t_align = umeyama_sim3_np(pred_centers, gt_centers)
        pred_pose_aligned = pred_pose_w_c.copy()
        for i in range(T):
            R_pred = pred_pose_w_c[i, :3, :3]
            t_pred = pred_pose_w_c[i, :3, 3]
            pred_pose_aligned[i, :3, :3] = R_align @ R_pred
            pred_pose_aligned[i, :3, 3] = s * (R_align @ t_pred) + t_align
        print(f"  [rrd predicted] Sim(3) scale s = {s:.4f}")

    # --- Build global point clouds (merged across all frames) ---
    pred_pts_all, pred_cols_all = [], []
    gt_pts_all, gt_cols_all = [], []
    for i in range(T):
        rgb_u8 = (images[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)     # (H, W, 3)

        # predicted: backproject pred_depth * s through pred K with pred_pose_aligned
        K_pred = pred_intrinsics[i]
        pred_pts = backproject_depth_to_world(
            pred_depth[i] * s, valid[i], K_pred, pred_pose_aligned[i]
        )
        valid_mask = valid[i] & (pred_depth[i] > 1e-3)
        pred_cols = rgb_u8[valid_mask]
        pred_pts_all.append(pred_pts)
        pred_cols_all.append(pred_cols)

        # GT: backproject gt_depth through gt_K with gt_pose
        gt_pts = backproject_depth_to_world(gt_depth[i], valid[i], gt_K, gt_pose_w_c[i])
        gt_valid_mask = valid[i] & (gt_depth[i] > 1e-3)
        gt_cols = rgb_u8[gt_valid_mask]
        gt_pts_all.append(gt_pts)
        gt_cols_all.append(gt_cols)

    # Concatenate + cap at 2M points each for viewer perf
    def cap(pts_list, cols_list, max_n=2_000_000):
        pts = np.concatenate(pts_list)
        cols = np.concatenate(cols_list)
        if len(pts) > max_n:
            idx = np.random.choice(len(pts), max_n, replace=False)
            pts, cols = pts[idx], cols[idx]
        return pts, cols

    pred_pts_g, pred_cols_g = cap(pred_pts_all, pred_cols_all)
    gt_pts_g, gt_cols_g = cap(gt_pts_all, gt_cols_all)

    rr.log("world/points_pred_aligned",
            rr.Points3D(pred_pts_g, colors=pred_cols_g, radii=0.003), static=True)
    rr.log("world/points_gt",
            rr.Points3D(gt_pts_g, colors=gt_cols_g, radii=0.003), static=True)

    # --- Per-frame cameras on the timeline ---
    for i in range(T):
        rr.set_time_sequence("frame", i)
        rgb_u8 = (images[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)

        # Predicted aligned camera
        P = pred_pose_aligned[i]
        rr.log("world/cam_pred_aligned",
                rr.Transform3D(translation=P[:3, 3], mat3x3=P[:3, :3]))
        rr.log("world/cam_pred_aligned/image",
                rr.Pinhole(image_from_camera=pred_intrinsics[i],
                           width=img_size, height=img_size, image_plane_distance=0.1))
        rr.log("world/cam_pred_aligned/image", rr.Image(rgb_u8))

        # GT camera
        P = gt_pose_w_c[i]
        rr.log("world/cam_gt",
                rr.Transform3D(translation=P[:3, 3], mat3x3=P[:3, :3]))
        rr.log("world/cam_gt/image",
                rr.Pinhole(image_from_camera=gt_K,
                           width=img_size, height=img_size, image_plane_distance=0.1))
        rr.log("world/cam_gt/image", rr.Image(rgb_u8))


if __name__ == "__main__":
    main()
