"""Sanity-check our TUM data pipeline by running off-the-shelf VGGT and logging
to Rerun. If VGGT's predicted poses and point cloud line up with TUM GT, our
RGB/intrinsics/pose handling is fine. If VGGT also fails on this data, the
problem is in TUM-side handling and we find it here before any rewrite.

What gets logged to Rerun:
  world (right-handed Z-up)
    camera_pred[i] — VGGT's predicted camera frustum + RGB image at frame i
    camera_gt[i]   — TUM ground-truth camera frustum at frame i (different color)
    points_pred[i] — VGGT's per-pixel point cloud at frame i (colored by RGB)
    points_gt[i]   — TUM GT depth back-projected to 3D using GT pose (colored by RGB)

Output:
  viz/output/vggt_tum_rerun/recording.rrd  — open with: rerun viz/output/vggt_tum_rerun/recording.rrd
  stdout: per-frame VGGT vs GT pose error + depth error
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/vggt")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import rerun as rr                                                                  # noqa: E402
from PIL import Image                                                                # noqa: E402

from vggt.models.vggt import VGGT                                                    # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images                            # noqa: E402
from vggt.utils.geometry import unproject_depth_map_to_point_map                     # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri                         # noqa: E402

from vggt_mamba.data.tum_rgbd import sync_sequence, intrinsics_for                   # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path,
                    default=Path("/workspace/datasets/weights/vggt/model.pt"))
    p.add_argument("--seq", default="rgbd_dataset_freiburg2_desk")
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--n-frames", type=int, default=24,
                    help="Number of frames to sample evenly through the sequence.")
    p.add_argument("--out-dir", type=Path,
                    default=Path("viz/output/vggt_tum_rerun"))
    p.add_argument("--vggt-conf-thresh", type=float, default=2.0,
                    help="Keep only points with VGGT depth-conf >= this (filter low-conf noise).")
    return p.parse_args()


def load_gt_depth(rec, depth_max_m: float = 8.0):
    """Load TUM GT depth — uint16 PNG / 5000 = meters, 0 = invalid."""
    d = np.asarray(Image.open(rec.depth_path), dtype=np.float32) / 5000.0           # (H, W)
    d = np.where((d > 0) & (d < depth_max_m), d, np.nan)
    return d


def rgb_for_rerun(rec, target_size: int):
    """Return float [0,1] RGB at original aspect, resized to target_size (square)."""
    img = Image.open(rec.rgb_path).convert("RGB").resize((target_size, target_size))
    return np.asarray(img, dtype=np.float32) / 255.0


def pose_error_m(R_pred: np.ndarray, t_pred: np.ndarray,
                  R_gt: np.ndarray, t_gt: np.ndarray) -> tuple[float, float]:
    """Translation L2 (meters) + rotation geodesic (radians)."""
    t_err = float(np.linalg.norm(t_pred - t_gt))
    # Rotation error: angle of R_pred R_gt^T
    R_err_mat = R_pred @ R_gt.T
    cos_theta = (np.trace(R_err_mat) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    r_err = float(np.arccos(cos_theta))
    return t_err, r_err


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # === Load TUM frame list, sample N evenly ===
    recs_all = sync_sequence(args.data_root / args.seq)
    idx = np.linspace(0, len(recs_all) - 1, args.n_frames).astype(int).tolist()
    recs = [recs_all[i] for i in idx]
    print(f"[vggt-tum] {args.seq}: {len(recs_all)} total frames; sampled {len(recs)} for VGGT")

    # === Get GT poses for the sampled frames, expressed relative to first sampled frame ===
    gt_poses_abs = np.stack([r.pose_w_c for r in recs])                              # (N, 4, 4) world-from-cam, absolute
    P0_inv = np.linalg.inv(gt_poses_abs[0])
    gt_poses = np.einsum("ij,njk->nik", P0_inv, gt_poses_abs)                        # (N, 4, 4) relative to frame 0

    # === Load TUM RGB at VGGT's expected input size (the VGGT loader handles resize) ===
    # We'll save the original-aspect RGBs separately for Rerun visualization.
    rgb_paths = [str(r.rgb_path) for r in recs]
    # VGGT preprocessing: stacks (N, 3, H, W), the loader resizes to its preferred resolution.
    images_t = load_and_preprocess_images(rgb_paths).cuda()                          # (N, 3, H, W)
    print(f"[vggt-tum] preprocessed images shape: {tuple(images_t.shape)}")

    # === Load model from local ckpt ===
    print(f"[vggt-tum] loading VGGT from {args.ckpt}")
    model = VGGT()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval().cuda()

    # === Forward pass ===
    print(f"[vggt-tum] running VGGT forward pass...")
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            preds = model(images_t.unsqueeze(0))                                     # batched
    # Predictions:
    #   preds["pose_enc"]: (1, N, 9)  encoded poses
    #   preds["depth"]: (1, N, H, W, 1)
    #   preds["depth_conf"]: (1, N, H, W)
    #   preds["world_points"]: (1, N, H, W, 3)
    #   preds["world_points_conf"]: (1, N, H, W)
    print(f"[vggt-tum] pred keys: {list(preds.keys())}")
    pose_enc = preds["pose_enc"][0].float().cpu()                                    # (N, 9)
    img_h, img_w = images_t.shape[-2:]
    extr, intr = pose_encoding_to_extri_intri(pose_enc.unsqueeze(0),
                                                (img_h, img_w))                       # (1, N, 3, 4), (1, N, 3, 3)
    extr_np = extr[0].numpy()                                                        # (N, 3, 4)  cam-from-world
    intr_np = intr[0].numpy()                                                        # (N, 3, 3)
    depth_np = preds["depth"][0, :, :, :, 0].float().cpu().numpy()                   # (N, H, W)
    depth_conf_np = preds["depth_conf"][0].float().cpu().numpy()                     # (N, H, W)
    world_points_np = preds["world_points"][0].float().cpu().numpy()                 # (N, H, W, 3)

    # VGGT extrinsics are 3×4 (cam-from-world). Build 4×4 world-from-cam for logging.
    pred_pose_wc = np.zeros((len(recs), 4, 4))
    pred_pose_wc[:, 3, 3] = 1.0
    for i in range(len(recs)):
        # cam-from-world → world-from-cam = inverse
        R_cw = extr_np[i, :3, :3]
        t_cw = extr_np[i, :3, 3]
        R_wc = R_cw.T
        t_wc = -R_wc @ t_cw
        pred_pose_wc[i, :3, :3] = R_wc
        pred_pose_wc[i, :3, 3] = t_wc
    # Express relative to first frame, matching how we expressed GT poses.
    P0_pred_inv = np.linalg.inv(pred_pose_wc[0])
    pred_poses = np.einsum("ij,njk->nik", P0_pred_inv, pred_pose_wc)

    # === Pose error per frame ===
    print(f"\n[vggt-tum] === POSE ERROR (relative to first sampled frame) ===")
    print(f"  {'i':>3}  {'frame':>5}  {'|t|_pred':>10}  {'|t|_gt':>10}  "
           f"{'t_err':>10}  {'r_err(deg)':>10}")
    pose_errs_t = []
    pose_errs_r = []
    for i, fi in enumerate(idx):
        t_p = pred_poses[i, :3, 3]
        t_g = gt_poses[i, :3, 3]
        R_p = pred_poses[i, :3, :3]
        R_g = gt_poses[i, :3, :3]
        t_err, r_err = pose_error_m(R_p, t_p, R_g, t_g)
        pose_errs_t.append(t_err)
        pose_errs_r.append(r_err)
        print(f"  {i:>3}  {fi:>5}  {np.linalg.norm(t_p):>10.3f}  {np.linalg.norm(t_g):>10.3f}  "
               f"{t_err:>10.3f}  {np.degrees(r_err):>10.2f}")
    print(f"  Mean translation error: {float(np.mean(pose_errs_t)):.3f} m")
    print(f"  Mean rotation error:    {np.degrees(np.mean(pose_errs_r)):.2f} deg")

    # === Depth error per frame (where both GT and VGGT-confident pixels exist) ===
    print(f"\n[vggt-tum] === DEPTH ERROR ===")
    print(f"  {'i':>3}  {'abs_rel':>10}  {'mean_err_m':>12}  {'n_valid':>10}")
    abs_rels = []
    for i, rec in enumerate(recs):
        # Load GT depth at VGGT's resolution
        gt_d = np.asarray(Image.open(rec.depth_path).resize((img_w, img_h), Image.NEAREST),
                           dtype=np.float32) / 5000.0
        gt_valid = (gt_d > 0.1) & (gt_d < 8.0)
        vggt_conf_mask = depth_conf_np[i] > args.vggt_conf_thresh
        mask = gt_valid & vggt_conf_mask
        if mask.sum() < 100:
            print(f"  {i:>3}  {'-':>10}  {'-':>12}  {int(mask.sum()):>10}  (too few valid)")
            continue
        pred_d = depth_np[i][mask]
        gt_d_m = gt_d[mask]
        abs_rel = float(np.mean(np.abs(pred_d - gt_d_m) / np.clip(gt_d_m, 0.1, None)))
        mean_err = float(np.mean(np.abs(pred_d - gt_d_m)))
        abs_rels.append(abs_rel)
        print(f"  {i:>3}  {abs_rel:>10.3f}  {mean_err:>12.3f}  {int(mask.sum()):>10}")
    if abs_rels:
        print(f"  Mean abs_rel: {float(np.mean(abs_rels)):.3f}")

    # === Log to Rerun ===
    print(f"\n[vggt-tum] writing Rerun recording to {args.out_dir}/recording.rrd ...")
    rr.init("vggt_tum_data_check", spawn=False)
    rr.save(str(args.out_dir / "recording.rrd"))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)                # TUM convention: +y down, +z forward

    for i, rec in enumerate(recs):
        rr.set_time_sequence("frame", i)
        rgb_for_log = images_t[i].permute(1, 2, 0).float().cpu().numpy()             # (H, W, 3) in [0,1]
        # convert to uint8 for log
        rgb_u8 = (rgb_for_log * 255).clip(0, 255).astype(np.uint8)

        # ----- VGGT predicted camera -----
        P_pred = pred_poses[i]                                                       # 4x4 world-from-cam, relative to first frame
        rr.log("world/cam_pred",
                rr.Transform3D(translation=P_pred[:3, 3], mat3x3=P_pred[:3, :3]))
        rr.log("world/cam_pred/image",
                rr.Pinhole(focal_length=[intr_np[i, 0, 0], intr_np[i, 1, 1]],
                            principal_point=[intr_np[i, 0, 2], intr_np[i, 1, 2]],
                            width=img_w, height=img_h, image_plane_distance=0.3))
        rr.log("world/cam_pred/image", rr.Image(rgb_u8))

        # ----- GT camera (TUM) -----
        P_gt = gt_poses[i]
        rr.log("world/cam_gt",
                rr.Transform3D(translation=P_gt[:3, 3], mat3x3=P_gt[:3, :3]))
        # TUM intrinsics scaled to VGGT's input resolution for plausible frustum drawing
        fx, fy, cx, cy = intrinsics_for(args.seq)
        sx, sy = img_w / 640.0, img_h / 480.0
        rr.log("world/cam_gt/image",
                rr.Pinhole(focal_length=[fx * sx, fy * sy],
                            principal_point=[cx * sx, cy * sy],
                            width=img_w, height=img_h, image_plane_distance=0.2))

        # ----- Predicted point cloud (VGGT depth back-projected) -----
        pts_pred = world_points_np[i]                                                # (H, W, 3) in VGGT's coords (relative to first cam by VGGT convention)
        conf_mask = depth_conf_np[i] > args.vggt_conf_thresh
        if conf_mask.sum() > 0:
            pts = pts_pred[conf_mask]                                                # (M, 3)
            colors = rgb_u8[conf_mask]                                               # (M, 3)
            rr.log(f"world/points_pred", rr.Points3D(pts, colors=colors, radii=0.005))

        # ----- GT point cloud (TUM GT depth back-projected with GT pose) -----
        gt_d = np.asarray(Image.open(rec.depth_path).resize((img_w, img_h), Image.NEAREST),
                           dtype=np.float32) / 5000.0
        gt_valid = (gt_d > 0.1) & (gt_d < 8.0)
        if gt_valid.sum() > 0:
            us, vs = np.meshgrid(np.arange(img_w), np.arange(img_h))
            uu = us[gt_valid].astype(np.float32)
            vv = vs[gt_valid].astype(np.float32)
            dd = gt_d[gt_valid]
            # backproject in GT camera frame
            K_inv = np.linalg.inv(np.array([[fx*sx, 0, cx*sx], [0, fy*sy, cy*sy], [0, 0, 1.0]]))
            pix_h = np.stack([uu, vv, np.ones_like(uu)], axis=-1)                    # (M, 3)
            cam_dir = pix_h @ K_inv.T                                                # (M, 3)  z=1
            P_cam = cam_dir * dd[:, None]                                            # (M, 3)
            R = P_gt[:3, :3]; t = P_gt[:3, 3]
            P_world = P_cam @ R.T + t                                                # (M, 3)
            colors = rgb_u8[gt_valid]
            rr.log(f"world/points_gt", rr.Points3D(P_world, colors=colors, radii=0.005))

    print(f"[vggt-tum] DONE — open the recording with:")
    print(f"    rerun {args.out_dir / 'recording.rrd'}")
    print(f"[vggt-tum] (or copy the file to host machine if running viewer there)")


if __name__ == "__main__":
    main()
