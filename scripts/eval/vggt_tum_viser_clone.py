"""Clone of VGGT's demo_viser.py pipeline, but for TUM + Rerun.

This script mirrors the inference path of VGGT's official demo_viser.py
EXACTLY, using only VGGT's own helpers — no hand-rolled extrinsic inversion,
no hand-rolled back-projection, no custom alignment.

Pipeline (verbatim from demo_viser.py + README quick-start):
    images = load_and_preprocess_images(rgb_paths).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"],
                                                        images.shape[-2:])
    world_points = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    cam_to_world = closed_form_inverse_se3(extrinsic)
    # then demo_viser recenters at scene mean and visualises

GT comparison: we ALSO log TUM's GT poses and back-projected GT depth in the
same world frame (after the same scene-centering transform), so the viewer
can see VGGT's reconstruction vs TUM ground truth side by side WITHOUT any
custom alignment hack.

Notes:
    - Output to viz/output/vggt_tum_viser_clone/recording.rrd
    - N_FRAMES default 16 to fit GPU memory (24 OOMs on the 15GB card)
    - VGGT's predictions are scale-normalized (see training-time
      normalize_camera_extrinsics_and_points_batch), so the predicted scene
      will be at a smaller scale than TUM metric — this is by design.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import rerun as rr
from PIL import Image

sys.path.insert(0, "/workspace/vggt")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt.models.vggt import VGGT                                                    # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images                            # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri                         # noqa: E402
from vggt.utils.geometry import (                                                    # noqa: E402
    closed_form_inverse_se3,
    unproject_depth_map_to_point_map,
)

from vggt_mamba.data.tum_rgbd import sync_sequence, intrinsics_for                   # noqa: E402

SEQ = "rgbd_dataset_freiburg2_desk"
DATA_ROOT = Path("/workspace/datasets/tum_rgbd")
N_FRAMES = 16
CONF_PCTILE = 25.0                                                                   # demo_viser default
CKPT_PATH = Path("/workspace/datasets/weights/vggt/model.pt")
OUT_DIR = Path("/workspace/vggt-mamba/viz/output/vggt_tum_viser_clone")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"device={device}, dtype={dtype}")

    # === Pick N evenly-sampled fr2/desk frames ===
    recs_all = sync_sequence(DATA_ROOT / SEQ)
    idx = np.linspace(0, len(recs_all) - 1, N_FRAMES).astype(int).tolist()
    recs = [recs_all[i] for i in idx]
    rgb_paths = [str(r.rgb_path) for r in recs]
    print(f"seq={SEQ}, total={len(recs_all)}, sampled={len(recs)}")

    # === Load model — same way as README + demo_viser ===
    print(f"loading VGGT from {CKPT_PATH} ...")
    model = VGGT()
    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval().to(device)

    # === Preprocess — verbatim load_and_preprocess_images, crop mode ===
    images = load_and_preprocess_images(rgb_paths).to(device)                         # (S, 3, H, W)
    print(f"images shape: {tuple(images.shape)}")

    # === Forward pass — verbatim README quick-start pattern (no unsqueeze) ===
    print("running VGGT...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    print(f"output keys: {list(predictions.keys())}")

    # === Get extrinsics + intrinsics via VGGT's helper ===
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )                                                                                # extrinsic (1, S, 3, 4), intrinsic (1, S, 3, 3)
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # === Numpy + squeeze batch dim (demo_viser pattern) ===
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)             # drop batch

    images_np = predictions["images"]                                                # (S, 3, H, W)
    depth_map = predictions["depth"]                                                 # (S, H, W, 1)
    depth_conf = predictions["depth_conf"]                                           # (S, H, W)
    extrinsics_np = predictions["extrinsic"]                                         # (S, 3, 4)
    intrinsics_np = predictions["intrinsic"]                                         # (S, 3, 3)

    # === Build VGGT predicted point cloud — via VGGT's own unproject helper ===
    world_points = unproject_depth_map_to_point_map(
        depth_map, extrinsics_np, intrinsics_np
    )                                                                                # (S, H, W, 3)
    colors_per_frame = images_np.transpose(0, 2, 3, 1)                               # (S, H, W, 3) in [0,1]
    S, H, W, _ = world_points.shape

    points = world_points.reshape(-1, 3)                                             # (S*H*W, 3)
    colors_flat = (colors_per_frame.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = depth_conf.reshape(-1)

    # === Get cam-to-world for each predicted camera (demo_viser line 102-104) ===
    cam_to_world_mat = closed_form_inverse_se3(extrinsics_np)                         # (S, 4, 4)
    cam_to_world_pred = cam_to_world_mat[:, :3, :]                                    # (S, 3, 4)

    # === Recenter scene at mean of predicted points (demo_viser line 106-109) ===
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world_pred_centered = cam_to_world_pred.copy()
    cam_to_world_pred_centered[..., -1] -= scene_center
    print(f"scene_center (pred frame): {scene_center}")

    # === Confidence filter (demo_viser default 25%) ===
    init_threshold_val = np.percentile(conf_flat, CONF_PCTILE)
    conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    pts_pred_kept = points_centered[conf_mask]
    cols_pred_kept = colors_flat[conf_mask]
    print(f"predicted points kept after conf {CONF_PCTILE}% filter: {len(pts_pred_kept):,}")

    # === Build TUM GT in the SAME normalized + scene-centered frame ===
    # First put GT in cam-0's frame (same convention VGGT uses internally for
    # output: world frame == cam 0 of the input sequence).
    gt_abs = np.stack([r.pose_w_c for r in recs])                                    # (S, 4, 4)
    P0_inv = np.linalg.inv(gt_abs[0])
    gt_rel = np.einsum("ij,njk->nik", P0_inv, gt_abs)                                # (S, 4, 4), cam 0 = I

    # Back-project TUM GT depth into 3D using TUM intrinsics + GT poses
    fx, fy, cx, cy = intrinsics_for(SEQ)
    img_h, img_w = images.shape[-2:]
    sx, sy = img_w / 640.0, img_h / 480.0
    K_gt = np.array([[fx*sx, 0, cx*sx], [0, fy*sy, cy*sy], [0, 0, 1.0]])
    K_inv_gt = np.linalg.inv(K_gt)

    pts_gt_all, cols_gt_all = [], []
    for i, rec in enumerate(recs):
        gt_d = np.asarray(
            Image.open(rec.depth_path).resize((img_w, img_h), Image.NEAREST),
            dtype=np.float32,
        ) / 5000.0                                                                    # meters
        valid = (gt_d > 0.1) & (gt_d < 8.0)
        if valid.sum() == 0:
            continue
        us, vs = np.meshgrid(np.arange(img_w), np.arange(img_h))
        uu = us[valid].astype(np.float32)
        vv = vs[valid].astype(np.float32)
        dd = gt_d[valid]
        pix_h = np.stack([uu, vv, np.ones_like(uu)], axis=-1)
        cam_dir = pix_h @ K_inv_gt.T
        P_cam = cam_dir * dd[:, None]                                                # (M, 3) in cam i
        P_world = P_cam @ gt_rel[i, :3, :3].T + gt_rel[i, :3, 3]                     # (M, 3) in cam 0 frame, TUM metric
        rgb = (colors_per_frame[i] * 255).astype(np.uint8)
        col = rgb[valid]
        pts_gt_all.append(P_world)
        cols_gt_all.append(col)

    pts_gt_metric = np.concatenate(pts_gt_all)
    cols_gt_metric = np.concatenate(cols_gt_all)

    # GT is in TUM METRIC scale; predicted is in VGGT NORMALIZED scale.
    # We log both as-is in the same cam-0-centered frame so the user can SEE
    # the scale gap and the trajectory shape. demo_viser only recentered the
    # predicted cloud — we'll do the same: subtract pred scene_center from GT
    # so the two clouds share an origin in the viewer.
    pts_gt_centered = pts_gt_metric - scene_center
    # GT poses (4x4 -> 3x4) in same centered frame
    gt_rel_centered = gt_rel.copy()
    gt_rel_centered[:, :3, 3] -= scene_center
    cam_to_world_gt = gt_rel_centered[:, :3, :]                                       # (S, 3, 4)

    # Subsample GT cloud to 2M points if too dense
    if len(pts_gt_centered) > 2_000_000:
        sub = np.random.choice(len(pts_gt_centered), 2_000_000, replace=False)
        pts_gt_centered = pts_gt_centered[sub]
        cols_gt_metric = cols_gt_metric[sub]
    print(f"GT points: {len(pts_gt_centered):,} (TUM metric scale)")

    # === Log to Rerun ===
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rrd_path = OUT_DIR / "recording.rrd"
    print(f"writing Rerun recording to {rrd_path} ...")
    rr.init("vggt_tum_viser_clone", spawn=False)
    rr.save(str(rrd_path))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # Predicted global cloud (VGGT scale, scene-centered)
    rr.log("world/points_pred", rr.Points3D(pts_pred_kept, colors=cols_pred_kept, radii=0.001),
            static=True)
    # GT global cloud (TUM metric, scene-centered)
    rr.log("world/points_gt", rr.Points3D(pts_gt_centered, colors=cols_gt_metric, radii=0.005),
            static=True)

    # Per-frame camera frustums (animated by timeline)
    for i, rec in enumerate(recs):
        rr.set_time_sequence("frame", i)
        # Predicted camera (VGGT scale, centered)
        P_pred = np.eye(4)
        P_pred[:3, :] = cam_to_world_pred_centered[i]
        rr.log("world/cam_pred",
                rr.Transform3D(translation=P_pred[:3, 3], mat3x3=P_pred[:3, :3]))
        rr.log("world/cam_pred/image",
                rr.Pinhole(image_from_camera=intrinsics_np[i],
                           width=W, height=H, image_plane_distance=0.05))
        rgb_log = (images_np[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        rr.log("world/cam_pred/image", rr.Image(rgb_log))

        # GT camera (TUM metric, centered)
        P_gt = np.eye(4)
        P_gt[:3, :] = cam_to_world_gt[i]
        rr.log("world/cam_gt",
                rr.Transform3D(translation=P_gt[:3, 3], mat3x3=P_gt[:3, :3]))
        rr.log("world/cam_gt/image",
                rr.Pinhole(image_from_camera=K_gt,
                           width=W, height=H, image_plane_distance=0.2))
        rr.log("world/cam_gt/image", rr.Image(rgb_log))

    print(f"DONE. open with: rerun {rrd_path}")
    print(f"  - world/points_pred  : VGGT's reconstruction, VGGT-normalized scale (~0.44× metric)")
    print(f"  - world/points_gt    : TUM ground truth, METRIC scale")
    print(f"  - world/cam_pred[*]  : VGGT-predicted camera trajectory (smaller scale)")
    print(f"  - world/cam_gt[*]    : TUM ground truth camera trajectory (metric scale)")
    print(f"  expected: same shape, different sizes — scale is by design of VGGT training")


if __name__ == "__main__":
    main()
