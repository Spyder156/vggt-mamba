"""Quick ColorHead-not-collapsed check: stream 30 frames of fr1/room with the
photometric ckpt, measure per-frame patch-RGB std (predicted) vs the same
quantity for the target. PASS: patch_rgb_pred std > 0.01 (the bypass-guard
threshold from the pre-registered gate)."""
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.data.tum_rgbd import sync_sequence, intrinsics_for
from vggt_mamba.models.aggregators.anchor_pool import cam9_to_pose_w_c
from vggt_mamba.models.terrawm_d import build_terrawm_d
from vggt_mamba.models.voxel_grid import (
    backproject_patches_to_world, build_rays_from_pose,
    render_rays_volumetric, write_voxels_trilinear,
)

ckpt_path = "experiments/phase4_terrawm_d_seed0_photometric/dinov3_terrawm_d/ckpt_008000.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg = ckpt["config"]
m = build_terrawm_d(
    cfg["encoder"], "/workspace/datasets/weights",
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
    use_photometric=True, photometric_hidden=64,
    photometric_pose_gradient=cfg["model"].get("photometric_pose_gradient", True),
)
m.load_state_dict(ckpt["model"], strict=False)
m = m.cuda().eval()

data_root = Path("/workspace/datasets/tum_rgbd")
seq = "rgbd_dataset_freiburg1_room"
recs = sync_sequence(data_root / seq)[:30]
img_size = m.img_size
fx, fy, cx, cy = intrinsics_for(seq)
sx, sy = img_size / 640.0, img_size / 480.0
K = torch.tensor([[[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]]], device="cuda")
fov = torch.tensor([[1.0, 1.0]], device="cuda")
patch_pixel = m._patch_pixel_grid.cuda().unsqueeze(0).float()
voxel_state = m.init_voxel_state(1, "cuda", torch.float32)
prev_pose_9 = torch.tensor([[0., 0, 0, 0, 0, 0, 1, 1.0, 1.0]], device="cuda")

patch_stds, target_stds, photo_diffs = [], [], []
mean_pred_rgb, mean_target_rgb = [], []
with torch.no_grad():
    for i, rec in enumerate(recs):
        img = Image.open(rec.rgb_path).convert("RGB").resize((img_size, img_size))
        rgb = torch.from_numpy(np.asarray(img, np.float32) / 255.).permute(2, 0, 1).unsqueeze(0).cuda()
        initial_T = cam9_to_pose_w_c(prev_pose_9)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            patches = m._encode_frame(rgb)
            bootstrap_d = m.bootstrap_depth(patches).float()
            voxel_feat = m.patch_to_voxel(patches).float()
            wc = m.write_confidence(patches).float() if m.use_write_confidence else None
            ray_o1, ray_d1 = build_rays_from_pose(initial_T, K, patch_pixel)
            r1 = render_rays_volumetric(voxel_state, ray_o1, ray_d1,
                n_samples=m.n_render_samples, near=m.render_near, far=m.render_far)
            rendered_feat = r1["feature"]
            patch_rgb_pred = m.color_head(rendered_feat)
            target_patch = torch.nn.functional.adaptive_avg_pool2d(
                rgb.float(), (m.grid_h, m.grid_w)).flatten(2).transpose(1, 2)
            ray_total_w = r1["total_weight"]
            init9 = torch.cat([prev_pose_9[:, :3], prev_pose_9[:, 3:7], fov], -1)
            d9 = m.pose_head(patches, rendered_feat, ray_total_w, init9)
            dT = cam9_to_pose_w_c(d9)
            cT = initial_T.float() @ dT
            wp = backproject_patches_to_world(patch_pixel, bootstrap_d, K, cT.detach())
            write_voxels_trilinear(voxel_state, wp, voxel_feat, weights=wc)
            prev_pose_9 = torch.cat([cT[:, :3, 3], d9[:, 3:7], fov], -1).float()
        if i < 1:
            continue
        # patch-RGB spatial std per frame (variation across the 1024 patches), avg over R,G,B
        pred_std = float(patch_rgb_pred[0].float().std(dim=0).mean())
        tgt_std = float(target_patch[0].float().std(dim=0).mean())
        patch_stds.append(pred_std)
        target_stds.append(tgt_std)
        photo_diffs.append(float((patch_rgb_pred.float() - target_patch.float()).abs().mean()))
        mean_pred_rgb.append([float(patch_rgb_pred[0].float()[..., c].mean()) for c in range(3)])
        mean_target_rgb.append([float(target_patch[0].float()[..., c].mean()) for c in range(3)])

print(f"ColorHead per-frame patch-RGB std (mean over frames 1..29): {float(np.mean(patch_stds)):.4f}")
print(f"  range: [{min(patch_stds):.4f}, {max(patch_stds):.4f}]")
print(f"Target RGB patch std for comparison: {float(np.mean(target_stds)):.4f}")
print(f"Photometric diff (L1 per pixel): {float(np.mean(photo_diffs)):.4f}")
mean_pred = np.array(mean_pred_rgb).mean(axis=0)
mean_target = np.array(mean_target_rgb).mean(axis=0)
print(f"Mean RGB — pred: ({mean_pred[0]:.3f}, {mean_pred[1]:.3f}, {mean_pred[2]:.3f})")
print(f"Mean RGB — target: ({mean_target[0]:.3f}, {mean_target[1]:.3f}, {mean_target[2]:.3f})")
print(f"\nGate: ColorHead patch-RGB std > 0.01 → "
      f"{'PASS' if np.mean(patch_stds) > 0.01 else 'FAIL'}")
print(f"Ratio pred/target std: {float(np.mean(patch_stds))/float(np.mean(target_stds)):.3f}  "
      f"(1.0 = ColorHead reproduces full scene variation)")
