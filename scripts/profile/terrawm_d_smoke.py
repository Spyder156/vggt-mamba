"""TerraWM-D smoke + the locked no-bypass verification.

Tests, in order:

  1. End-to-end forward + loss + backward on synthetic data.
  2. Real-batch forward on TUM data + dense-mask coverage statistic.
  3. **NO-BYPASS POSE TEST** (the load-bearing structural guarantee):
     run forward TWICE — once with the voxel grid populated normally,
     once with the voxel grid forced to zero before the second render.
     The corrected pose in the zero-grid case must equal the initial
     pose estimate exactly (zero-init final layer + no input information
     to the comparison MLP). If they differ, there's a bypass somewhere
     and the architecture has the same disease as the recurrent latent.

Saves no plots — this is pure verification. Output is text PASS/FAIL.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.data.tum_rgbd import TUMRGBDDataset, unproject_depth_to_pointmap   # noqa: E402
from vggt_mamba.models.heads.bootstrap_depth import gt_per_patch_depth              # noqa: E402
from vggt_mamba.losses.multitask import terrawm_d_loss                              # noqa: E402
from vggt_mamba.models.pose_utils import gt_relative_motion_from_abs_poses          # noqa: E402
from vggt_mamba.models.terrawm_d import build_terrawm_d                              # noqa: E402
from vggt_mamba.models.voxel_grid import init_voxel_state, reset_voxel_state         # noqa: E402

from torch.utils.data import DataLoader


def main():
    torch.manual_seed(0)
    print("[d-smoke] building TerraWM-D...")
    m = build_terrawm_d(
        "dinov3", "/workspace/datasets/weights",
        n_intraframe_layers=4,
        voxel_resolution=(64, 64, 64),
    ).cuda()
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"[d-smoke] trainable: {n/1e6:.2f}M")

    # ===== 1. Synthetic forward + loss + backward =====
    print("\n[d-smoke] === (1) synthetic forward + loss + backward ===")
    B, T = 1, 4
    s = m.img_size
    rgb = torch.rand(B, T, 3, s, s, device="cuda")
    K = torch.tensor([[[420., 0., 256.], [0., 575., 264.], [0., 0., 1.]]], device="cuda")
    gt_poses = torch.eye(4, device="cuda").expand(B, T, 4, 4).contiguous()
    gt_depth = torch.rand(B, T, s, s, device="cuda") * 4.0 + 0.5
    gt_valid = torch.ones(B, T, s, s, device="cuda", dtype=torch.bool)
    patch_d, patch_v = gt_per_patch_depth(gt_depth, m.grid_h, m.grid_w)
    gt_delta_7 = gt_relative_motion_from_abs_poses(gt_poses.float())
    fov = torch.ones(B, T, 2, device="cuda")
    camera_delta_gt = torch.cat([gt_delta_7, fov], dim=-1)
    t0 = time.perf_counter()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = m(rgb, K_intrinsics=K, gt_poses_w_c=gt_poses)
        targets = {
            "gt_depth_full": gt_depth, "valid": gt_valid,
            "gt_depth_patch": patch_d, "gt_depth_patch_valid": patch_v,
            "poses_w_c": gt_poses, "camera_delta_gt": camera_delta_gt,
        }
        loss, log = terrawm_d_loss(out, targets)
    torch.cuda.synchronize()
    fwd_t = time.perf_counter() - t0
    loss.backward()
    torch.cuda.synchronize()
    full_t = time.perf_counter() - t0
    print(f"[d-smoke]   forward {fwd_t:.2f}s, full step {full_t:.2f}s")
    for k, v in log.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
    print(f"[d-smoke]   peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # ===== 2. Real-batch forward on TUM =====
    print("\n[d-smoke] === (2) real TUM batch forward ===")
    ds = TUMRGBDDataset(
        Path("/workspace/datasets/tum_rgbd"), split="train",
        n_frames=4, stride=8, frame_stride=10, img_size=512,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out_real = m(batch["rgb"], K_intrinsics=batch["K"],
                     gt_poses_w_c=batch["poses_w_c"].float())
    coverage = out_real["depth_mask"].float().mean().item() * 100
    print(f"[d-smoke]   depth_mask coverage on TUM: {coverage:.1f}% "
          f"(should be substantially > 0 with real depth — first frame writes don't render anywhere yet)")
    if coverage < 1.0:
        print(f"[d-smoke]   WARNING: very low coverage at random init; should improve after training")

    # ===== 3. NO-BYPASS POSE TEST =====
    print("\n[d-smoke] === (3) NO-BYPASS POSE TEST (locked structural guarantee) ===")
    # Run a single _frame_step with a populated voxel state vs an empty one;
    # the pose head's correction in the empty case must be exactly initial pose.
    m.eval()
    with torch.no_grad():
        patches = m._encode_frame(rgb[:, 0])                                # (B, P, D)
        initial_T = gt_poses[:, 0]                                          # (B, 4, 4)
        fov_t = torch.ones(B, 2, device="cuda")
        # Case A: empty voxel state (the no-bypass case).
        empty_state = init_voxel_state(m.voxel_cfg, batch_size=B, device="cuda", dtype=torch.float32)
        step_empty = m._frame_step(patches, empty_state, initial_T, K, fov_t)
        cam_empty = step_empty["camera"][0].cpu().numpy()                   # (9,)
        # Reset and write some fake content into the voxel state.
        populated = init_voxel_state(m.voxel_cfg, batch_size=B, device="cuda", dtype=torch.float32)
        # Inject random content into a few voxels so render returns non-zero feat.
        populated.features.normal_(std=0.5)
        populated.write_mass.fill_(0.5)
        step_pop = m._frame_step(patches, populated, initial_T, K, fov_t)
        cam_pop = step_pop["camera"][0].cpu().numpy()
        # Initial pose as 9-vec.
        from vggt_mamba.models.terrawm_d import _pose_T_to_cam9
        init_9 = _pose_T_to_cam9(initial_T, fov_t)[0].cpu().numpy()

    diff_empty_to_init = float(np.abs(cam_empty - init_9).max())
    diff_pop_to_init = float(np.abs(cam_pop - init_9).max())
    diff_empty_pop = float(np.abs(cam_empty - cam_pop).max())
    print(f"   empty-grid corrected pose diff to init: {diff_empty_to_init:.2e} "
          f"(MUST be ~0 — zero-init final layer + no input info)")
    print(f"   populated-grid corrected pose diff to init: {diff_pop_to_init:.2e} "
          f"(also ~0 pre-training — final layer is zero-init; trains to be non-zero)")
    print(f"   empty vs populated corrected pose diff: {diff_empty_pop:.2e} "
          f"(at init: ~0 since both are init pose; AFTER training: should be substantial)")

    if diff_empty_to_init < 1e-4:
        print(f"[d-smoke]   NO-BYPASS pre-training: PASS — empty grid produces init pose.")
    else:
        print(f"[d-smoke]   NO-BYPASS pre-training: FAIL — empty grid SHOULD give init pose at random init!")
        return 1

    # Stress: inject random gradient into pose head's final layer to simulate
    # "after training" with non-zero corrections. Then re-test empty vs populated.
    print("\n[d-smoke] === (3b) NO-BYPASS POSE TEST after fake-training MLP ===")
    with torch.no_grad():
        m.pose_head.compare[-1].weight.normal_(std=0.01)
        m.pose_head.compare[-1].bias.normal_(std=0.01)
    with torch.no_grad():
        step_empty = m._frame_step(patches, init_voxel_state(m.voxel_cfg, B, "cuda", torch.float32), initial_T, K, fov_t)
        pop2 = init_voxel_state(m.voxel_cfg, B, "cuda", torch.float32)
        pop2.features.normal_(std=0.5); pop2.write_mass.fill_(0.5)
        step_pop = m._frame_step(patches, pop2, initial_T, K, fov_t)
    cam_e2 = step_empty["camera"][0].cpu().numpy()
    cam_p2 = step_pop["camera"][0].cpu().numpy()
    diff_e2_init = float(np.abs(cam_e2 - init_9).max())
    diff_p2_init = float(np.abs(cam_p2 - init_9).max())
    diff_e2_p2 = float(np.abs(cam_e2 - cam_p2).max())
    print(f"   empty-grid corrected diff to init: {diff_e2_init:.2e}  (must STILL be ~0 — no info, no correction)")
    print(f"   populated-grid corrected diff to init: {diff_p2_init:.2e}  (should be > 0 — has info)")
    print(f"   empty vs populated diff: {diff_e2_p2:.2e}  (should be > 0 — voxel grid drives the correction)")

    if diff_e2_init < 1e-4 and diff_p2_init > 1e-4:
        print(f"[d-smoke]   NO-BYPASS structural guarantee: PASS — "
              f"empty grid produces no correction even with non-zero MLP weights.")
    else:
        print(f"[d-smoke]   NO-BYPASS structural guarantee: FAIL — bypass detected!")
        return 1

    print(f"\n[d-smoke] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
