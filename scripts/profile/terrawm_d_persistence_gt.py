"""TerraWM-D persistence test — redone right.

The old test (terrawm_d_persistence.py) measured divergence between two free-
running predicted-pose trajectories. With cam_l1 ≈ 0.029m/frame, two free
runners separate from EACH OTHER far faster than either drifts from GT. The
5-8m divergences in that test were pose-chaos artifacts, not scene memory.

This test isolates the persistence question from pose chaos:

  - Writes use GROUND-TRUTH poses (pose head removed from the loop)
  - For each reset point R, snapshot the grid at frame `eval_frame`
  - PRIMARY VERDICT: render every snapshot from the SAME fixed pose
    (GT pose of frame `eval_frame`) and compare rendered depth pairwise
    (continuous vs each reset_R) within their MUTUAL coverage mask.
    Differences here are purely from grid content — pose is held fixed.
  - CORROBORATING SCREEN: grid-IoU of populated voxels (continuous vs each
    reset_R) at frame `eval_frame`. Low IoU → buffer (different footprint).
    High IoU + high render-diff → same footprint but different content.

Render-diff is decisive. IoU is a footprint check.
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
from vggt_mamba.models.voxel_grid import (                                          # noqa: E402
    backproject_patches_to_world, build_rays_from_pose, init_voxel_state,
    render_rays_volumetric, reset_voxel_state, write_voxels_trilinear,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--seq", default="rgbd_dataset_freiburg3_sitting_xyz")
    p.add_argument("--reset-points", type=int, nargs="+", default=[100, 200, 300, 400, 500])
    p.add_argument("--eval-frame", type=int, default=600)
    p.add_argument("--coverage-eps", type=float, default=1e-3,
                   help="per-pixel mass threshold for 'covered'")
    p.add_argument("--voxel-occ-thresh", type=float, default=1e-3,
                   help="per-voxel mass threshold for 'populated' (IoU calc)")
    p.add_argument("--out-dir", type=Path,
                   default=Path("viz/output/terrawm_d_persistence_gt"))
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
def stream_with_gt_poses(model, recs, gt_rel_T, K, fov, reset_frame: int | None,
                          eval_frame: int):
    """Stream the sequence writing at GT poses (pose head bypassed). Snapshot
    voxel state at eval_frame. If reset_frame is given, zero voxel state at
    that frame index BEFORE writing for that frame.
    """
    img_size = model.img_size
    voxel_state = model.init_voxel_state(batch_size=1, device="cuda", dtype=torch.float32)
    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0).float()           # (1, P, 2)
    snap = None
    for i, rec in enumerate(recs):
        if reset_frame is not None and i == reset_frame:
            reset_voxel_state(voxel_state)
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        gt_pose_T = gt_rel_T[i:i + 1]                                            # (1, 4, 4)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            patches = model._encode_frame(rgb)
            bootstrap_d = model.bootstrap_depth(patches).float()                # (1, P)
            voxel_feat = model.patch_to_voxel(patches).float()                  # (1, P, voxel_dim)
        world_pts = backproject_patches_to_world(patch_pixel, bootstrap_d, K, gt_pose_T)
        write_voxels_trilinear(voxel_state, world_pts, voxel_feat)
        if i == eval_frame:
            snap = {
                "features": voxel_state.features.detach().clone().cpu(),
                "write_mass": voxel_state.write_mass.detach().clone().cpu(),
            }
    return snap


def restore_voxel_state(model, snapshot, device="cuda"):
    state = model.init_voxel_state(batch_size=1, device=device, dtype=torch.float32)
    state.features.copy_(snapshot["features"].to(device))
    state.write_mass.copy_(snapshot["write_mass"].to(device))
    return state


@torch.no_grad()
def render_at_pose(model, voxel_state, pose_T, K, img_size: int) -> dict:
    """Render the voxel grid from a fixed pose. Returns dense depth + mass."""
    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0)                    # (1, P, 2)
    ray_o, ray_d = build_rays_from_pose(pose_T, K, patch_pixel)
    out = render_rays_volumetric(
        voxel_state, ray_o, ray_d,
        n_samples=model.n_render_samples, near=model.render_near, far=model.render_far,
    )
    import torch.nn.functional as F
    H = W = img_size
    dense_d = F.interpolate(
        out["depth"].view(1, 1, model.grid_h, model.grid_w),
        size=(H, W), mode="bilinear", align_corners=True
    ).squeeze().cpu().numpy()
    dense_m = F.interpolate(
        out["total_weight"].view(1, 1, model.grid_h, model.grid_w),
        size=(H, W), mode="bilinear", align_corners=True
    ).squeeze().cpu().numpy()
    return {"depth": dense_d, "mass": dense_m,
            "patch_depth": out["depth"][0].cpu().numpy(),
            "patch_mass": out["total_weight"][0].cpu().numpy()}


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model, cfg = load_model(args.ckpt, args.weights_root)
    img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]
    fx, fy, cx, cy = intrinsics_for(args.seq)
    sx, sy = img_size / 640.0, img_size / 480.0
    K = torch.tensor([[[fx * sx, 0., cx * sx], [0., fy * sy, cy * sy], [0., 0., 1.]]],
                     device="cuda")
    fov = torch.tensor([[1.0, 1.0]], device="cuda")
    recs = sync_sequence(args.data_root / args.seq)
    n_stream = args.eval_frame + 1
    recs = recs[:n_stream]

    # Relativize GT poses to frame 0 (so frame-0 cam = world origin, matching training).
    gt_poses_w_c = np.stack([r.pose_w_c for r in recs])                         # (T, 4, 4)
    P0_inv = np.linalg.inv(gt_poses_w_c[0])
    gt_rel = np.einsum("ij,njk->nik", P0_inv, gt_poses_w_c)                      # (T, 4, 4)
    gt_rel_T = torch.from_numpy(gt_rel).float().cuda()                           # (T, 4, 4)
    print(f"[d-persist-gt] {args.seq}: {len(recs)} frames, eval @ frame {args.eval_frame}")
    print(f"[d-persist-gt] reset points {args.reset_points}")
    print(f"[d-persist-gt] GT cam bbox: "
          f"x{gt_rel[:, 0, 3].min():+.2f}/{gt_rel[:, 0, 3].max():+.2f} "
          f"y{gt_rel[:, 1, 3].min():+.2f}/{gt_rel[:, 1, 3].max():+.2f} "
          f"z{gt_rel[:, 2, 3].min():+.2f}/{gt_rel[:, 2, 3].max():+.2f}  m")

    # Sweep: snapshot grid at eval_frame for each reset point.
    R_to_run = [("continuous", None)] + [(f"reset_{R}", R) for R in args.reset_points]
    snapshots = {}
    for label, R in R_to_run:
        t0 = time.perf_counter()
        snap = stream_with_gt_poses(model, recs, gt_rel_T, K, fov,
                                     reset_frame=R, eval_frame=args.eval_frame)
        dt = time.perf_counter() - t0
        mass = snap["write_mass"].sum().item()
        nz = (snap["write_mass"] > args.voxel_occ_thresh).sum().item()
        snapshots[label] = snap
        print(f"[d-persist-gt]   {label:15s}  streamed {dt:.1f}s  "
              f"mass={mass:.0f}  nonzero_voxels={nz}")

    # === Render every snapshot from the SAME fixed pose = GT pose at eval_frame ===
    eval_pose_T = gt_rel_T[args.eval_frame:args.eval_frame + 1]                  # (1, 4, 4)
    renders = {}
    for label, _ in R_to_run:
        state = restore_voxel_state(model, snapshots[label])
        renders[label] = render_at_pose(model, state, eval_pose_T, K, img_size)
        print(f"[d-persist-gt]   render {label:15s}: "
              f"mass>{args.coverage_eps:.0e} coverage = "
              f"{(renders[label]['mass'] > args.coverage_eps).mean():.3f}")

    # === PRIMARY METRIC: pairwise render comparison, coverage-masked ===
    print(f"\n[d-persist-gt] === PRIMARY: fixed-pose render-diff (continuous vs reset_R) ===")
    print(f"[d-persist-gt] all renders are at the same pose (GT @ frame {args.eval_frame}).")
    print(f"[d-persist-gt] mutual-coverage mask: pixels where BOTH grids have mass > {args.coverage_eps}.")
    cont_d = renders["continuous"]["depth"]
    cont_m = renders["continuous"]["mass"]
    primary = []
    for label, R in R_to_run[1:]:
        rst_d = renders[label]["depth"]
        rst_m = renders[label]["mass"]
        mutual = (cont_m > args.coverage_eps) & (rst_m > args.coverage_eps)
        mutual_frac = float(mutual.mean())
        if mutual.sum() == 0:
            print(f"[d-persist-gt]   R={R:3d}: no mutual coverage, skipping")
            primary.append({"R": R, "render_l1_m": float("nan"),
                            "render_relative": float("nan"),
                            "mutual_coverage": mutual_frac})
            continue
        diff_abs = np.abs(cont_d - rst_d)
        l1 = float(diff_abs[mutual].mean())
        relative = float((diff_abs[mutual] / np.clip(cont_d[mutual], 1e-3, None)).mean())
        print(f"[d-persist-gt]   R={R:3d}: render-L1 {l1:.4f} m   "
              f"relative {relative:.3f}   mutual_coverage {mutual_frac:.3f}")
        primary.append({"R": R, "render_l1_m": l1, "render_relative": relative,
                         "mutual_coverage": mutual_frac})

    # === SCREEN METRIC: grid-IoU of populated voxels ===
    print(f"\n[d-persist-gt] === SCREEN: grid IoU (populated-voxel overlap) ===")
    print(f"[d-persist-gt] voxel 'populated' iff write_mass > {args.voxel_occ_thresh}")
    cont_occ = (snapshots["continuous"]["write_mass"] > args.voxel_occ_thresh).squeeze(-1).squeeze(0).numpy()
    screen = []
    for label, R in R_to_run[1:]:
        rst_occ = (snapshots[label]["write_mass"] > args.voxel_occ_thresh).squeeze(-1).squeeze(0).numpy()
        inter = (cont_occ & rst_occ).sum()
        union = (cont_occ | rst_occ).sum()
        iou = float(inter) / max(union, 1)
        # Also compute "asymmetric": fraction of continuous-populated voxels that reset_R also populates.
        cont_in_rst = float(inter) / max(cont_occ.sum(), 1)
        rst_in_cont = float(inter) / max(rst_occ.sum(), 1)
        print(f"[d-persist-gt]   R={R:3d}: IoU={iou:.3f}   "
              f"cont∩rst/cont={cont_in_rst:.3f}   cont∩rst/rst={rst_in_cont:.3f}")
        screen.append({"R": R, "iou": iou,
                        "intersection_over_continuous": cont_in_rst,
                        "intersection_over_reset": rst_in_cont})

    # === VERDICT (combined) ===
    # PERSISTENT: high IoU AND high render_diff → footprint same, content differs (early
    #   writes leave a trace).  Wait — that's BUFFER actually.  Let me re-think.
    # Restated correctly:
    #   PERSISTENT scene memory means: early writes' INFLUENCE persists into the late
    #   render. So a continuous grid (which has the early writes) renders DIFFERENTLY from
    #   a reset grid (which doesn't) at matched pose — *render-L1 should grow with earlier
    #   reset R*. That's the signature: monotonic decrease in render-L1 as R increases.
    #   Buffer: render-L1 is ~flat across R (only the most recent ~100 frames of writes
    #   matter; older ones are overwritten or irrelevant).
    # IoU complements: low IoU at small R → buffer rebuilt different footprint (settled,
    #   strong non-persistence signal).  High IoU + non-flat render-L1 → footprint same,
    #   content differs (a specific persistent-features story).  High IoU + flat render-L1
    #   → buffer/redundant writes.
    print(f"\n[d-persist-gt] === VERDICT ===")
    verdict = "INCONCLUSIVE"
    spread = slope = 0.0
    is_monotonic_up = False
    primary_valid = [p for p in primary if not np.isnan(p["render_l1_m"])]
    if len(primary_valid) >= 3:
        R_vals = np.array([p["R"] for p in primary_valid])
        L1_vals = np.array([p["render_l1_m"] for p in primary_valid])
        spread = float(L1_vals.max() - L1_vals.min())
        slope = float(np.polyfit(R_vals, L1_vals, 1)[0])                         # m per frame of R
        # PERSISTENT signature: render-L1 INCREASES with R (larger R → less time
        # to rebuild after reset → bigger diff from continuous because the early
        # writes' content is genuinely missing). BUFFER signature: flat in R.
        is_monotonic_up = bool(np.all(np.diff(L1_vals) >= -1e-4))
        is_increasing_in_R = slope > 0
        if is_increasing_in_R and spread > 0.1 and is_monotonic_up:
            verdict = "PERSISTENT"
        elif spread < 0.05:
            verdict = "BUFFER (flat render-diff across R)"
        elif is_increasing_in_R and spread > 0.1:
            verdict = "PERSISTENT_NON_MONOTONIC  (overall slope right, but non-monotonic)"
        else:
            verdict = "INCONCLUSIVE"
        print(f"[d-persist-gt]   render-L1 spread: {spread:.4f} m")
        print(f"[d-persist-gt]   render-L1 slope:  {slope:+.5f} m per R-frame")
        print(f"[d-persist-gt]   monotonic in R:   {is_monotonic_up}")
        print(f"[d-persist-gt]   VERDICT:          {verdict}")
    else:
        verdict = "INSUFFICIENT_DATA"
        print(f"[d-persist-gt]   too few valid R points to call (need ≥3 with mutual coverage)")

    # === Visualizations ===
    # 1. Render-L1 vs R, plus IoU vs R, side-by-side.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if primary_valid:
        R_v = [p["R"] for p in primary_valid]
        L1_v = [p["render_l1_m"] for p in primary_valid]
        cov_v = [p["mutual_coverage"] for p in primary_valid]
        axes[0].plot(R_v, L1_v, "o-", linewidth=2, markersize=8, color="tab:blue", label="render-L1 (m)")
        ax2 = axes[0].twinx()
        ax2.plot(R_v, cov_v, "s--", color="gray", alpha=0.6, label="mutual coverage")
        ax2.set_ylabel("mutual coverage", color="gray")
        ax2.set_ylim(0, 1.05)
        axes[0].set_xlabel("reset frame R")
        axes[0].set_ylabel("render-L1 (continuous vs reset_R) at GT pose @ eval (m)", color="tab:blue")
        axes[0].set_title("PRIMARY: fixed-pose render-diff vs R\n"
                          "PERSISTENT: monotonic INCREASE (larger R → less time to rebuild → bigger diff)")
        axes[0].grid(alpha=0.3); axes[0].legend(loc="upper left"); ax2.legend(loc="upper right")
    R_v = [p["R"] for p in screen]
    iou_v = [p["iou"] for p in screen]
    cont_in_v = [p["intersection_over_continuous"] for p in screen]
    axes[1].plot(R_v, iou_v, "o-", linewidth=2, markersize=8, color="tab:green", label="IoU")
    axes[1].plot(R_v, cont_in_v, "s--", color="tab:olive", alpha=0.7, label="cont∩rst / cont")
    axes[1].set_xlabel("reset frame R")
    axes[1].set_ylabel("populated-voxel overlap")
    axes[1].set_title("SCREEN: grid IoU vs R\n"
                       "Low IoU → buffer (rebuilt with different footprint)")
    axes[1].set_ylim(0, 1.05); axes[1].grid(alpha=0.3); axes[1].legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "verdict.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[d-persist-gt]   saved verdict.png")

    # 2. Per-pixel render comparison: continuous depth, each reset_R depth,
    #    and |diff| heatmap masked on mutual coverage.
    n_runs = len(R_to_run)
    fig, axes = plt.subplots(3, n_runs, figsize=(3 * n_runs, 9))
    if n_runs == 1:
        axes = axes[:, None]
    for col, (label, _) in enumerate(R_to_run):
        d = renders[label]["depth"]
        m = renders[label]["mass"]
        axes[0, col].imshow(d, cmap="turbo", vmin=0, vmax=8.0)
        axes[0, col].set_title(label, fontsize=9); axes[0, col].axis("off")
        axes[1, col].imshow(m, cmap="hot", vmin=0, vmax=1.5)
        axes[1, col].axis("off")
        if label == "continuous":
            axes[2, col].axis("off")
        else:
            mutual = (cont_m > args.coverage_eps) & (m > args.coverage_eps)
            diff = np.abs(cont_d - d) * mutual
            axes[2, col].imshow(diff, cmap="magma", vmin=0, vmax=2.0)
            axes[2, col].axis("off")
    axes[0, 0].set_ylabel("depth (m)", fontsize=10)
    axes[1, 0].set_ylabel("ray total weight", fontsize=10)
    axes[2, 0].set_ylabel("|cont - reset| × mutual_mask", fontsize=10)
    fig.suptitle(
        f"Fixed-pose render comparison @ GT pose of frame {args.eval_frame}.\n"
        "Row 1: depth.  Row 2: coverage.  Row 3: |continuous - reset_R| masked on mutual coverage.",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(args.out_dir / "fixed_pose_renders.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[d-persist-gt]   saved fixed_pose_renders.png")

    # === Summary JSON ===
    summary = {
        "ckpt": str(args.ckpt),
        "seq": args.seq,
        "eval_frame": args.eval_frame,
        "reset_points": args.reset_points,
        "voxel_mass_at_eval_frame": {
            label: float(snapshots[label]["write_mass"].sum())
            for label, _ in R_to_run
        },
        "voxel_nonzero_at_eval_frame": {
            label: int((snapshots[label]["write_mass"] > args.voxel_occ_thresh).sum())
            for label, _ in R_to_run
        },
        "primary_render_diff": primary,
        "screen_grid_iou": screen,
        "render_l1_spread_m": spread if 'spread' in dir() else 0.0,
        "render_l1_slope_m_per_R": slope if 'slope' in dir() else 0.0,
        "verdict": verdict,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[d-persist-gt] saved {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
