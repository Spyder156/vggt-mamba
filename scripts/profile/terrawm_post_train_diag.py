"""Post-training TerraWM diagnostics — pre-committed before launch.

Runs two checks on the trained TerraWM ckpt:

  (A) Predictor health: did VICReg actually fix the collapse seen at init?
      Smoke pre-training: per-dim std = 0.034 (severely collapsed).
      Healthy after training: ≥ 0.5 (and ideally ~1.0, the VICReg γ).

  (B) Scene-state ablation: does the recurrent state actually carry scene
      memory that matters for downstream predictions? Two streaming passes
      over the same sequence — one continuous (state evolves normally),
      one with the Mamba state reset to zero at frame K. If the state is
      genuinely encoding scene info, post-reset predictions should differ
      meaningfully. If predictions converge quickly after the reset, the
      model is essentially memoryless per-frame and the "world model"
      framing isn't actually being used.

Outputs:
  viz/output/terrawm_post_train/predictor_dim_std_trained.png
  viz/output/terrawm_post_train/scene_state_ablation.png
  viz/output/terrawm_post_train/diag.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from torch.utils.data import DataLoader

from vggt_mamba.data.tum_rgbd import (                                          # noqa: E402
    TUMRGBDDataset, sync_sequence, unproject_depth_to_pointmap, _quat_to_rot,
)
from vggt_mamba.losses.multitask import terrawm_loss                            # noqa: E402
from vggt_mamba.models.terrawm import build_terrawm                            # noqa: E402
from vggt_mamba.models.pose_utils import gt_relative_motion_from_abs_poses       # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--seq", default="rgbd_dataset_freiburg3_sitting_xyz")
    p.add_argument("--ablation-reset-frame", type=int, default=200,
                   help="frame at which to reset the Mamba state in the ablation run")
    p.add_argument("--n-frames", type=int, default=500)
    p.add_argument("--out-dir", type=Path,
                   default=Path("viz/output/terrawm_post_train"))
    return p.parse_args()


def load_model(ckpt_path, weights_root):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_terrawm(
        cfg["encoder"], str(weights_root),
        n_intraframe_layers=cfg["model"]["n_intraframe_layers"],
        n_summary_tokens=cfg["model"]["n_summary_tokens"],
        n_summary_dynamic=cfg["model"].get("n_summary_dynamic"),
        n_xfm_layers=cfg["model"]["n_xfm_layers"],
        d_state=cfg["model"]["d_state"],
        bidirectional=False,
        track_enabled=cfg["model"]["track_enabled"],
        max_frames=ckpt["model"]["frame_embed"].shape[1],
        dense_residual_to_patches=cfg["model"].get("dense_residual_to_patches", True),
        predict_next_latent=cfg["model"].get("predict_next_latent", False),
        ema_momentum=cfg["model"].get("ema_momentum", 0.99),
        cross_frame_target=cfg["model"].get("cross_frame_target", "summary"),
        delta_pose=cfg["model"].get("delta_pose", cfg["model"].get("terrawm", False)),
        motion_enc_freqs=cfg["model"].get("motion_enc_freqs", cfg["model"].get("terrawm_motion_freqs", 64)),
    )
    model.load_state_dict(ckpt["model"], strict=False)
    return model.cuda().eval(), cfg


def load_rgb(rec, img_size):
    from PIL import Image
    img = Image.open(rec.rgb_path).convert("RGB").resize((img_size, img_size))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0
                            ).permute(2, 0, 1).unsqueeze(0).contiguous()


@torch.no_grad()
def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model, cfg = load_model(args.ckpt, args.weights_root)
    img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]
    print(f"[diag] loaded TerraWM ckpt: {args.ckpt}")
    print(f"[diag]   terrawm={model.delta_pose}, n_summary={model.n_summary}, "
          f"n_dynamic={model.n_dynamic}")

    diag = {}

    # ===== (A) Predictor health =====
    print("\n[A] PREDICTOR HEALTH — running 1 training-mode forward to get predictor output")
    ds = TUMRGBDDataset(
        args.data_root, split="train",
        n_frames=cfg["data"]["n_frames"], stride=cfg["data"]["stride_train"],
        frame_stride=cfg["data"]["frame_stride"], img_size=img_size,
        randomize_stride=tuple(cfg["data"]["randomize_stride"])
        if cfg["data"].get("randomize_stride") else None,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
    # Need GT delta motion for predictor conditioning
    gt_delta_7 = gt_relative_motion_from_abs_poses(batch["poses_w_c"].float())
    gt_motion_pred_input = gt_delta_7[:, 1:].contiguous()
    model.train()                       # so predictor fires
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        preds = model(
            batch["rgb"], K_intrinsics=batch.get("K"),
            gt_relative_motion=gt_motion_pred_input,
        )
    model.eval()
    pn = preds["predicted_next"].float().detach()                                # (B, T-1, K, D)
    tn = preds["target_next"].float().detach()
    pn_flat = pn.reshape(-1, pn.shape[-1])
    tn_flat = tn.reshape(-1, tn.shape[-1])
    std_pred = pn_flat.std(dim=0).cpu().numpy()                                  # (D,)
    std_target = tn_flat.std(dim=0).cpu().numpy()

    INIT_STD = 0.034    # measured at random init in smoke
    diag["predictor"] = {
        "mean_dim_std_trained": float(std_pred.mean()),
        "mean_dim_std_target": float(std_target.mean()),
        "p10_dim_std_trained": float(np.percentile(std_pred, 10)),
        "p90_dim_std_trained": float(np.percentile(std_pred, 90)),
        "init_dim_std_for_comparison": INIT_STD,
        "vicreg_gamma": 1.0,
    }
    print(f"  predictor per-dim std: mean {std_pred.mean():.4f}  "
          f"(was {INIT_STD} at init; VICReg γ=1)")
    print(f"  target per-dim std:    mean {std_target.mean():.4f}")
    print(f"  pct of dims above γ/2=0.5: {(std_pred > 0.5).mean() * 100:.1f}%")
    print(f"  pct of dims above γ=1.0:   {(std_pred > 1.0).mean() * 100:.1f}%")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.hist(std_pred, bins=50, alpha=0.7, color="tab:blue",
            label=f"predictor (trained)  mean={std_pred.mean():.3f}")
    ax.hist(std_target, bins=50, alpha=0.5, color="tab:orange",
            label=f"EMA target  mean={std_target.mean():.3f}")
    ax.axvline(INIT_STD, color="red", linestyle="--",
               label=f"at init: {INIT_STD}")
    ax.axvline(0.5, color="gray", linestyle=":", label="γ/2 health floor")
    ax.axvline(1.0, color="green", linestyle=":", label="γ=1 VICReg target")
    ax.set_xlabel("per-dim std across (B × T-1 × K) samples")
    ax.set_ylabel("count of dims")
    ax.set_title(f"(A) Predictor health post-training — did VICReg fix collapse?\n"
                 f"At init mean was {INIT_STD}; after training mean is {std_pred.mean():.3f}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_dir / "predictor_dim_std_trained.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved predictor_dim_std_trained.png")

    # ===== (B) Scene-state ablation =====
    print(f"\n[B] SCENE-STATE ABLATION — stream {args.seq} normally vs with state reset at frame {args.ablation_reset_frame}")
    recs = sync_sequence(args.data_root / args.seq)[:args.n_frames]
    print(f"  {len(recs)} frames")

    # Run 1: continuous streaming.
    state1 = model.init_streaming_state(use_cuda_graphs=False)
    cam_continuous = []
    pmap_z_continuous = []
    for i, rec in enumerate(recs):
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        out, state1 = model.streaming_forward(rgb, state1, frame_idx=i)
        cam_continuous.append(out["camera"][0, 0].float().cpu().numpy())
        # mean depth per frame (rough single-number signature)
        pmap_z_continuous.append(float(out["pointmap"][0, 0, 2].float().mean()))
    cam_continuous = np.stack(cam_continuous)
    pmap_z_continuous = np.array(pmap_z_continuous)

    # Run 2: identical until reset, then re-init state.
    state2 = model.init_streaming_state(use_cuda_graphs=False)
    cam_ablation = []
    pmap_z_ablation = []
    for i, rec in enumerate(recs):
        if i == args.ablation_reset_frame:
            # Reset state — mimic starting from scratch, but with the same
            # frame index so frame_embed is unchanged.
            state2 = model.init_streaming_state(use_cuda_graphs=False)
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        out, state2 = model.streaming_forward(rgb, state2, frame_idx=i)
        cam_ablation.append(out["camera"][0, 0].float().cpu().numpy())
        pmap_z_ablation.append(float(out["pointmap"][0, 0, 2].float().mean()))
    cam_ablation = np.stack(cam_ablation)
    pmap_z_ablation = np.array(pmap_z_ablation)

    # Compare: per-frame divergence in cam-output and mean-depth.
    cam_diff = np.linalg.norm(cam_continuous[:, :3] - cam_ablation[:, :3], axis=-1)  # (N,)
    depth_diff = np.abs(pmap_z_continuous - pmap_z_ablation)

    # Aggregate: pre-reset (should be 0) and post-reset (signal of state-dependence).
    pre_reset = slice(0, args.ablation_reset_frame)
    post_reset = slice(args.ablation_reset_frame, None)
    recovery_window = slice(args.ablation_reset_frame,
                            min(args.ablation_reset_frame + 50, len(recs)))
    late_window = slice(max(args.ablation_reset_frame + 200, len(recs) - 50), len(recs))

    diag["state_ablation"] = {
        "reset_frame": args.ablation_reset_frame,
        "n_frames": len(recs),
        "cam_diff_pre_reset_max_m": float(cam_diff[pre_reset].max()),
        "cam_diff_post_reset_immediate_mean_m": float(cam_diff[recovery_window].mean()),
        "cam_diff_post_reset_late_mean_m": float(cam_diff[late_window].mean()),
        "depth_diff_pre_reset_max_m": float(depth_diff[pre_reset].max()),
        "depth_diff_post_reset_immediate_mean_m": float(depth_diff[recovery_window].mean()),
        "depth_diff_post_reset_late_mean_m": float(depth_diff[late_window].mean()),
    }
    print(f"  pre-reset cam diff max: {diag['state_ablation']['cam_diff_pre_reset_max_m']:.5f} m  (should be ~0)")
    print(f"  post-reset (immediate 50 frames):")
    print(f"    cam diff mean:   {diag['state_ablation']['cam_diff_post_reset_immediate_mean_m']:.5f} m")
    print(f"    depth diff mean: {diag['state_ablation']['depth_diff_post_reset_immediate_mean_m']:.5f} m")
    print(f"  post-reset (late, after recovery):")
    print(f"    cam diff mean:   {diag['state_ablation']['cam_diff_post_reset_late_mean_m']:.5f} m")
    print(f"    depth diff mean: {diag['state_ablation']['depth_diff_post_reset_late_mean_m']:.5f} m")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x = np.arange(len(recs))
    axes[0].plot(x, cam_diff, color="tab:blue", linewidth=1)
    axes[0].axvline(args.ablation_reset_frame, color="red", linestyle="--",
                    label=f"state reset @ frame {args.ablation_reset_frame}")
    axes[0].set_ylabel("||cam_continuous − cam_reset||  (Δt in m)")
    axes[0].set_title(f"(B) Scene-state ablation — TerraWM on {args.seq}\n"
                      "If state encodes scene memory: post-reset divergence should be non-trivial. "
                      "If state is irrelevant: convergence within a few frames.")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(x, depth_diff, color="tab:green", linewidth=1)
    axes[1].axvline(args.ablation_reset_frame, color="red", linestyle="--")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("|mean depth_continuous − mean depth_reset|  (m)")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_dir / "scene_state_ablation.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved scene_state_ablation.png")

    (args.out_dir / "diag.json").write_text(json.dumps(diag, indent=2))
    print(f"\n[diag] saved diag.json")


if __name__ == "__main__":
    main()
