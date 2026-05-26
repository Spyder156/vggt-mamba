"""TerraWM smoke test + visualizations for green-flag review.

Loads a TerraWM-configured model (warm-started from 1b ckpt), runs one
training-shaped forward + loss + backward on real TUM data, and produces
four plots for visual sanity-checking BEFORE we commit to training:

  1. GT delta-pose distribution — translation magnitude histogram over a
     small training batch. Confirms the delta supervision target is in a
     sensible range (cm-scale, not garbage).
  2. Predicted vs GT delta translation per frame (one window) — confirms
     the camera-head outputs are interpreted as deltas correctly.
  3. Predictor output per-dim std histogram — the diagnostic that exp 2/2b
     never measured. With VICReg γ=1, the bulk of dims should be above ~0.5
     even at random init (before training).
  4. Camera-motion conditioning sensitivity — run the predictor with motion
     and with zero-motion, plot ||Δ output|| per token. If conditioning is
     working, output should change measurably with motion input.

All plots saved to viz/output/terrawm_smoke/*.png for IDE review.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from torch.utils.data import DataLoader

from vggt_mamba.data.tum_rgbd import TUMRGBDDataset, unproject_depth_to_pointmap   # noqa: E402
from vggt_mamba.losses.multitask import terrawm_loss                               # noqa: E402
from vggt_mamba.models.terrawm import build_terrawm                               # noqa: E402
from vggt_mamba.models.pose_utils import gt_relative_motion_from_abs_poses          # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path,
                   default=Path("configs/phase3_streaming_terrawm.yaml"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("viz/output/terrawm_smoke"))
    p.add_argument("--n-windows", type=int, default=8,
                   help="number of windows to sample for delta distribution plot")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(args.config.read_text())
    device = "cuda"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ===== Dataset =====
    ds = TUMRGBDDataset(
        Path(cfg["data"]["root"]), split="train",
        n_frames=cfg["data"]["n_frames"], stride=cfg["data"]["stride_train"],
        frame_stride=cfg["data"]["frame_stride"], img_size=512,
        randomize_stride=tuple(cfg["data"]["randomize_stride"])
        if cfg["data"].get("randomize_stride") else None,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    print(f"[smoke] dataset: {len(ds.sequences)} seqs, {len(ds)} windows of N={cfg['data']['n_frames']}")

    # ===== Plot 1: GT delta-pose distribution across many windows =====
    print(f"[smoke] sampling {args.n_windows} windows for GT delta distribution...")
    delta_trans_mags = []      # per-pair translation magnitudes (in m)
    delta_rot_angles = []      # per-pair rotation angles (in degrees)
    for k, batch in enumerate(loader):
        if k >= args.n_windows:
            break
        poses = batch["poses_w_c"][0].float()              # (T, 4, 4)
        gt_delta = gt_relative_motion_from_abs_poses(poses.unsqueeze(0))[0]  # (T, 7)
        # Skip frame 0 (identity by construction).
        dt = gt_delta[1:, :3].norm(dim=-1).numpy()         # (T-1,)
        dq = gt_delta[1:, 3:7].numpy()
        # Convert quaternion to angle: 2 * acos(|qw|)
        dq_norm = dq / np.linalg.norm(dq, axis=-1, keepdims=True).clip(1e-12)
        angle_rad = 2.0 * np.arccos(np.clip(np.abs(dq_norm[:, 3]), 0, 1))
        delta_trans_mags.extend(dt.tolist())
        delta_rot_angles.extend(np.degrees(angle_rad).tolist())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(delta_trans_mags, bins=30, color="tab:blue", alpha=0.8)
    axes[0].set_xlabel("||Δt|| per frame pair (m)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"GT delta translation magnitude\n"
                      f"median {np.median(delta_trans_mags):.3f} m, max {max(delta_trans_mags):.3f} m")
    axes[0].grid(alpha=0.3)
    axes[1].hist(delta_rot_angles, bins=30, color="tab:orange", alpha=0.8)
    axes[1].set_xlabel("Δrotation angle per frame pair (°)")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"GT delta rotation angle\n"
                      f"median {np.median(delta_rot_angles):.2f}°, max {max(delta_rot_angles):.2f}°")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Plot 1: GT delta-pose distribution — sanity check supervision targets")
    plt.tight_layout()
    plt.savefig(args.out_dir / "1_gt_delta_distribution.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[smoke]   saved 1_gt_delta_distribution.png")

    # ===== Build model =====
    print("[smoke] building TerraWM model + loading 1b warm-start...")
    model = build_terrawm(
        cfg["encoder"], "/workspace/datasets/weights",
        n_intraframe_layers=cfg["model"]["n_intraframe_layers"],
        n_summary_tokens=cfg["model"]["n_summary_tokens"],
        n_summary_dynamic=cfg["model"].get("n_summary_dynamic"),
        n_xfm_layers=cfg["model"]["n_xfm_layers"],
        d_state=cfg["model"]["d_state"],
        bidirectional=False,
        track_enabled=cfg["model"]["track_enabled"],
        max_frames=max(cfg["data"]["n_frames"] * 16, 256),
        dense_residual_to_patches=cfg["model"]["dense_residual_to_patches"],
        predict_next_latent=cfg["model"]["predict_next_latent"],
        ema_momentum=cfg["model"]["ema_momentum"],
        cross_frame_target=cfg["model"]["cross_frame_target"],
        delta_pose=True,
        motion_enc_freqs=cfg["model"]["motion_enc_freqs"],
    ).to(device)
    warm = torch.load(cfg["load_ckpt"], map_location=device, weights_only=False)
    warm_sd = warm["model"]
    msd = model.state_dict()
    if "frame_embed" in warm_sd and warm_sd["frame_embed"].shape != msd["frame_embed"].shape:
        wb, mb = warm_sd["frame_embed"], msd["frame_embed"]
        n = min(wb.shape[1], mb.shape[1])
        new = mb.clone(); new[:, :n] = wb[:, :n]
        warm_sd["frame_embed"] = new
        print(f"[smoke]   frame_embed: copied {n} from ckpt")
    # TerraWM uses ConditionedNextLatentPredictor with different shape than the
    # ckpt's LatentPredictor. Drop incompatible predictor keys so the new
    # predictor stays at its random init.
    incompat = [k for k in list(warm_sd.keys())
                if k.startswith("latent_predictor.") and k in msd
                and warm_sd[k].shape != msd[k].shape]
    for k in incompat:
        del warm_sd[k]
    if incompat:
        print(f"[smoke]   dropped {len(incompat)} incompatible latent_predictor keys from ckpt")
    msg = model.load_state_dict(warm_sd, strict=False)
    non_enc = [k for k in msg.missing_keys if not k.startswith("encoder.")]
    print(f"[smoke]   missing(non-enc) after load: {len(non_enc)}")
    print(f"[smoke]   sample missing keys: {non_enc[:6]}")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke]   trainable: {n_train/1e6:.2f}M")

    # ===== Forward + loss + backward on one batch =====
    model.train()
    batch = next(iter(loader))
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    with torch.no_grad():
        gt_delta_7 = gt_relative_motion_from_abs_poses(batch["poses_w_c"].float())
        gt_motion_pred_input = gt_delta_7[:, 1:].contiguous()                  # (B, T-1, 7)
        fov = batch["camera_gt"][..., 7:]
        camera_delta_gt = torch.cat([gt_delta_7, fov], dim=-1)                 # (B, T, 9)

    t0 = time.perf_counter()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        preds = model(
            batch["rgb"], K_intrinsics=batch.get("K"),
            gt_relative_motion=gt_motion_pred_input,
        )
        gt_pmap = unproject_depth_to_pointmap(
            batch["depth"], batch["K"].unsqueeze(1).expand(-1, batch["depth"].shape[1], 3, 3)
        )
        tcam = batch["camera_gt"][..., :3]
        diff_t = tcam[:, 1:] - tcam[:, :-1]
        pmw = diff_t.norm(dim=-1) / diff_t.norm(dim=-1).median().clamp_min(1e-6)
        targets = {
            "gt_pointmap_cam": gt_pmap,
            "valid": batch["valid"],
            "poses_w_c": batch["poses_w_c"],
            "camera_gt": batch["camera_gt"],
            "camera_delta_gt": camera_delta_gt,
        }
        loss, log = terrawm_loss(
            preds, targets, w_track=0,
            w_pred=cfg["loss"]["w_pred"],
            w_vic_var=cfg["loss"]["w_vic_var"],
            w_vic_cov=cfg["loss"]["w_vic_cov"],
            pred_motion_weights=pmw,
            cam_target_key="camera_delta_gt",
        )
    torch.cuda.synchronize()
    fwd_t = time.perf_counter() - t0
    loss.backward()
    torch.cuda.synchronize()
    step_t = time.perf_counter() - t0
    print(f"[smoke]   forward+loss {fwd_t:.2f}s   full step {step_t:.2f}s   "
          f"est 2000 steps {step_t*2000/60:.1f} min")
    print(f"[smoke]   loss components:")
    for k, v in log.items():
        if isinstance(v, float):
            print(f"     {k}: {v:.4f}")
    print(f"[smoke]   peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # ===== Plot 2: Predicted vs GT delta translation (per frame, one window) =====
    pred_cam = preds["camera"][0].float().detach().cpu().numpy()              # (T, 9)
    gt_cam_delta = camera_delta_gt[0].float().cpu().numpy()                    # (T, 9)
    T = pred_cam.shape[0]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for k, label in enumerate(["x", "y", "z"]):
        axes[0].plot(pred_cam[:, k], "o--", label=f"pred Δt_{label}", alpha=0.7)
        axes[0].plot(gt_cam_delta[:, k], "x-", label=f"gt Δt_{label}", alpha=0.7)
    axes[0].set_ylabel("Δ translation per frame (m)")
    axes[0].set_title(f"Plot 2: Predicted vs GT delta camera motion (one window, frame 0 = identity)\n"
                      f"pre-training; expect predictions ≠ GT but in right ballpark")
    axes[0].legend(ncols=2, fontsize=8); axes[0].grid(alpha=0.3)
    # Quaternion component-wise
    for k, label in enumerate(["qx", "qy", "qz", "qw"]):
        axes[1].plot(pred_cam[:, 3 + k], "o--", label=f"pred Δ{label}", alpha=0.7)
        axes[1].plot(gt_cam_delta[:, 3 + k], "x-", label=f"gt Δ{label}", alpha=0.7)
    axes[1].set_xlabel("frame index")
    axes[1].set_ylabel("Δ quaternion components")
    axes[1].legend(ncols=4, fontsize=7); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_dir / "2_pred_vs_gt_delta.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[smoke]   saved 2_pred_vs_gt_delta.png")

    # ===== Plot 3: predictor output per-dim std =====
    if "predicted_next" in preds:
        pn = preds["predicted_next"].float().detach()                         # (B, T-1, K, D)
        pn_flat = pn.reshape(-1, pn.shape[-1])
        std_per_dim = pn_flat.std(dim=0).cpu().numpy()                         # (D,)
        target_pn = preds["target_next"].float().detach().reshape(-1, pn.shape[-1])
        tgt_std_per_dim = target_pn.std(dim=0).cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(std_per_dim, bins=40, color="tab:blue", alpha=0.7,
                label=f"predictor output (mean {std_per_dim.mean():.3f})")
        ax.hist(tgt_std_per_dim, bins=40, color="tab:orange", alpha=0.5,
                label=f"EMA target (mean {tgt_std_per_dim.mean():.3f})")
        ax.axvline(0.5, color="red", linestyle="--",
                   label="health floor ~0.5 (VICReg γ=1)")
        ax.set_xlabel("per-dim std across (B × T-1 × K) samples")
        ax.set_ylabel("count of dims (D=1024)")
        ax.set_title("Plot 3: Predictor output per-dim std — collapse diagnostic\n"
                     "(at random init, before any training)")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out_dir / "3_predictor_dim_std.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[smoke]   saved 3_predictor_dim_std.png")
    else:
        print(f"[smoke]   skipping plot 3 — predicted_next not in outputs")

    # ===== Plot 4: motion-conditioning sensitivity =====
    print("[smoke] motion-conditioning sensitivity probe...")
    model.eval()
    # Forward once with real motion, once with zero motion. Compare predictor outputs.
    # Need to re-run predictor manually since at eval, the model skips it.
    # Use the saved state_per_frame indirectly: re-run forward with model.train() momentarily.
    # Simpler: directly invoke the predictor module with the same state we already have.
    if "predicted_next" in preds:
        # Pre-recorded state was internal. Rebuild by running a tiny manual probe:
        # Re-run model.train() forward with zero motion, compare predicted_next.
        zero_motion = torch.zeros_like(gt_motion_pred_input)
        model.train()  # re-enable predictor
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            preds_zero = model(
                batch["rgb"], K_intrinsics=batch.get("K"),
                gt_relative_motion=zero_motion,
            )
        pred_with = preds["predicted_next"].float().detach()                  # (B, T-1, K, D)
        pred_zero = preds_zero["predicted_next"].float().detach()
        # Magnitude of the change per (frame, token).
        per_tok_change = (pred_with - pred_zero).norm(dim=-1)                 # (B, T-1, K)
        per_frame_change = per_tok_change[0].mean(dim=-1).cpu().numpy()       # (T-1,)
        motion_mag = gt_motion_pred_input[0, :, :3].norm(dim=-1).cpu().numpy()  # (T-1,)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(per_frame_change, "o-", color="tab:blue")
        axes[0].set_xlabel("frame index (t → t+1)")
        axes[0].set_ylabel("||predictor(motion) − predictor(0)||")
        axes[0].set_title(f"Plot 4a: per-frame change with motion conditioning\n"
                          f"mean change {per_frame_change.mean():.4f} "
                          f"(non-zero ⇒ predictor uses the motion input)")
        axes[0].grid(alpha=0.3)
        axes[1].scatter(motion_mag, per_frame_change, color="tab:blue", alpha=0.7)
        axes[1].set_xlabel("||GT motion|| per frame (m)")
        axes[1].set_ylabel("||predictor output change||")
        axes[1].set_title("Plot 4b: change vs motion magnitude\n"
                          "(should correlate positively for working conditioning)")
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out_dir / "4_motion_sensitivity.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[smoke]   saved 4_motion_sensitivity.png")
        print(f"[smoke]   mean change with vs without motion: {per_frame_change.mean():.4f}")
        if per_frame_change.mean() < 1e-3:
            print(f"[smoke]   *** WARNING *** motion conditioning produces ~0 change. Check encoder.")

    print(f"\n[smoke] all plots saved to {args.out_dir}/")
    print("[smoke] PASS")


if __name__ == "__main__":
    main()
