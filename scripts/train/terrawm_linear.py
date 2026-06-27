"""Trainer for TerraWM-Linear V1 (Path A).

Pipeline per step:
    1. Sample a window of N frames from TUM (TUMRGBDDataset).
    2. Normalize GT scene by mean-point-distance (eval/normalize.py).
       Re-centers world at cam 0, divides translations + depth by avg-distance.
    3. Forward TerraWMLinear → cameras (B, T, 9) + depths (B, T, H, W).
    4. Encode normalized GT pose to 9-d pose enc (eval/pose_enc.py).
    5. Loss = w_trans * L1(t)
            + w_quat  * L1(q)  [sign-disambiguated: min(||q-q_gt||, ||q+q_gt||)]
            + w_fov   * L1(fov)
            + w_depth * L1(depth)[valid]
    6. AdamW step (only on trainable params; encoder frozen).
    7. Every eval_every steps, snapshot a ckpt + run held-out eval.

Usage:
    python -m scripts.train.terrawm_linear --config configs/terrawm_linear.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.data.tum_rgbd import TUMRGBDDataset
from vggt_mamba.eval.metrics import chamfer_distance_torch, sample_valid_points
from vggt_mamba.eval.normalize import (
    back_project_depth_to_world,
    normalize_scene_by_mean_distance,
)
from vggt_mamba.eval.pose_enc import (
    fov_from_intrinsics,
    world_from_cam_to_pose_enc,
)
from vggt_mamba.models.terrawm_linear import TerraWMConfig, TerraWMLinear


# ---------- loss helpers --------------------------------------------------------

def quaternion_l1_loss(pred_q: torch.Tensor, gt_q: torch.Tensor) -> torch.Tensor:
    """Sign-disambiguated L1 on (qx, qy, qz, qw). q and -q encode the same rotation."""
    diff_pos = (pred_q - gt_q).abs().sum(dim=-1)
    diff_neg = (pred_q + gt_q).abs().sum(dim=-1)
    return torch.minimum(diff_pos, diff_neg).mean()


def multi_view_3d_chamfer(
    pred_depth: torch.Tensor,        # (B, T, H, W) — model's normalized depth
    valid: torch.Tensor,             # (B, T, H, W) bool
    pose_normed_w_c: torch.Tensor,   # (B, T, 4, 4) — normalized GT world-from-cam
    K: torch.Tensor,                 # (B, 3, 3)
    n_samples: int = 512,
) -> torch.Tensor:
    """Multi-view 3D consistency: back-project predicted depth via GT pose,
    sample valid points per frame, average pairwise Chamfer across (i, j).

    Forces depth to be 3D-coherent across views — the single strongest signal
    for monocular depth. Uses GT pose (cheat-pose Flavor A) so the constraint
    only depends on whether depth is right.
    """
    world_pts = back_project_depth_to_world(pred_depth, valid, K, pose_normed_w_c)  # (B, T, H, W, 3)
    # Move to (B, T, 3, H, W) for sample_valid_points
    pts_for_sample = world_pts.permute(0, 1, 4, 2, 3).contiguous()
    sampled = sample_valid_points(pts_for_sample, valid, n_samples)                  # (B, T, N, 3)

    T = sampled.shape[1]
    pair_losses = []
    for i in range(T):
        for j in range(i + 1, T):
            pair_losses.append(chamfer_distance_torch(sampled[:, i], sampled[:, j]))
    if not pair_losses:
        return sampled.new_zeros(())
    return torch.stack(pair_losses).mean()


def cosine_warmup_lr(step: int, n_steps: int, warmup: int, max_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, n_steps - warmup)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


# ---------- eval helper ---------------------------------------------------------

def run_held_out_eval(ckpt_path: Path, cfg: dict, log_path: Path) -> dict | None:
    """Spawn the held-out eval script as a subprocess. Returns parsed metrics or None."""
    cmd = [
        sys.executable, "-m", "scripts.eval.terrawm_linear_held_out",
        "--ckpt", str(ckpt_path),
        "--seq", cfg["eval"]["seq"],
        "--n-frames", str(cfg["eval"]["n_frames"]),
        "--img-size", str(cfg["model"]["img_size"]),
        "--d-model", str(cfg["model"]["d_model"]),
        "--n-latents", str(cfg["model"]["n_latents"]),
        "--n-write-blocks", str(cfg["model"]["n_write_blocks"]),
        "--n-decode-blocks", str(cfg["model"]["n_decode_blocks"]),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
        log_path.write_text(out.stdout + "\n--- stderr ---\n" + out.stderr)
        return parse_eval_stdout(out.stdout)
    except subprocess.CalledProcessError as e:
        log_path.write_text((e.stdout or "") + "\n--- stderr ---\n" + (e.stderr or ""))
        return None


def parse_eval_stdout(s: str) -> dict:
    """Pull numeric metrics out of held-out script stdout (rough but works)."""
    out: dict[str, float] = {}
    for line in s.splitlines():
        line = line.strip()
        if "ATE Sim(3) rmse:" in line:
            out["ate_sim3_rmse"] = float(line.split("rmse:")[1].split("m")[0].strip())
        elif "ATE SE(3) rmse:" in line:
            out["ate_se3_rmse"] = float(line.split("rmse:")[1].split("m")[0].strip())
        elif "RPE@1 trans rmse:" in line:
            out["rpe1_trans_rmse"] = float(line.split("rmse:")[1].split("m")[0].strip())
        elif "RPE@1 rot rmse:" in line:
            out["rpe1_rot_rmse_deg"] = float(line.split("rmse:")[1].split("deg")[0].strip())
        elif "abs_rel after scale s:" in line:
            out["abs_rel_scaled"] = float(line.split("scale s:")[1].split("(")[0].strip())
    return out


# ---------- main ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    torch.manual_seed(cfg["experiment"]["seed"])
    np.random.seed(cfg["experiment"]["seed"])

    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ckpts").mkdir(exist_ok=True)
    (out_dir / "eval_logs").mkdir(exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.dump(cfg))
    print(f"output_dir: {out_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- model ---
    model_cfg = TerraWMConfig(
        img_size=cfg["model"]["img_size"],
        d_enc=cfg["model"]["d_enc"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_latents=cfg["model"]["n_latents"],
        n_write_blocks=cfg["model"]["n_write_blocks"],
        n_decode_blocks=cfg["model"]["n_decode_blocks"],
        encoder_repo=cfg["model"]["encoder_repo"],
        freeze_encoder=cfg["model"]["freeze_encoder"],
        max_frames=cfg["model"]["max_frames"],
    )
    model = TerraWMLinear(model_cfg).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"trainable params: {n_train/1e6:.2f}M  |  frozen: {n_frozen/1e6:.2f}M")

    opt = torch.optim.AdamW(trainable, lr=cfg["train"]["lr"],
                              weight_decay=cfg["train"]["weight_decay"])

    start_step = 0
    if args.resume:
        sd = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        start_step = sd["step"]
        print(f"resumed from step {start_step}")

    # --- data ---
    train_ds = TUMRGBDDataset(
        data_root=cfg["data"]["data_root"],
        split="train",
        n_frames=cfg["data"]["n_frames"],
        stride=cfg["data"]["stride"],
        frame_stride=cfg["data"]["frame_stride"],
        img_size=cfg["data"]["img_size"],
        depth_max_m=cfg["data"]["depth_max_m"],
        randomize_stride=tuple(cfg["data"]["randomize_stride"]) if cfg["data"].get("randomize_stride") else None,
    )
    print(f"train dataset: {len(train_ds)} windows")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        drop_last=True,
        persistent_workers=cfg["data"]["num_workers"] > 0,
    )

    # --- loss weights + mode ---
    w_t = cfg["loss"]["w_trans"]
    w_q = cfg["loss"]["w_quat"]
    w_f = cfg["loss"]["w_fov"]
    w_d = cfg["loss"]["w_depth"]
    w_3d = cfg["loss"].get("w_3d", 0.0)
    n_3d_samples = cfg["loss"].get("n_3d_samples", 512)

    pose_mode = cfg["model"].get("pose_supervision_mode", "predicted")
    print(f"pose_supervision_mode: {pose_mode}")
    print(f"loss weights: t={w_t} q={w_q} fov={w_f} d={w_d} 3d={w_3d}")

    use_bf16 = cfg["train"]["use_bf16"]
    n_steps = cfg["train"]["n_steps"]
    eval_every = cfg["train"]["eval_every"]
    ckpt_every = cfg["train"]["ckpt_every"]
    log_every = cfg["train"]["log_every"]
    grad_clip = cfg["train"]["grad_clip"]
    warmup = cfg["train"]["warmup_steps"]
    max_lr = cfg["train"]["lr"]
    img_size = cfg["model"]["img_size"]

    train_iter = iter(train_loader)
    step = start_step
    t0 = time.time()
    last_log = t0
    print(f"starting training: {n_steps} steps")
    print(f"  log every {log_every}, ckpt+eval every {ckpt_every}/{eval_every}")

    while step < n_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        rgb = batch["rgb"].to(device, non_blocking=True)                                # (B, T, 3, H, W)
        depth = batch["depth"].to(device, non_blocking=True)                            # (B, T, H, W)
        valid = batch["valid"].to(device, non_blocking=True)                            # (B, T, H, W)
        pose_w_c = batch["poses_w_c"].to(device, non_blocking=True)                     # (B, T, 4, 4)
        K = batch["K"].to(device, non_blocking=True)                                    # (B, 3, 3)
        B, T = rgb.shape[:2]

        # --- normalize GT (Path A) ---
        # Re-center at cam 0, divide by mean-distance.
        pose_normed, depth_normed, scale = normalize_scene_by_mean_distance(
            pose_w_c.float(), depth.float(), valid, K.float()
        )

        # --- encode normalized GT pose to 9-d ---
        fov_h_gt, fov_w_gt = fov_from_intrinsics(K, (img_size, img_size))                # (B,) each
        fov_h_gt_t = fov_h_gt[:, None].expand(B, T)                                     # (B, T)
        fov_w_gt_t = fov_w_gt[:, None].expand(B, T)                                     # (B, T)
        gt_pose_enc = world_from_cam_to_pose_enc(pose_normed, fov_h_gt_t, fov_w_gt_t)    # (B, T, 9)

        # --- forward ---
        opt.zero_grad(set_to_none=True)
        # Set LR via cosine warmup
        lr_now = cosine_warmup_lr(step, n_steps, warmup, max_lr)
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            if pose_mode == "gt_replace":
                out = model(rgb, gt_pose_enc=gt_pose_enc)
            else:
                out = model(rgb)

        cams_pred = out["cameras"].float()                                              # (B, T, 9)
        depths_pred = out["depths"].float()                                              # (B, T, H, W)

        # --- pose losses (disabled in gt_replace mode) ---
        if pose_mode == "gt_replace":
            zero = depths_pred.new_zeros(())
            loss_trans = zero
            loss_quat = zero
            loss_fov = zero
        else:
            loss_trans = (cams_pred[..., :3] - gt_pose_enc[..., :3]).abs().mean()
            loss_quat = quaternion_l1_loss(cams_pred[..., 3:7], gt_pose_enc[..., 3:7])
            loss_fov = (cams_pred[..., 7:] - gt_pose_enc[..., 7:]).abs().mean()

        # --- depth L1 ---
        valid_mask = valid.float()
        denom = valid_mask.sum().clamp(min=1.0)
        depth_diff = (depths_pred - depth_normed).abs() * valid_mask
        loss_depth = depth_diff.sum() / denom

        # --- 3D consistency (back-project pred depth via GT pose) ---
        if w_3d > 0:
            loss_3d = multi_view_3d_chamfer(
                depths_pred, valid, pose_normed, K.float(), n_samples=n_3d_samples
            )
        else:
            loss_3d = depths_pred.new_zeros(())

        loss = (
            w_t * loss_trans
            + w_q * loss_quat
            + w_f * loss_fov
            + w_d * loss_depth
            + w_3d * loss_3d
        )

        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        opt.step()
        step += 1

        if step % log_every == 0:
            now = time.time()
            dt = now - last_log
            sps = log_every / max(dt, 1e-6)
            last_log = now
            elapsed = now - t0
            print(
                f"[step {step:>5d}/{n_steps}] "
                f"loss={loss.item():.4f}  "
                f"(t={loss_trans.item():.3f} q={loss_quat.item():.3f} "
                f"fov={loss_fov.item():.3f} d={loss_depth.item():.3f} "
                f"3d={loss_3d.item():.3f})  "
                f"lr={lr_now:.2e}  "
                f"sps={sps:.2f}  "
                f"elapsed={elapsed/60:.1f}m"
            )

        if step % ckpt_every == 0 or step == n_steps:
            ckpt_path = out_dir / "ckpts" / f"step_{step:06d}.pt"
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                 "cfg": cfg},
                ckpt_path,
            )
            print(f"  saved ckpt -> {ckpt_path}")

        if step % eval_every == 0 or step == n_steps:
            ckpt_path = out_dir / "ckpts" / f"step_{step:06d}.pt"
            log_path = out_dir / "eval_logs" / f"step_{step:06d}.log"
            metrics = run_held_out_eval(ckpt_path, cfg, log_path)
            if metrics:
                metrics["step"] = step
                with (out_dir / "eval_logs" / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(metrics) + "\n")
                msg = "  EVAL  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "step")
                print(msg)
            else:
                print(f"  EVAL FAILED — see {log_path}")

    print(f"training done. total time: {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
