"""Train TerraWM-D (voxel-grid world model) on TUM-RGBD.

Usage:
    ./docker/run.sh python scripts/train/phase4_terrawm_d.py \\
        --config configs/phase4_terrawm_d.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.data.tum_rgbd import TUMRGBDDataset                              # noqa: E402
from vggt_mamba.losses.multitask import terrawm_d_loss                            # noqa: E402
from vggt_mamba.models.heads.bootstrap_depth import gt_per_patch_depth            # noqa: E402
from vggt_mamba.models.pose_utils import gt_relative_motion_from_abs_poses        # noqa: E402
from vggt_mamba.models.terrawm_d import build_terrawm_d                            # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--data-root", type=Path, default=None)
    return p.parse_args()


def to_device(batch: dict, device: str) -> dict:
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[d-train] device={device} encoder={cfg['encoder']}")

    data_root = args.data_root or Path(cfg["data"]["root"])
    weights_root = Path(os.environ.get("VGGT_MAMBA_DATA_ROOT", "/workspace/datasets")) / "weights"

    img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]
    rand_stride = cfg["data"].get("randomize_stride")
    if rand_stride is not None:
        rand_stride = tuple(rand_stride)
    train_ds = TUMRGBDDataset(
        data_root, split="train",
        n_frames=cfg["data"]["n_frames"],
        stride=cfg["data"]["stride_train"],
        frame_stride=cfg["data"]["frame_stride"],
        img_size=img_size,
        depth_max_m=cfg["data"]["depth_max_m"],
        randomize_stride=rand_stride,
    )
    eval_ds = TUMRGBDDataset(
        data_root, split="eval",
        n_frames=cfg["data"]["n_frames"],
        stride=cfg["data"]["stride_eval"],
        frame_stride=cfg["data"]["frame_stride"],
        img_size=img_size,
        depth_max_m=cfg["data"]["depth_max_m"],
    )
    print(f"[d-train] train: {len(train_ds.sequences)} seq, {len(train_ds)} windows  "
          f"eval: {len(eval_ds.sequences)} seq, {len(eval_ds)} windows")

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=cfg["train"]["num_workers"], pin_memory=True,
                              drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2,
                             pin_memory=True)

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
    ).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[d-train] trainable params: {n_train/1e6:.2f}M  "
          f"voxel grid: {model.voxel_cfg.resolution} × {model.voxel_cfg.feature_dim}-dim")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["optim"]["lr"],
                            betas=tuple(cfg["optim"]["betas"]),
                            weight_decay=cfg["optim"]["weight_decay"])

    def lr_at(step: int) -> float:
        warmup = cfg["optim"]["warmup_steps"]
        if step < warmup:
            return cfg["optim"]["lr"] * (step + 1) / max(warmup, 1)
        return cfg["optim"]["lr"]

    tag = f"{cfg['encoder']}_terrawm_d"
    out_dir = Path(cfg["output_dir"]) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.yaml").write_text(yaml.safe_dump(cfg))
    log_f = (out_dir / "metrics.jsonl").open("a")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[cfg["train"]["precision"]]
    autocast = torch.amp.autocast(device_type="cuda", dtype=dtype)

    step = 0
    train_iter = iter(train_loader)
    t_start = time.perf_counter()

    while step < cfg["train"]["steps"]:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = to_device(batch, device)
        rgb = batch["rgb"]                                   # (B, T, 3, H, W)
        poses = batch["poses_w_c"].float()                   # (B, T, 4, 4)
        depth = batch["depth"]                               # (B, T, H, W)
        valid = batch["valid"]
        K = batch["K"]                                       # (B, 3, 3)

        # GT per-patch depth (for bootstrap loss).
        with torch.no_grad():
            patch_d, patch_v = gt_per_patch_depth(depth, model.grid_h, model.grid_w)
            # GT delta motion + fov for pose supervision.
            gt_delta_7 = gt_relative_motion_from_abs_poses(poses)
            fov = batch["camera_gt"][..., 7:]                # (B, T, 2)
            camera_delta_gt = torch.cat([gt_delta_7, fov], dim=-1)

        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        opt.zero_grad(set_to_none=True)
        with autocast:
            preds = model(rgb, K_intrinsics=K, gt_poses_w_c=poses, fov=fov)
            targets = {
                "gt_depth_full": depth,
                "valid": valid,
                "gt_depth_patch": patch_d,
                "gt_depth_patch_valid": patch_v,
                "poses_w_c": poses,
                "camera_delta_gt": camera_delta_gt,
            }
            loss, log_dict = terrawm_d_loss(
                preds, targets,
                w_render_l1=cfg["loss"]["w_render_l1"],
                w_render_log=cfg["loss"]["w_render_log"],
                w_bootstrap=cfg["loss"]["w_bootstrap"],
                w_pose=cfg["loss"]["w_pose"],
                w_mvc=cfg["loss"]["w_mvc"],
                mvc_samples=cfg["loss"]["mvc_samples"],
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1

        if step % cfg["train"]["log_every"] == 0 or step == 1:
            elapsed = time.perf_counter() - t_start
            rec = {"step": step, "lr": lr_at(step), "elapsed_s": elapsed, **log_dict}
            print(f"[d-train] step={step:5d}  lr={rec['lr']:.2e}  "
                  f"render_l1={log_dict['loss_render_l1']:.4f}  "
                  f"bootstrap={log_dict['loss_bootstrap']:.4f}  "
                  f"pose={log_dict['loss_pose']:.4f}  "
                  f"mask_cov={log_dict['depth_mask_coverage']:.2f}  "
                  f"total={log_dict['loss_total']:.4f}")
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

        if step % cfg["train"]["eval_every"] == 0 or step == cfg["train"]["steps"]:
            model.eval()
            abs_rels, cams, covs = [], [], []
            with torch.no_grad():
                for i, eb in enumerate(eval_loader):
                    if i >= 20:
                        break
                    eb = to_device(eb, device)
                    with autocast:
                        ep = model(eb["rgb"], K_intrinsics=eb["K"],
                                   gt_poses_w_c=eb["poses_w_c"].float(),
                                   fov=eb["camera_gt"][..., 7:])
                    # abs_rel: per-pixel rendered-depth vs GT, masked on intersection of
                    # depth_mask AND GT valid.
                    pred_d = ep["depth"][0].float()                          # (T, H, W)
                    gt_d = eb["depth"][0]
                    v = eb["valid"][0]
                    m = (ep["depth_mask"][0] & v).flatten()
                    if m.sum() > 0:
                        p = pred_d.flatten()[m].clamp_min(1e-6)
                        g = gt_d.flatten()[m].clamp_min(1e-6)
                        abs_rels.append(float((p - g).abs().div(g).mean()))
                    covs.append(float(ep["depth_mask"][0].float().mean()))
                    # Pose: delta cam_l1
                    pgt_delta = gt_relative_motion_from_abs_poses(eb["poses_w_c"][:1].float())
                    cam_pred_t = ep["camera"][0, :, :3].float().cpu()
                    cam_gt_t = pgt_delta[0, :, :3].cpu()
                    cams.append(float((cam_pred_t - cam_gt_t).abs().mean()))
            model.train()
            mean_ar = sum(abs_rels) / max(len(abs_rels), 1)
            mean_cam = sum(cams) / max(len(cams), 1)
            mean_cov = sum(covs) / max(len(covs), 1)
            ev = {"step": step, "kind": "eval", "abs_rel_mean": mean_ar,
                  "cam_l1_mean": mean_cam, "depth_mask_coverage": mean_cov,
                  "n": len(abs_rels)}
            print(f"[d-train] EVAL step={step}  abs_rel={mean_ar:.4f}  "
                  f"cam_l1={mean_cam:.4f}  mask_cov={mean_cov:.2f}  n={len(abs_rels)}")
            log_f.write(json.dumps(ev) + "\n")
            log_f.flush()

        if step % cfg["train"]["ckpt_every"] == 0 or step == cfg["train"]["steps"]:
            ckpt = {
                "step": step,
                "model": {k: v.cpu() for k, v in model.state_dict().items()
                          if "encoder.backbone" not in k},
                "opt": opt.state_dict(),
                "config": cfg,
            }
            torch.save(ckpt, out_dir / f"ckpt_{step:06d}.pt")
            print(f"[d-train] saved ckpt step={step}")

    log_f.close()
    print(f"[d-train] done. logs at {out_dir}")


if __name__ == "__main__":
    main()
