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
from vggt_mamba.eval.tb_logger import TBLogger                                    # noqa: E402
from vggt_mamba.losses.multitask import terrawm_d_loss                            # noqa: E402
from vggt_mamba.models.heads.bootstrap_depth import gt_per_patch_depth            # noqa: E402
from vggt_mamba.models.pose_utils import gt_relative_motion_from_abs_poses        # noqa: E402
from vggt_mamba.models.terrawm_d import build_terrawm_d                            # noqa: E402


def _train_diagnostics(preds: dict, targets: dict, step: int) -> dict:
    """Per-step training diagnostics that adjudicate cold-start vs bias-integration.

    Grid mass + per-axis edge fractions tell the cold-start story (is the grid
    populating?). Predicted-delta vs GT-delta stats tell the bias story (are
    predictions converging or noise?). Plot these trajectories to read the
    failure mode without waiting 2000 steps and an eval run.

    Requires forward() called with return_voxel_state=True so "voxel_write_mass"
    is in preds.
    """
    rec = {"step": step, "kind": "diag"}
    # Pose delta stats: predicted vs GT.
    pred_t = preds["camera"][..., :3].detach().float().cpu()                  # (B, T, 3)
    gt_t = targets["camera_delta_gt"][..., :3].detach().float().cpu()
    pred_mag = pred_t.norm(dim=-1).flatten()                                  # (B*T,)
    gt_mag = gt_t.norm(dim=-1).flatten()
    bias = (pred_t - gt_t).mean(dim=(0, 1))                                   # (3,) per-axis mean bias
    # Per-axis Pearson correlation pred_dt vs gt_dt across the batch's (B*T) frames.
    # This is the PRE-REGISTERED VERDICT METRIC for the bound+tracking-loss experiment:
    # a pose head that has stopped emitting a lazy constant should show
    # correlation > 0.5 on at least one of the active-motion sequences.
    pred_flat = pred_t.view(-1, 3)                                            # (B*T, 3)
    gt_flat = gt_t.view(-1, 3)
    corr = []
    for ax in range(3):
        p, g = pred_flat[:, ax], gt_flat[:, ax]
        if g.std() > 1e-6 and p.std() > 1e-6:
            c = float(torch.corrcoef(torch.stack([p, g]))[0, 1])
        else:
            c = float("nan")
        corr.append(c)
    # Scale ratio: pred magnitude over GT magnitude. Should approach 1.0 after fix.
    scale_ratio = float(pred_mag.mean() / max(gt_mag.mean().item(), 1e-6))
    rec.update({
        "pred_dt_mean":  float(pred_mag.mean()),
        "pred_dt_p95":   float(pred_mag.quantile(0.95)),
        "gt_dt_mean":    float(gt_mag.mean()),
        "gt_dt_p95":     float(gt_mag.quantile(0.95)),
        "pose_bias_x":   float(bias[0]),
        "pose_bias_y":   float(bias[1]),
        "pose_bias_z":   float(bias[2]),
        "abs_err_per_frame": float((pred_t - gt_t).norm(dim=-1).mean()),
        "pose_corr_x":  corr[0],
        "pose_corr_y":  corr[1],
        "pose_corr_z":  corr[2],
        "pose_scale_ratio": scale_ratio,
    })
    # Grid mass stats (requires return_voxel_state=True on this forward).
    if "voxel_write_mass" in preds:
        wm = preds["voxel_write_mass"]                                         # (B, V_x, V_y, V_z, 1)
        wm = wm[0, ..., 0].float().cpu()                                       # (V_x, V_y, V_z)
        v_x, v_y, v_z = wm.shape
        total = float(wm.sum())
        nonzero = int((wm > 0).sum())
        # Edge mass fractions: mass in outer 10% along each axis / total.
        ex = max(1, v_x // 10)
        ey = max(1, v_y // 10)
        ez = max(1, v_z // 10)
        edge_x = float((wm.sum(dim=(1, 2))[:ex].sum() + wm.sum(dim=(1, 2))[-ex:].sum()) / max(total, 1e-9))
        edge_y = float((wm.sum(dim=(0, 2))[:ey].sum() + wm.sum(dim=(0, 2))[-ey:].sum()) / max(total, 1e-9))
        edge_z = float((wm.sum(dim=(0, 1))[:ez].sum() + wm.sum(dim=(0, 1))[-ez:].sum()) / max(total, 1e-9))
        rec.update({
            "grid_mass_total":    total,
            "grid_nonzero":       nonzero,
            "grid_fill_fraction": nonzero / (v_x * v_y * v_z),
            "grid_edge_frac_x":   edge_x,
            "grid_edge_frac_y":   edge_y,
            "grid_edge_frac_z":   edge_z,
        })
    return rec


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
    train_split = cfg["data"].get("train_seqs") or "train"
    eval_split = cfg["data"].get("eval_seqs") or "eval"
    train_ds = TUMRGBDDataset(
        data_root, split=train_split,
        n_frames=cfg["data"]["n_frames"],
        stride=cfg["data"]["stride_train"],
        frame_stride=cfg["data"]["frame_stride"],
        img_size=img_size,
        depth_max_m=cfg["data"]["depth_max_m"],
        randomize_stride=rand_stride,
    )
    eval_ds = TUMRGBDDataset(
        data_root, split=eval_split,
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
        use_write_confidence=cfg["model"].get("use_write_confidence", False),
        write_confidence_hidden=cfg["model"].get("write_confidence_hidden", 64),
        differentiable_write_geometry=cfg["model"].get("differentiable_write_geometry", False),
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
    tb = TBLogger(out_dir / "tb") if cfg["train"].get("tensorboard", True) else None

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

        # Periodically request voxel state from forward (lightweight: just a
        # detached tensor reference, no extra compute).
        diag_every = cfg["train"].get("diag_every", 100)
        img_every = cfg["train"].get("img_every", 500)
        next_step = step + 1
        want_voxel_state = next_step % diag_every == 0 or step == 0 or next_step % img_every == 0

        opt.zero_grad(set_to_none=True)
        with autocast:
            preds = model(rgb, K_intrinsics=K, gt_poses_w_c=poses, fov=fov,
                          return_voxel_state=want_voxel_state)
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
                pose_tracking=cfg["loss"].get("pose_tracking", False),
                pose_w_l1=cfg["loss"].get("pose_w_l1", 0.1),
                pose_w_rel=cfg["loss"].get("pose_w_rel", 1.0),
                pose_w_cos=cfg["loss"].get("pose_w_cos", 1.0),
                pose_cos_mag_floor_m=cfg["loss"].get("pose_cos_mag_floor_m", 0.002),
                pose_scale_invariant=cfg["loss"].get("pose_scale_invariant", False),
                pose_w_scale_inv=cfg["loss"].get("pose_w_scale_inv", 1.0),
                pose_w_rot=cfg["loss"].get("pose_w_rot", 1.0),
                pose_w_fov=cfg["loss"].get("pose_w_fov", 0.1),
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
            if tb is not None:
                tb.log_scalars("loss", {k: v for k, v in log_dict.items() if k.startswith("loss_")}, step)
                tb.log_scalars("train", {
                    "lr": rec["lr"],
                    "cam_trans": log_dict.get("cam_trans", 0.0),
                    "cam_rot": log_dict.get("cam_rot", 0.0),
                    "cam_rel": log_dict.get("cam_rel", 0.0),
                    "cam_direction": log_dict.get("cam_direction", 0.0),
                    "cam_scale_ratio": log_dict.get("cam_scale_ratio", 0.0),
                    "cam_cos_tracked": log_dict.get("cam_cos_tracked", 0.0),
                    "cam_track_frac": log_dict.get("cam_track_frac", 0.0),
                    # Scale-invariant verdict metrics (live):
                    "cam_scale_inv": log_dict.get("cam_scale_inv", 0.0),
                    "cam_mag_corr": log_dict.get("cam_mag_corr", 0.0),       # PRIMARY VERDICT
                    "cam_pred_mag_mean": log_dict.get("cam_pred_mag_mean", 0.0),  # GUARD against collapse
                    "cam_gt_mag_mean": log_dict.get("cam_gt_mag_mean", 0.0),
                    "depth_mask_coverage": log_dict.get("depth_mask_coverage", 0.0),
                }, step)

        if want_voxel_state:
            diag_rec = _train_diagnostics(preds, targets, step)
            print(f"[d-diag]  step={step:5d}  "
                  f"pred_dt={diag_rec['pred_dt_mean']:.4f}  gt_dt={diag_rec['gt_dt_mean']:.4f}  "
                  f"bias=({diag_rec['pose_bias_x']:+.4f},{diag_rec['pose_bias_y']:+.4f},{diag_rec['pose_bias_z']:+.4f})  "
                  f"grid_mass={diag_rec.get('grid_mass_total', 0):.0f}  "
                  f"grid_nz={diag_rec.get('grid_nonzero', 0)}  "
                  f"edge_z={diag_rec.get('grid_edge_frac_z', 0):.3f}")
            log_f.write(json.dumps(diag_rec) + "\n")
            log_f.flush()
            if tb is not None:
                # Scalars for the diagnostic.
                tb.log_scalars("diag", {k: v for k, v in diag_rec.items()
                                         if isinstance(v, (int, float)) and k not in ("step",)}, step)
                # Histograms: pred/GT pose deltas + bootstrap depth + rendered depth + voxel mass.
                tb.log_histograms("hist", {
                    "pred_dt_x":      preds["camera"][..., 0],
                    "pred_dt_y":      preds["camera"][..., 1],
                    "pred_dt_z":      preds["camera"][..., 2],
                    "gt_dt_x":        targets["camera_delta_gt"][..., 0],
                    "gt_dt_y":        targets["camera_delta_gt"][..., 1],
                    "gt_dt_z":        targets["camera_delta_gt"][..., 2],
                    "pred_dq":        preds["camera"][..., 3:7],
                    "gt_dq":          targets["camera_delta_gt"][..., 3:7],
                    "bootstrap_depth": preds["bootstrap_depth_patch"],
                    "rendered_depth": preds["depth"],
                    "depth_mass":     preds["depth_mass"],
                }, step)
                if "voxel_write_mass" in preds:
                    wm = preds["voxel_write_mass"][..., 0]                              # (B, V_x, V_y, V_z)
                    nz = wm[wm > 0]
                    if nz.numel() > 0:
                        tb.log_histograms("hist", {"voxel_mass_nonzero": nz}, step)

        if want_voxel_state and (next_step % img_every == 0 or step == 0):
            if tb is not None:
                # Heavy: image panels.
                tb.log_train_batch(
                    rgb=rgb, gt_depth=depth, pred_depth=preds["depth"],
                    depth_mask=preds["depth_mask"], depth_mass=preds["depth_mass"],
                    bootstrap_depth_patch=preds["bootstrap_depth_patch"],
                    grid_h=model.grid_h, grid_w=model.grid_w,
                    gt_valid=valid, step=step,
                    depth_max_m=cfg["data"]["depth_max_m"],
                )
                if "voxel_write_mass" in preds:
                    tb.log_voxel_grid(
                        voxel_write_mass=preds["voxel_write_mass"],
                        voxel_bounds=tuple(cfg["model"]["voxel_bounds"]),
                        step=step,
                    )
                tb.log_trajectory(
                    pred_delta_9=preds["camera"],
                    gt_delta_9=targets["camera_delta_gt"],
                    step=step,
                )

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
                    cam_diff = (cam_pred_t - cam_gt_t).abs().mean().item()
                    if not (cam_diff != cam_diff or cam_diff == float("inf")):
                        cams.append(cam_diff)
                    else:
                        n_pred_nan = torch.isnan(cam_pred_t).sum().item()
                        n_gt_nan = torch.isnan(cam_gt_t).sum().item()
                        print(f"[d-train] EVAL batch {i}: cam_l1 non-finite ({cam_diff})  "
                              f"pred_nan={n_pred_nan}  gt_nan={n_gt_nan}  "
                              f"pred_max={cam_pred_t.abs().max():.3f}  gt_max={cam_gt_t.abs().max():.3f}")
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
            if tb is not None:
                tb.log_scalars("eval", {
                    "abs_rel": mean_ar, "cam_l1": mean_cam,
                    "depth_mask_coverage": mean_cov, "n_windows": len(abs_rels),
                }, step)

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
    if tb is not None:
        tb.close()
    print(f"[d-train] done. logs at {out_dir}")


if __name__ == "__main__":
    main()
