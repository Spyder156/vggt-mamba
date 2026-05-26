"""Phase 3 — train GeoMamba (full architecture) on TUM-RGBD.

Usage:
    ./docker/run.sh python scripts/train/phase3_geomamba.py \\
        --config configs/phase3_geomamba.yaml
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

from vggt_mamba.data.tum_rgbd import TUMRGBDDataset, unproject_depth_to_pointmap  # noqa: E402
from vggt_mamba.losses.multitask import geomamba_loss                              # noqa: E402
from vggt_mamba.models.geomamba import build_geomamba                              # noqa: E402
from vggt_mamba.models.pose_utils import gt_relative_motion_from_abs_poses         # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--data-root", type=Path, default=None)
    return p.parse_args()


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def to_device(batch: dict, device: str) -> dict:
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def gt_pointmap_cam(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """(B, T, H, W), (B, 3, 3) -> (B, T, 3, H, W) gt camera-frame pointmap."""
    b, t = depth.shape[:2]
    K_bt = K.unsqueeze(1).expand(b, t, 3, 3)
    return unproject_depth_to_pointmap(depth, K_bt)


@torch.no_grad()
def _dump_eval_viz(batch, predictions, step: int, viz_dir: Path):
    """One eval batch -> RGB / GT depth / pred depth side-by-side PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    viz_dir.mkdir(parents=True, exist_ok=True)
    rgb = batch["rgb"][0].permute(0, 2, 3, 1).cpu().numpy()
    gt = batch["depth"][0].cpu().numpy()
    v = batch["valid"][0].cpu().numpy()
    pred = predictions["pointmap"][0, :, 2].float().cpu().numpy()
    t = rgb.shape[0]
    # sample at most 8 columns
    if t > 8:
        idx = np.linspace(0, t - 1, 8).round().astype(int)
    else:
        idx = np.arange(t)
    vmax = float(np.nanmax(np.where(v, gt, np.nan)))
    fig, axes = plt.subplots(3, len(idx), figsize=(2.4 * len(idx), 7))
    if len(idx) == 1:
        axes = axes[:, None]
    for k, i in enumerate(idx):
        axes[0, k].imshow(rgb[i]); axes[0, k].axis("off")
        axes[0, k].set_title(f"frame {i}", fontsize=8)
        gtm = np.where(v[i], gt[i], np.nan)
        axes[1, k].imshow(gtm, cmap="viridis", vmin=0, vmax=vmax); axes[1, k].axis("off")
        axes[2, k].imshow(np.clip(pred[i], 0, vmax), cmap="viridis", vmin=0, vmax=vmax)
        axes[2, k].axis("off")
    fig.suptitle(f"GeoMamba  step={step}", fontsize=10)
    plt.tight_layout()
    plt.savefig(viz_dir / f"step_{step:06d}_preds.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 30, viz_dir: Path | None = None,
             step: int = 0) -> dict:
    model.eval()
    terrawm = getattr(model, "terrawm", False)
    abs_rels, cams = [], []
    first = None
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = to_device(batch, device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            preds = model(batch["rgb"], K_intrinsics=batch.get("K"))
        # depth abs-rel mean over the window
        pred_d = preds["pointmap"][0, :, 2].float().cpu()
        gt_d = batch["depth"][0].cpu()
        v = batch["valid"][0].cpu()
        m = v.flatten()
        if m.sum() > 0:
            p = pred_d.flatten()[m].clamp_min(1e-6)
            g = gt_d.flatten()[m].clamp_min(1e-6)
            abs_rels.append(float((p - g).abs().div(g).mean()))
        # Camera L1: in TerraWM mode, predicted output is per-frame delta motion,
        # so we compare to GT per-frame delta motion (not absolute).
        if terrawm:
            gt_delta = gt_relative_motion_from_abs_poses(
                batch["poses_w_c"][:1].float()
            )                                                                  # (1, T, 7)
            cam_pred = preds["camera"][0, :, :3].float().cpu()
            gt_delta_t = gt_delta[0, :, :3].cpu()
            cams.append(float((cam_pred - gt_delta_t).abs().mean()))
        else:
            cam_pred = preds["camera"][0, :, :3].float().cpu()
            cam_gt = batch["camera_gt"][0, :, :3].cpu()
            cams.append(float((cam_pred - cam_gt).abs().mean()))
        if i == 0 and viz_dir is not None:
            first = (batch, preds)
    if first is not None:
        _dump_eval_viz(first[0], first[1], step, viz_dir)
    model.train()
    if not abs_rels:
        return {"abs_rel_mean": float("nan"), "cam_l1_mean": float("nan"), "n": 0}
    return {
        "abs_rel_mean": sum(abs_rels) / len(abs_rels),
        "cam_l1_mean": sum(cams) / len(cams),
        "n": len(abs_rels),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[phase3] device={device} encoder={cfg['encoder']}")

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
    print(f"[phase3] train: {len(train_ds.sequences)} seq, {len(train_ds)} windows")
    print(f"[phase3] eval:  {len(eval_ds.sequences)} seq, {len(eval_ds)} windows")

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=cfg["train"]["num_workers"], pin_memory=True,
                              drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2,
                             pin_memory=True)

    model = build_geomamba(
        cfg["encoder"], str(weights_root),
        n_intraframe_layers=cfg["model"]["n_intraframe_layers"],
        n_summary_tokens=cfg["model"]["n_summary_tokens"],
        n_summary_dynamic=cfg["model"].get("n_summary_dynamic"),
        n_xfm_layers=cfg["model"]["n_xfm_layers"],
        d_state=cfg["model"]["d_state"],
        bidirectional=cfg["model"]["bidirectional"],
        aggregator_name=cfg.get("aggregator", "mamba"),
        track_enabled=cfg["model"]["track_enabled"],
        # Allow inference-time extrapolation up to 16x training N.
        max_frames=max(cfg["data"]["n_frames"] * 16, 256),
        dense_residual_to_patches=cfg["model"].get("dense_residual_to_patches", True),
        predict_next_latent=cfg["model"].get("predict_next_latent", False),
        ema_momentum=cfg["model"].get("ema_momentum", 0.99),
        cross_frame_target=cfg["model"].get("cross_frame_target", "summary"),
        use_anchor_pool=cfg["model"].get("use_anchor_pool", False),
        n_anchors=cfg["model"].get("n_anchors", 32),
        n_anchor_writes=cfg["model"].get("n_anchor_writes", 4),
        anchor_match_threshold=cfg["model"].get("anchor_match_threshold", 0.5),
        terrawm=cfg["model"].get("terrawm", False),
        terrawm_motion_freqs=cfg["model"].get("terrawm_motion_freqs", 64),
    ).to(device)

    # Optional: load a warm-start checkpoint (e.g., the 1b backbone for Exp 2).
    if cfg.get("load_ckpt"):
        warm = torch.load(cfg["load_ckpt"], map_location=device, weights_only=False)
        warm_sd = warm["model"]
        # Resolve size mismatches for frame_embed (longer N at inference/finetune
        # extends it). Partial copy: 0..min(old, new) from ckpt, keep init for rest.
        model_sd = model.state_dict()
        if "frame_embed" in warm_sd and "frame_embed" in model_sd:
            wb = warm_sd["frame_embed"]
            mb = model_sd["frame_embed"]
            if wb.shape != mb.shape:
                n_copy = min(wb.shape[1], mb.shape[1])
                resized = mb.clone()
                resized[:, :n_copy] = wb[:, :n_copy]
                warm_sd["frame_embed"] = resized
                print(f"[phase3] frame_embed: ckpt has {wb.shape[1]} positions, "
                      f"new model has {mb.shape[1]}; copied {n_copy}, "
                      f"{mb.shape[1] - n_copy} kept at init")
        # Drop any other shape-mismatched key (e.g., latent_predictor shape
        # changes between vanilla and TerraWM-conditioned variants) so the
        # new component stays at its random init.
        dropped = []
        for k in list(warm_sd.keys()):
            if k in model_sd and warm_sd[k].shape != model_sd[k].shape:
                dropped.append(k)
                del warm_sd[k]
        if dropped:
            print(f"[phase3] dropped {len(dropped)} shape-mismatched keys "
                  f"(sample: {dropped[:4]})")
        msg = model.load_state_dict(warm_sd, strict=False)
        non_enc_missing = [k for k in msg.missing_keys if not k.startswith("encoder.")]
        print(f"[phase3] warm-start from {cfg['load_ckpt']}: "
              f"missing(non-encoder)={len(non_enc_missing)} unexpected={len(msg.unexpected_keys)}")
        if non_enc_missing:
            print(f"  missing (sample): {non_enc_missing[:5]}")

    # Optional: freeze backbone — only train anchor pool params (Exp 2).
    if cfg.get("freeze_backbone"):
        n_frozen = 0
        n_unfrozen = 0
        for name, p in model.named_parameters():
            if name.startswith("anchor_pool."):
                p.requires_grad_(True)
                n_unfrozen += p.numel()
            else:
                p.requires_grad_(False)
                n_frozen += p.numel()
        print(f"[phase3] freeze_backbone=True: frozen {n_frozen/1e6:.2f}M, "
              f"trainable {n_unfrozen/1e6:.3f}M (anchor pool only)")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[phase3] trainable params: {n_train/1e6:.2f}M")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["optim"]["lr"],
                            betas=tuple(cfg["optim"]["betas"]),
                            weight_decay=cfg["optim"]["weight_decay"])

    def lr_at(step: int) -> float:
        warmup = cfg["optim"]["warmup_steps"]
        if step < warmup:
            return cfg["optim"]["lr"] * (step + 1) / max(warmup, 1)
        return cfg["optim"]["lr"]

    tag = f"{cfg['encoder']}_{cfg.get('aggregator', 'mamba')}"
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
        gt_pmap = gt_pointmap_cam(batch["depth"], batch["K"])

        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        # Optional masked-frame regularizer — randomly zero one frame's RGB
        # so the model has to rely on Mamba state + cross-frame readout to
        # predict that frame's depth. Forces the state to carry useful info.
        mask_p = cfg["loss"].get("mask_frame_prob", 0.0)
        if mask_p > 0 and torch.rand(1).item() < mask_p:
            t_mask = int(torch.randint(0, batch["rgb"].shape[1], (1,)).item())
            batch["rgb"][:, t_mask] = 0.0

        # TerraWM: precompute GT per-frame relative motion (Δt, Δq).
        # Used both as predictor conditioning input AND as camera supervision target.
        terrawm_mode = cfg["model"].get("terrawm", False)
        gt_delta_motion = None
        camera_delta_gt = None
        if terrawm_mode:
            with torch.no_grad():
                gt_delta_motion_7 = gt_relative_motion_from_abs_poses(batch["poses_w_c"].float())
                # (B, T, 7). For the predictor input we take frames 1..T-1 (= motion t→t+1 for t=0..T-2).
                gt_delta_motion = gt_delta_motion_7[:, 1:].contiguous()           # (B, T-1, 7)
                # Build camera_delta_gt = [Δt, Δq, fovx, fovy] per frame.
                # Keep FOV absolute (from original camera_gt).
                fov = batch["camera_gt"][..., 7:]                                  # (B, T, 2)
                camera_delta_gt = torch.cat([gt_delta_motion_7, fov], dim=-1)      # (B, T, 9)

        opt.zero_grad(set_to_none=True)
        with autocast:
            preds = model(
                batch["rgb"],
                K_intrinsics=batch.get("K"),
                train_pose_noise_std_m=cfg["model"].get("train_pose_noise_std_m", 0.0),
                gt_relative_motion=gt_delta_motion,
            )
            targets = {
                "gt_pointmap_cam": gt_pmap,
                "valid": batch["valid"],
                "poses_w_c": batch["poses_w_c"],
                "camera_gt": batch["camera_gt"],
            }
            if terrawm_mode:
                targets["camera_delta_gt"] = camera_delta_gt
            # Pre-registered Experiment 1a loss-weighting: weight each (b, t)
            # pred-loss pair by ‖t_gt(b, t+1) − t_gt(b, t)‖₂ / batch-median.
            # Locked formula — do not tune.
            pmw = None
            if cfg["loss"].get("pred_loss_weighting") == "motion_norm":
                tcam = batch["camera_gt"][..., :3]              # (B, T, 3) translations
                diff = tcam[:, 1:] - tcam[:, :-1]               # (B, T-1, 3)
                norms = diff.norm(dim=-1)                       # (B, T-1)
                med = norms.median().clamp_min(1e-6)
                pmw = norms / med                               # (B, T-1)
            loss, log_dict = geomamba_loss(
                preds, targets,
                w_l1=cfg["loss"]["w_pmap_l1"],
                w_log=cfg["loss"]["w_pmap_log"],
                w_mvc=cfg["loss"]["w_mvc"],
                w_cam=cfg["loss"]["w_cam"],
                w_track=cfg["loss"]["w_track"],
                w_pred=cfg["loss"].get("w_pred", 0.5),
                w_anchor=cfg["loss"].get("w_anchor", 0.5),
                w_vic_var=cfg["loss"].get("w_vic_var", 0.0),
                w_vic_cov=cfg["loss"].get("w_vic_cov", 0.0),
                anchor_threshold=cfg["loss"].get("anchor_threshold", 0.5),
                mvc_samples=cfg["loss"]["mvc_samples"],
                pred_motion_weights=pmw,
                cam_target_key="camera_delta_gt" if terrawm_mode else "camera_gt",
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if hasattr(model, "update_ema_target"):
            model.update_ema_target()
        step += 1

        if step % cfg["train"]["log_every"] == 0 or step == 1:
            elapsed = time.perf_counter() - t_start
            rec = {"step": step, "lr": lr_at(step), "elapsed_s": elapsed, **log_dict}
            pred_s = f"  pred={log_dict['loss_pred']:.4f}" if "loss_pred" in log_dict else ""
            anc_s = f"  anc={log_dict['loss_anchor_consistency']:.4f}" if "loss_anchor_consistency" in log_dict else ""
            print(f"[phase3] step={step:5d}  lr={rec['lr']:.2e}  "
                  f"L1={log_dict['loss_pmap_l1']:.4f}  log={log_dict['loss_pmap_log']:.4f}  "
                  f"mvc={log_dict['loss_mvc']:.4f}  cam={log_dict['loss_cam']:.4f}"
                  f"{pred_s}{anc_s}  total={log_dict['loss_total']:.4f}")
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

        if step % cfg["train"]["eval_every"] == 0 or step == cfg["train"]["steps"]:
            phase_name = Path(cfg["output_dir"]).name
            viz_dir = Path(__file__).resolve().parents[2] / "viz" / "output" / phase_name / tag
            metrics = evaluate(model, eval_loader, device, viz_dir=viz_dir, step=step)
            metrics.update({"step": step, "kind": "eval"})
            print(f"[phase3] EVAL step={step}  abs_rel_mean={metrics['abs_rel_mean']:.4f}  "
                  f"cam_l1_mean={metrics['cam_l1_mean']:.4f}  n={metrics['n']}")
            log_f.write(json.dumps(metrics) + "\n")
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
            print(f"[phase3] saved ckpt step={step}")

    log_f.close()
    print(f"[phase3] done. logs at {out_dir}")


if __name__ == "__main__":
    main()
