"""Phase 1 tokenizer probe: train Mini-3R on TUM, report multi-view consistency.

Run inside the docker container:

    ./docker/run.sh python scripts/train/phase1_tokenizer_probe.py \\
        --config configs/phase1_tokenizer_probe.yaml \\
        --encoder vjepa
    ./docker/run.sh python scripts/train/phase1_tokenizer_probe.py \\
        --config configs/phase1_tokenizer_probe.yaml \\
        --encoder dinov2

Outputs go to <output_dir>/<encoder>/ with metrics.jsonl, args.yaml, ckpt_*.pt.
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

# allow `python scripts/train/...` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.data.tum_rgbd import TUMRGBDDataset, unproject_depth_to_pointmap  # noqa: E402
from vggt_mamba.eval.metrics import multi_view_consistency                        # noqa: E402
from vggt_mamba.losses.pointmap import phase1_loss                                # noqa: E402
from vggt_mamba.models.mini3r import build_mini3r                                 # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--encoder", choices=["vjepa", "dinov2", "dinov3"], default=None,
                   help="override encoder in config")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--steps", type=int, default=None, help="override train.steps")
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
def _dump_pred_viz(rgb, depth_gt, valid, pred_pmap, step: int, viz_dir: Path, encoder: str):
    """Save a side-by-side RGB / GT-depth / pred-depth PNG. User-only viewing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    viz_dir.mkdir(parents=True, exist_ok=True)
    rgb_np = rgb[0].permute(0, 2, 3, 1).cpu().numpy()
    gt = depth_gt[0].cpu().numpy()
    v = valid[0].cpu().numpy()
    pred_d = pred_pmap[0, :, 2].float().cpu().numpy()
    t = rgb_np.shape[0]
    vmax = max(float(np.nanmax(np.where(v, gt, np.nan))), float(np.percentile(pred_d, 99)))
    fig, axes = plt.subplots(3, t, figsize=(3 * t, 9))
    if t == 1:
        axes = axes[:, None]
    for i in range(t):
        axes[0, i].imshow(rgb_np[i]); axes[0, i].axis("off")
        axes[0, i].set_title(f"frame {i}")
        gt_masked = np.where(v[i], gt[i], np.nan)
        axes[1, i].imshow(gt_masked, cmap="viridis", vmin=0, vmax=vmax)
        axes[1, i].axis("off"); axes[1, i].set_title("GT depth")
        axes[2, i].imshow(pred_d[i], cmap="viridis", vmin=0, vmax=vmax)
        axes[2, i].axis("off"); axes[2, i].set_title("pred depth")
    fig.suptitle(f"{encoder}  step={step}")
    plt.tight_layout()
    plt.savefig(viz_dir / f"step_{step:06d}_preds.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    max_batches: int = 50,
    viz_dir: Path | None = None,
    step: int = 0,
    encoder: str = "?",
):
    model.eval()
    mvc_vals = []
    first_batch_cache = None
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = to_device(batch, device)
        pred = model(batch["rgb"])
        mvc = multi_view_consistency(pred, batch["valid"], batch["poses_w_c"], n_samples=2048)
        mvc_vals.append(float(mvc))
        if i == 0 and viz_dir is not None:
            first_batch_cache = (batch["rgb"], batch["depth"], batch["valid"], pred)
    if first_batch_cache is not None:
        rgb, depth_gt, valid, pred = first_batch_cache
        _dump_pred_viz(rgb, depth_gt, valid, pred, step, viz_dir, encoder)
    model.train()
    if not mvc_vals:
        return {"mvc": float("nan"), "n_batches": 0}
    return {
        "mvc_mean": sum(mvc_vals) / len(mvc_vals),
        "mvc_min": min(mvc_vals),
        "mvc_max": max(mvc_vals),
        "n_batches": len(mvc_vals),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.encoder is not None:
        cfg["encoder"] = args.encoder
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps

    torch.manual_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[phase1] device={device} encoder={cfg['encoder']}")

    data_root = args.data_root or Path(cfg["data"]["root"])
    weights_root = Path(os.environ.get("VGGT_MAMBA_DATA_ROOT", "/workspace/datasets")) / "weights"

    # --- Datasets ---
    frame_stride = cfg["data"].get("frame_stride", 10)
    img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]
    train_ds = TUMRGBDDataset(
        data_root,
        split="train",
        n_frames=cfg["data"]["n_frames"],
        stride=cfg["data"]["stride_train"],
        frame_stride=frame_stride,
        img_size=img_size,
        depth_max_m=cfg["data"]["depth_max_m"],
    )
    eval_ds = TUMRGBDDataset(
        data_root,
        split="eval",
        n_frames=cfg["data"]["n_frames"],
        stride=cfg["data"]["stride_eval"],
        frame_stride=frame_stride,
        img_size=img_size,
        depth_max_m=cfg["data"]["depth_max_m"],
    )
    print(f"[phase1] train: {len(train_ds.sequences)} seq, {len(train_ds)} windows")
    print(f"[phase1] eval:  {len(eval_ds.sequences)} seq, {len(eval_ds)} windows")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # --- Model ---
    model = build_mini3r(
        cfg["encoder"],
        weights_root=str(weights_root),
        aggregator_name=cfg.get("aggregator", "attention"),
        n_xfm_layers=cfg["model"]["n_xfm_layers"],
        d_state=cfg.get("d_state", 128),
    ).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[phase1] trainable params: {n_train/1e6:.2f}M "
          f"(encoder={cfg['encoder']}, aggregator={cfg.get('aggregator', 'attention')}"
          f"{', d_state=' + str(cfg.get('d_state', 128)) if cfg.get('aggregator') == 'mamba' else ''})")

    # --- Optimizer ---
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params,
        lr=cfg["optim"]["lr"],
        betas=tuple(cfg["optim"]["betas"]),
        weight_decay=cfg["optim"]["weight_decay"],
    )

    def lr_at(step: int) -> float:
        warmup = cfg["optim"]["warmup_steps"]
        if step < warmup:
            return cfg["optim"]["lr"] * (step + 1) / max(warmup, 1)
        return cfg["optim"]["lr"]

    # --- Output ---
    tag = f"{cfg['encoder']}_{cfg.get('aggregator', 'attention')}"
    out_dir = Path(cfg["output_dir"]) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.yaml").write_text(yaml.safe_dump(cfg))
    log_f = (out_dir / "metrics.jsonl").open("a")

    # --- Train ---
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        cfg["train"]["precision"]
    ]
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
        gt_pmap = gt_pointmap_cam(batch["depth"], batch["K"])  # (B, T, 3, H, W)

        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        opt.zero_grad(set_to_none=True)
        with autocast:
            pred = model(batch["rgb"])
            loss, log_dict = phase1_loss(
                pred,
                gt_pmap,
                batch["valid"],
                batch["poses_w_c"],
                consistency_weight=cfg["loss"]["consistency_weight"],
                consistency_samples=cfg["loss"]["consistency_samples"],
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1

        if step % cfg["train"]["log_every"] == 0 or step == 1:
            elapsed = time.perf_counter() - t_start
            rec = {"step": step, "lr": lr_at(step), "elapsed_s": elapsed, **log_dict}
            print(f"[phase1] step={step:5d}  lr={rec['lr']:.2e}  "
                  f"L1={log_dict['loss_l1']:.4f}  log={log_dict['loss_log']:.4f}  "
                  f"mvc={log_dict['loss_mvc']:.4f}  total={log_dict['loss_total']:.4f}")
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

        if step % cfg["train"]["eval_every"] == 0 or step == cfg["train"]["steps"]:
            # Viz dir derived from output_dir name + encoder+aggregator tag.
            tag = f"{cfg['encoder']}_{cfg.get('aggregator', 'attention')}"
            phase_name = Path(cfg["output_dir"]).name  # e.g. phase1_tokenizer_probe
            viz_dir = Path(__file__).resolve().parents[2] / "viz" / "output" / phase_name / tag
            metrics = evaluate(model, eval_loader, device, viz_dir=viz_dir,
                               step=step, encoder=cfg["encoder"])
            metrics.update({"step": step, "kind": "eval"})
            print(f"[phase1] EVAL step={step}  mvc_mean={metrics.get('mvc_mean'):.4f}  "
                  f"n_batches={metrics['n_batches']}")
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
            print(f"[phase1] saved ckpt step={step}")

    log_f.close()
    print(f"[phase1] done. logs at {out_dir}")


if __name__ == "__main__":
    main()
