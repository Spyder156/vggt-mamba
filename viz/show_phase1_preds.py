"""Dump predicted-vs-GT depth side-by-side for a Phase 1 checkpoint.

Loads a checkpoint, runs the model on a fixed window from the eval split,
and writes a PNG comparing predicted depth (Z channel of pointmap) against
GT depth, plus the RGB frames.

Usage:
    ./docker/run.sh python viz/show_phase1_preds.py \\
        --ckpt experiments/phase1_tokenizer_probe/dinov2/ckpt_004000.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vggt_mamba.data.tum_rgbd import TUMRGBDDataset, unproject_depth_to_pointmap  # noqa: E402
from vggt_mamba.models.mini3r import build_mini3r                                  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data-root", type=Path,
                   default=Path(os.environ.get("VGGT_MAMBA_DATA_ROOT", "/workspace/datasets")) / "tum_rgbd")
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--weights-root", type=Path,
                   default=Path(os.environ.get("VGGT_MAMBA_DATA_ROOT", "/workspace/datasets")) / "weights")
    p.add_argument("--out", type=Path, default=None,
                   help="output PNG path (default: viz/output/<run>/step_<N>_preds.png)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    encoder = cfg["encoder"]
    img_size = 384 if encoder == "vjepa" else 518
    step = ckpt["step"]
    print(f"[viz/preds] loaded ckpt encoder={encoder} step={step}")

    model = build_mini3r(encoder, str(args.weights_root), cfg["model"]["n_xfm_layers"])
    # Load only the trainable params we saved (encoder backbone is frozen).
    msg = model.load_state_dict(ckpt["model"], strict=False)
    print(f"[viz/preds] state load: missing={len(msg.missing_keys)} "
          f"unexpected={len(msg.unexpected_keys)}")
    model = model.to(device).eval()

    ds = TUMRGBDDataset(args.data_root, split="eval", n_frames=cfg["data"]["n_frames"],
                        stride=cfg["data"]["stride_eval"], img_size=img_size)
    s = ds[args.idx]
    rgb = s["rgb"].unsqueeze(0).to(device)        # (1, T, 3, H, W)
    depth_gt = s["depth"].numpy()                  # (T, H, W)
    valid = s["valid"].numpy()

    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        pmap = model(rgb)[0]                       # (T, 3, H, W) in camera frame
    pred_depth = pmap[:, 2].float().cpu().numpy()

    if args.out is None:
        args.out = Path(__file__).resolve().parent / f"output/phase1_{encoder}" \
                   / f"step_{step:06d}_preds.png"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    t = rgb.shape[1]
    fig, axes = plt.subplots(3, t, figsize=(3 * t, 9))
    if t == 1:
        axes = axes[:, None]
    rgb_np = rgb[0].permute(0, 2, 3, 1).cpu().numpy()
    vmax = max(np.nanmax(depth_gt[valid]), float(np.percentile(pred_depth, 99)))
    for i in range(t):
        axes[0, i].imshow(rgb_np[i]); axes[0, i].set_title(f"frame {i}"); axes[0, i].axis("off")
        gt_masked = np.where(valid[i], depth_gt[i], np.nan)
        axes[1, i].imshow(gt_masked, cmap="viridis", vmin=0, vmax=vmax)
        axes[1, i].set_title("GT depth"); axes[1, i].axis("off")
        axes[2, i].imshow(pred_depth[i], cmap="viridis", vmin=0, vmax=vmax)
        axes[2, i].set_title("pred depth"); axes[2, i].axis("off")
    fig.suptitle(f"{encoder} step={step}  seq={s['seq_name']}  idx={args.idx}")
    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"[viz/preds] saved {args.out}")


if __name__ == "__main__":
    main()
