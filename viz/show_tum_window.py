"""Visualize one TUM-RGBD sliding window: RGB + depth heatmap + valid mask.

Usage:
    ./docker/run.sh python viz/show_tum_window.py
    ./docker/run.sh python viz/show_tum_window.py --idx 100 --img-size 384
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running directly from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vggt_mamba.data.tum_rgbd import TUMRGBDDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path,
                   default=Path(os.environ.get("VGGT_MAMBA_DATA_ROOT", "/workspace/datasets")) / "tum_rgbd")
    p.add_argument("--idx", type=int, default=0, help="window index in train split")
    p.add_argument("--n-frames", type=int, default=4)
    p.add_argument("--frame-stride", type=int, default=10,
                   help="gap between frames inside one window (30 fps source)")
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent / "output/tum_window")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = TUMRGBDDataset(
        args.data_root, split="train", n_frames=args.n_frames,
        stride=8, frame_stride=args.frame_stride, img_size=args.img_size,
    )
    s = ds[args.idx]
    rgb = s["rgb"].permute(0, 2, 3, 1).numpy()   # (T, H, W, 3)
    depth = s["depth"].numpy()                    # (T, H, W)
    valid = s["valid"].numpy()                    # (T, H, W)
    seq = s["seq_name"]
    start = s["start_frame"]

    args.out.mkdir(parents=True, exist_ok=True)

    t = rgb.shape[0]
    fig, axes = plt.subplots(3, t, figsize=(3 * t, 9))
    if t == 1:
        axes = axes[:, None]
    for i in range(t):
        axes[0, i].imshow(rgb[i])
        axes[0, i].set_title(f"frame {start + i} rgb")
        axes[0, i].axis("off")

        masked = np.where(valid[i], depth[i], np.nan)
        im = axes[1, i].imshow(masked, cmap="viridis", vmin=0.0, vmax=5.0)
        axes[1, i].set_title("depth (m)")
        axes[1, i].axis("off")
        plt.colorbar(im, ax=axes[1, i], fraction=0.04)

        axes[2, i].imshow(valid[i], cmap="gray", vmin=0, vmax=1)
        axes[2, i].set_title(f"valid ({valid[i].mean()*100:.1f}%)")
        axes[2, i].axis("off")
    fig.suptitle(f"{seq}  /  idx={args.idx}  /  start={start}")
    plt.tight_layout()

    out_path = args.out / f"window_{args.idx:05d}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"[viz/tum] saved {out_path}")
    print(f"[viz/tum] seq={seq} start={start} n_frames={t} valid={valid.mean()*100:.1f}%")
    print(f"[viz/tum] depth range over valid: "
          f"[{depth[valid].min():.3f}, {depth[valid].max():.3f}] m")


if __name__ == "__main__":
    main()
