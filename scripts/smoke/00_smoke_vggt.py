"""Phase 0 smoke test: VGGT teacher inference (direct .pt load, no HF lib).

Loads the local VGGT-1B checkpoint, runs it on a small batch of frames from
a CO3D-v2 scene, and saves the predicted pointmap as a .ply.

Pass = the model loads, produces non-NaN outputs, and the .ply opens in a
viewer with recognizable geometry.

Usage:
    python scripts/smoke/00_smoke_vggt.py \\
        --images data/co3d_v2/apple/<scene_id>/images \\
        --weights data/weights/vggt/model.pt \\
        --out experiments/smoke/vggt_pointmap.ply
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True, help="dir of frames to feed VGGT")
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("experiments/smoke/vggt_pointmap.ply"))
    p.add_argument(
        "--weights",
        type=Path,
        default=Path(os.environ.get("VGGT_MAMBA_DATA_ROOT", "data")) / "weights/vggt/model.pt",
    )
    return p.parse_args()


def load_images(images_dir: Path, n: int) -> torch.Tensor:
    files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}])[:n]
    if not files:
        raise FileNotFoundError(f"no images in {images_dir}")
    arrs = []
    for f in files:
        img = Image.open(f).convert("RGB").resize((518, 518))
        arrs.append(np.asarray(img, dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)


def save_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = points.shape[0]
    header = (
        f"ply\nformat ascii 1.0\nelement vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("w") as f:
        f.write(header)
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke-vggt] device={device}")

    # `vggt` package = editable install of third_party/vggt (entrypoint.sh does it).
    from vggt.models.vggt import VGGT  # type: ignore

    print(f"[smoke-vggt] instantiating VGGT and loading {args.weights}")
    model = VGGT()
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"[smoke-vggt] load msg: "
          f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    model = model.to(device).eval()

    print(f"[smoke-vggt] loading {args.n_frames} frames from {args.images}")
    images = load_images(args.images, args.n_frames).to(device)

    print("[smoke-vggt] forward")
    with torch.inference_mode(), torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        out = model(images.unsqueeze(0))

    if "pointmaps" not in out:
        print(f"[smoke-vggt] available keys: {list(out.keys())}")
        raise KeyError("expected 'pointmaps' in VGGT output")

    pmap = out["pointmaps"][0, 0].float().cpu().numpy()
    rgb = (images[0].cpu().numpy() * 255).astype(np.uint8)
    pts = pmap.reshape(3, -1).T
    cols = rgb.reshape(3, -1).T

    print(f"[smoke-vggt] saving {pts.shape[0]} points to {args.out}")
    save_ply(pts, cols, args.out)
    print("[smoke-vggt] OK")


if __name__ == "__main__":
    main()
