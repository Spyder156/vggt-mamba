"""Phase 0 GATE: V-JEPA 2.1 vs DINOv2 PCA visualization.

For a given image, extract patch features from both encoders, run PCA on the
patch tokens, map the top 3 components to RGB, save a side-by-side PNG.

Uses the encoder wrappers in `vggt_mamba.models.encoders` so loading is
correct (V-JEPA needs num_frames=2 + tubelet_size=2; DINOv2 needs
block_chunks=0 and init_values for LayerScale).

Output goes to `viz/output/phase0_pca/`. User-only viewing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.models.encoders import DINOv2Encoder, VJEPAEncoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parents[2] / "viz/output/phase0_pca/pca_compare.png",
    )
    p.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("VGGT_MAMBA_DATA_ROOT", "/workspace/datasets")),
    )
    return p.parse_args()


def load_image(path: Path, size: int) -> tuple[np.ndarray, torch.Tensor]:
    pil = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(pil)
    tens = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return arr, tens


def pca_rgb(feats: np.ndarray, h: int, w: int) -> np.ndarray:
    pca = PCA(n_components=3)
    proj = pca.fit_transform(feats)
    proj = (proj - proj.min(axis=0)) / (proj.ptp(axis=0) + 1e-9)
    return proj.reshape(h, w, 3)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[pca] device={device}")

    # Load each encoder at its native resolution.
    vjepa_w = args.data_root / "weights/vjepa2-vitl/vjepa2_1_vitl_dist_vitG_384.pt"
    dinov2_w = args.data_root / "weights/dinov2-large/dinov2_vitl14_pretrain.pth"

    print("[pca] loading V-JEPA 2.1 ViT-L  (384, grid 24)")
    vjepa = VJEPAEncoder(vjepa_w, img_size=384).to(device)
    print("[pca] loading DINOv2 ViT-L/14   (518, grid 37)")
    dinov2 = DINOv2Encoder(dinov2_w, img_size=518).to(device)

    # Display image at common size; resize tensors per encoder native res.
    display_arr, _ = load_image(args.image, 384)
    _, tens_384 = load_image(args.image, 384)
    _, tens_518 = load_image(args.image, 518)

    print("[pca] V-JEPA forward")
    vj_out = vjepa(tens_384.to(device))
    print(f"[pca]   patches={tuple(vj_out.patches.shape)} grid={vj_out.grid_h}x{vj_out.grid_w}")
    vj_feats = vj_out.patches[0].float().cpu().numpy()
    vj_rgb = pca_rgb(vj_feats, vj_out.grid_h, vj_out.grid_w)

    print("[pca] DINOv2 forward")
    dn_out = dinov2(tens_518.to(device))
    print(f"[pca]   patches={tuple(dn_out.patches.shape)} grid={dn_out.grid_h}x{dn_out.grid_w}")
    dn_feats = dn_out.patches[0].float().cpu().numpy()
    dn_rgb = pca_rgb(dn_feats, dn_out.grid_h, dn_out.grid_w)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(display_arr)
    axes[0].set_title("input image")
    axes[0].axis("off")
    axes[1].imshow(vj_rgb)
    axes[1].set_title(f"V-JEPA 2.1 PCA  ({vj_out.grid_h}x{vj_out.grid_w})")
    axes[1].axis("off")
    axes[2].imshow(dn_rgb)
    axes[2].set_title(f"DINOv2 PCA  ({dn_out.grid_h}x{dn_out.grid_w})")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"[pca] saved {args.out}")


if __name__ == "__main__":
    main()
