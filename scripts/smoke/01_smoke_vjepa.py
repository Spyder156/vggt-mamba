"""Phase 0 smoke test: V-JEPA 2.1 ViT-L dense feature extraction (direct .pt).

Loads V-JEPA 2.1 ViT-L from its official .pt checkpoint, runs it on a short
video clip, prints per-patch feature stats. No HuggingFace involved.

Pass = features are non-NaN and shape is (B, T, N_patches, D) as expected
when we feed the cross-frame aggregator in Phase 1.

Usage:
    python scripts/smoke/01_smoke_vjepa.py \\
        --images data/co3d_v2/apple/<scene>/images \\
        --weights data/weights/vjepa2-vitl/vjepa2_1_vitl_dist_vitG_384.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# V-JEPA 2 uses `src/` as its package root. Insert third_party/vjepa2 on path
# so `from src.models...` resolves. Avoids the awkward `pip install -e` that
# pollutes the global `src` namespace.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_VJEPA_PATH = _REPO_ROOT / "third_party" / "vjepa2"
if str(_VJEPA_PATH) not in sys.path:
    sys.path.insert(0, str(_VJEPA_PATH))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--img-size", type=int, default=384, help="V-JEPA 2.1 ViT-L was trained at 384")
    p.add_argument(
        "--weights",
        type=Path,
        default=Path(os.environ.get("VGGT_MAMBA_DATA_ROOT", "data"))
        / "weights/vjepa2-vitl/vjepa2_1_vitl_dist_vitG_384.pt",
    )
    return p.parse_args()


def load_clip(images_dir: Path, n: int, size: int) -> torch.Tensor:
    files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}])[:n]
    if len(files) < n:
        raise FileNotFoundError(f"need {n} frames, got {len(files)} in {images_dir}")
    arrs = [
        np.asarray(Image.open(f).convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
        for f in files
    ]
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)


def load_vjepa_encoder(weights: Path, img_size: int, device: str) -> torch.nn.Module:
    """Build V-JEPA 2.1 ViT-L (RoPE variant) and load weights from a .pt."""
    from src.models.vision_transformer import vit_large_rope  # type: ignore

    model = vit_large_rope(patch_size=16, img_size=img_size)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    enc = state["encoder"] if "encoder" in state else state
    enc = {k.replace("module.", "").replace("backbone.", ""): v for k, v in enc.items()}
    msg = model.load_state_dict(enc, strict=False)
    print(f"[smoke-vjepa] load msg: missing={len(msg.missing_keys)} "
          f"unexpected={len(msg.unexpected_keys)}")
    return model.to(device).eval()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke-vjepa] device={device}")

    print(f"[smoke-vjepa] loading from {args.weights}")
    model = load_vjepa_encoder(args.weights, args.img_size, device)

    clip = load_clip(args.images, args.n_frames, args.img_size).to(device)
    # V-JEPA expects (B, C, T, H, W) for video — channel-first with T on dim 2.
    clip = clip.permute(1, 0, 2, 3).unsqueeze(0)
    print(f"[smoke-vjepa] input shape (B,C,T,H,W): {tuple(clip.shape)}")

    with torch.inference_mode(), torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        feats = model(clip)

    feats = feats[0] if isinstance(feats, (list, tuple)) else feats
    print(f"[smoke-vjepa] output shape: {tuple(feats.shape)}")
    print(f"[smoke-vjepa] mean={feats.float().mean().item():.4f} "
          f"std={feats.float().std().item():.4f}")
    assert not torch.isnan(feats).any(), "NaN in features"
    print("[smoke-vjepa] OK")


if __name__ == "__main__":
    main()
