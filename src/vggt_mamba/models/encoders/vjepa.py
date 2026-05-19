"""V-JEPA 2.1 ViT-L (RoPE) wrapper for single-image patch features.

V-JEPA is a video model; we feed each image as a 1-frame clip
(B, 3, 1, H, W) so it sees a clip of length 1. Patch grid is H/16 x W/16.

The repo expects `from src.models.vision_transformer import vit_large_rope`,
which assumes `third_party/vjepa2` is on sys.path. We add it locally so we
don't pollute global site-packages with a top-level `src/` module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from .base import EncoderOutput

_REPO = Path(__file__).resolve().parents[4]
_VJEPA = _REPO / "third_party" / "vjepa2"
if str(_VJEPA) not in sys.path:
    sys.path.insert(0, str(_VJEPA))


def _load_vjepa_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    enc = state["encoder"] if isinstance(state, dict) and "encoder" in state else state
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in enc.items()}


class VJEPAEncoder(nn.Module):
    """V-JEPA 2.1 ViT-L (RoPE) frozen feature extractor.

    Input:  (B, 3, H, W) RGB in [0, 1]. H and W should equal `img_size`.
    Output: EncoderOutput with patches (B, N, D), grid = (img_size/16)^2.
    """

    # ImageNet normalization (V-JEPA training default).
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, weights: str | Path, img_size: int = 384, freeze: bool = True):
        super().__init__()
        from src.models.vision_transformer import vit_large_rope  # type: ignore

        self.img_size = img_size
        self.patch_size = 16
        self.tubelet_size = 2
        self.grid = img_size // 16

        # V-JEPA's patch_embed is Conv3d (tubelet_size=2 × patch_size=16 × patch_size=16).
        # We construct as a "video" model with num_frames=2 so the conv kernel matches
        # the checkpoint, then feed each still image as a 2-frame clip (duplicated).
        self.backbone = vit_large_rope(
            patch_size=16, img_size=img_size, num_frames=2, tubelet_size=2
        )
        state = _load_vjepa_state(Path(weights))
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if len(missing) > 0:
            print(f"[vjepa] missing keys: {len(missing)} (first 5: {missing[:5]})")
        if len(unexpected) > 0:
            print(f"[vjepa] unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")

        # infer feature dim by running a tiny dummy forward (2-frame clip)
        with torch.no_grad():
            x = torch.zeros(1, 3, 2, img_size, img_size)
            out = self.backbone(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            self.dim = int(out.shape[-1])

        if freeze:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    @torch.no_grad()
    def _frozen_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W). Duplicate to a 2-frame clip so the Conv3d patch_embed
        # produces 1 temporal token per pair => spatial-only patch grid.
        x = self.normalize(x).unsqueeze(2).expand(-1, -1, 2, -1, -1)  # (B, 3, 2, H, W)
        out = self.backbone(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out  # (B, grid*grid, D)

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        feats = self._frozen_forward(x)
        return EncoderOutput(patches=feats, grid_h=self.grid, grid_w=self.grid, dim=self.dim)


if __name__ == "__main__":
    import os
    root = Path(os.environ.get("VGGT_MAMBA_DATA_ROOT", "/workspace/datasets"))
    w = root / "weights/vjepa2-vitl/vjepa2_1_vitl_dist_vitG_384.pt"
    enc = VJEPAEncoder(w).cuda()
    x = torch.rand(2, 3, 384, 384, device="cuda")
    out = enc(x)
    print(f"[vjepa] patches={tuple(out.patches.shape)} grid={out.grid_h}x{out.grid_w} dim={out.dim}")
