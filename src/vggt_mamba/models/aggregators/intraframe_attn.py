"""Per-frame self-attention refinement.

Stacked transformer encoder layers that operate independently on each
frame's P patches — no cross-frame mixing. Mirrors VGGT's "frame-attention"
layers; cross-frame mixing happens later via the Mamba block.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IntraFrameTransformer(nn.Module):
    """patches (B, T, P, D) -> refined (B, T, P, D), per-frame self-attn only.

    Supports gradient checkpointing per layer to trade compute for memory —
    activations of each layer are dropped and re-computed during backward.
    Essential on 16 GB GPUs at N=16.
    """

    def __init__(self, dim: int, n_layers: int = 12, n_heads: int = 16,
                 mlp_ratio: float = 4.0, grad_checkpoint: bool = True):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads,
            dim_feedforward=int(dim * mlp_ratio),
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(
                d_model=dim, nhead=n_heads,
                dim_feedforward=int(dim * mlp_ratio),
                activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(n_layers)]
        )
        self.dim = dim
        self.n_layers = n_layers
        self.grad_checkpoint = grad_checkpoint

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        b, t, p, d = patches.shape
        x = patches.reshape(b * t, p, d)
        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x.reshape(b, t, p, d)


if __name__ == "__main__":
    m = IntraFrameTransformer(dim=1024, n_layers=4).cuda()
    x = torch.randn(1, 4, 1024, 1024, device="cuda")
    y = m(x)
    print(f"[intraframe] in {tuple(x.shape)} -> out {tuple(y.shape)}")
    print(f"[intraframe] params (4 layers): "
          f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M")
