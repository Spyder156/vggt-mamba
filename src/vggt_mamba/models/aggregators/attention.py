"""Cross-frame transformer aggregator — Phase 1 baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossFrameTransformer(nn.Module):
    """Plain transformer encoder over all patches of all frames in a window.

    Input/output: (B, T*N, D)
    """

    def __init__(self, dim: int, n_layers: int = 4, n_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=int(dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)
