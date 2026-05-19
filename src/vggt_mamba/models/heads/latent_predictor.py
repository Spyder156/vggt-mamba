"""Next-frame summary-token predictor.

Given the Mamba state at frame t (= K summary tokens after the Mamba scan),
predict frame (t+1)'s summary tokens. The target comes from an EMA-copy of
the online summary encoder; gradients are stopped on the target side (JEPA
recipe). This is the world-model regularizer: it forces the latent state
to be predictive about future observations, not just a passive read of past.

Predictor is a tiny per-token MLP. Same predictor applies to all K positions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentPredictor(nn.Module):
    """state_t at one frame -> predicted next-frame summary tokens."""

    def __init__(self, dim: int, hidden: int = 512):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, state_per_frame: torch.Tensor) -> torch.Tensor:
        """state_per_frame: (B, T, K, D) -> (B, T, K, D) predicted next-frame summaries."""
        return self.predictor(state_per_frame)


if __name__ == "__main__":
    p = LatentPredictor(dim=1024).cuda()
    x = torch.randn(2, 8, 4, 1024, device="cuda")
    y = p(x)
    print(f"[latent_predictor] in {tuple(x.shape)} -> out {tuple(y.shape)}")
    print(f"[latent_predictor] params: {sum(t.numel() for t in p.parameters())/1e6:.2f}M")
