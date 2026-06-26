"""TerraWM conditioned next-latent predictor.

The CrossJEPA mechanism: condition the predictor on the camera motion
between frame t and t+1, so the encoder/state can specialize on scene
structure and let the predictor handle view-change physics.

Inputs:
  state_per_frame: (B, T-1, K_dyn, D)  — current state (dynamic channel)
  motion_t_to_tp1: (B, T-1, 7)         — relative pose (Δt_3, Δq_4)
Output:
  predicted_next: (B, T-1, K_dyn, D)   — predicted next-frame summary tokens

The motion is encoded via a small MLP into the same space as the state,
then concatenated per-token before the predictor MLP. Per-token processing
matches LatentPredictor.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_pose_encoding(motion: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """Sinusoidal encoding of (Δt, Δq) at multiple frequencies.

    motion: (..., 7) [tx, ty, tz, qx, qy, qz, qw]
    Returns: (..., 7 * dim)  — sin+cos at dim/2 frequencies per component.
    """
    *batch_shape, n_dim = motion.shape
    assert n_dim == 7, f"expected 7 motion dims, got {n_dim}"
    # Frequency bands, geometric: 2^0, 2^1, ..., 2^(n_freqs-1).
    n_freqs = dim // 2
    freqs = torch.pow(
        2.0, torch.arange(n_freqs, device=motion.device, dtype=motion.dtype)
    )
    # (..., 7, 1) * (n_freqs,) -> (..., 7, n_freqs)
    scaled = motion.unsqueeze(-1) * freqs                          # (..., 7, n_freqs)
    enc = torch.cat([scaled.sin(), scaled.cos()], dim=-1)          # (..., 7, 2*n_freqs)
    return enc.reshape(*batch_shape, 7 * dim)


class ConditionedNextLatentPredictor(nn.Module):
    """Predict next-frame summary tokens, conditioned on camera motion.

    The state encoder/recurrence is freed from learning view-change physics —
    that information is fed to the predictor explicitly. Encoder learns scene
    structure; predictor handles the camera-relative re-rendering.
    """

    def __init__(self, dim: int, hidden: int = 512, motion_enc_freqs: int = 64):
        super().__init__()
        self.dim = dim
        self.motion_enc_dim = 7 * motion_enc_freqs
        self.motion_freqs = motion_enc_freqs
        # Project motion encoding into the state dim for stable concat.
        self.motion_proj = nn.Sequential(
            nn.Linear(self.motion_enc_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.predictor = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        state_per_frame: torch.Tensor,    # (B, T-1, K_dyn, D)
        motion: torch.Tensor,             # (B, T-1, 7)
    ) -> torch.Tensor:
        """Returns (B, T-1, K_dyn, D) predicted next-frame summary tokens."""
        B, Tm1, K, D = state_per_frame.shape
        # Encode motion → (B, T-1, 7*freqs) → MLP project → (B, T-1, D).
        motion_enc = sinusoidal_pose_encoding(motion.float(), dim=self.motion_freqs)
        motion_feat = self.motion_proj(motion_enc.to(state_per_frame.dtype))   # (B, T-1, D)
        # Broadcast over K tokens.
        motion_feat = motion_feat.unsqueeze(2).expand(B, Tm1, K, D)             # (B, T-1, K, D)
        # Concat with state per token, predict per token.
        cat_in = torch.cat([state_per_frame, motion_feat], dim=-1)              # (B, T-1, K, 2D)
        return self.predictor(cat_in)


if __name__ == "__main__":
    torch.manual_seed(0)
    p = ConditionedNextLatentPredictor(dim=1024).cuda()
    state = torch.randn(2, 7, 8, 1024, device="cuda")
    motion = torch.randn(2, 7, 7, device="cuda") * 0.1
    out = p(state, motion)
    print(f"input state: {tuple(state.shape)}  motion: {tuple(motion.shape)}")
    print(f"output:      {tuple(out.shape)}")
    print(f"params: {sum(t.numel() for t in p.parameters()) / 1e6:.2f}M")

    # Sensitivity check: varying motion should change output.
    motion_zero = torch.zeros_like(motion)
    out_zero = p(state, motion_zero)
    diff = (out - out_zero).abs().mean()
    print(f"sensitivity to motion (||out(motion) - out(0)||): {diff:.4f} (should be non-zero)")
