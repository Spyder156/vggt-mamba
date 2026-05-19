"""Per-patch readout from the per-frame state tokens.

Each frame's patches cross-attend to that frame's K state tokens. After
the Mamba scan (which propagated info across frames), each frame's state
encodes scene context; this module distributes that context back to all
patches for dense decoding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchStateCrossAttn(nn.Module):
    """patches (B, T, P, D), state_per_frame (B, T, K, D) -> (B, T, P, D).

    `residual_to_patches=True` (default) keeps the original transformer-style
    residual: `dense = patches + cross_attn(patches, state)`.  Safe but lets
    the model route around state when state is uninformative.

    `residual_to_patches=False` removes that residual:
    `dense = cross_attn(patches, state)`.  Patches are still used as the
    *queries* for attention (their spatial identity is preserved), but the
    only information flowing forward is what the cross-attention extracts
    from state.  This forces state to be load-bearing — if state is
    uninformative, quality crashes.  Used in intervention B4.

    FFN residual is always kept (standard transformer post-attention block).
    """

    def __init__(self, dim: int, n_heads: int = 8, residual_to_patches: bool = True):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim),
        )
        self.residual_to_patches = residual_to_patches

    def forward(self, patches: torch.Tensor, state_per_frame: torch.Tensor) -> torch.Tensor:
        b, t, p, d = patches.shape
        k = state_per_frame.shape[2]
        flat_p = patches.reshape(b * t, p, d)
        flat_s = state_per_frame.reshape(b * t, k, d)
        attn_out, _ = self.attn(self.norm_q(flat_p), self.norm_kv(flat_s),
                                self.norm_kv(flat_s))
        x = (flat_p + attn_out) if self.residual_to_patches else attn_out
        x = x + self.mlp(self.norm_ff(x))   # FFN residual always
        return x.reshape(b, t, p, d)


if __name__ == "__main__":
    m = PatchStateCrossAttn(dim=1024).cuda()
    patches = torch.randn(1, 4, 1024, 1024, device="cuda")
    state = torch.randn(1, 4, 4, 1024, device="cuda")
    y = m(patches, state)
    print(f"[state_readout] patches {tuple(patches.shape)}  state {tuple(state.shape)}  "
          f"-> {tuple(y.shape)}")
    print(f"[state_readout] params: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
