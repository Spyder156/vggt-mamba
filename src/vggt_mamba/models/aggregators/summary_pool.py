"""Per-frame summary token pooler.

K learnable query tokens cross-attend to each frame's P patch tokens,
producing K compact summary tokens per frame. The Mamba scan then runs
over T*K (rather than T*P) tokens, so state updates per frame instead of
per patch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SummaryTokenPooler(nn.Module):
    """patches (B, T, P, D) -> summaries (B, T, K, D)."""

    def __init__(self, dim: int, n_summary: int = 4, n_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.n_summary = n_summary

        self.queries = nn.Parameter(torch.zeros(1, n_summary, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_out = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        b, t, p, d = patches.shape
        flat_kv = patches.reshape(b * t, p, d)
        q = self.queries.expand(b * t, -1, -1)              # (B*T, K, D)
        attn_out, _ = self.attn(self.norm_q(q), self.norm_kv(flat_kv),
                                self.norm_kv(flat_kv))
        x = q + attn_out
        x = x + self.mlp(self.norm_out(x))
        return x.reshape(b, t, self.n_summary, d)


if __name__ == "__main__":
    pool = SummaryTokenPooler(dim=1024, n_summary=4).cuda()
    x = torch.randn(1, 4, 1024, 1024, device="cuda")
    y = pool(x)
    print(f"[summary_pool] in {tuple(x.shape)} -> out {tuple(y.shape)}")
    print(f"[summary_pool] params: {sum(p.numel() for p in pool.parameters())/1e6:.2f}M")
