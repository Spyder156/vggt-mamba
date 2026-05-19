"""Track head — query-based 2D point trajectory across frames.

Given a query 2D image location `(u, v) ∈ [0, 1]²` inside frame `query_frame`,
predict the corresponding 2D location of that physical 3D point in every
other frame.

The query is encoded by combining its 2D coordinate with the pooled state
of its frame, projected to D dims. Then in parallel across all frames, the
query cross-attends to each frame's K state tokens and an MLP predicts the
2D output.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TrackHead(nn.Module):
    """Predict 2D tracks across all T frames given a query point in one frame."""

    def __init__(self, dim: int, n_heads: int = 8, hidden: int = 256):
        super().__init__()
        self.query_proj = nn.Linear(2 + dim, dim)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(
        self,
        query_xy: torch.Tensor,           # (B, 2) in [0,1]
        query_frame: int,                  # which frame the query is from
        state_per_frame: torch.Tensor,    # (B, T, K, D)
    ) -> torch.Tensor:
        b, t, k, d = state_per_frame.shape

        # 1) Build a query embedding from (2D coord, frame state).
        pooled_qf = state_per_frame[:, query_frame].mean(dim=1)        # (B, D)
        q_in = torch.cat([query_xy, pooled_qf], dim=-1)                # (B, 2 + D)
        query = self.query_proj(q_in)                                  # (B, D)

        # 2) Broadcast the same query across all T frames; cross-attend to each
        #    frame's K state tokens in parallel.
        q_per_frame = query.unsqueeze(1).expand(b, t, d).unsqueeze(2)  # (B, T, 1, D)
        q_flat = q_per_frame.reshape(b * t, 1, d)
        kv_flat = state_per_frame.reshape(b * t, k, d)

        attn_out, _ = self.attn(self.norm_q(q_flat),
                                self.norm_kv(kv_flat),
                                self.norm_kv(kv_flat))
        x = q_flat + attn_out
        x = self.norm_ff(x).reshape(b, t, d)
        tracks = self.mlp(x)                                            # (B, T, 2)
        return tracks


if __name__ == "__main__":
    h = TrackHead(dim=1024).cuda()
    s = torch.randn(1, 8, 4, 1024, device="cuda")
    q = torch.rand(1, 2, device="cuda")
    y = h(q, query_frame=3, state_per_frame=s)
    print(f"[track_head] query (1,2) + frame_idx + state {tuple(s.shape)} -> {tuple(y.shape)}")
    print(f"[track_head] params: {sum(p.numel() for p in h.parameters())/1e6:.2f}M")
