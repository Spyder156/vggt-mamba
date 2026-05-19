"""GeoMamba — Phase 3 full architecture.

Frozen encoder → per-frame self-attn → summary tokens → cross-frame
Mamba (optionally bidirectional) → multi-task heads (camera + dense
pointmap + optional track).

Same encoder zoo as Mini-3R (DINOv3/DINOv2/V-JEPA). Mini-3R is kept for
back-compat with Phase 1/2 work; GeoMamba is the production model.

Forward returns a dict:
    {
        "camera":    (B, T, 9)    [tx,ty,tz, qx,qy,qz,qw, fovx,fovy]
        "pointmap":  (B, T, 3, H, W)
        "tracks":    (B, T, 2)    only if track_xy + track_frame given
    }
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from .aggregators import (
    CrossFrameMamba,
    CrossFrameTransformer,
    IntraFrameTransformer,
    PatchStateCrossAttn,
    SummaryTokenPooler,
)
from .encoders import DINOv2Encoder, DINOv3Encoder, EncoderOutput, VJEPAEncoder
from .heads.camera import CameraHead
from .heads.dpt import PointmapHead as DPTUpsampleHead
from .heads.latent_predictor import LatentPredictor
from .heads.track import TrackHead


class GeoMamba(nn.Module):
    """Full streaming 3D reconstruction model."""

    def __init__(
        self,
        encoder: VJEPAEncoder | DINOv2Encoder | DINOv3Encoder,
        n_intraframe_layers: int = 12,
        n_summary_tokens: int = 4,
        n_xfm_layers: int = 12,
        d_state: int = 128,
        bidirectional: bool = True,
        aggregator_name: Literal["mamba", "attention"] = "mamba",
        head_hidden: int = 256,
        track_enabled: bool = True,
        max_frames: int = 256,
        dense_residual_to_patches: bool = True,
        predict_next_latent: bool = False,
        ema_momentum: float = 0.99,
    ):
        super().__init__()
        self.encoder = encoder
        self.dim = encoder.dim
        self.img_size = encoder.img_size
        self.grid_h = encoder.grid
        self.grid_w = encoder.grid
        self.n_summary = n_summary_tokens

        # 1. Per-frame self-attention refinement.
        self.intraframe = IntraFrameTransformer(dim=self.dim, n_layers=n_intraframe_layers)

        # 2. Summary token pooling.
        self.summary_pool = SummaryTokenPooler(dim=self.dim, n_summary=n_summary_tokens)

        # 3. Frame-position embedding for the Mamba scan.
        self.frame_embed = nn.Parameter(torch.zeros(1, max_frames, 1, self.dim))
        nn.init.trunc_normal_(self.frame_embed, std=0.02)

        # 4. Cross-frame aggregator: Mamba (default) or attention (ablation).
        self.aggregator_name = aggregator_name
        if aggregator_name == "mamba":
            self.cross_frame = CrossFrameMamba(
                dim=self.dim, n_layers=n_xfm_layers, d_state=d_state,
                bidirectional=bidirectional,
            )
        elif aggregator_name == "attention":
            self.cross_frame = CrossFrameTransformer(
                dim=self.dim, n_layers=n_xfm_layers, n_heads=8,
            )
        else:
            raise ValueError(f"unknown aggregator {aggregator_name!r}")

        # 5. Heads.
        self.camera_head = CameraHead(dim=self.dim, hidden=head_hidden)
        # Dense head = patches read from state, then DPT upsample to pointmap.
        self.dense_readout = PatchStateCrossAttn(
            dim=self.dim, n_heads=8,
            residual_to_patches=dense_residual_to_patches,
        )
        self.dpt = DPTUpsampleHead(
            in_dim=self.dim, hidden=head_hidden, out_size=self.img_size,
        )
        self.track_head: TrackHead | None = TrackHead(dim=self.dim, hidden=head_hidden) \
            if track_enabled else None

        # 6. Optional world-model regularizer: predict next-frame summary tokens
        # from the current state. Targets come from EMA copies of the online
        # intraframe + summary encoders, with stopgrad — JEPA recipe to avoid
        # collapse.
        self.predict_next_latent = predict_next_latent
        self.ema_momentum = ema_momentum
        if predict_next_latent:
            self.latent_predictor = LatentPredictor(dim=self.dim, hidden=head_hidden * 2)
            # EMA-target copies of intraframe + summary_pool. requires_grad=False
            # and .eval() so they're never updated by SGD and never use grad_ckpt.
            self.target_intraframe = copy.deepcopy(self.intraframe)
            self.target_summary_pool = copy.deepcopy(self.summary_pool)
            for p in self.target_intraframe.parameters():
                p.requires_grad_(False)
            for p in self.target_summary_pool.parameters():
                p.requires_grad_(False)
            self.target_intraframe.eval()
            self.target_summary_pool.eval()
        else:
            self.latent_predictor = None
            self.target_intraframe = None
            self.target_summary_pool = None

    def forward(
        self,
        rgb: torch.Tensor,
        track_xy: torch.Tensor | None = None,
        track_frame: int | None = None,
    ) -> dict[str, torch.Tensor]:
        b, t, _, h, w = rgb.shape
        assert h == self.img_size and w == self.img_size, \
            f"expected {self.img_size}, got {h}x{w}"

        # 1. Encoder (frozen, per-frame).
        flat_rgb = rgb.reshape(b * t, 3, h, w)
        with torch.no_grad():
            enc_out: EncoderOutput = self.encoder(flat_rgb)
        patches = enc_out.patches.reshape(b, t, -1, self.dim)            # (B, T, P, D)

        # 2. Per-frame self-attn refinement.
        refined = self.intraframe(patches)                                # (B, T, P, D)

        # 3. Summary tokens per frame.
        summaries = self.summary_pool(refined)                            # (B, T, K, D)

        # 4. Add frame embed, flatten, scan with Mamba.
        summaries = summaries + self.frame_embed[:, :t]                   # broadcasts over K
        seq = summaries.reshape(b, t * self.n_summary, self.dim)
        state_seq = self.cross_frame(seq)                                 # (B, T*K, D)
        state_per_frame = state_seq.reshape(b, t, self.n_summary, self.dim)

        # 5a. Camera head — state-only.
        cam = self.camera_head(state_per_frame)                           # (B, T, 9)

        # 5b. Dense (pointmap) head — patches read from state, then DPT.
        dense_in = self.dense_readout(refined, state_per_frame)           # (B, T, P, D)
        # reshape to spatial grid for DPT, then chunk over frames so the
        # bilinear upsample stays under PyTorch's INT_MAX per-call limit.
        grid = dense_in.reshape(b * t, -1, self.dim).transpose(1, 2)
        grid = grid.reshape(b * t, self.dim, self.grid_h, self.grid_w)
        chunk = 8
        pmap_chunks = [self.dpt(grid[i:i + chunk]) for i in range(0, b * t, chunk)]
        pmap = torch.cat(pmap_chunks, dim=0).reshape(b, t, 3, h, w)

        out = {"camera": cam, "pointmap": pmap}

        # 5c. Optional track head.
        if self.track_head is not None and track_xy is not None and track_frame is not None:
            out["tracks"] = self.track_head(track_xy, track_frame, state_per_frame)

        # 5d. World-model regularizer: predict next-frame summary tokens from state.
        # Target = EMA-target encoder output on the same patches, stopgrad.
        # Loss is consumed by the train script via geomamba_loss.
        if self.predict_next_latent and self.latent_predictor is not None and t >= 2:
            with torch.no_grad():
                tgt_refined = self.target_intraframe(patches)                # (B, T, P, D)
                tgt_summaries = self.target_summary_pool(tgt_refined)        # (B, T, K, D)
            # For each t in 0..T-2: predict frame (t+1)'s summary from state_t.
            predicted_next = self.latent_predictor(state_per_frame[:, :-1])  # (B, T-1, K, D)
            target_next = tgt_summaries[:, 1:].detach()                       # (B, T-1, K, D)
            out["predicted_next"] = predicted_next
            out["target_next"] = target_next

        return out

    @torch.no_grad()
    def update_ema_target(self) -> None:
        """Move the EMA-target encoders toward the online encoders.
        Call once after every optimizer step when training with predict_next_latent.
        """
        if not self.predict_next_latent:
            return
        m = self.ema_momentum
        for p_online, p_target in zip(self.intraframe.parameters(),
                                      self.target_intraframe.parameters()):
            p_target.data.mul_(m).add_(p_online.data, alpha=1.0 - m)
        for p_online, p_target in zip(self.summary_pool.parameters(),
                                      self.target_summary_pool.parameters()):
            p_target.data.mul_(m).add_(p_online.data, alpha=1.0 - m)

    # ---------- streaming inference ----------

    @torch.no_grad()
    def init_streaming_state(self, batch_size: int = 1, dtype=torch.bfloat16,
                             device="cuda") -> list[dict]:
        """Zero-initialized per-layer Mamba state for streaming inference.

        Only valid for causal-only models (bidirectional=False).
        """
        if not isinstance(self.cross_frame, CrossFrameMamba):
            raise RuntimeError("streaming_forward requires a Mamba cross-frame aggregator")
        return self.cross_frame.init_streaming_state(
            batch_size=batch_size, dtype=dtype, device=device,
        )

    @torch.no_grad()
    def streaming_forward(
        self,
        rgb_frame: torch.Tensor,
        state: list[dict],
        frame_idx: int,
        autocast_dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[dict[str, torch.Tensor], list[dict]]:
        """Process one frame, return predictions for that frame + updated state.

        rgb_frame: (1, 3, H, W).
        state: from `init_streaming_state` or a previous `streaming_forward` return.
        frame_idx: which frame in the global stream (for the frame embedding).

        Returns ({camera, pointmap}, new_state). Memory cost should be roughly
        constant in the number of frames processed so far — the only thing
        that depends on history is `state`, which is fixed-size.
        """
        assert rgb_frame.dim() == 4 and rgb_frame.shape[0] == 1, \
            f"streaming_forward expects (1, 3, H, W); got {tuple(rgb_frame.shape)}"
        h, w = rgb_frame.shape[-2:]

        with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
            # 1. Encoder on this single frame.
            with torch.no_grad():
                enc_out = self.encoder(rgb_frame)                            # (1, P, D)
            patches = enc_out.patches.unsqueeze(0).to(autocast_dtype)        # (1, 1, P, D)

            # 2. Per-frame self-attn refinement.
            refined = self.intraframe(patches)                               # (1, 1, P, D)

            # 3. Summary tokens for this frame.
            summaries = self.summary_pool(refined)                           # (1, 1, K, D)

            # 4. Frame embed + advance Mamba state by K tokens.
            if frame_idx < self.frame_embed.shape[1]:
                summaries = summaries + self.frame_embed[:, frame_idx:frame_idx + 1]
            tokens = summaries.reshape(1, self.n_summary, self.dim).to(autocast_dtype)
            state_seq, state = self.cross_frame.streaming_step(tokens, state)
            state_per_frame = state_seq.unsqueeze(1)                         # (1, 1, K, D)

            # 5a. Camera head.
            cam = self.camera_head(state_per_frame)                          # (1, 1, 9)

            # 5b. Dense head.
            dense_in = self.dense_readout(refined, state_per_frame)          # (1, 1, P, D)
            grid = dense_in.reshape(1, -1, self.dim).transpose(1, 2)
            grid = grid.reshape(1, self.dim, self.grid_h, self.grid_w)
            pmap = self.dpt(grid).reshape(1, 1, 3, h, w)

        return {"camera": cam, "pointmap": pmap}, state


def build_geomamba(
    encoder_name: Literal["vjepa", "dinov2", "dinov3"],
    weights_root: str,
    n_intraframe_layers: int = 12,
    n_summary_tokens: int = 4,
    n_xfm_layers: int = 12,
    d_state: int = 128,
    bidirectional: bool = True,
    aggregator_name: Literal["mamba", "attention"] = "mamba",
    track_enabled: bool = True,
    max_frames: int = 256,
    dense_residual_to_patches: bool = True,
    predict_next_latent: bool = False,
    ema_momentum: float = 0.99,
) -> GeoMamba:
    weights_root = Path(weights_root)
    if encoder_name == "dinov3":
        enc = DINOv3Encoder(
            repo_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
            img_size=512, freeze=True,
        )
    elif encoder_name == "dinov2":
        enc = DINOv2Encoder(
            weights_root / "dinov2-large/dinov2_vitl14_pretrain.pth",
            img_size=518, freeze=True,
        )
    elif encoder_name == "vjepa":
        enc = VJEPAEncoder(
            weights_root / "vjepa2-vitl/vjepa2_1_vitl_dist_vitG_384.pt",
            img_size=384, freeze=True,
        )
    else:
        raise ValueError(f"unknown encoder {encoder_name!r}")

    return GeoMamba(
        enc,
        n_intraframe_layers=n_intraframe_layers,
        n_summary_tokens=n_summary_tokens,
        n_xfm_layers=n_xfm_layers,
        d_state=d_state,
        bidirectional=bidirectional,
        aggregator_name=aggregator_name,
        track_enabled=track_enabled,
        max_frames=max_frames,
        dense_residual_to_patches=dense_residual_to_patches,
        predict_next_latent=predict_next_latent,
        ema_momentum=ema_momentum,
    )


if __name__ == "__main__":
    import os
    root = os.environ.get("VGGT_MAMBA_DATA_ROOT", "/workspace/datasets") + "/weights"
    # Small variant for smoke test
    m = build_geomamba(
        "dinov3", root,
        n_intraframe_layers=2, n_xfm_layers=2,
        track_enabled=True,
    ).cuda()
    s = m.img_size
    x = torch.rand(1, 4, 3, s, s, device="cuda")
    q = torch.rand(1, 2, device="cuda")
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = m(x, track_xy=q, track_frame=2)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"[geomamba:dinov3 small] trainable {n_train/1e6:.2f}M")
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
