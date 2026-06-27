"""TerraWM-Linear — streaming patch-token aggregator with cross-attention.

Architecture inspired by VGGT but with a FIXED-SIZE LATENT STATE in place of
VGGT's all-to-all cross-frame "global" attention. This trades quadratic
cross-attention over frames for O(T) per-frame cost with constant memory.

For each frame t in the sequence:
    1. DINOv3 patch embed (frozen) → (B, P, D_enc)
    2. Project D_enc → D_model + add 2D patch-position embedding
    3. Add frame-index embedding
    4. WRITE: cross-attn latents (Q) ← patches (K, V), then self-attn within
       latents, repeated n_write_blocks times. The latent state is updated
       in place.

After ingesting frames (or at any point during streaming), we DECODE:
    5. Per-frame camera query → cross-attn over latents → camera head →
       9-d pose enc (translation 3 + quaternion 4 + FoV 2), VGGT-style.
    6. Per-patch queries (one per output patch position, per frame) → cross-
       attn over latents → depth head → per-pixel depth via conv upsample.

Path A scale handling: model outputs are AT WHATEVER SCALE TRAINING TARGETS
HAVE — we apply VGGT-style mean-point-distance normalization to GT before
computing loss (see eval/normalize.py + losses.py to be added), so the model
learns to predict in normalized space and we Sim(3)-align at eval.

Notes:
    - V1 omits a separate point-map head; depth + camera + intrinsics is
      enough to back-project, matching what we use from VGGT.
    - V1 uses learned 2D pos embeddings and learned frame-index embeddings.
      Can swap to 2D RoPE / sinusoidal later for length generalization.
    - Cheat-pose mechanism not wired yet; in V1 the pose head is purely an
      output, not used downstream. We'll add cheat-pose when we wire pose
      back into the WRITE step (e.g., to bias geometry queries).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.dinov3 import DINOv3Encoder


# ---------- small building blocks ----------------------------------------------

class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, out_dim: int | None = None):
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class CrossAttnBlock(nn.Module):
    """q_seq <- kv_seq via cross-attention + MLP. Pre-norm."""
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv),
                                  need_weights=False)
        q = q + attn_out
        q = q + self.mlp(self.norm_mlp(q))
        return q


class SelfAttnBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_attn(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm_mlp(x))
        return x


# ---------- WRITE step ----------------------------------------------------------

class WriteBlock(nn.Module):
    """One write block: cross-attn (latents ← patches) + self-attn on latents."""
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.cross = CrossAttnBlock(dim, n_heads)
        self.self_attn = SelfAttnBlock(dim, n_heads)

    def forward(self, latents: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        latents = self.cross(latents, patches)
        latents = self.self_attn(latents)
        return latents


# ---------- DECODE step ---------------------------------------------------------

class DecodeBlock(nn.Module):
    """One decode block: cross-attn (query ← latents) + MLP."""
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.cross = CrossAttnBlock(dim, n_heads)

    def forward(self, query: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        return self.cross(query, latents)


# ---------- heads ---------------------------------------------------------------

class CameraHead(nn.Module):
    """1 query token per frame → 9-d pose enc [tx, ty, tz, qx, qy, qz, qw, fov_h, fov_w]."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.head = MLP(dim, dim, 9)

    def forward(self, query_out: torch.Tensor) -> torch.Tensor:
        # query_out: (B, T, D)
        raw = self.head(self.norm(query_out))                                         # (B, T, 9)
        t = raw[..., :3]
        q = F.normalize(raw[..., 3:7], dim=-1)
        fov = F.softplus(raw[..., 7:])                                                # keep FoV positive
        return torch.cat([t, q, fov], dim=-1)


class DepthHead(nn.Module):
    """Per-patch query outputs → upsample → per-pixel depth.

    V1 is intentionally lightweight: MLP per token, reshape to (B*T, hidden, gh, gw),
    then a small conv stack + bilinear upsample. Real DPT can replace this later.
    """
    def __init__(self, dim: int, grid: int, img_size: int, hidden: int = 256):
        super().__init__()
        self.grid = grid
        self.img_size = img_size
        self.token_mlp = MLP(dim, dim, hidden)
        self.norm = nn.LayerNorm(dim)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden // 2, hidden // 2, 3, padding=1), nn.GELU(),
        )
        self.out = nn.Conv2d(hidden // 2, 1, 1)

    def forward(self, query_out: torch.Tensor) -> torch.Tensor:
        # query_out: (B, T, P, D), P = grid * grid
        B, T, P, D = query_out.shape
        x = self.token_mlp(self.norm(query_out))                                       # (B, T, P, hidden)
        x = x.view(B * T, self.grid, self.grid, -1).permute(0, 3, 1, 2)                # (B*T, hidden, gh, gw)
        x = self.refine(x)
        x = F.interpolate(x, size=self.img_size, mode="bilinear", align_corners=False)
        depth = F.softplus(self.out(x))                                                 # (B*T, 1, H, W), positive
        return depth.view(B, T, 1, self.img_size, self.img_size)                       # (B, T, 1, H, W)


# ---------- TerraWM-Linear ------------------------------------------------------

@dataclass
class TerraWMConfig:
    img_size: int = 512                                                                # divisible by 16
    d_enc: int = 1024                                                                  # DINOv3 ViT-L/16 dim
    d_model: int = 768
    n_heads: int = 12
    n_latents: int = 512
    n_write_blocks: int = 4
    n_decode_blocks: int = 2
    encoder_repo: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    freeze_encoder: bool = True
    max_frames: int = 64                                                               # for frame-index embedding table
    # Cheat-pose mode. "predicted": pose head is the only source of pose at
    # train/eval. "gt_replace": GT pose enc is fed INTO the WRITE step as a
    # per-frame conditioning vector — the depth half learns with geometry as
    # given. Pose head still runs (and its output is returned) but its loss
    # is up to the trainer.
    pose_supervision_mode: str = "predicted"                                          # "predicted" | "gt_replace"


class TerraWMLinear(nn.Module):
    def __init__(self, cfg: TerraWMConfig | None = None):
        super().__init__()
        cfg = cfg or TerraWMConfig()
        self.cfg = cfg

        # Frozen encoder
        self.encoder = DINOv3Encoder(
            repo_id=cfg.encoder_repo, img_size=cfg.img_size, freeze=cfg.freeze_encoder
        )
        self.grid = self.encoder.grid                                                  # img_size // 16
        self.n_patches_per_frame = self.grid * self.grid

        # Patch projection D_enc -> D_model
        self.patch_proj = nn.Linear(cfg.d_enc, cfg.d_model)

        # 2D learned position embedding (one per patch position)
        self.patch_pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches_per_frame, cfg.d_model)
        )
        nn.init.normal_(self.patch_pos_embed, std=0.02)

        # Frame-index embedding (added to patches at WRITE and to queries at DECODE)
        self.frame_embed = nn.Parameter(torch.zeros(cfg.max_frames, cfg.d_model))
        nn.init.normal_(self.frame_embed, std=0.02)

        # Latent set — learned initial state
        self.latents_init = nn.Parameter(torch.zeros(1, cfg.n_latents, cfg.d_model))
        nn.init.normal_(self.latents_init, std=0.02)

        # WRITE blocks
        self.write_blocks = nn.ModuleList([
            WriteBlock(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_write_blocks)
        ])

        # DECODE blocks (shared for camera + depth queries)
        self.decode_blocks = nn.ModuleList([
            DecodeBlock(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_decode_blocks)
        ])

        # Query parameters
        self.camera_query = nn.Parameter(torch.zeros(1, 1, cfg.d_model))                # one per frame
        nn.init.normal_(self.camera_query, std=0.02)
        self.depth_query = nn.Parameter(
            torch.zeros(1, self.n_patches_per_frame, cfg.d_model)                      # per patch
        )
        nn.init.normal_(self.depth_query, std=0.02)

        # Heads
        self.camera_head = CameraHead(cfg.d_model)
        self.depth_head = DepthHead(cfg.d_model, self.grid, cfg.img_size)

        # Pose embedder — used only when pose_supervision_mode='gt_replace'.
        # 9-d normalized pose enc → d_model. Added to each frame's patches at
        # WRITE time, so the latent state sees both image content AND geometry.
        self.pose_embedder = nn.Linear(9, cfg.d_model)
        # Zero-init so a model trained without it still works after we add it.
        nn.init.zeros_(self.pose_embedder.weight)
        nn.init.zeros_(self.pose_embedder.bias)

    # ------------------------------------------------------------------
    def init_state(self, batch_size: int) -> torch.Tensor:
        """Return a fresh latent state for a new sequence."""
        return self.latents_init.expand(batch_size, -1, -1).contiguous()

    # ------------------------------------------------------------------
    def encode_frame(
        self,
        rgb: torch.Tensor,
        frame_idx: int,
        gt_pose_enc_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """rgb: (B, 3, H, W) -> patches: (B, P, D_model) with pos + frame +
        (optional) pose embedding.

        gt_pose_enc_t: (B, 9) — only used when pose_supervision_mode='gt_replace'.
            When supplied, its embedding is added to every patch so the WRITE
            step sees pose context.
        """
        enc_out = self.encoder(rgb)                                                    # (B, P, D_enc)
        patches = self.patch_proj(enc_out.patches)                                     # (B, P, D_model)
        patches = patches + self.patch_pos_embed
        patches = patches + self.frame_embed[frame_idx][None, None, :]                 # broadcast (D,) -> (1,1,D)
        if gt_pose_enc_t is not None and self.cfg.pose_supervision_mode == "gt_replace":
            pose_emb = self.pose_embedder(gt_pose_enc_t)                               # (B, D_model)
            patches = patches + pose_emb[:, None, :]                                   # broadcast over patches
        return patches

    # ------------------------------------------------------------------
    def write_step(self, latents: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """One WRITE pass — update latents in place with one frame's patches."""
        for blk in self.write_blocks:
            latents = blk(latents, patches)
        return latents

    # ------------------------------------------------------------------
    def decode_frame(self, latents: torch.Tensor, frame_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode camera + depth for one frame from the current latent state.

        Returns (camera_pose_enc (B, 9), depth (B, 1, H, W)).
        """
        B = latents.shape[0]
        frame_emb = self.frame_embed[frame_idx][None, None, :]                          # (1, 1, D)

        # Camera query
        cam_q = self.camera_query.expand(B, -1, -1) + frame_emb                         # (B, 1, D)
        cam_out = cam_q
        for blk in self.decode_blocks:
            cam_out = blk(cam_out, latents)
        cam_pred = self.camera_head(cam_out[:, None, 0, :])                             # (B, 1, 9)
        # squeeze the artificial 1-frame dim before returning
        cam_pred = cam_pred.squeeze(1)                                                  # (B, 9)

        # Depth queries (one per patch position)
        depth_q = self.depth_query.expand(B, -1, -1) + frame_emb                        # (B, P, D)
        depth_out = depth_q
        for blk in self.decode_blocks:
            depth_out = blk(depth_out, latents)
        depth = self.depth_head(depth_out[:, None, :, :])                                # (B, 1, 1, H, W)
        depth = depth.squeeze(1).squeeze(1)                                              # (B, H, W)

        return cam_pred, depth

    # ------------------------------------------------------------------
    def forward(
        self, rgb_seq: torch.Tensor, gt_pose_enc: torch.Tensor | None = None
    ) -> dict:
        """rgb_seq: (B, T, 3, H, W) — sequence forward.

        gt_pose_enc: (B, T, 9) — only used when pose_supervision_mode='gt_replace'.
            Should be the NORMALIZED GT pose encoding (after the train-time
            mean-distance normalization) so its scale matches model outputs.

        Returns dict with:
            cameras: (B, T, 9)
            depths:  (B, T, H, W)
            latents: (B, M, D)
        """
        B, T, C, H, W = rgb_seq.shape
        assert H == self.cfg.img_size and W == self.cfg.img_size, (
            f"input must be square {self.cfg.img_size}x{self.cfg.img_size}, got {H}x{W}"
        )
        if self.cfg.pose_supervision_mode == "gt_replace" and gt_pose_enc is None:
            raise ValueError("pose_supervision_mode='gt_replace' requires gt_pose_enc")
        latents = self.init_state(B)

        # WRITE all frames
        for t in range(T):
            pose_t = gt_pose_enc[:, t] if gt_pose_enc is not None else None
            patches = self.encode_frame(rgb_seq[:, t], t, gt_pose_enc_t=pose_t)         # (B, P, D)
            latents = self.write_step(latents, patches)

        # DECODE per frame (camera + depth)
        cams = []
        depths = []
        for t in range(T):
            cam_t, depth_t = self.decode_frame(latents, t)
            cams.append(cam_t)
            depths.append(depth_t)
        cameras = torch.stack(cams, dim=1)                                              # (B, T, 9)
        depth_out = torch.stack(depths, dim=1)                                          # (B, T, H, W)

        return {
            "cameras": cameras,
            "depths": depth_out,
            "latents": latents,
        }


def build_terrawm_linear(**overrides) -> TerraWMLinear:
    cfg = TerraWMConfig(**overrides)
    return TerraWMLinear(cfg)


if __name__ == "__main__":
    import sys
    print("[terrawm-linear] smoke test (forward + backward, no training)")
    torch.manual_seed(0)
    cfg = TerraWMConfig(img_size=512, d_model=384, n_heads=8, n_latents=128,
                          n_write_blocks=2, n_decode_blocks=2)
    model = TerraWMLinear(cfg).cuda()
    B, T = 1, 4
    rgb = torch.rand(B, T, 3, cfg.img_size, cfg.img_size, device="cuda")
    out = model(rgb)
    print(f"  cameras: {tuple(out['cameras'].shape)}")
    print(f"  depths : {tuple(out['depths'].shape)}")
    print(f"  latents: {tuple(out['latents'].shape)}")

    # Sanity loss + backward
    target_depth = torch.rand_like(out["depths"])
    target_cam = torch.randn_like(out["cameras"])
    loss = (out["depths"] - target_depth).abs().mean() + \
            (out["cameras"] - target_cam).abs().mean()
    print(f"  loss : {loss.item():.4f}")
    loss.backward()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  trainable params: {trainable/1e6:.2f} M")
    print(f"  frozen    params: {frozen/1e6:.2f} M (DINOv3 ViT-L/16)")
    print("[terrawm-linear] OK")
    sys.exit(0)
