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
    GraphedStreamingScan,
    IntraFrameTransformer,
    PatchStateCrossAttn,
    SummaryTokenPooler,
)
from .aggregators.anchor_pool import (
    AnchorPool,
    build_patch_pixel_grid,
    cam9_to_pose_w_c,
)
from .encoders import DINOv2Encoder, DINOv3Encoder, EncoderOutput, VJEPAEncoder
from .heads.camera import CameraHead
from .heads.conditioned_predictor import ConditionedNextLatentPredictor
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
        n_summary_dynamic: int | None = None,
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
        cross_frame_target: Literal["summary", "patch"] = "summary",
        use_anchor_pool: bool = False,
        n_anchors: int = 32,
        n_anchor_writes: int = 4,
        anchor_match_threshold: float = 0.5,
        terrawm: bool = False,
        terrawm_motion_freqs: int = 64,
    ):
        super().__init__()
        self.encoder = encoder
        self.dim = encoder.dim
        self.img_size = encoder.img_size
        self.grid_h = encoder.grid
        self.grid_w = encoder.grid
        self.n_summary = n_summary_tokens
        # Dual-channel split: the first `n_dynamic` summary tokens drive the
        # camera head + world-model prediction objective (smooth, predictive
        # role); the remaining `n_summary - n_dynamic` tokens are free to
        # specialize for the dense head (sharp, observation role).
        # Default = n_summary_tokens (single-channel, backward compat).
        if n_summary_dynamic is None:
            n_summary_dynamic = n_summary_tokens
        assert 0 < n_summary_dynamic <= n_summary_tokens, \
            f"n_summary_dynamic={n_summary_dynamic} must be in (0, {n_summary_tokens}]"
        self.n_dynamic = n_summary_dynamic
        # Cross-frame Mamba target: either the K compact summary tokens per
        # frame (cheap, but compressed) or the P refined patch tokens per
        # frame (expensive, but spatially aligned — the DPT head reads from
        # these directly without going through summary pooling).
        self.cross_frame_target = cross_frame_target

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
        self.terrawm = terrawm
        if predict_next_latent:
            if terrawm:
                # TerraWM: predictor is conditioned on camera motion between
                # frame t and frame t+1. Frees the scene-state encoder from
                # learning view-change physics.
                self.latent_predictor = ConditionedNextLatentPredictor(
                    dim=self.dim, hidden=head_hidden * 2,
                    motion_enc_freqs=terrawm_motion_freqs,
                )
            else:
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

        # 7. Optional anchor pool for feedforward re-grounding (Experiment 2).
        # When enabled, after the camera head emits a coarse pose per frame,
        # the anchor pool reads matching anchors from previously-stored scene
        # observations and emits a (Δt, Δq) correction. The corrected pose
        # replaces the coarse one in out["camera"]; the coarse pose is kept
        # in out["camera_coarse"] for monitoring.
        self.use_anchor_pool = use_anchor_pool
        self.anchor_match_threshold = anchor_match_threshold
        if use_anchor_pool:
            self.anchor_pool = AnchorPool(
                dim=self.dim, n_anchors=n_anchors, n_writes=n_anchor_writes,
                match_threshold=anchor_match_threshold,
            )
            # Precompute the per-patch pixel grid (constant across batch/frame).
            self.register_buffer(
                "_patch_pixel_grid",
                build_patch_pixel_grid(self.grid_h, self.grid_w, self.img_size, device="cpu"),
                persistent=False,
            )
        else:
            self.anchor_pool = None

    def forward(
        self,
        rgb: torch.Tensor,
        track_xy: torch.Tensor | None = None,
        track_frame: int | None = None,
        K_intrinsics: torch.Tensor | None = None,   # (B, 3, 3) — required if use_anchor_pool
        train_pose_noise_std_m: float = 0.0,        # simulated cumulative drift per frame
        gt_relative_motion: torch.Tensor | None = None,  # (B, T-1, 7) for TerraWM predictor conditioning
    ) -> dict[str, torch.Tensor]:
        b, t, _, h, w = rgb.shape
        assert h == self.img_size and w == self.img_size, \
            f"expected {self.img_size}, got {h}x{w}"

        # 1. Encoder (frozen, per-frame).
        flat_rgb = rgb.reshape(b * t, 3, h, w)
        with torch.no_grad():
            enc_out: EncoderOutput = self.encoder(flat_rgb)
        patches = enc_out.patches.reshape(b, t, -1, self.dim)            # (B, T, P, D)
        p = patches.shape[2]

        # 2. Per-frame self-attn refinement.
        refined = self.intraframe(patches)                                # (B, T, P, D)

        if self.cross_frame_target == "summary":
            # Original path: pool to K summaries, scan T*K tokens, dense head
            # cross-attends patches to per-frame state tokens.
            summaries = self.summary_pool(refined)                        # (B, T, K, D)
            summaries = summaries + self.frame_embed[:, :t]               # broadcasts over K
            seq = summaries.reshape(b, t * self.n_summary, self.dim)
            state_seq = self.cross_frame(seq)                             # (B, T*K, D)
            state_per_frame = state_seq.reshape(b, t, self.n_summary, self.dim)
            dense_in = self.dense_readout(refined, state_per_frame)       # (B, T, P, D)
        else:
            # Patch-scan path: scan T*P tokens directly, then pool to
            # K summaries for the camera+predict heads. DPT reads scanned
            # patches directly — spatially aligned, no summary bottleneck.
            patch_in = refined + self.frame_embed[:, :t]                  # broadcasts over P
            seq = patch_in.reshape(b, t * p, self.dim)
            scanned = self.cross_frame(seq).reshape(b, t, p, self.dim)    # (B, T, P, D)
            state_per_frame = self.summary_pool(scanned)                  # (B, T, K, D)
            dense_in = scanned                                            # DPT reads patch hiddens

        # 5a. Camera head — reads only the dynamic channel of state.
        # When n_dynamic == n_summary this is a no-op slice (single-channel mode).
        state_dynamic = state_per_frame[:, :, :self.n_dynamic]            # (B, T, K_dyn, D)
        cam_coarse = self.camera_head(state_dynamic)                      # (B, T, 9)

        # 5b. Dense (pointmap) head — reshape to spatial grid for DPT, then
        # chunk over frames so the bilinear upsample stays under PyTorch's
        # INT_MAX per-call limit.
        grid = dense_in.reshape(b * t, -1, self.dim).transpose(1, 2)
        grid = grid.reshape(b * t, self.dim, self.grid_h, self.grid_w)
        chunk = 8
        pmap_chunks = [self.dpt(grid[i:i + chunk]) for i in range(0, b * t, chunk)]
        pmap = torch.cat(pmap_chunks, dim=0).reshape(b, t, 3, h, w)

        out = {"camera": cam_coarse, "pointmap": pmap}

        # 5b'. Optional anchor pool re-grounding (Experiment 2).
        # Frame-sequential: write/read/correct one frame at a time, carrying
        # the anchor pool state forward. The Mamba scan, summary pool, DPT,
        # and coarse pose are already computed in parallel; only the anchor
        # bookkeeping is sequential. K_a × P per frame is small (~32K ops).
        if self.use_anchor_pool and self.anchor_pool is not None:
            assert K_intrinsics is not None, "anchor pool needs K_intrinsics (B, 3, 3)"
            # Per-patch depth: take Z from pointmap, downsample to patch grid.
            # pmap is (B, T, 3, H, W); avg-pool the Z channel to (grid_h, grid_w).
            patch_size_h = h // self.grid_h
            patch_size_w = w // self.grid_w
            z_full = pmap[:, :, 2]                                         # (B, T, H, W)
            patch_depth_grid = torch.nn.functional.avg_pool2d(
                z_full.reshape(b * t, 1, h, w), kernel_size=(patch_size_h, patch_size_w)
            ).reshape(b, t, self.grid_h * self.grid_w)                     # (B, T, P)
            # Patch pixel grid (static, broadcast over batch).
            patch_pixel = self._patch_pixel_grid.to(rgb.device).unsqueeze(0).expand(b, -1, -1)
            # Take patches from the (post-Mamba scanned) tensor for descriptors.
            if self.cross_frame_target == "patch":
                anchor_descs_src = scanned                                  # (B, T, P, D)
            else:
                anchor_descs_src = refined                                  # fallback for summary mode
            # Carry state per-batch. write_idx and valid mask evolve in place.
            anchor_state = self.anchor_pool.init_state(b, device=rgb.device,
                                                        dtype=anchor_descs_src.dtype)
            # Optional training-time cumulative pose drift, to teach the MLP
            # what inference-time drift looks like (B). At inference, drift
            # accumulates over hundreds of frames; in a 32-frame training
            # window, accumulated drift is naturally tiny — the correction MLP
            # would learn "do nothing" because there's nothing to correct.
            # Inject a synthetic random-walk drift trajectory so the MLP
            # sees the regime it'll face at deployment.
            if self.training and train_pose_noise_std_m > 0.0:
                # Per-frame translation increments ~ N(0, σ² I), accumulated.
                # Frame 0 drift = 0 (anchor of the trajectory).
                step = torch.randn(b, t, 3, device=rgb.device,
                                   dtype=torch.float32) * train_pose_noise_std_m
                step[:, 0] = 0.0
                drift_traj = step.cumsum(dim=1)                             # (B, T, 3)
            else:
                drift_traj = None
            cam_corrected_list = []
            scores_list = []
            anchor_pos_history = []
            for ti in range(t):
                # READ + CORRECT (gradient flows through the correction MLP).
                cam_t = cam_coarse[:, ti]                                   # (B, 9)
                # Apply per-frame drift (translation only) for the anchor path.
                # The cam loss target is still GT — so the MLP has to learn
                # to output a correction that undoes the drift.
                if drift_traj is not None:
                    drift_t = drift_traj[:, ti].to(cam_t.dtype)             # (B, 3)
                    cam_t_for_pool = cam_t.clone()
                    cam_t_for_pool[:, :3] = cam_t_for_pool[:, :3] + drift_t
                else:
                    cam_t_for_pool = cam_t
                patches_t = anchor_descs_src[:, ti]                         # (B, P, D)
                corrected_t, scores_t = self.anchor_pool.correct_pose(
                    anchor_state, patches_t, cam_t_for_pool
                )
                cam_corrected_list.append(corrected_t)                      # (B, 9)
                scores_list.append(scores_t)                                # (B, P, K_a)
                # Snapshot anchor positions BEFORE this frame's write (for consistency loss).
                anchor_pos_history.append(anchor_state.positions.clone())
                # WRITE using the CORRECTED pose (no gradient through write).
                pose_w_c = cam9_to_pose_w_c(corrected_t.detach())
                self.anchor_pool.write(
                    anchor_state,
                    patches_t.detach(),
                    patch_pixel,
                    patch_depth_grid[:, ti].detach(),
                    K_intrinsics,
                    pose_w_c,
                )
            cam_corrected = torch.stack(cam_corrected_list, dim=1)          # (B, T, 9)
            scores = torch.stack(scores_list, dim=1)                        # (B, T, P, K_a)
            anchor_pos_history_t = torch.stack(anchor_pos_history, dim=1)   # (B, T, K_a, 3)

            out["camera"] = cam_corrected
            out["camera_coarse"] = cam_coarse
            out["anchor_scores"] = scores
            out["anchor_positions"] = anchor_pos_history_t
            out["patch_pixel"] = patch_pixel                                # (B, P, 2)
            out["K_intrinsics"] = K_intrinsics                              # (B, 3, 3)

        # 5c. Optional track head.
        if self.track_head is not None and track_xy is not None and track_frame is not None:
            out["tracks"] = self.track_head(track_xy, track_frame, state_per_frame)

        # 5d. World-model regularizer: predict next-frame summary tokens from state.
        # Target = EMA-target encoder output on the same patches, stopgrad.
        # In dual-channel mode, the prediction objective applies only to the
        # first K_dyn ("dynamic") tokens — the observation channel is free of
        # the smoothness-inducing prediction loss.
        # Loss is consumed by the train script via geomamba_loss.
        # Skip the predictor entirely at inference (eval-only forward passes
        # don't compute the pred loss). For TerraWM specifically, the predictor
        # also requires motion conditioning that eval callers don't supply.
        predictor_active = (
            self.predict_next_latent
            and self.latent_predictor is not None
            and t >= 2
            and self.training
            and (not self.terrawm or gt_relative_motion is not None)
        )
        if predictor_active:
            with torch.no_grad():
                tgt_refined = self.target_intraframe(patches)                # (B, T, P, D)
                tgt_summaries = self.target_summary_pool(tgt_refined)        # (B, T, K, D)
            kd = self.n_dynamic
            if self.terrawm:
                predicted_next = self.latent_predictor(
                    state_per_frame[:, :-1, :kd], gt_relative_motion
                )                                                              # (B, T-1, K_dyn, D)
            else:
                predicted_next = self.latent_predictor(state_per_frame[:, :-1, :kd])  # (B, T-1, K_dyn, D)
            target_next = tgt_summaries[:, 1:, :kd].detach()                       # (B, T-1, K_dyn, D)
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
                             device="cuda", use_cuda_graphs: bool = False):
        """Zero-initialized per-layer Mamba state for streaming inference.

        Only valid for causal-only models (bidirectional=False).

        If `use_cuda_graphs=True`, returns a GraphedStreamingScan wrapper that
        captures the per-frame scan in a CUDA graph (Speed-B). Bit-perfect
        equivalent to the loop path (parity verified to atol=0 over 50 frames);
        ~6× faster on the patch-scan path because it eliminates per-kernel
        cudaLaunchKernel overhead. streaming_forward auto-detects the wrapper
        type and routes accordingly.
        """
        if not isinstance(self.cross_frame, CrossFrameMamba):
            raise RuntimeError("streaming_forward requires a Mamba cross-frame aggregator")
        if not use_cuda_graphs:
            return self.cross_frame.init_streaming_state(
                batch_size=batch_size, dtype=dtype, device=device,
            )
        # Graphed path: K = patches-per-frame in patch mode, n_summary in summary mode.
        K = (self.grid_h * self.grid_w) if self.cross_frame_target == "patch" else self.n_summary
        gs = GraphedStreamingScan(
            self.cross_frame, batch_size=batch_size, K=K, dim=self.dim,
            dtype=dtype, device=device,
        )
        gs.capture()
        return gs

    @torch.no_grad()
    def init_anchor_state(self, batch_size: int = 1, dtype=torch.bfloat16, device="cuda"):
        """Initialize the anchor pool state for streaming inference."""
        if not self.use_anchor_pool or self.anchor_pool is None:
            return None
        return self.anchor_pool.init_state(batch_size=batch_size, device=device, dtype=dtype)

    @torch.no_grad()
    def streaming_forward(
        self,
        rgb_frame: torch.Tensor,
        state: list[dict],
        frame_idx: int,
        autocast_dtype: torch.dtype = torch.bfloat16,
        anchor_state=None,
        K_intrinsics: torch.Tensor | None = None,
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
            p = patches.shape[2]

            # 2. Per-frame self-attn refinement.
            refined = self.intraframe(patches)                               # (1, 1, P, D)

            is_graphed = isinstance(state, GraphedStreamingScan)
            if self.cross_frame_target == "summary":
                summaries = self.summary_pool(refined)                       # (1, 1, K, D)
                if frame_idx < self.frame_embed.shape[1]:
                    summaries = summaries + self.frame_embed[:, frame_idx:frame_idx + 1]
                tokens = summaries.reshape(1, self.n_summary, self.dim).to(autocast_dtype)
                if is_graphed:
                    state_seq, state = state.step(tokens)
                else:
                    state_seq, state = self.cross_frame.streaming_step(tokens, state)
                state_per_frame = state_seq.unsqueeze(1)                     # (1, 1, K, D)
                dense_in = self.dense_readout(refined, state_per_frame)      # (1, 1, P, D)
            else:
                # Patch-scan streaming: advance state by the whole frame's P patches.
                patch_in = refined
                if frame_idx < self.frame_embed.shape[1]:
                    patch_in = patch_in + self.frame_embed[:, frame_idx:frame_idx + 1]
                tokens = patch_in.reshape(1, p, self.dim).to(autocast_dtype)
                if is_graphed:
                    scanned, state = state.step(tokens)
                else:
                    scanned, state = self.cross_frame.streaming_step(tokens, state)
                scanned = scanned.unsqueeze(1)                               # (1, 1, P, D)
                state_per_frame = self.summary_pool(scanned)                 # (1, 1, K, D)
                dense_in = scanned                                           # (1, 1, P, D)

            # 5a. Camera head — dynamic channel only.
            state_dynamic = state_per_frame[:, :, :self.n_dynamic]           # (1, 1, K_dyn, D)
            cam = self.camera_head(state_dynamic)                            # (1, 1, 9)

            # 5b. Dense head.
            grid = dense_in.reshape(1, -1, self.dim).transpose(1, 2)
            grid = grid.reshape(1, self.dim, self.grid_h, self.grid_w)
            pmap = self.dpt(grid).reshape(1, 1, 3, h, w)

            # 5c. Optional anchor pool re-grounding.
            anchor_diag = None
            if self.use_anchor_pool and self.anchor_pool is not None:
                assert anchor_state is not None and K_intrinsics is not None, \
                    "streaming_forward with use_anchor_pool needs anchor_state and K_intrinsics"
                # Per-frame patches for descriptors: use scanned (patch mode) or refined (summary mode).
                if self.cross_frame_target == "patch":
                    patches_t = scanned.squeeze(1) if scanned.dim() == 4 else scanned   # (1, P, D)
                else:
                    patches_t = refined.squeeze(1)
                # Patch-grid pixel coords (static).
                patch_pixel = self._patch_pixel_grid.to(rgb_frame.device).unsqueeze(0)  # (1, P, 2)
                # Per-patch depth from this frame's pointmap.
                z = pmap[:, 0, 2]                                              # (1, H, W)
                ph = h // self.grid_h
                pw_ = w // self.grid_w
                patch_depth = torch.nn.functional.avg_pool2d(
                    z.unsqueeze(1), kernel_size=(ph, pw_)
                ).reshape(1, self.grid_h * self.grid_w)                        # (1, P)
                # READ + CORRECT.
                cam_coarse_t = cam[0, 0].unsqueeze(0)                          # (1, 9)
                corrected_t, scores_t = self.anchor_pool.correct_pose(
                    anchor_state, patches_t, cam_coarse_t
                )
                # Diagnostic snapshot BEFORE the write (write mutates state.positions).
                anchor_diag = {
                    "camera_coarse": cam_coarse_t.clone(),                     # (1, 9)
                    "scores": scores_t.clone(),                                # (1, P, K_a)
                    "anchor_positions_pre_write": anchor_state.positions.clone(),  # (1, K_a, 3)
                    "anchor_valid_pre_write": anchor_state.valid.clone(),      # (1, K_a)
                    "patch_pixel": patch_pixel.clone(),                        # (1, P, 2)
                }
                # WRITE using corrected pose.
                pose_w_c = cam9_to_pose_w_c(corrected_t)
                self.anchor_pool.write(
                    anchor_state, patches_t, patch_pixel, patch_depth,
                    K_intrinsics, pose_w_c,
                )
                cam = corrected_t.unsqueeze(0)                                 # (1, 1, 9)

        out = {"camera": cam, "pointmap": pmap}
        if anchor_diag is not None:
            out.update(anchor_diag)
        return out, state


def build_geomamba(
    encoder_name: Literal["vjepa", "dinov2", "dinov3"],
    weights_root: str,
    n_intraframe_layers: int = 12,
    n_summary_tokens: int = 4,
    n_summary_dynamic: int | None = None,
    n_xfm_layers: int = 12,
    d_state: int = 128,
    bidirectional: bool = True,
    aggregator_name: Literal["mamba", "attention"] = "mamba",
    track_enabled: bool = True,
    max_frames: int = 256,
    dense_residual_to_patches: bool = True,
    predict_next_latent: bool = False,
    ema_momentum: float = 0.99,
    cross_frame_target: Literal["summary", "patch"] = "summary",
    use_anchor_pool: bool = False,
    n_anchors: int = 32,
    n_anchor_writes: int = 4,
    anchor_match_threshold: float = 0.5,
    terrawm: bool = False,
    terrawm_motion_freqs: int = 64,
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
        n_summary_dynamic=n_summary_dynamic,
        n_xfm_layers=n_xfm_layers,
        d_state=d_state,
        bidirectional=bidirectional,
        aggregator_name=aggregator_name,
        track_enabled=track_enabled,
        max_frames=max_frames,
        dense_residual_to_patches=dense_residual_to_patches,
        predict_next_latent=predict_next_latent,
        ema_momentum=ema_momentum,
        cross_frame_target=cross_frame_target,
        use_anchor_pool=use_anchor_pool,
        n_anchors=n_anchors,
        n_anchor_writes=n_anchor_writes,
        anchor_match_threshold=anchor_match_threshold,
        terrawm=terrawm,
        terrawm_motion_freqs=terrawm_motion_freqs,
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
