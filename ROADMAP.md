# GeoMamba — Engineering Roadmap

> **Project**: A linearly-scaling, feed-forward 3D reconstruction model for sequential video.
> Linear-memory streaming alternative to VGGT, anchored on a Mamba-2 SSM state that serves as
> a Dreamer/RSSM-style world model. V-JEPA 2.1 tokenizer, multi-task heads (camera, depth,
> pointmap, tracks), bidirectional refinement for loop-closure-like behaviour.

## Thesis

VGGT and VGG-T³ treat inputs as permutation-invariant *sets*. The data that actually matters
in practice — SLAM, AR, robotics, dashcam, drone — is **temporally ordered video**. A
sequential model with a fixed-size SSM state exploits smooth motion / frame coherence /
causality natively, which is *why* it can be cheaper without quality loss. Linearity is a
consequence of the right inductive bias, not the goal.

## Competitive landscape (as of May 2026)

| Model | Memory in N | Inner-loop at inference? | Bidirectional? | Unordered? | Camera pose |
|---|---|---|---|---|---|
| VGGT | O(N²) | no | n/a | yes | strong |
| VGG-T³ | O(1) | **yes (TTT, Muon, 2 inner steps)** | n/a | yes | **weak** |
| StreamVGGT | O(N) (growing KV) | no | no | no | OK |
| Stream3R | O(N) (growing KV) | no | no | no | OK |
| CUT3R / TTT3R | O(1) | yes (TTT3R) | no | no (TTT3R degrades) | weak on unordered |
| **GeoMamba (ours)** | **O(1)** | **no** | **yes** | **yes** | **target: ≥ StreamVGGT** |

The four-way win is: O(1) memory + no test-time inner loop + bidirectional refinement +
handles both ordered/unordered.

---

## Phase 0 — Setup & baseline plumbing
**Duration:** ~1 week · **Compute:** single GPU · **Risk:** none

**Build:**
- Repo skeleton: `data/`, `src/geomamba/`, `third_party/`, `scripts/`, `notebooks/`, `viz/`, `configs/`, `experiments/`.
- Env via uv: `mamba-ssm`, `causal-conv1d`, `vjepa2`, `vggt`, `torch>=2.4`, `xformers`, `rerun-sdk`, `open3d`.
- Git submodules for `third_party/`: VGGT, V-JEPA 2, Mamba (state-spaces/mamba).
- Data download scripts for Phase 1 only (don't download the full VGGT training mix yet):
  - CO3D-v2 subset (10–20 categories, ~50–80 GB)
  - Sintel-stereo (depth GT, ~5 GB)
  - 7-Scenes (~20 GB, eval sanity)
- Eval harness: pointmap CD, ATE, RPE, depth Abs Rel — wrappers around existing implementations.

**Tests (smoke):**
- VGGT inference on a CO3D scene → save reference pointmap.
- V-JEPA 2.1 ViT-L produces dense features on a held-out video.
- A single Mamba-2 block forward+backward on toy `(B, T, D)` input.

**Visualizations:**
- Reproduce V-JEPA 2.1 Figure 1 PCA on your own images. **If you can't reproduce the
  qualitative result, stop and debug — everything downstream assumes this works.**

**Gate:** All three smoke tests pass.

---

## Phase 1 — Tokenizer probe (the cheap kill-switch)
**Duration:** ~2 weeks · **Compute:** 1 A100 for ~2–3 days · **Risk:** validates a free standalone contribution

**Build:**
- "Mini-3R": frozen encoder → 4-layer cross-frame transformer (no Mamba yet — plain self-attn so we isolate the tokenizer variable) → DPT pointmap head.
- Two encoder variants: V-JEPA 2.1 ViT-L vs DINOv2 ViT-L. Same dim, same patch size, same head.

**Train:**
- CO3D-v2 slice (~2–5k scenes, 8 frames each).
- 1–2 days on 1× A100.
- Identical hyperparameters across both variants.

**Tests:**
- Multi-view consistency: predicted pointmaps from frames 1 and 5 should agree where they overlap (mask-aware Chamfer between the two clouds).
- Single-frame depth Abs Rel on Sintel.
- 7-Scenes Acc/Comp on a tiny eval split.

**Visualizations:**
- Side-by-side PCA of V-JEPA vs DINOv2 features on the SAME image.
- Pointmap renderings from each variant on identical test scenes.
- Loss-curve overlay; CD vs training step.

**Gate:** Does V-JEPA 2.1 beat DINOv2 by ≥10% on multi-view consistency?
- **Yes** → tokenizer is a contribution; proceed with V-JEPA in the full model.
- **No** → drop the tokenizer claim, use DINOv2 to align with prior art, proceed with the Mamba aggregator as the sole contribution.

---

## Phase 2 — State capacity validation
**Duration:** ~3–4 weeks · **Compute:** 2–4 A100s intermittent · **Risk:** kills the O(1) memory claim if it fails

**Build:**
- Cross-frame Mamba-2 block (start with one layer).
- Per-frame: K=4 learnable summary tokens via cross-attn over patches.
- Mamba-2 scans summary tokens across the sequence.
- Patch-to-state readout: cross-attn(query = `P_t`, kv = `s_t`).
- Drop into the Mini-3R skeleton from Phase 1.

**Train:**
- Same CO3D slice.
- Sweep state size S ∈ {1024, 2048, 4096, 8192, 16384}.
- One small run per S; total ~5 days on 2 A100s.

**Tests:**
- Pointmap CD vs S → find the knee.
- Memory profile: confirm peak memory is **flat in N** for fixed S.
- Long-sequence stability: train at 8 frames, infer at 64, 256, 1024 frames. Does quality collapse?
- Same on a same-compute KV-cache baseline (mini-StreamVGGT) for head-to-head.

**Visualizations:**
- Pareto curve: pointmap CD vs S (log-x).
- Memory comparison plot: Yours (flat) vs StreamVGGT (linear) vs VGGT (quadratic) at N ∈ {10, 100, 500, 1000}.
- Per-patch attention heatmaps: which patches read most strongly from state.
- t-SNE of `s_t` trajectory across a sequence.

**Gate:** Any S ≤ 16k within 20% of the KV-cache baseline on multi-view CD?
- **Yes** → Mamba aggregator is viable; proceed.
- **No** → switch to "design B" (sliding-window patch cache + long-term Mamba state).

---

## Phase 3 — Full architecture build
**Duration:** ~4–6 weeks · **Compute:** 4–8 A100s · **Risk:** ablation correctness

**Build:**
- Full L-layer model (start with L=12 for budget, target L=24): alternating per-frame self-attn + cross-frame Mamba-2 blocks.
- Four heads (mirror VGGT): camera, depth, pointmap, tracks.
- Camera head reads state only (no patches).
- Depth + pointmap heads: cross-attn(`P_t` ↔ `s_t`) + residual `P_t`.
- Track head: similar, with query token input.
- Three inference modes from the same weights: causal · bidirectional refine · windowed-lookahead (W=8).

**Tests:**
- Per-head sanity on a single scene before any benchmarking.
- Causal vs bidirectional vs windowed quality gap on Sintel and CO3D val.
- Per-frame inference latency at frame 1, 100, 500, 1000.
- Full memory profile across modes.

**Visualizations:**
- Production-quality architecture diagram for the paper.
- Latency-per-frame timeline.
- Quality-vs-latency Pareto across three modes.

**Gate:** Does bidirectional refine close ≥70% of the gap between causal-mode and offline VGGT?
- **Yes** → "near-streaming with loop closure" is the headline.
- **No** → ship as "linear-time offline alternative to VGGT."

---

## Phase 4 — Distillation training (the heavy phase)
**Duration:** ~6–8 weeks · **Compute:** ~60–120 A100-days

**Build:**
- VGGT teacher pipeline generating pseudo-GT for camera/depth/pointmap/tracks.
- KD loss following StreamVGGT (Huber + confidence-weighted, soft targets on all four heads).
- Multi-dataset loader: VGGT's standard mix (Co3Dv2, BlendedMVS, ARKitScenes, MegaDepth, WildRGB, ScanNet, HyperSim, MVS-Synth, OmniObject3D, PointOdyssey, Virtual KITTI, Spring, Waymo).

**Train (two stages):**
- **4a:** Distill the cross-frame Mamba block from VGGT global attention. Freeze the V-JEPA encoder (LoRA optional) and the prediction heads. ~50% of budget.
- **4b:** End-to-end finetune with full multi-task loss. ~50% of budget.

**Tests:**
- KD convergence vs raw-supervision baseline.
- Per-dataset held-out performance.
- Indoor → outdoor domain shift robustness.

**Visualizations:**
- Loss curves for 4a vs 4b.
- Per-task loss balance over training.
- Reconstruction quality progression: epoch 1, 5, 10, final.

**Gate:** Within 20% of StreamVGGT on 7-Scenes/NRGBD/ETH3D after 4a alone?

---

## Phase 5 — Benchmarks & differentiator demos
**Duration:** ~3–4 weeks · **Compute:** inference only (~1–2k GPU-hours)

**Benchmarks:**
- 3D recon: 7-Scenes, NRGBD, ETH3D, DTU (Acc/Comp/NC/CD).
- Depth: Sintel, Bonn, KITTI, NYUv2 (Abs Rel, δ<1.25).
- Camera pose: ScanNet, Sintel, TUM-dynamics (ATE, RPE).
- 4D dynamic: TUM-dynamics, Bonn-dynamic.

**Baselines (head-to-head):**
- Offline ceiling: VGGT, VGG-T³.
- Streaming: StreamVGGT, Stream3R, CUT3R, TTT3R.
- Classic: DUSt3R, MASt3R.

**Differentiator demos (project-page hero clips):**
1. **1000-frame single-pass reconstruction** at constant memory (VGGT OOMs).
2. **Loop closure** — causal vs bidirectional-refine side-by-side on TUM dynamic.
3. **Visual localization** — freeze final `s_N`, query with held-out frame.
4. **Resumable scene state** — checkpoint `s_N`, reload, continue.
5. **(Optional, high-risk):** NVS preview — small splat/NeRF head conditioned on `(s_t, pose)`.

**Visualizations:**
- Memory-vs-N money plot: ours (flat) vs StreamVGGT (linear) vs VGGT (quadratic).
- Camera trajectory recovery overlays.
- Loop-closure before/after.
- Long-sequence reconstruction time-lapse video.

---

## Phase 6 — Paper, code, project page
**Duration:** ~3–4 weeks · **Compute:** minimal

- Draft → internal review → submission. Target ~9 pages.
- Code release with reproduction scripts.
- Model weights (V-JEPA encoder LoRA + Mamba aggregator + heads).
- Project page: hero video, memory plot, four interactive demos.
- Supplementary video walkthrough.

---

## Compute & timeline summary

| Phase | Weeks | GPU-days | Risk-reduction |
|---|---|---|---|
| 0 — Setup | 1 | <1 | smoke tests pass |
| 1 — Tokenizer probe | 2 | ~3 | V-JEPA validated or dropped |
| 2 — State capacity | 3–4 | ~10–15 | O(1) state validated or fallback |
| 3 — Full architecture | 4–6 | ~20–30 | bidirectional refine validated |
| 4 — Distillation | 6–8 | **~60–120** | competitive on benchmarks |
| 5 — Benchmarks & demos | 3–4 | ~5–10 | paper-ready numbers |
| 6 — Paper & release | 3–4 | minimal | submission |
| **Total** | **22–29** | **~100–180** | |

---

## Operating rules

1. **Don't skip the gates.** Each is a cheap kill-switch.
2. **Move fast.** Scoop risk is real — Mamba-VGGT is the obvious next paper after VGG-T³.
3. **Conservative path for the main paper.** NVS is a clearly-labeled "preliminary" sidebar at best.
4. **Track ablations from day one.** Every architectural choice needs a same-compute baseline.
