# Competitor comparison — scoping document

**Goal:** identify what it actually costs to produce a matched-protocol Table 1, vs. shipping with our-numbers-only and careful caveats.

**Method:** for each published competitor we'd compare against, catalog the public-code status, hardware/data requirements, exact eval protocol, and P50 / P90 days-to-first-comparable-number.

## Per-competitor catalog

For each row below: (a) public code repo, (b) checkpoints, (c) hardware they ran on, (d) eval datasets, (e) consumer-GPU portability risk, (f) data we'd need locally, (g) **exact metric definitions used**.

---

### StreamVGGT  (Zhuo et al., 2026)

| dim | value |
|---|---|
| (a) repo | https://github.com/jucic/StreamVGGT — public |
| (b) checkpoints | published on HuggingFace (model card, FP16 weights) |
| (c) hardware | A100 80GB (paper Section 5.1) |
| (d) eval datasets | TUM-dynamics, ScanNetv2, KITTI, 7-Scenes |
| (e) Blackwell sm_120 portability | medium risk — uses xformers + flash-attn-2 which historically pin CUDA versions; latest flash-attn-2 ships sm_90 only, may need source build for sm_120 |
| (f) data needed locally | TUM-dynamics ✓ (have), ScanNetv2 (gated), KITTI (~80GB) |
| (g) ATE definition | **ATE Sim(3) per-sequence RMSE** per their Table 5 caption; uses evo_traj for alignment (`evo_ape kitti --align --correct_scale`) |
| (g) Abs-Rel definition | per-frame median scale alignment + per-frame masking by `gt > 1e-3`; uses depth_max = 10.0 m (NOT 8.0 like ours — re-eval ours at 10.0 for fair comparison) |
| (g) RPE definition | δ = 1 frame, translation RMSE in meters; rotation reported separately in degrees |
| P50 days to first comparable number | 3-4 days |
| P90 days | 7-10 days (if flash-attn-2 source build is required) |
| Critical blocker | depth_max mismatch — re-evaluating ours at depth_max=10.0 is half-day; matching their masking is another half-day |

### CUT3R  (Wang et al., 2025)

| dim | value |
|---|---|
| (a) repo | https://github.com/CUT3R/CUT3R — public |
| (b) checkpoints | published, multiple variants (offline, online) |
| (c) hardware | A100 80GB |
| (d) eval datasets | TUM-dynamics, ScanNet, 7-Scenes, KITTI, NYUv2 |
| (e) Blackwell portability | low risk — pure PyTorch + standard ops |
| (f) data needed | TUM ✓, NYUv2 (~3GB) |
| (g) ATE definition | **ATE Sim(3) RMSE per-sequence**, evo_ape, scale correction enabled |
| (g) Abs-Rel definition | per-sequence median scale alignment (not per-frame!) → different from StreamVGGT; depth_max = 10.0 m |
| (g) RPE definition | δ = 1 frame, translation only (no rotation in main table) |
| P50 days | 2 days |
| P90 days | 4 days |
| Critical blocker | per-sequence vs per-frame scale alignment changes Abs-Rel by ~10-15% relative |

### Stream3R  (Yang et al., 2026)

| dim | value |
|---|---|
| (a) repo | https://github.com/stream3r/Stream3R — public |
| (b) checkpoints | one variant published |
| (c) hardware | 4× A100 (training); single A100 inference |
| (d) eval datasets | TUM-dynamics, ScanNet, 7-Scenes |
| (e) Blackwell portability | low — vanilla PyTorch |
| (f) data needed | TUM ✓ |
| (g) ATE definition | ATE Sim(3) RMSE, same conventions as CUT3R |
| (g) Abs-Rel definition | per-sequence median scale; depth_max = 10.0 m |
| (g) RPE | not reported in main table |
| P50 days | 2 days |
| P90 days | 4 days |
| Critical blocker | ATE = 0.213 m in their paper is suspicious vs CUT3R (0.046) — confirm we're reading the same column |

### Spann3R  (Wang et al., 2024)

| dim | value |
|---|---|
| (a) repo | https://github.com/HengyiWang/spann3r — public |
| (b) checkpoints | published |
| (c) hardware | A100 |
| (d) eval datasets | 7-Scenes, NRGBD, TUM (limited) |
| (e) Blackwell portability | medium — depends on DUSt3R fork compatibility |
| (f) data needed | 7-Scenes (~2GB), NRGBD (~5GB), TUM ✓ |
| (g) ATE definition | their paper does NOT report TUM-dynamics ATE directly — they evaluate on 7-Scenes and NRGBD instead. **Comparing on TUM requires running their code ourselves** |
| (g) Abs-Rel | per-frame median scale alignment |
| P50 days | 3-5 days (more because their primary benchmarks differ from ours) |
| P90 days | 7 days |
| Critical blocker | benchmark mismatch — we'd be reporting Spann3R-on-TUM numbers they themselves don't report, which reviewers may treat as cherry-picked |

### Point3R  (Lin et al., 2025)

| dim | value |
|---|---|
| (a) repo | https://github.com/YkiWu/Point3R — public |
| (b) checkpoints | published; ray-aware variant also published |
| (c) hardware | A100 |
| (d) eval datasets | TUM-dynamics, ScanNet, 7-Scenes |
| (e) Blackwell portability | low |
| (f) data needed | TUM ✓ |
| (g) ATE definition | ATE Sim(3) RMSE; per-sequence Sim(3) align |
| (g) Abs-Rel | not reported on TUM in main tables; depth_max = 10.0 |
| (g) RPE | not reported |
| P50 days | 2 days |
| P90 days | 4 days |
| Critical blocker | minimal — most-comparable competitor for ATE alone |

### VGGT (offline)  (Wang et al., 2025)

| dim | value |
|---|---|
| (a) repo | https://github.com/facebookresearch/vggt — public |
| (b) checkpoints | published (1.2B model) |
| (c) hardware | H100 80GB |
| (d) eval datasets | TUM-dynamics, all standard benchmarks |
| (e) Blackwell portability | medium-high — uses xformers; OOMs at N≈200 on 16GB GPU per paper |
| (f) data needed | TUM ✓ |
| (g) ATE definition | ATE Sim(3) on FULL trajectory (offline = global), not per-sequence streaming |
| (g) Abs-Rel definition | per-frame median scale; depth_max = 10.0 |
| P50 days | 3 days |
| P90 days | 5 days (mostly memory wrangling — 1.2B model needs chunked eval on 16GB) |
| Critical blocker | offline vs streaming mismatch — VGGT processes the whole window jointly; our streaming inference is online. Direct comparison is inherently asymmetric |

---

## Cross-cutting metric definition gotchas

These items affect ALL competitor comparisons; pin them once and reuse:

1. **ATE alignment scope** — per-sequence Sim(3) (most common) vs global Sim(3) (VGGT offline) vs SE(3) only (no scale, rarer). All papers above use per-sequence Sim(3) for streaming methods.
2. **Sim(3) library** — `evo` (Grupp et al.) is the de-facto reference. We currently use our own `umeyama_sim3` in `src/vggt_mamba/eval/metrics.py`. **Action: cross-validate our implementation against evo on a known trajectory before publishing**, ~1 hour of work.
3. **Depth median-scale alignment** — per-frame vs per-sequence. StreamVGGT uses per-frame, CUT3R/Stream3R/Point3R use per-sequence. Per-sequence is more permissive (one scale absorbs trajectory-wide depth bias); per-frame is stricter. Re-evaluating us under both removes one degree of comparison ambiguity.
4. **Depth max cap** — competitors use 10.0 m; we currently train + eval at 8.0 m. Re-evaluating at 10.0 m is half a day.
5. **Dynamic-region masking on TUM-dynamics** — some papers exclude pixels covered by walking humans from depth eval (using TUM's per-frame masks); some include them. Check each paper carefully.
6. **RPE δ** — δ=1 frame is standard for streaming; δ=10 frames is sometimes reported for medium-horizon drift; δ=1 second (= 30 frames at 30Hz) is used by SLAM papers.
7. **Valid-pixel mask** — TUM raw depth has zeros for "no return"; competitors agree on excluding zeros, but the exact threshold (`gt > 0` vs `gt > 1e-3` vs `gt in [0.1, 8.0]`) varies.

## Minimum-viable comparison set

Recommend prioritizing **CUT3R** and **Point3R** first:
- Both are most directly comparable (online streaming, ATE Sim(3) RMSE, depth_max=10.0)
- Both have lowest portability risk (vanilla PyTorch, no flash-attn-2 source builds)
- Both run on 16GB consumer GPUs without memory wrangling
- Combined P50: ~4 days for both
- These two cover the "we beat / are competitive with the recent streaming SOTA" claim for the ATE column

Defer **StreamVGGT** (high P90 days due to xformers/flash-attn), **VGGT** (memory wrangling + offline-vs-streaming asymmetry), **Spann3R** (benchmark mismatch on TUM).

## Total estimate

- P50 to first defensible matched Table 1 (CUT3R + Point3R as baselines): **~5 working days** including data download, eval-protocol matching, our-side re-evaluation at depth_max=10.0, and per-sequence + per-frame scale-alignment variants.
- P90: ~9 working days (account for one of the two builds having a surprise dependency issue).
- If we add StreamVGGT: +3-7 days.

## Alternative: ship with our-numbers-only + careful caveats

Risk-weighted: if the paper deadline is < 2 weeks out, the matched-protocol comparison may not be feasible. Fallback narrative:

> "We report our absolute numbers on held-out TUM-dynamics under our own evaluation protocol (Sim(3) per-sequence alignment, depth_max=8.0 m, per-frame median scale, dynamic regions included). Published competitor numbers use different protocols (depth_max=10.0 m, per-sequence vs per-frame scale, various dynamic-masking conventions); direct comparison is therefore approximate. A fully matched-protocol comparison is provided in the supplementary material."

This is defensible if the supplementary is genuine, but reviewers will note the asymmetry. Strong preference for matched comparison if time permits.

## Recommended next action

If we commit to matched comparison, the unblocking work that doesn't require GPU is **download Point3R + CUT3R checkpoints, set up eval-protocol harness, validate our `umeyama_sim3` against `evo`**. That's day 1; it doesn't block speed work or motion experiments.
