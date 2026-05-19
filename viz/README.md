# viz/

Visualizations for the user. Claude doesn't look at these — the user is the
only viewer. Their visual reports are ground truth.

Every experiment in this project writes outputs here. Layout:

```
viz/output/
├── phase0_pca/             V-JEPA vs DINOv2 PCA comparison (Phase 0 gate)
├── tum_window/             TUM-RGBD sample windows (rgb + depth + valid)
├── phase1_dinov2/          DINOv2 run — curves + per-eval prediction renders
└── phase1_vjepa/           V-JEPA  run — curves + per-eval prediction renders
```

Each script writes to a fixed subdir under `output/` so the user can just
open the folder and look.

## Scripts

- `show_tum_window.py` — dump one TUM data window as a grid of RGB/depth/valid PNGs.
- `plot_phase1_curves.py` — read a `metrics.jsonl` and plot L1/log/MVC curves over steps.
- `show_phase1_preds.py` — load a Phase 1 checkpoint, run on a fixed eval batch, dump pred depth vs GT depth side-by-side.
- `../scripts/smoke/03_pca_reproduce.py` — PCA comparison of encoders (Phase 0 gate); saves to `viz/output/phase0_pca/`.

## Convention

Inside any training script, every eval step also writes a prediction PNG to
the run's viz dir. So `viz/output/phase1_dinov2/step_000500_preds.png` shows
what the model looked like at step 500.
