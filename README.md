# vggt-mamba

Linear-scaling, sequential-video 3D reconstruction via a Mamba-2 SSM world-model state.
CVPR submission target. See `ROADMAP.md` for the full plan.

## Thesis (one sentence)

VGGT throws away the temporal structure of video by treating frames as a permutation-invariant
set; we model the stream with a Mamba-2 state that *is* the running scene, giving constant
memory in sequence length without the quality cost.

## Quickstart (Docker)

The dev env is a Docker image. Host requirements:
NVIDIA driver supporting CUDA 12.8+, Docker 27+, `nvidia-container-toolkit`, `docker compose`.

```bash
cd /home/raghav/workspace/CVPR/vggt-mamba

# 1. Build the image (~15–30 min first time; subsequent builds are cached)
./docker/build.sh

# 2. Open an interactive shell inside the container
./docker/run.sh

# Inside the container, the repo is mounted at /workspace/vggt-mamba
# and datasets at /workspace/datasets. The entrypoint installs the
# editable third_party packages (VGGT, V-JEPA 2, DINOv2) on first start.
```

Other usage patterns:

```bash
# Run a script directly
./docker/run.sh python scripts/smoke/02_smoke_mamba.py

# Launch Jupyter on host port 8888
./docker/run.sh jupyter
```

## What's in the image

- `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel` base — sm_120 supported.
- Built from source for Blackwell: `causal-conv1d`, `mamba-ssm` (`TORCH_CUDA_ARCH_LIST=12.0`).
- All Python deps from `pyproject.toml` (einops, transformers, timm, open3d, rerun-sdk, …).
- VGGT / V-JEPA 2 / DINOv2 installed editable from `third_party/` at runtime.

Verified on RTX 5070 Ti / Blackwell sm_120 / driver 580.126 / host CUDA 12.8.

## Data layout

`data/` is a symlink to `/home/raghav/workspace/datasets/` (shared across projects).
Inside the container it shows up at `/workspace/datasets`.

```
/home/raghav/workspace/datasets/
├── co3d_v2/        (Phase 1 training)
├── tum_rgbd/       (Phase 1 training + Phase 5 dynamic-scene eval)
├── sintel/         (Phase 1 depth eval)
├── 7scenes/        (Phase 1 recon eval)
└── weights/        (VGGT, V-JEPA 2.1, DINOv2 ViT-L .pt files)
```

Download via scripts in `scripts/data/` — none auto-run. Weight URLs are direct
(`fbaipublicfiles.com`, `huggingface.co/.../resolve/...`), no `huggingface_hub` library needed.

## Repo layout

```
src/vggt_mamba/         Python package
  data/                 dataset loaders
  models/
    encoders/           V-JEPA, DINOv2, VGGT-tokenizer wrappers
    aggregators/        Mamba block, attention block
    heads/              camera, depth, pointmap, track
  losses/
  train/                training loops
  eval/                 metrics + benchmark wrappers
  utils/

docker/                 Dockerfile, docker-compose.yml, helper scripts
third_party/            git clones (vggt, vjepa2, mamba, dinov2 — gitignored)
scripts/
  data/                 dataset download scripts
  smoke/                Phase 0 smoke tests
configs/                Hydra configs per phase
notebooks/              exploratory work
viz/                    rerun/open3d viewers
experiments/            run outputs (gitignored)
Relevant_Papers/        reference PDFs

ROADMAP.md              the plan
```

## Phase status

- Phase 0 (setup) — **in progress**
- Phase 1 (tokenizer probe) — not started
- Phases 2–6 — see ROADMAP.md

## Notes for editing on the host (VS Code etc.)

The pyproject.toml lists deps but they're only installed *inside* the container.
The IDE will show "package not installed" hints for them — that's expected, ignore.
If you want the IDE to resolve them, point its Python interpreter at the dev
container (VS Code's "Dev Containers" extension does this automatically using
`.devcontainer/devcontainer.json` — happy to add that if you want).
