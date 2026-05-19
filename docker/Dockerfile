# vggt-mamba dev image
# Base: official PyTorch 2.7.1 + CUDA 12.8 + cuDNN 9 (Blackwell sm_120 supported)
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# Avoid interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive

# System deps — git, build tools, video/image I/O, headless OpenGL for open3d
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        unzip \
        ca-certificates \
        build-essential \
        ninja-build \
        cmake \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxkbcommon-x11-0 \
        libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

# Workspace layout — repo and datasets are bind-mounted at runtime
WORKDIR /workspace/vggt-mamba

# Python deps — install everything *except* torch (already in base) and the
# CUDA-kernel packages (mamba-ssm, causal-conv1d), which we build from source.
RUN python -m pip install --no-cache-dir --upgrade pip wheel packaging ninja

# Bulk Python deps. Pinned loosely; pyproject.toml is the source of truth.
RUN python -m pip install --no-cache-dir \
        einops>=0.8.0 \
        "numpy>=1.26,<2.0" \
        scipy>=1.13 \
        scikit-learn>=1.5 \
        pillow>=10.0 \
        opencv-python-headless>=4.10 \
        imageio>=2.34 \
        imageio-ffmpeg>=0.5 \
        matplotlib>=3.9 \
        tqdm>=4.66 \
        rich>=13.7 \
        hydra-core>=1.3 \
        omegaconf>=2.3 \
        wandb>=0.17 \
        tensorboard>=2.16 \
        huggingface-hub>=0.24 \
        transformers>=4.44 \
        timm>=1.0 \
        safetensors>=0.4 \
        rerun-sdk>=0.18 \
        open3d>=0.18 \
        trimesh>=4.4 \
        plyfile>=1.0 \
        iopath>=0.1.10 \
        pyyaml>=6.0 \
        jupyterlab>=4.2 \
        ipykernel>=6.29 \
        ipywidgets>=8.1

# Build causal-conv1d and mamba-ssm from source for Blackwell sm_120.
# The PyPI wheels do not ship sm_120 kernels as of mid-2026.
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV MAMBA_FORCE_BUILD=TRUE
ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE
ENV MAX_JOBS=4

RUN git clone --depth 1 https://github.com/Dao-AILab/causal-conv1d.git /tmp/causal-conv1d \
    && cd /tmp/causal-conv1d \
    && python -m pip install --no-build-isolation --no-cache-dir . \
    && cd / && rm -rf /tmp/causal-conv1d

# Pin to v2.2.6.post3 — the last pure Mamba-2 release. v2.3+ adds Mamba-3
# layers that pull tilelang/quack-kernels/cuda-toolkit-13.x which forcibly
# upgrade torch to 2.11+cu130 and break the base image's torchvision.
RUN git clone --branch v2.2.6.post3 --depth 1 \
        https://github.com/state-spaces/mamba.git /tmp/mamba \
    && cd /tmp/mamba \
    && python -m pip install --no-build-isolation --no-cache-dir . \
    && cd / && rm -rf /tmp/mamba \
    # Patch mamba_ssm/__init__.py to NOT import the LMHead model at package
    # load. The LMHead pulls `transformers.generation.GreedySearchDecoderOnlyOutput`
    # which was removed in transformers >=4.40. We only need the Mamba2 layer.
    && MAMBA_INIT=$(python -c "import sysconfig,os; print(os.path.join(sysconfig.get_paths()['purelib'], 'mamba_ssm', '__init__.py'))") \
    && echo "patching ${MAMBA_INIT}" \
    && sed -i 's|^from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel|# patched: &|' "${MAMBA_INIT}"

# Entry script: installs the third_party editable packages from the
# bind-mounted repo on first container start (then short-circuits).
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV VGGT_MAMBA_DATA_ROOT=/workspace/datasets

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
