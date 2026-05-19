#!/usr/bin/env bash
# Installs editable third_party packages (VGGT, V-JEPA 2, DINOv2) on first
# container start, then short-circuits on subsequent starts using a sentinel
# file in the bind-mounted repo. Then exec's whatever CMD was passed.

set -euo pipefail

REPO=/workspace/vggt-mamba
SENTINEL="${REPO}/.docker-third-party-installed"

if [ ! -f "${SENTINEL}" ]; then
    echo "[entrypoint] first start — installing third_party editable packages"

    for pkg in vggt vjepa2 dinov2; do
        DIR="${REPO}/third_party/${pkg}"
        if [ -d "${DIR}" ]; then
            echo "[entrypoint] pip install -e ${DIR}"
            python -m pip install --no-deps --no-cache-dir -e "${DIR}" || {
                echo "[entrypoint] WARN: failed to install ${pkg} (continuing)"
            }
        else
            echo "[entrypoint] missing ${DIR} — skipping"
        fi
    done

    # Install the local package itself
    if [ -f "${REPO}/pyproject.toml" ]; then
        echo "[entrypoint] pip install -e ${REPO}"
        python -m pip install --no-deps --no-cache-dir -e "${REPO}" || {
            echo "[entrypoint] WARN: failed to install vggt_mamba (continuing)"
        }
    fi

    touch "${SENTINEL}"
    echo "[entrypoint] setup done. (delete ${SENTINEL} to re-run)"
fi

exec "$@"
