#!/usr/bin/env bash
# Build the vggt-mamba dev image.
# Run from the repo root or anywhere — paths are resolved.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${REPO_ROOT}"

echo "[build] building vggt-mamba:latest from ${REPO_ROOT}"
docker compose -f docker/docker-compose.yml build "$@"
echo "[build] done"
