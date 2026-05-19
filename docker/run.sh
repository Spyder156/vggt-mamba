#!/usr/bin/env bash
# Start (or re-attach to) the dev container.
#   ./docker/run.sh              -> opens an interactive bash inside
#   ./docker/run.sh python -c .. -> runs a one-shot command
#   ./docker/run.sh jupyter      -> launches jupyterlab on host port 8888

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${REPO_ROOT}"

if [ $# -eq 0 ]; then
    docker compose -f docker/docker-compose.yml run --rm --service-ports dev bash
elif [ "$1" = "jupyter" ]; then
    docker compose -f docker/docker-compose.yml run --rm --service-ports dev \
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
                    --NotebookApp.token='' --NotebookApp.password=''
else
    docker compose -f docker/docker-compose.yml run --rm --service-ports dev "$@"
fi
