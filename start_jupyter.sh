#!/bin/bash
# JupyterLab on this node in the background (pixi environment, repo root as
# the file browser root). Port from JUPYTER_PORT (default 8889).
cd "$(dirname "$0")" || exit 1
nohup pixi run jupyter > jupyter.log 2>&1 < /dev/null &
echo "JupyterLab starting on port ${JUPYTER_PORT:-8889}; log: jupyter.log"
