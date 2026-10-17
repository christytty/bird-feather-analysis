#!/bin/bash
# Memory pressure on every node, then an interactive asitop on this one
# (Apple silicon). Nodes come from .env.
set -uo pipefail
# shellcheck source=cluster_env.sh
. "$(dirname "$0")/cluster_env.sh"

echo "=== Feather cluster monitor ==="
for n in "${NODES[@]}"; do
  printf 'Node %s: ' "$n"
  node_ssh "$n" "if command -v vm_stat >/dev/null; then vm_stat | grep 'Pages active'; else free -h | sed -n 2p; fi" 2>/dev/null || echo "offline"
done
echo "----------------------------------------"
if [ "$(uname -s)" = "Darwin" ]; then
  command -v asitop >/dev/null 2>&1 || pipx install asitop >/dev/null 2>&1 || pip install --user asitop >/dev/null 2>&1
  echo "Launching asitop on this node (press q to quit)..."
  sudo asitop
fi
