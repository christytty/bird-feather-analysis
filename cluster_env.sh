# shellcheck shell=bash
# Shared settings for the cluster scripts. Sourced, not run.
# Everything comes from the environment or .env in the repo root (see .env.example).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT/.env"
  set +a
fi

: "${FEATHER_HEAD_HOST:?set FEATHER_HEAD_HOST in .env}"
HEAD="$FEATHER_HEAD_HOST"
# All nodes that run workers, head included (comma-separated).
IFS=',' read -r -a NODES <<< "${FEATHER_CLUSTER_HOSTS:-$HEAD}"
WORKERS=()
for n in "${NODES[@]}"; do [ "$n" != "$HEAD" ] && WORKERS+=("$n"); done

SSH_USER="${FEATHER_SSH_USER:-$USER}"
REPO_DIR="${FEATHER_REMOTE_REPO_DIR:-~/bird-feather-analysis}"
REPO_URL="${FEATHER_REPO_URL:-https://github.com/ns-mkusper/bird-feather-analysis.git}"
REPO_BRANCH="${FEATHER_REPO_BRANCH:-main}"
# pixi on the nodes; installed into ~/.pixi by node_prepare when missing.
PIXI='$HOME/.pixi/bin/pixi'

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)
if [ -n "${FEATHER_SSH_KEY:-}" ]; then
  SSH_OPTS+=(-i "${FEATHER_SSH_KEY/#\~/$HOME}" -o IdentitiesOnly=yes)
fi

node_ssh() {  # node_ssh <host> <command...>
  local host="$1"; shift
  ssh "${SSH_OPTS[@]}" "$SSH_USER@$host" "$@"
}

# Clone or update the repo, install pixi and the environment, copy .env.
node_prepare() {
  local host="$1"
  node_ssh "$host" "
    set -e
    [ -x $PIXI ] || curl -fsSL https://pixi.sh/install.sh | PIXI_NO_PATH_UPDATE=1 bash >/dev/null
    if [ ! -d $REPO_DIR/.git ]; then git clone -q '$REPO_URL' $REPO_DIR; fi
    cd $REPO_DIR && git fetch -q origin '$REPO_BRANCH' && git checkout -q '$REPO_BRANCH' && git reset -q --hard 'origin/$REPO_BRANCH'
    $PIXI install -q
  "
  if [ -f "$ROOT/.env" ]; then
    node_ssh "$host" "umask 077 && cat > $REPO_DIR/.env" < "$ROOT/.env"
  fi
}
