#!/bin/bash
# Deploy the repo to every node, start the broker, workers and dashboard, and
# launch the distributed pipeline. Hosts, SSH and broker settings come from
# .env (see .env.example); each node runs the same pixi environment.
#
#   ./run_cluster.sh
#
# FEATHER_MANAGE_WORKERS=0 leaves the workers alone (for nodes whose workers are
# kept running by a service manager); FEATHER_START_REDIS=0 uses an existing broker.
set -euo pipefail
# shellcheck source=cluster_env.sh
. "$(dirname "$0")/cluster_env.sh"

: "${BROKER_URL:?set BROKER_URL in .env}"
SYNC_RAW_DATA="${SYNC_RAW_DATA:-1}"
MANAGE_WORKERS="${FEATHER_MANAGE_WORKERS:-1}"
START_REDIS="${FEATHER_START_REDIS:-1}"

echo "== Feather cluster: head $HEAD, ${#NODES[@]} worker node(s)"

echo "1. Preparing nodes (repo, pixi environment, .env)..."
for n in "${NODES[@]}"; do
  ( node_prepare "$n" && echo "   -> $n ready" ) &
done
wait

if [ "$SYNC_RAW_DATA" = "1" ] && [ "${#WORKERS[@]}" -gt 0 ]; then
  echo "2. Syncing raw images from the head to workers (when counts differ)..."
  head_count=$(node_ssh "$HEAD" "find $REPO_DIR/data/raw -maxdepth 1 -type f | wc -l" | tr -d ' ')
  for n in "${WORKERS[@]}"; do
    count=$(node_ssh "$n" "find $REPO_DIR/data/raw -maxdepth 1 -type f 2>/dev/null | wc -l" | tr -d ' ')
    if [ "$count" = "$head_count" ]; then
      echo "   -> $n already has $count images"
      continue
    fi
    echo "   -> $n: $count -> $head_count images"
    node_ssh "$n" "mkdir -p $REPO_DIR/data/raw"
    node_ssh "$HEAD" "cd $REPO_DIR/data && tar -cf - raw" | node_ssh "$n" "cd $REPO_DIR/data && tar -xf -"
  done
fi

if [ "$START_REDIS" = "1" ]; then
  echo "3. Starting the broker on the head..."
  node_ssh "$HEAD" "
    cd $REPO_DIR
    pkill -f 'redis-server .*feather_dump.rdb' >/dev/null 2>&1 || true
    nohup $PIXI run redis > redis.log 2>&1 < /dev/null &
  "
  sleep 2
fi

if [ "$MANAGE_WORKERS" = "1" ]; then
  echo "4. (Re)starting workers..."
  for n in "${NODES[@]}"; do
    node_ssh "$n" "
      cd $REPO_DIR
      pkill -f 'celery -A src.celery_app worker' >/dev/null 2>&1 || true
      sleep 2
      nohup $PIXI run worker > celery_worker.log 2>&1 < /dev/null &
    " &
  done
  wait
fi

echo "5. Starting the Flower dashboard on the head..."
node_ssh "$HEAD" "
  cd $REPO_DIR
  pkill -f 'celery -A src.celery_app flower' >/dev/null 2>&1 || true
  FLOWER_UNAUTHENTICATED_API=true nohup $PIXI run flower > flower.log 2>&1 < /dev/null &
"

echo "6. Launching the distributed pipeline on the head..."
node_ssh "$HEAD" "
  cd $REPO_DIR
  pkill -f 'src/full_run_distributed.py' >/dev/null 2>&1 || true
  nohup $PIXI run python src/full_run_distributed.py > distributed_pipeline.log 2>&1 < /dev/null &
"

echo "== Deployed"
echo "Flower:        http://$HEAD:5555"
echo "Pipeline log:  $REPO_DIR/distributed_pipeline.log (head)"
echo "Worker logs:   $REPO_DIR/celery_worker.log (each node)"
