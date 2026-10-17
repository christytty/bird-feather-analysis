"""Archive finished cluster runs from the head node.

Asks the head node's `sync-runs` command (run over SSH) to copy
data/runs/<run_id>/ to the run archive, gathering the workers' logs and
metrics first, or to report how the last sync went. Test runs (max_images
other than "all", or ids containing test/smoke/debug/tmp) and runs without
processed images are skipped. The archive target and schedule are configured
on the head node, not here.

    python -m src.sync_runs           # start a sync
    python -m src.sync_runs status    # last result and log lines
"""
from __future__ import annotations

import subprocess
import sys

from src.notebook_remote_runtime import RemoteClusterIO, load_notebook_config_from_env


def main() -> int:
    action = "sync-runs status" if sys.argv[1:] == ["status"] else "sync-runs"
    config = load_notebook_config_from_env()
    io = RemoteClusterIO(config)
    return subprocess.call([*io._ssh_base_cmd(config.head_host), action])


if __name__ == "__main__":
    sys.exit(main())
