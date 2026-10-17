"""Copy finished cluster runs from the head node to Google Drive.

The head node uploads data/runs/<run_id>/ to the shared Drive folder
Feather_Molt_Project/runs/<run_id>/, gathering the workers' logs and metrics
first. Test runs (max_images other than "all", or ids containing test/smoke/
debug/tmp) and runs without processed images are skipped. It also runs on its
own every hour; this just starts it now or shows how it went.

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
