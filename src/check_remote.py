"""Check that this machine can drive the cluster: broker, workers and SSH.

Reads the same settings as the notebooks (BROKER_URL, RESULT_BACKEND,
FEATHER_SSH_KEY, ...), loading .env from the repo root first.

    python -m src.check_remote
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.notebook_remote_runtime import (  # noqa: E402
    RemoteClusterIO,
    assert_tcp_reachable,
    create_celery_client,
    load_notebook_config_from_env,
)


def main() -> int:
    config = load_notebook_config_from_env()
    failed = False

    try:
        assert_tcp_reachable(config.broker_url, "Broker")
        app = create_celery_client(config)
        with app.connection_for_write() as conn:
            conn.ensure_connection(max_retries=1)
        print(f"broker   ok   {config.broker_url.split('@')[-1]}")
        replies = app.control.inspect(timeout=3).ping() or {}
        print(f"workers  {'ok' if replies else 'NONE'}   {', '.join(sorted(replies)) or 'no worker answered'}")
        failed |= not replies
    except Exception as exc:  # noqa: BLE001
        print(f"broker   FAIL {type(exc).__name__}: {exc}")
        failed = True

    io = RemoteClusterIO(config)
    for host in config.cluster_hosts:
        try:
            if host == config.head_host:
                n = len(io.list_remote_images(host))
                print(f"ssh      ok   {host} ({n} input images)")
            else:
                n = len(io.list_remote_outputs(host))
                print(f"ssh      ok   {host} ({n} outputs)")
        except Exception as exc:  # noqa: BLE001
            print(f"ssh      FAIL {host}: {exc}")
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
