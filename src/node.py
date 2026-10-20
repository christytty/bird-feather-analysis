"""Start one of the cluster roles on this node (used by the pixi tasks).

    python -m src.node redis        # head: Celery broker (password from BROKER_URL)
    python -m src.node worker       # worker: segmentation + VLM tasks
    python -m src.node flower       # head: Celery dashboard on :5555
    python -m src.node vlm-server   # Apple-silicon node: OpenAI-compatible VLM
    python -m src.node jupyter      # JupyterLab

Settings come from the environment or the repo's .env (see .env.example).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
DEFAULT_VLM_MODEL = "mlx-community/Qwen3.8-27B-4bit"


def _exec(argv: list[str]) -> None:
    exe = shutil.which(argv[0]) or argv[0]
    os.execv(exe, [exe, *argv[1:]])


def redis() -> None:
    broker = urlparse(os.getenv("BROKER_URL", "redis://localhost:6379/0"))
    argv = ["redis-server", "--port", str(broker.port or 6379), "--bind", "0.0.0.0",
            "--dir", str(REPO / "data"), "--dbfilename", "feather_dump.rdb", "--appendonly", "no"]
    if broker.password:
        argv += ["--requirepass", broker.password]
    else:
        argv += ["--protected-mode", "no"]
    (REPO / "data").mkdir(exist_ok=True)
    _exec(argv)


def worker() -> None:
    os.environ.setdefault("FEATHER_VLM_MODEL", DEFAULT_VLM_MODEL)
    if int(os.getenv("WORKER_CONCURRENCY", "1")) > 1:
        # the threads would share one FeatherProcessor: SAM's predictor holds the image it was
        # set to, so two slides at once can get each other's masks. For more throughput start a
        # second worker process instead (its own models): FEATHER_WORKER_NAME=feather2@%h.
        raise SystemExit("WORKER_CONCURRENCY > 1 is unsafe (shared SAM state); start a second worker process")
    _exec([sys.executable, "-m", "celery", "-A", "src.celery_app", "worker", "--pool=threads",
           f"--concurrency={os.getenv('WORKER_CONCURRENCY', '1')}", "--without-mingle",
           "--loglevel=INFO", f"--hostname={os.getenv('FEATHER_WORKER_NAME', 'feather@%h')}"])


def flower() -> None:
    _exec([sys.executable, "-m", "celery", "-A", "src.celery_app", "flower",
           "--address=0.0.0.0", f"--port={os.getenv('FLOWER_PORT', '5555')}"])


def vlm_server() -> None:
    # mlx_vlm.server's own main() forces uvicorn reload=True, which watches the
    # working directory tree; run its app through uvicorn directly instead.
    os.environ["PRELOAD_MODEL"] = os.getenv("FEATHER_VLM_MODEL", DEFAULT_VLM_MODEL)
    import uvicorn

    uvicorn.run("mlx_vlm.server:app", host="0.0.0.0", port=int(os.getenv("FEATHER_VLM_PORT", "8790")),
                workers=1)


def jupyter() -> None:
    _exec(["jupyter", "lab", "--ip=0.0.0.0", f"--port={os.getenv('JUPYTER_PORT', '8889')}",
           "--no-browser", f"--ServerApp.root_dir={REPO}"])


ROLES = {"redis": redis, "worker": worker, "flower": flower, "vlm-server": vlm_server, "jupyter": jupyter}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ROLES:
        sys.exit(__doc__)
    load_dotenv(REPO / ".env", override=False)
    os.chdir(REPO)
    os.environ["PYTHONPATH"] = str(REPO)
    ROLES[sys.argv[1]]()


if __name__ == "__main__":
    main()
