"""The weekly automated trigger, now that Kubernetes has been retired in
favor of Docker Compose (see TODO.md's Phase 5 note - a CronJob doesn't
exist without a cluster). Invoked by the "FPL pipeline wake" Windows Task
Scheduler task (WakeToRun, daily) once it's confirmed Docker Desktop is up;
this script's own job starts there, not with waking the machine.

Brings the Compose stack up, then runs the same deadline-aware CLI that
used to run inside the CronJob's pod (ingestion/__main__.py's `auto` mode:
ingestion/deadline.py's 24-36h check, then ingestion/trigger.py's real call
to the orchestrator's POST /run if it decides to trigger) - unchanged
logic, just invoked via `docker compose run` instead of a Job container.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("weekly_pipeline")


def main() -> int:
    log.info("bringing up the Compose stack")
    subprocess.run(["docker", "compose", "up", "-d", "--wait"], cwd=REPO_ROOT, check=True)

    log.info("running the deadline-aware ingestion check")
    result = subprocess.run(
        ["docker", "compose", "run", "--rm", "ingestion", "auto"], cwd=REPO_ROOT
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
