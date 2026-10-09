"""The weekly automated trigger, now that Kubernetes has been retired in
favor of Docker Compose (see ARCHITECTURE.md 8b - a CronJob doesn't exist
without a cluster). Invoked by the "FPL pipeline wake" Windows Task
Scheduler task (WakeToRun, daily) once it's confirmed Docker Desktop is up;
this script's own job starts there, not with waking the machine.

Brings the Compose stack up, then runs the same deadline-aware CLI that
used to run inside the CronJob's pod (ingestion/__main__.py's `auto` mode:
ingestion/deadline.py's 24-36h check, then ingestion/trigger.py's real call
to the orchestrator's POST /run if it decides to trigger) - unchanged
logic, just invoked via `docker compose run` instead of a Job container.

Then exports the pending proposal (scripts/export_live_gameweek.py) and, if
the static JSON changed, commits and pushes it - the push is what fires
.github/workflows/deploy-pages.yml, so the site updates with no manual step.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = "frontend/public/data"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("weekly_pipeline")


def main() -> int:
    log.info("bringing up the Compose stack")
    subprocess.run(["docker", "compose", "up", "-d", "--wait"], cwd=REPO_ROOT, check=True)

    log.info("running the deadline-aware ingestion check")
    result = subprocess.run(
        ["docker", "compose", "run", "--rm", "ingestion", "auto"], cwd=REPO_ROOT
    )
    if result.returncode != 0:
        return result.returncode

    log.info("exporting the pending proposal for the frontend")
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "export_live_gameweek.py")], cwd=REPO_ROOT, check=True)
    changed = subprocess.run(
        ["git", "status", "--porcelain", "--", DATA_DIR], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    if not changed:
        log.info("frontend data unchanged - nothing to publish")
        return 0

    log.info("publishing updated frontend data")
    # Pathspec-limited commit: never sweeps up unrelated work in the tree.
    subprocess.run(["git", "add", "--", DATA_DIR], cwd=REPO_ROOT, check=True)
    subprocess.run(["git", "commit", "-m", "data: weekly export of the pending proposal", "--", DATA_DIR], cwd=REPO_ROOT, check=True)
    subprocess.run(["git", "push", "origin", "HEAD:main"], cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
