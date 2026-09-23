"""ARCHITECTURE.md 10's lightweight approve/reject action: a thin CLI
wrapping orchestrator/service.py's POST /resume,
for whoever finds that more convenient than a raw curl command. Does
nothing else - the actual interrupt/resume mechanism this calls is already
built and tested (orchestrator/graph.py, orchestrator/service.py).

Deliberately not a frontend button - CLAUDE.md's hard constraint is that
the frontend never calls any backend, directly or indirectly, so this
stays a separate, human-run tool. See docs/human_in_the_loop.md for the
full recommend-and-confirm workflow this is one step of.

Usage:
    python scripts/approve_gameweek.py 12 approve
    python scripts/approve_gameweek.py 12 reject --url http://orchestrator.example:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

DEFAULT_URL = "http://localhost:8000"


def resume_gameweek(gameweek: int, decision: str, base_url: str = DEFAULT_URL) -> dict:
    body = json.dumps({"gameweek": gameweek, "decision": decision}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/resume", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gameweek", type=int)
    parser.add_argument("decision", choices=["approve", "reject"])
    parser.add_argument("--url", default=DEFAULT_URL, help="orchestrator base URL (default: %(default)s)")
    args = parser.parse_args(argv)

    result = resume_gameweek(args.gameweek, args.decision, args.url)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
