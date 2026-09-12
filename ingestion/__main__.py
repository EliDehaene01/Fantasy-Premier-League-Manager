"""CLI entrypoint: ``python -m ingestion backfill`` / ``python -m ingestion weekly``.

This is the thing a Kubernetes CronJob actually execs (ARCHITECTURE.md
section 8's ``ingestion`` Job) - a plain argparse CLI over the two functions
in backfill.py/weekly.py, not a notebook or an interactive script, so
containerizing it later is "point the container at this module", no rewrite.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .backfill import backfill_current_season
from .db import get_connection
from .weekly import run_weekly

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingestion")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ingestion")
    parser.add_argument(
        "--database-url", default=None, help="Postgres DSN; defaults to $DATABASE_URL"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_backfill = sub.add_parser("backfill", help="populate silver from GW1 to the latest finished GW")
    p_backfill.add_argument("--from-gw", type=int, default=1)
    p_backfill.add_argument("--upto-gw", type=int, default=None)

    p_weekly = sub.add_parser("weekly", help="ingest the newly-finished GW + fixtures + our entry state")
    p_weekly.add_argument("--team-id", type=int, default=None, help="defaults to $FPL_TEAM_ID")

    args = parser.parse_args(argv)
    conn = get_connection(args.database_url)

    if args.mode == "backfill":
        written = backfill_current_season(conn, from_gw=args.from_gw, upto_gw=args.upto_gw)
        log.info("backfill wrote gameweeks: %s", written)
    elif args.mode == "weekly":
        summary = run_weekly(conn, team_id=args.team_id)
        log.info("weekly run summary: %s", summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
