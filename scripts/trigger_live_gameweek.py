"""The missing "something calls POST /run" from docs/human_in_the_loop.md
step 2: assembles a real gameweek's player_pool from the live Postgres
tables ingestion/weekly.py just wrote (players + player_gameweek_stats for
price, my_team_state for the current squad/bank) and POSTs it to the
orchestrator Deployment's /run endpoint - the actual live-mode trigger, not
a backtest/demo stand-in (see backtest/export_sample_gameweek.py's own
docstring: that one seeds a *backtest* schema and fakes "live" for frontend
demo data only).

Usage:
    python scripts/trigger_live_gameweek.py 5 --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

import psycopg
from psycopg.rows import dict_row

from ingestion.fpl_client import get_bootstrap_static

DEFAULT_URL = "http://localhost:8000"


def get_connection():
    # ingestion.db.get_connection() pulls in psycopg2, which this machine's
    # Application Control policy currently blocks (psycopg2's compiled
    # _psycopg DLL specifically - confirmed via a direct import attempt,
    # while psycopg v3 imports fine) - connecting directly with psycopg v3
    # here sidesteps that without touching ingestion/db.py's real,
    # psycopg2-based production path.
    return psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row)


def build_initial_state(conn) -> dict:
    team_row = conn.execute(
        "SELECT gw, bank, team_value, picks FROM my_team_state ORDER BY gw DESC LIMIT 1"
    ).fetchone()
    squad_ids = {pick["element"] for pick in team_row["picks"]}

    rows = conn.execute(
        "SELECT p.player_id, p.team_id AS club_id, p.position, s.price "
        "FROM players p JOIN player_gameweek_stats s "
        "  ON s.player_id = p.player_id AND s.gw = %s "
        "WHERE s.price IS NOT NULL",
        (team_row["gw"],),
    ).fetchall()

    # chance_of_playing_this_round / news are live-only bootstrap-static
    # fields (ingestion/silver.py never persists them - see its own schema)
    # that the News agent's Tier 1 (agents/news/tier1.py::check_tier1) reads
    # straight off each PlayerEntry to decide whether a player is even
    # ambiguous enough to hand to Tier 2. Omitting them here would make
    # every player look like "no news, fully fit" and Tier 1 would resolve
    # the entire pool by itself - exactly the silent Tier-1-only fallback
    # this run needs to NOT reproduce.
    availability = {e["id"]: e for e in get_bootstrap_static()["elements"]}

    pool = [
        {
            "player_id": r["player_id"],
            "club_id": r["club_id"],
            "position": r["position"],
            "price": r["price"],
            "in_current_squad": r["player_id"] in squad_ids,
            "chance_of_playing_this_round": availability.get(r["player_id"], {}).get("chance_of_playing_this_round"),
            "news": availability.get(r["player_id"], {}).get("news") or None,
        }
        for r in rows
    ]
    budget = round(team_row["team_value"] + team_row["bank"], 1)
    return {"player_pool": pool, "budget": budget, "free_transfers": 1, "hit_cost": 4.0}


def trigger(gameweek: int, base_url: str = DEFAULT_URL) -> dict:
    conn = get_connection()
    initial_state = build_initial_state(conn)
    body = json.dumps({"gameweek": gameweek, "initial_state": initial_state}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/run", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gameweek", type=int)
    parser.add_argument("--url", default=DEFAULT_URL, help="orchestrator base URL (default: %(default)s)")
    args = parser.parse_args(argv)

    result = trigger(args.gameweek, args.url)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
