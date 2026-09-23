"""Weekly incremental mode: pull just the newly-finished gameweek, refreshed
fixtures, and our own current squad/bank/transfer state. This is the same
bronze/silver code backfill.py uses, scoped down to "what's new since last
time" - it's what the weekly scheduled pipeline actually runs
(ARCHITECTURE.md 8b; invoked via `docker compose run --rm ingestion weekly`
or `auto`, see scripts/weekly_pipeline.py).

Team id comes from the ``FPL_TEAM_ID`` environment variable, not a
hardcoded default - nothing in this module should need editing to point
at a different account.
"""

from __future__ import annotations

import os

from . import bronze, db, silver  # noqa: F401 - db only used for the type hint below
from .backfill import finished_gameweeks


def _now_cost_by_player(bootstrap: dict) -> dict[int, float]:
    return {e["id"]: e["now_cost"] / 10.0 for e in bootstrap["elements"]}


def run_weekly(conn: "db.Connection", *, team_id: int | None = None) -> dict:
    """Ingest the latest finished gameweek + fixtures + our entry state.

    Returns a small summary dict (mostly useful for the CLI / CronJob logs):
    which gameweek was ingested, how many player rows were written, and
    which gameweek our squad state was pulled for.
    """
    team_id = team_id or int(os.environ["FPL_TEAM_ID"])

    bootstrap = bronze.fetch_and_land_bootstrap_static(conn)
    silver.upsert_teams(conn, bootstrap)
    silver.upsert_players(conn, bootstrap)
    silver.upsert_gameweeks(conn, bootstrap)

    fixtures = bronze.fetch_and_land_fixtures(conn)
    silver.upsert_fixtures(conn, fixtures)

    finished = finished_gameweeks(bootstrap)
    latest_finished = max(finished) if finished else None

    rows_written = 0
    if latest_finished is not None:
        live = bronze.fetch_and_land_event_live(conn, latest_finished)
        # Unlike backfill, we DO pass today's prices here: this gameweek just
        # finished, so "current price" and "price at its deadline" are close
        # enough to be a real feature rather than a guess (see the docstring
        # on silver.upsert_player_gameweek_stats).
        rows_written = silver.upsert_player_gameweek_stats(
            conn, latest_finished, live, now_cost_by_player=_now_cost_by_player(bootstrap)
        )

    # Our own state: current-event picks if FPL has already rolled the
    # "current" gameweek forward, otherwise the one that just finished.
    current_events = [ev for ev in bootstrap["events"] if ev.get("is_current")]
    state_gw = current_events[0]["id"] if current_events else latest_finished

    entry = bronze.fetch_and_land_entry(conn, team_id)
    transfers = bronze.fetch_and_land_entry_transfers(conn, team_id)

    # Chip usage: from entry/history's authoritative `chips` list, not
    # derived from this gameweek's active_chip below - see
    # silver.upsert_chip_usage's docstring for why.
    history = bronze.fetch_and_land_entry_history(conn, team_id)
    silver.upsert_chip_usage(conn, history.get("chips", []))

    squad_ids: list[int] = []
    bank = team_value = 0.0
    if state_gw is not None:
        picks = bronze.fetch_and_land_entry_picks(conn, team_id, state_gw)
        silver.upsert_my_team_state(conn, state_gw, entry, picks, transfers)
        squad_ids = [p["element"] for p in picks.get("picks", [])]
        # Same entry_history fields silver.upsert_my_team_state itself reads
        # for the my_team_state table - returned here too so a caller (the
        # deadline-aware `auto` CLI mode, to trigger the orchestrator's real
        # /run) doesn't need a second Postgres round trip just to re-read
        # what this function already has in hand.
        history = picks.get("entry_history", {})
        bank = (history.get("bank") or 0) / 10.0
        team_value = (history.get("value") or 0) / 10.0

    # Refine OUR squad's price/ownership/transfer-balance history via
    # element-summary - bounded to ~15 players, not the full pool (see
    # silver.refine_player_gameweek_from_element_summary's docstring).
    for player_id in squad_ids:
        summary = bronze.fetch_and_land_element_summary(conn, player_id)
        silver.refine_player_gameweek_from_element_summary(conn, player_id, summary)

    return {
        "latest_finished_gw": latest_finished,
        "player_gameweek_rows_written": rows_written,
        "my_team_state_gw": state_gw,
        "squad_size": len(squad_ids),
        "bootstrap": bootstrap,
        "squad_ids": set(squad_ids),
        "bank": bank,
        "team_value": team_value,
    }
