"""Backfill mode: populate silver for every finished gameweek of the CURRENT
season, from gameweek 1 through whatever's most recently finished.

WHY THIS DOESN'T ALSO BACKFILL LAST SEASON
-------------------------------------------
The official FPL API has no gameweek-level view of a completed season. Once
a season ends, ``element-summary/{id}``'s ``history`` (per-gameweek) is
cleared and only ``history_past`` remains - one row per player per SEASON
(total points, minutes, etc. for the whole year), not per gameweek. There is
no endpoint that gives a finished season's ``event/{gw}/live`` for gw in
1..38 after the fact. So last season simply cannot be backfilled this way,
at any cost in API calls - it's not that it's expensive, the data doesn't
exist here any more. The vaastav archive (``merged_gws_2025-26.csv`` for the
season already completed) remains the sole source for that, frozen, and
stays exactly as-is (see ``data/reconcile_archive.py`` for how the two get
reconciled into one shape for training).

The CURRENT season is different: it's in progress, so every gameweek that
HAS finished is still available at full per-player granularity via
``event/{gw}/live`` - one call per gameweek, not one call per player - which
is what this function loops.
"""

from __future__ import annotations

from . import bronze, db, silver  # noqa: F401 - db only used for the type hint below


def finished_gameweeks(bootstrap: dict) -> list[int]:
    return sorted(ev["id"] for ev in bootstrap["events"] if ev.get("finished"))


def backfill_current_season(
    conn: "db.Connection", *, from_gw: int = 1, upto_gw: int | None = None
) -> list[int]:
    """Populate silver for gameweeks ``from_gw..upto_gw`` of the current
    season (``upto_gw`` defaults to the most recently finished gameweek).

    Idempotent: every table this touches is upserted (see silver.py), so
    running this twice - or re-running it mid-season once more gameweeks
    have finished - never duplicates a row; it just replaces gameweeks it
    already had and adds the new ones.

    Returns the list of gameweeks actually written.
    """
    bootstrap = bronze.fetch_and_land_bootstrap_static(conn)
    silver.upsert_teams(conn, bootstrap)
    silver.upsert_players(conn, bootstrap)
    silver.upsert_gameweeks(conn, bootstrap)

    fixtures = bronze.fetch_and_land_fixtures(conn)
    silver.upsert_fixtures(conn, fixtures)

    finished = finished_gameweeks(bootstrap)
    if upto_gw is not None:
        finished = [gw for gw in finished if gw <= upto_gw]
    finished = [gw for gw in finished if gw >= from_gw]

    written: list[int] = []
    for gw in finished:
        # Deliberately no now_cost_by_player here - see the docstring on
        # upsert_player_gameweek_stats: today's price would be wrong for a
        # gameweek from weeks/months ago, so backfilled rows get NULL price
        # rather than a plausible-looking guess.
        live = bronze.fetch_and_land_event_live(conn, gw)
        silver.upsert_player_gameweek_stats(conn, gw, live)
        written.append(gw)

    return written
