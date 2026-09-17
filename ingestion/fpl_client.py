"""Thin HTTP client for the official FPL public API - the bronze layer's only
network boundary. Every other ingestion module calls the API exclusively
through the functions here, never with a bare ``requests.get``, so the
politeness rules (User-Agent, delay, timeout) apply everywhere automatically.

This API has no auth and no published rate limit, but it is also not an
officially documented developer product (see CLAUDE.md) - there is no SLA to
lean on if we hammer it. Being a reasonable citizen here means: identify
ourselves, and never fire requests back-to-back.
"""

from __future__ import annotations

import time
from typing import Any

import requests

BASE_URL = "https://fantasy.premierleague.com/api"

# Identifies this project to FPL's ops team without embedding anyone's
# personal contact details in source that ends up in a public repo.
USER_AGENT = (
    "FPL-Agents-Manager/0.1 "
    "(+https://github.com/EliDehaene01/Fantasy-Premier-League-Manager; "
    "personal portfolio project, read-only public API use)"
)

# ponytail: a flat sleep-before-every-call is the simplest thing that is
# unambiguously polite. Upgrade to a real token-bucket/backoff scheme only if
# this ever starts getting 429s - it hasn't.
REQUEST_DELAY_SECONDS = 0.75
TIMEOUT_SECONDS = 15

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def _get(path: str) -> Any:
    """GET one endpoint under ``BASE_URL``, decoded JSON.

    Sleeps first (not after) so the delay applies even if the caller is only
    making one request - the failure mode we're avoiding is a tight loop, not
    the time between two isolated calls.
    """
    time.sleep(REQUEST_DELAY_SECONDS)
    resp = _session.get(f"{BASE_URL}/{path}", timeout=TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()


def get_bootstrap_static() -> dict:
    """Players, teams, gameweeks (``events``) and position master data.

    The one endpoint almost everything else depends on: player/team ids,
    names, current price, and which gameweeks are finished.
    """
    return _get("bootstrap-static/")


def get_fixtures() -> list:
    """Every fixture for the season: gw, teams, kickoff time, score, FDR."""
    return _get("fixtures/")


def get_event_live(gw: int) -> dict:
    """Every player's actual stats for one *finished* gameweek, in one call.

    This is what backfill uses instead of looping ``element-summary`` per
    player (~700 calls) - see CLAUDE.md / the backfill module docstring.
    """
    return _get(f"event/{gw}/live/")


def get_element_summary(player_id: int) -> dict:
    """One player's own per-gameweek history plus upcoming fixtures.

    Used sparingly (our ~15-player squad, not the full player pool) for the
    fields ``event/{gw}/live`` doesn't carry: historical price and ownership
    for a specific player. See ``ingestion/silver.py``.
    """
    return _get(f"element-summary/{player_id}/")


def get_entry(team_id: int) -> dict:
    """Our own manager summary: overall points/rank, bank, squad value."""
    return _get(f"entry/{team_id}/")


def get_entry_picks(team_id: int, gw: int) -> dict:
    """Our squad's 15 picks and active chip for one gameweek."""
    return _get(f"entry/{team_id}/event/{gw}/picks/")


def get_entry_transfers(team_id: int) -> list:
    """Every transfer we've made this season, in chronological order."""
    return _get(f"entry/{team_id}/transfers/")


def get_entry_history(team_id: int) -> dict:
    """Season summary for our entry, including the authoritative ``chips``
    list: every chip played this season, each as ``{name, time, event}``.

    This is the one source that's complete regardless of which individual
    gameweeks the weekly job has actually run for - unlike
    ``my_team_state.active_chip`` (populated per-gameweek from
    ``get_entry_picks``), which only reflects whatever gameweek happened to
    be current when the job ran that week. See ingestion/silver.py's
    ``upsert_chip_usage`` and docs/ingestion_schema.md for why this endpoint,
    not that column, is what the Chips agent reads chip availability from.
    """
    return _get(f"entry/{team_id}/history/")
