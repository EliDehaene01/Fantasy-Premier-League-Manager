"""Bronze layer: land a raw API response exactly as received.

No parsing, no renaming, no type coercion - that is deliberate. If a later
silver-parsing bug is found, the fix is "reprocess bronze", never "we lost
the original data and have to wait for the API to give it back". Every
function here does one thing: call the FPL API through ``fpl_client`` and
insert the raw JSON into ``bronze_responses`` alongside which endpoint it
came from, what parameters were used, and when it was pulled.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from psycopg2.extras import Json

from . import db, fpl_client


def land_raw(conn, endpoint: str, params: dict, payload: Any) -> None:
    """Insert one raw response. Append-only - see the note in db.py.

    ``params``/``payload`` go in as ``Json(...)`` (psycopg2's adapter for
    Postgres's ``json``/``jsonb`` types) rather than a ``json.dumps(...)``
    string, so Postgres receives them typed as JSON directly instead of a
    plain-text parameter it would have to cast.
    """
    conn.execute(
        "INSERT INTO bronze_responses (endpoint, params, payload, ingested_at) "
        "VALUES (%s, %s, %s, %s)",
        (
            endpoint,
            Json(params),
            Json(payload),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# One "fetch_and_land_*" per endpoint: calls fpl_client, lands the raw
# response in bronze, and returns the parsed JSON so the caller (silver.py)
# doesn't have to hit the API again to get a Python object to parse.
# ---------------------------------------------------------------------------
def fetch_and_land_bootstrap_static(conn: "db.Connection") -> dict:
    payload = fpl_client.get_bootstrap_static()
    land_raw(conn, "bootstrap-static", {}, payload)
    return payload


def fetch_and_land_fixtures(conn: "db.Connection") -> list:
    payload = fpl_client.get_fixtures()
    land_raw(conn, "fixtures", {}, payload)
    return payload


def fetch_and_land_event_live(conn: "db.Connection", gw: int) -> dict:
    payload = fpl_client.get_event_live(gw)
    land_raw(conn, "event-live", {"gw": gw}, payload)
    return payload


def fetch_and_land_element_summary(conn: "db.Connection", player_id: int) -> dict:
    payload = fpl_client.get_element_summary(player_id)
    land_raw(conn, "element-summary", {"player_id": player_id}, payload)
    return payload


def fetch_and_land_entry(conn: "db.Connection", team_id: int) -> dict:
    payload = fpl_client.get_entry(team_id)
    land_raw(conn, "entry", {"team_id": team_id}, payload)
    return payload


def fetch_and_land_entry_picks(conn: "db.Connection", team_id: int, gw: int) -> dict:
    payload = fpl_client.get_entry_picks(team_id, gw)
    land_raw(conn, "entry-picks", {"team_id": team_id, "gw": gw}, payload)
    return payload


def fetch_and_land_entry_transfers(conn: "db.Connection", team_id: int) -> list:
    payload = fpl_client.get_entry_transfers(team_id)
    land_raw(conn, "entry-transfers", {"team_id": team_id}, payload)
    return payload
