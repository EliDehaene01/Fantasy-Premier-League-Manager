"""Tests for the bronze/silver ingestion pipeline.

No real network calls anywhere here - every ``fpl_client.get_*`` function is
monkeypatched to return small, synthetic, FPL-API-shaped fixtures (2 teams,
2 players, 2 finished gameweeks). That keeps these tests fast, deterministic,
and independent of the live FPL API or our real squad, while still
exercising the actual bronze -> silver code path end to end - now against a
real local Postgres instance (``$DATABASE_URL``) rather than sqlite
``:memory:``.

MIGRATION NOTE (sqlite -> Postgres): everything below still tests the exact
same idempotency/shape guarantees the sqlite version did, but the FIXTURE
PLUMBING for getting a clean, isolated connection had to change, since
Postgres has no ``:memory:`` equivalent - it's one running server, not a new
throwaway database per connection:

  * The ``conn`` fixture now connects to a dedicated ``test_ingestion``
    schema (via ``get_connection(schema=...)``) rather than the real
    ``public`` schema production ingestion writes to, and truncates it
    before every test - a stand-in for "a fresh empty database", not a
    change to what's being asserted.
  * ``test_backfill_vs_weekly_same_row_shape`` used to open a SECOND,
    completely independent ``:memory:`` database to run ``weekly`` against,
    so it could compare two truly separate ingests. Against one shared
    Postgres instance there is no second independent database to open - so
    this test now runs backfill, captures its row, truncates the tables,
    then runs weekly on the SAME connection/schema and captures its row.
    This is arguably more representative of real usage (there is only ever
    one production database; backfill and weekly are never actually run
    against two different databases), but it IS a different mechanism for
    getting "two independent ingests" than the sqlite version used, so it's
    called out here rather than left silent.

Neither change touches what any test asserts - same expected columns, same
expected price behaviour, same expected counts.

Covers the three things the task asks for:
  1. Idempotency: ingesting the same gameweek twice doesn't duplicate rows.
  2. Cold-start is covered separately in features/test_engineering.py.
  3. Backfill vs. weekly mode produce the same silver row shape for a
     gameweek both could cover.
"""

from __future__ import annotations

import pytest

from ingestion import backfill, fpl_client, silver, weekly
from ingestion.db import get_connection

TEST_SCHEMA = "test_ingestion"
_ALL_TABLES = [
    "bronze_responses", "teams", "players", "gameweeks",
    "fixtures", "player_gameweek_stats", "my_team_state", "chip_usage",
]


def _truncate_all(conn) -> None:
    conn.execute(f"TRUNCATE TABLE {', '.join(_ALL_TABLES)} RESTART IDENTITY CASCADE")
    conn.commit()

TEAMS = [
    {"id": 1, "name": "Team A", "short_name": "TMA", "strength": 4},
    {"id": 2, "name": "Team B", "short_name": "TMB", "strength": 3},
]
ELEMENTS = [
    {"id": 101, "web_name": "Player One", "first_name": "P", "second_name": "One",
     "team": 1, "element_type": 3, "now_cost": 75},
    {"id": 102, "web_name": "Player Two", "first_name": "P", "second_name": "Two",
     "team": 2, "element_type": 4, "now_cost": 60},
]
EVENTS = [
    {"id": 1, "name": "Gameweek 1", "deadline_time": "2025-08-15T10:00:00Z",
     "finished": True, "is_current": False, "average_entry_score": 50},
    {"id": 2, "name": "Gameweek 2", "deadline_time": "2025-08-22T10:00:00Z",
     "finished": True, "is_current": True, "average_entry_score": 45},
]
BOOTSTRAP = {"teams": TEAMS, "elements": ELEMENTS, "events": EVENTS}

FIXTURES = [
    {"id": 501, "event": 1, "team_h": 1, "team_a": 2, "team_h_score": 2, "team_a_score": 1,
     "kickoff_time": "2025-08-15T14:00:00Z", "finished": True,
     "team_h_difficulty": 2, "team_a_difficulty": 3},
    {"id": 502, "event": 2, "team_h": 2, "team_a": 1, "team_h_score": 0, "team_a_score": 0,
     "kickoff_time": "2025-08-22T14:00:00Z", "finished": True,
     "team_h_difficulty": 3, "team_a_difficulty": 2},
]


def _player_stats(minutes=90, total_points=6):
    return {
        "minutes": minutes, "total_points": total_points, "goals_scored": 1, "assists": 0,
        "clean_sheets": 0, "goals_conceded": 1, "own_goals": 0, "penalties_saved": 0,
        "penalties_missed": 0, "yellow_cards": 0, "red_cards": 0, "saves": 0, "bonus": 1,
        "bps": 30, "influence": 20.0, "creativity": 10.0, "threat": 15.0, "ict_index": 4.5,
        "starts": 1, "expected_goals": 0.5, "expected_assists": 0.1,
        "expected_goal_involvements": 0.6, "expected_goals_conceded": 1.2,
    }


def event_live(fixture_id: int) -> dict:
    return {
        "elements": [
            {"id": 101, "stats": _player_stats(), "explain": [{"fixture": fixture_id, "stats": []}]},
            {"id": 102, "stats": _player_stats(minutes=80, total_points=2), "explain": [{"fixture": fixture_id, "stats": []}]},
        ]
    }


ENTRY = {"id": 555, "summary_overall_points": 120, "summary_overall_rank": 500_000}
ENTRY_PICKS_GW2 = {
    "active_chip": None,
    "entry_history": {
        "event": 2, "points": 60, "total_points": 120, "rank": 100, "overall_rank": 500_000,
        "bank": 5, "value": 1005, "event_transfers": 1, "event_transfers_cost": 0, "points_on_bench": 2,
    },
    "picks": [
        {"element": 101, "position": 1, "multiplier": 1, "is_captain": True, "is_vice_captain": False},
        {"element": 102, "position": 2, "multiplier": 1, "is_captain": False, "is_vice_captain": True},
    ],
}
ENTRY_TRANSFERS = [
    {"element_in": 102, "element_in_cost": 60, "element_out": 103, "element_out_cost": 55,
     "entry": 555, "event": 2, "time": "2025-08-20T12:00:00Z"},
]
ENTRY_HISTORY = {
    "current": [], "past": [],
    "chips": [{"name": "wildcard", "time": "2025-08-18T09:00:00Z", "event": 2}],
}


def _element_summary(player_id: int) -> dict:
    return {
        "history": [
            {"round": 1, "value": 75, "selected": 100_000, "transfers_balance": 500},
            {"round": 2, "value": 76, "selected": 105_000, "transfers_balance": 300},
        ],
        "history_past": [],
        "fixtures": [],
    }


@pytest.fixture
def patched_client(monkeypatch):
    """Replace every fpl_client network call with the fixtures above."""
    monkeypatch.setattr(fpl_client, "get_bootstrap_static", lambda: BOOTSTRAP)
    monkeypatch.setattr(fpl_client, "get_fixtures", lambda: FIXTURES)
    monkeypatch.setattr(fpl_client, "get_event_live", lambda gw: event_live(501 if gw == 1 else 502))
    monkeypatch.setattr(fpl_client, "get_entry", lambda team_id: ENTRY)
    monkeypatch.setattr(fpl_client, "get_entry_picks", lambda team_id, gw: ENTRY_PICKS_GW2)
    monkeypatch.setattr(fpl_client, "get_entry_transfers", lambda team_id: ENTRY_TRANSFERS)
    monkeypatch.setattr(fpl_client, "get_entry_history", lambda team_id: ENTRY_HISTORY)
    monkeypatch.setattr(fpl_client, "get_element_summary", lambda player_id: _element_summary(player_id))
    monkeypatch.setenv("FPL_TEAM_ID", "555")


@pytest.fixture
def conn():
    c = get_connection(schema=TEST_SCHEMA)
    _truncate_all(c)
    yield c
    c.close()


def _row_count(conn, gw: int) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM player_gameweek_stats WHERE gw = %s", (gw,)
    ).fetchone()[0]


def test_idempotent_backfill_does_not_duplicate_rows(patched_client, conn):
    backfill.backfill_current_season(conn, upto_gw=1)
    assert _row_count(conn, 1) == 2  # 2 players landed once

    backfill.backfill_current_season(conn, upto_gw=1)
    assert _row_count(conn, 1) == 2  # re-run: replaced, not duplicated


def test_idempotent_direct_silver_upsert(patched_client, conn):
    """Same guarantee at the unit level: calling the upsert function twice
    with identical input never grows the table."""
    live = event_live(501)
    silver.upsert_teams(conn, BOOTSTRAP)
    silver.upsert_players(conn, BOOTSTRAP)
    silver.upsert_fixtures(conn, FIXTURES)

    n1 = silver.upsert_player_gameweek_stats(conn, 1, live)
    n2 = silver.upsert_player_gameweek_stats(conn, 1, live)
    assert n1 == n2 == 2
    assert _row_count(conn, 1) == 2


def test_backfill_vs_weekly_same_row_shape(patched_client, conn):
    """Backfill (gw1+gw2) and weekly (gw2 only) both touch gw2 - the rows
    they produce for that gameweek must have the same columns, even though
    weekly fills in `price` (from a fresh bootstrap-static pull) where
    backfill leaves it NULL (see the docstrings on both modules)."""
    backfill.backfill_current_season(conn, upto_gw=2)
    backfill_row = conn.execute(
        "SELECT * FROM player_gameweek_stats WHERE gw = 2 AND player_id = 101"
    ).fetchone()

    # See the module docstring's MIGRATION NOTE: no second independent
    # database to open against Postgres, so we clear the tables and run
    # weekly on the same connection/schema instead.
    _truncate_all(conn)
    weekly.run_weekly(conn)
    weekly_row = conn.execute(
        "SELECT * FROM player_gameweek_stats WHERE gw = 2 AND player_id = 101"
    ).fetchone()

    # NOTE: psycopg2.extras.DictRow.keys() returns a one-shot OrderedDict
    # key ITERATOR, not a list/view like sqlite3.Row.keys() did - comparing
    # two of those directly with `==` is always False (identity, not
    # content), regardless of whether the columns actually match. list(...)
    # here restores the same ordered-list comparison the sqlite version
    # actually ran; it is not a weaker check, just one written so it
    # actually invokes equality instead of silently always failing/passing
    # on iterator identity.
    assert list(backfill_row.keys()) == list(weekly_row.keys())
    # Backfill never knows a historical price; weekly does (fresh bootstrap
    # pull for the gameweek that just finished) - this is the one expected
    # difference in VALUE, not shape.
    assert backfill_row["price"] is None
    # 7.6, not the bootstrap-snapshot 7.5: player 101 is in our squad, so
    # weekly's element-summary refinement (round 2's `value`) overwrites the
    # bootstrap-static price with the more precise per-gameweek figure - see
    # silver.refine_player_gameweek_from_element_summary.
    assert weekly_row["price"] == pytest.approx(7.6)


def test_weekly_populates_my_team_state(patched_client, conn):
    summary = weekly.run_weekly(conn)
    assert summary["latest_finished_gw"] == 2
    assert summary["my_team_state_gw"] == 2
    assert summary["squad_size"] == 2

    row = conn.execute("SELECT * FROM my_team_state WHERE gw = 2").fetchone()
    assert row is not None
    assert row["bank"] == pytest.approx(0.5)


def test_weekly_populates_chip_usage_from_entry_history(patched_client, conn):
    """The real gap this covers: chip_usage must come from entry/history's
    complete `chips` list, not be silently empty just because this is the
    first week the job has ever run for this account (my_team_state.active_chip
    for THIS gameweek is None in the fixture - see ENTRY_PICKS_GW2 - so a
    my_team_state-derived answer would have missed the wildcard entirely)."""
    weekly.run_weekly(conn)
    row = conn.execute("SELECT * FROM chip_usage WHERE chip = 'wildcard' AND gw = 2").fetchone()
    assert row is not None
    assert row["played_at"] == "2025-08-18T09:00:00Z"


def test_idempotent_chip_usage_upsert(patched_client, conn):
    silver.upsert_chip_usage(conn, ENTRY_HISTORY["chips"])
    silver.upsert_chip_usage(conn, ENTRY_HISTORY["chips"])
    count = conn.execute("SELECT COUNT(*) FROM chip_usage").fetchone()[0]
    assert count == 1  # re-upserting the same (chip, gw) replaces, not duplicates
